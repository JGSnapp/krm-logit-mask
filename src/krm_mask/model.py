from __future__ import annotations

from dataclasses import dataclass
import math
from math import exp
from typing import Iterable


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class LocalCausalLM:
    model: object
    tokenizer: object

    @classmethod
    def load(cls, model_name: str = DEFAULT_MODEL) -> "LocalCausalLM":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=False,
        )
        model.eval()
        return cls(model=model, tokenizer=tokenizer)

    @property
    def device(self) -> object:
        return next(self.model.parameters()).device

    def prompt_ids(self, messages: list[dict[str, str]]) -> object:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def generate(
        self,
        input_ids: object,
        processors: Iterable[object],
        max_new_tokens: int,
        return_scores: bool = False,
    ) -> list[int] | tuple[list[int], list[dict[str, float | int]]]:
        import torch
        from transformers import LogitsProcessorList

        result = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=LogitsProcessorList(list(processors)),
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=return_scores,
            output_scores=return_scores,
        )
        if not return_scores:
            return result[0, input_ids.shape[1] :].tolist()

        generated = result.sequences[0, input_ids.shape[1] :].tolist()
        step_scores: list[dict[str, float | int]] = []
        for token_id, logits in zip(generated, result.scores):
            probabilities = torch.softmax(logits[0].float(), dim=-1)
            chosen_probability = float(probabilities[token_id].item())
            top_probability, top_id = probabilities.max(dim=-1)
            finite_count = int(torch.isfinite(logits[0]).sum().item())
            step_scores.append(
                {
                    "probability": chosen_probability,
                    "log_probability": float(
                        torch.log(probabilities[token_id].clamp_min(1e-30)).item()
                    ),
                    "top_probability": float(top_probability.item()),
                    "top_token_id": int(top_id.item()),
                    "allowed_count_from_scores": finite_count,
                }
            )
        return generated, step_scores

    def score_options(self, prompt: str, options: list[str]) -> dict[str, float]:
        """Length-normalized conditional log likelihood; no free-form label output."""
        import torch
        import torch.nn.functional as functional

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        scores: dict[str, float] = {}
        with torch.inference_mode():
            for option in options:
                option_ids = self.tokenizer(
                    option, add_special_tokens=False, return_tensors="pt"
                ).input_ids.to(self.device)
                joined = torch.cat((prompt_ids, option_ids), dim=1)
                logits = self.model(joined).logits[:, :-1]
                start = prompt_ids.shape[1] - 1
                option_logits = logits[:, start : start + option_ids.shape[1]]
                log_probs = functional.log_softmax(option_logits, dim=-1)
                gathered = log_probs.gather(-1, option_ids.unsqueeze(-1)).squeeze(-1)
                scores[option] = float(gathered.mean().item())
        return scores

    def sample_decision_path(
        self, input_ids: object, mask: object, max_new_tokens: int,
    ) -> tuple[list[int], list[dict[str, object]]]:
        """Sample one path under the mask and score it the way DISC needs.

        DISC (Ye et al., 2025, Alg. 1, line 8) multiplies the legal probability
        mass of every step. At a forced step that mass is the model probability
        of the only legal token, so forced tokens are part of the weight. They
        get no forward of their own: they are fed with the next forward, and
        their probabilities are read from its per-position logits.

        The mask may stop early, once the class or phrase is already unique.
        The unwritten rest of the phrase is then scored in one extra forward
        (log-sum-exp over all compatible completions). Without it, the prefix
        probability would still include continuations outside the graph.
        The terminal EOS is a resolver control token and has unit weight.
        """
        import torch
        from .disc import DiscDecodingError

        prompt_length = int(input_ids.shape[1])
        generated: list[int] = []
        scores: list[dict[str, object]] = []
        pending: list[int] = []  # forced tokens whose probability is not read yet
        past = None
        fed = 0  # prompt + generated positions already inside the KV cache
        last_log_probs = None

        def forward(tokens: object, keep: int) -> object:
            nonlocal past
            output = self.model(
                input_ids=tokens, past_key_values=past, use_cache=True,
                logits_to_keep=keep,
            )
            past = output.past_key_values
            return torch.log_softmax(output.logits[0, -keep:].float(), dim=-1)

        def advance(upto: int) -> None:
            """Feed generated[:upto]; read pending forced tokens and next-token log-probs."""
            nonlocal fed, last_log_probs
            suffix = torch.tensor(
                [generated[:upto]], device=input_ids.device, dtype=input_ids.dtype
            )
            joined = torch.cat((input_ids, suffix), dim=1)
            total = int(joined.shape[1])
            if total <= fed:
                return
            # Output token i sits at position prompt_length + i, so the logits
            # that predict it are at position prompt_length + i - 1.
            first_needed = min(
                [prompt_length + index - 1 for index in pending
                 if prompt_length + index - 1 >= fed] + [total - 1]
            )
            rows = forward(joined[:, fed:], total - first_needed)
            for index in pending:
                position = prompt_length + index - 1
                row = last_log_probs if position < first_needed else rows[position - first_needed]
                log_probability = float(row[generated[index]].item())
                if not math.isfinite(log_probability):
                    raise DiscDecodingError("Forced token has zero model probability")
                scores[index]["disc_log_mass"] = min(0.0, log_probability)
                scores[index]["model_log_probability"] = log_probability
                scores[index]["model_probability"] = exp(min(0.0, log_probability))
            pending.clear()
            fed = total
            last_log_probs = rows[-1]

        def score_tails(tails: list[tuple[int, ...]]) -> float:
            """log P(any compatible completion | path); one short forward per tail."""
            base_length = past.get_seq_length()
            values: list[float] = []
            for tail in tails:
                value = float(last_log_probs[tail[0]].item())
                if len(tail) > 1:
                    tokens = torch.tensor(
                        [list(tail[:-1])], device=input_ids.device, dtype=input_ids.dtype
                    )
                    rows = forward(tokens, len(tail) - 1)
                    value += sum(
                        float(rows[j, tail[j + 1]].item()) for j in range(len(tail) - 1)
                    )
                    # Negative = number of tokens to drop; accepted by old and new transformers.
                    past.crop(base_length - past.get_seq_length())
                values.append(value)
            peak = max(values)
            if not math.isfinite(peak):
                raise DiscDecodingError("Completion has zero model probability")
            return peak + math.log(sum(math.exp(value - peak) for value in values))

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                allowed = sorted(mask.allowed_tokens(generated))
                # A resolved phrase/class ends immediately. Abort is an escape
                # during selection, not a new decision after resolution.
                if set(allowed) - mask.abort_token_ids == {mask.eos_token_id}:
                    allowed = [mask.eos_token_id]
                if not allowed:
                    raise DiscDecodingError("No legal continuation")
                if len(allowed) == 1:
                    token = allowed[0]
                    step = {
                        "forced": True, "autocompleted": True,
                        "disc_log_mass": 0.0, "allowed_count_from_scores": 1,
                    }
                    if token == mask.eos_token_id:
                        step["terminal"] = True
                    else:
                        pending.append(len(generated))
                else:
                    advance(len(generated))
                    log_probs = last_log_probs
                    indices = torch.tensor(allowed, device=log_probs.device)
                    legal_log_probs = log_probs[indices]
                    log_mass = torch.logsumexp(legal_log_probs, dim=0)
                    if not torch.isfinite(log_mass):
                        raise DiscDecodingError("Non-finite legal token probabilities")
                    probabilities = torch.softmax(legal_log_probs, dim=0)
                    choice = int(torch.multinomial(probabilities, 1).item())
                    token = allowed[choice]
                    top = int(probabilities.argmax().item())
                    step = {
                        "forced": False, "probability": float(probabilities[choice].item()),
                        "log_probability": float((legal_log_probs[choice] - log_mass).item()),
                        "top_probability": float(probabilities[top].item()),
                        "top_token_id": allowed[top],
                        "allowed_count_from_scores": len(allowed),
                        "disc_log_mass": min(0.0, float(log_mass.item())),
                        "disc_allowed_mass": exp(min(0.0, float(log_mass.item()))),
                    }
                generated.append(token)
                scores.append(step)
                if token == mask.eos_token_id:
                    completion_tails = getattr(mask, "completion_tails", None)
                    tails = [
                        tuple(tail)
                        for tail in (completion_tails(generated) if completion_tails else [])
                        if tail
                    ]
                    if pending or tails:
                        advance(len(generated) - 1)
                    if tails:
                        tail_log = min(0.0, score_tails(tails))
                        step["disc_log_mass"] = tail_log
                        step["tail_log_probability"] = tail_log
                        step["tail_probability"] = exp(tail_log)
                        step["tail_count"] = len(tails)
                    return generated, scores
        raise DiscDecodingError("Token limit reached before terminal state")

