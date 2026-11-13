"""Cheap shortlist of competencies from next-token distributions (experiments 10-11).

Word order does not matter, so any catalog word could start the keyword list.
The model writes a free keyword list (greedy, one word per line); at every line
start the whole next-token distribution is read, so every catalog word gets the
best log-probability of its first token over all line starts. The best first
tokens are refined with their second token in one batched forward. Word values
are PMI-corrected with the priors of the word-bag selector and aggregated per
competency. The cost does not depend on the catalog size.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .wordbag import WordBagSelector, lm_prompt_tensor


@dataclass
class FirstPassEvidence:
    first: object  # tensor [vocab]: best line-start log-prob of every token
    pair_logprob: dict[tuple[int, int], float]  # log P(second | first) at the list start
    free_list: str
    forward_calls: int


class FirstPassScorer:
    def __init__(self, selector: WordBagSelector, list_tokens: int = 96, expand: int = 128):
        self.selector = selector
        self.lm = selector.lm
        self.index = selector.index
        self.list_tokens = list_tokens
        self.expand = expand
        self.word_tokens = {s: seq[:-1] for s, seq in selector.tokens.items()}  # no delimiter
        self.lemma_surfaces: dict[str, list[str]] = defaultdict(list)
        for surface, lemma in self.index.surfaces.items():
            if self.word_tokens[surface]:
                self.lemma_surfaces[lemma].append(surface)
        self.phrases_of: dict[str, list[set[str]]] = defaultdict(list)
        for competency_ids, lemmas in self.index.phrase_lemmas:
            for competency_id in competency_ids:
                self.phrases_of[competency_id].append(lemmas)
        self.second_tokens: dict[int, list[int]] = defaultdict(list)
        for seq in self.word_tokens.values():
            if len(seq) > 1 and seq[1] not in self.second_tokens[seq[0]]:
                self.second_tokens[seq[0]].append(seq[1])

    def evidence(self, text: str) -> FirstPassEvidence:
        import torch

        lm = self.lm
        tokenizer = lm.tokenizer
        context = self.selector.prompt(text)
        newline = tokenizer.encode("\n", add_special_tokens=False)[0]
        ids = lm_prompt_tensor(lm, context)
        past, rows, generated, at_start, calls = None, [], [], True, 0
        with torch.inference_mode():
            for _ in range(self.list_tokens):
                out = lm.model(input_ids=ids, past_key_values=past, use_cache=True)
                calls += 1
                past = out.past_key_values
                logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                if at_start:
                    rows.append(logp)
                token = int(logp.argmax())
                if token in (tokenizer.all_special_ids or [tokenizer.eos_token_id]):
                    break
                piece = tokenizer.decode([token])
                if "," in piece or ";" in piece:  # One word per line, as for the priors.
                    token, piece = newline, "\n"
                at_start = "\n" in piece
                generated.append(token)
                ids = torch.tensor([[token]], device=lm.device)
        first = torch.stack(rows).max(dim=0).values
        ranked = sorted(self.second_tokens, key=lambda t: float(first[t]), reverse=True)
        expanded = ranked[: self.expand]
        wanted = [self.second_tokens[t] for t in expanded]
        pair_logprob: dict[tuple[int, int], float] = {}
        if expanded:
            second = lm.next_token_logprobs(context, [(t,) for t in expanded], wanted)
            calls += 1
            pair_logprob = {
                (t, t2): v for t, tokens, values in zip(expanded, wanted, second)
                for t2, v in zip(tokens, values)
            }
        return FirstPassEvidence(
            first.cpu(), pair_logprob, tokenizer.decode(generated, skip_special_tokens=True), calls,
        )

    def word_values(
        self, evidence: FirstPassEvidence, alpha: float = 1.0, beta: float = 0.3, depth: int = 2,
    ) -> dict[str, float]:
        prior = self.selector.prior
        first = evidence.first

        def prefix_value(seq: tuple[int, ...]) -> float:
            if depth == 2 and len(seq) > 1 and (seq[0], seq[1]) in evidence.pair_logprob:
                prefix = seq[:2]
                logp = float(first[seq[0]]) + evidence.pair_logprob[(seq[0], seq[1])]
            else:
                prefix = seq[:1]
                logp = float(first[seq[0]])
            return logp - alpha * prior.get(prefix, 0.0)

        return {
            lemma: max(prefix_value(self.word_tokens[s]) for s in surfaces)
            + beta * self.index.idf.get(lemma, 0.0)
            for lemma, surfaces in self.lemma_surfaces.items()
        }

    def competency_scores(
        self, values: dict[str, float], aggregate: str = "bag", k: int = 4,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for competency_id, lemmas in self.index.competency_lemmas.items():
            if aggregate == "bag":
                top = sorted((values[l] for l in lemmas if l in values), reverse=True)[:k]
            else:  # "phrases": top-2 words inside a phrase, then top-2 phrases.
                phrase_values = []
                for phrase in self.phrases_of[competency_id]:
                    inside = sorted((values[l] for l in phrase if l in values), reverse=True)[:2]
                    if inside:
                        phrase_values.append(sum(inside) / len(inside))
                top = sorted(phrase_values, reverse=True)[:2]
            if top:
                scores[competency_id] = sum(top) / len(top)
        return scores
