"""Competency selection without the catalog in the prompt (experiments 7-22).

1. Word-signal search (wordbag.py, experiments 7/9/20): the model writes key
   words of the resume under a mask of catalog words; words come in pairs from
   one catalog phrase and their competencies are intersected. It proposes a
   pool of candidates (15 by default) and never sees competency names.
2. Check (experiment 19): for each candidate one simple chat question, "Имеет
   ли кандидат указанную компетенцию? Ответь ДА или НЕТ.", read from the first
   answer token in one forward. Candidates are re-ranked by the margin
   log P(ДА) - log P(НЕТ); the best ones with ДА are selected.
3. Quote (experiment 22): for each selected competency the model copies the
   fragment of the resume that confirms it best; a source-span mask keeps the
   output verbatim.

The number of model calls depends on the resume and the pool size, not on the
size of the catalog.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from time import perf_counter

from .catalog import Catalog, Competency
from .graph import NONE_ID, PhraseGraph
from .masking import SourceSpanMask
from .wordbag import WordBagConfig, WordBagSelector, WordIndex, pymorphy_lemmatizer, word_messages

QUESTION = "Имеет ли кандидат указанную компетенцию? Ответь ДА или НЕТ."
QUOTE_REQUEST = (
    "Выведи из текста цитату, подтверждающую её больше всего. "
    "Пиши сразу дословно, как в тексте."
)
YES = ["ДА", "Да", "да"]
NO = ["НЕТ", "Нет", "нет"]


@dataclass
class SelectionConfig:
    pool: int = 15  # candidates proposed by the word search
    max_competencies: int = 3
    min_margin: float = 0.0  # log P(ДА) - log P(НЕТ) needed to confirm
    quotes: bool = True
    max_quote_tokens: int = 96


SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")


def split_sentences(text: str, min_chars: int = 12) -> list[tuple[int, int]]:
    """(start, end) of sentences and lines; very short pieces join the next one."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_END.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    merged: list[tuple[int, int]] = []
    pending: int | None = None
    for begin, end in spans:
        piece = text[begin:end].strip()
        if not piece:
            continue
        begin += len(text[begin:end]) - len(text[begin:end].lstrip())
        end -= len(text[begin:end]) - len(text[begin:end].rstrip())
        if pending is not None:
            begin = pending
            pending = None
        if end - begin < min_chars:
            pending = begin
            continue
        merged.append((begin, end))
    if pending is not None:
        if merged:
            merged[-1] = (merged[-1][0], len(text.rstrip()))
        else:
            merged.append((pending, len(text.rstrip())))
    return merged


def check_messages(text: str, competency_name: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": (
        f"{QUESTION}\nПрофиль кандидата: {text}\nКомпетенция: {competency_name}"
    )}]


def quote_messages(text: str, competency_name: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": (
            f"{QUESTION}\nПрофиль кандидата: {text}\nКомпетенция: {competency_name}"
        )},
        {"role": "assistant", "content": "ДА"},
        {"role": "user", "content": QUOTE_REQUEST},
    ]


class CompetencySelector:
    def __init__(self, lm: object, catalog: Catalog, config: SelectionConfig | None = None):
        self.lm = lm
        self.catalog = catalog
        self.config = config or SelectionConfig()
        self.competencies: dict[str, Competency] = {c.id: c for c in catalog.competencies}
        graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
        self.index = WordIndex.build(graph, pymorphy_lemmatizer())
        self.words = WordBagSelector(
            lm, self.index, self._word_prompt,
            WordBagConfig(alpha=0.0, beta=0.3, batch_paths=True),
        )
        tokenizer = lm.tokenizer
        self.yes_ids = sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in YES})
        self.no_ids = sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in NO})

    def _word_prompt(self, text: str) -> list[int]:
        return self.lm.prompt_ids(word_messages(text))[0].tolist()

    def check(self, text: str, competency: Competency) -> tuple[float, float]:
        """(margin, P(ДА) among ДА/НЕТ) for one competency, one forward."""
        import torch

        ids = self.lm.prompt_ids(check_messages(text, competency.name))
        with torch.inference_mode():
            logits = self.lm.model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        yes = torch.logsumexp(logp[self.yes_ids], 0)
        no = torch.logsumexp(logp[self.no_ids], 0)
        margin = float(yes - no)
        return margin, float(torch.sigmoid(yes - no))

    def quote(self, text: str, competency: Competency) -> dict[str, object]:
        """The sentence of `text` the model would answer with, verbatim.

        Every sentence of the resume is a possible answer to the quote request;
        its log P(sentence + end of turn | dialogue) is computed exactly for
        all sentences in one batch (teacher forcing) instead of greedy
        token-by-token copying. The answer is the sentence with the best mean
        log-probability per token; the plain sum prefers short generic
        sentences ("Знаю Python.") and is kept as `by_sum` (experiment 22).
        """
        sentences = split_sentences(text)
        if not sentences:
            return {"text": None, "start": None, "end": None}
        prompt = self.lm.prompt_ids(quote_messages(text, competency.name))
        tokenizer = self.lm.tokenizer
        eos = tokenizer.eos_token_id
        continuations = [
            tokenizer.encode(text[start:end], add_special_tokens=False) + [eos]
            for start, end in sentences
        ]
        totals = self.lm.score_continuations(prompt, continuations)
        means = [total / len(tokens) for total, tokens in zip(totals, continuations)]
        top = max(means)
        weights = [math.exp(mean - top) for mean in means]
        best = max(range(len(sentences)), key=lambda i: means[i])
        best_sum = max(range(len(sentences)), key=lambda i: totals[i])
        start, end = sentences[best]
        return {
            "text": text[start:end], "start": start, "end": end,
            "share": weights[best] / sum(weights),
            "mean_log_probability": means[best],
            "sentences": len(sentences),
            "by_sum": {"text": text[sentences[best_sum][0]:sentences[best_sum][1]],
                       "start": sentences[best_sum][0], "end": sentences[best_sum][1]},
        }

    def quote_span(self, text: str, competency: Competency) -> dict[str, object]:
        """Greedy copy under a source-span mask (first version, kept for comparison)."""
        prompt = self.lm.prompt_ids(quote_messages(text, competency.name))
        encoded = self.lm.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        source_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        mask = SourceSpanMask(
            source_ids, prompt_length=prompt.shape[1],
            eos_token_id=self.lm.tokenizer.eos_token_id,
            max_tokens=self.config.max_quote_tokens,
        )
        generated, steps = self.lm.generate(
            prompt, [mask], max_new_tokens=self.config.max_quote_tokens + 1, return_scores=True)
        eos = self.lm.tokenizer.eos_token_id
        quote_ids = generated[:-1] if generated and generated[-1] == eos else generated
        start = end = None
        occurrences = mask.occurrences(quote_ids)
        if occurrences and quote_ids:
            first = occurrences[0]
            start = offsets[first][0]
            end = offsets[first + len(quote_ids) - 1][1]
        probabilities = [step["probability"] for step in steps[: len(quote_ids)]]
        return {
            "text": text[start:end] if start is not None else
            self.lm.tokenizer.decode(quote_ids, skip_special_tokens=True).strip(),
            "start": start, "end": end, "tokens": len(quote_ids),
            "min_token_probability": min(probabilities) if probabilities else None,
        }

    def select(self, text: str, max_competencies: int | None = None) -> dict[str, object]:
        config = self.config
        limit = max_competencies or config.max_competencies
        timings: dict[str, float] = {}

        started = perf_counter()
        found, calls = self.words.select_paths(text, config.pool, coherent=True)
        timings["search_s"] = perf_counter() - started

        started = perf_counter()
        candidates = []
        for rank, (competency_id, path) in enumerate(found, start=1):
            competency = self.competencies[competency_id]
            margin, p_yes = self.check(text, competency)
            candidates.append({
                "id": competency_id, "name": competency.name, "search_rank": rank,
                "path_score": round(path.score, 3),
                "signals": [sorted(signal) for signal in path.signals],
                "words": [word.word for word in path.words],
                "margin": round(margin, 3), "p_yes": round(p_yes, 4),
                "confirmed": margin > config.min_margin,
            })
        timings["check_s"] = perf_counter() - started

        ranked = sorted(candidates, key=lambda item: -item["margin"])
        selected = [dict(item) for item in ranked if item["confirmed"]][:limit]
        started = perf_counter()
        if config.quotes:
            for item in selected:
                item["quote"] = self.quote(text, self.competencies[item["id"]])
        timings["quote_s"] = perf_counter() - started
        timings["total_s"] = sum(timings.values())
        return {
            "text": text,
            "catalog": {"framework_id": self.catalog.framework_id,
                        "competencies": len(self.catalog.competencies)},
            "config": asdict(config) | {"max_competencies": limit},
            "selected": selected or [{"id": NONE_ID, "name": "Подходящих компетенций не найдено"}],
            "candidates": ranked,
            "search_forward_calls": calls,
            "timings": timings,
        }
