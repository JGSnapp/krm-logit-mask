"""Competency selection without the catalog in the prompt (experiments 7-22).

1. Word-signal search (wordbag.py, experiments 7/9/20): the model writes key
   words of the resume under a mask of catalog words; words come in pairs from
   one catalog phrase and their competencies are intersected. It proposes a
   pool of candidates (15 by default) and never sees competency names.
2. Check (experiment 19): for each candidate one simple chat question, "Имеет
   ли кандидат указанную компетенцию? Ответь ДА или НЕТ.", read from the first
   answer token in one forward. Candidates are re-ranked by the margin
   log P(ДА) - log P(НЕТ); the best ones with ДА are selected.
3. Quote (experiments 22-23): for each selected competency the sentence of the
   resume the model would answer with, verbatim ("quote last", `select`).

"Quote first" (`select_quote_first`, experiment 25): the model first picks the
sentences that show a skill or experience, the word search runs on each of
them (narrowing to a group of competencies), all candidates are checked with
the same question on the whole resume, and a competency is quoted with the
sentence that proposed it.

The number of model calls depends on the resume and the pool size, not on the
size of the catalog.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace
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
# Reference for the contrastive quote score: the same request without a competency.
NEUTRAL_QUOTE_REQUEST = (
    "Выведи из текста цитату, которая лучше всего описывает опыт кандидата. "
    "Пиши сразу дословно, как в тексте."
)
# Quote-first mode: the sentences that show a skill or experience.
FRAGMENT_REQUEST = (
    "Выведи из текста цитату, которая показывает навык или опыт кандидата. "
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
    # Quote score per sentence: mean + w * (mean - neutral mean), where the
    # neutral mean is the same sentence under the request without a competency
    # (experiment 23). w = 0: the most likely sentence; large w: the sentence
    # whose probability grows most from the competency name.
    quote_contrast: float = 1.0
    # Quote-first mode (experiment 25): quotes chosen first, candidates per quote.
    first_quotes: int = 3
    per_quote: int = 5


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


def neutral_quote_messages(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": (
        f"Профиль кандидата: {text}\n{NEUTRAL_QUOTE_REQUEST}"
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
        self._neutral: tuple[str, list[tuple[int, int]], list[list[int]], list[float]] | None = None

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

    def _sentence_means(self, messages, continuations) -> list[float]:
        totals = self.lm.score_continuations(self.lm.prompt_ids(messages), continuations)
        return [total / len(tokens) for total, tokens in zip(totals, continuations)]

    def _neutral_scores(self, text: str):
        """Sentences of the resume and their mean log-probabilities under the
        neutral request; computed once per resume."""
        if self._neutral is None or self._neutral[0] != text:
            sentences = split_sentences(text)
            eos = self.lm.tokenizer.eos_token_id
            continuations = [
                self.lm.tokenizer.encode(text[start:end], add_special_tokens=False) + [eos]
                for start, end in sentences
            ]
            means = self._sentence_means(neutral_quote_messages(text), continuations) if sentences else []
            self._neutral = (text, sentences, continuations, means)
        return self._neutral[1:]

    def quote_scores(self, text: str, competency: Competency) -> dict[str, object]:
        """Per-sentence mean log-probabilities with the competency and neutral."""
        sentences, continuations, neutral = self._neutral_scores(text)
        means = self._sentence_means(quote_messages(text, competency.name), continuations) if sentences else []
        return {"sentences": sentences, "means": means, "neutral": neutral}

    def quote(self, text: str, competency: Competency) -> dict[str, object]:
        """The sentence of `text` that confirms the competency, verbatim.

        Every sentence of the resume is a possible answer to the quote request;
        its log P(sentence + end of turn | dialogue) per token is computed
        exactly for all sentences in one batch (teacher forcing). The plain
        most likely sentence tends to be the same for every competency of a
        resume (experiment 22), so the score adds how much the competency name
        raises the sentence over the same request without a competency:
            mean + w * (mean - neutral mean),  w = `quote_contrast`.
        """
        scores = self.quote_scores(text, competency)
        sentences, means, neutral = scores["sentences"], scores["means"], scores["neutral"]
        if not sentences:
            return {"text": None, "start": None, "end": None}
        weight = self.config.quote_contrast
        combined = [m + weight * (m - n) for m, n in zip(means, neutral)]
        top = max(combined)
        weights = [math.exp(value - top) for value in combined]
        best = max(range(len(sentences)), key=lambda i: combined[i])
        plain = max(range(len(sentences)), key=lambda i: means[i])
        start, end = sentences[best]
        return {
            "text": text[start:end], "start": start, "end": end,
            "share": weights[best] / sum(weights),
            "mean_log_probability": means[best],
            "lift": means[best] - neutral[best],
            "contrast_weight": weight,
            "sentences": len(sentences),
            "plain": {"text": text[sentences[plain][0]:sentences[plain][1]],
                      "start": sentences[plain][0], "end": sentences[plain][1]},
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
            "mode": "quote_last",
            "text": text,
            "catalog": {"framework_id": self.catalog.framework_id,
                        "competencies": len(self.catalog.competencies)},
            "config": asdict(config) | {"max_competencies": limit},
            "selected": selected or [{"id": NONE_ID, "name": "Подходящих компетенций не найдено"}],
            "candidates": ranked,
            "search_forward_calls": calls,
            "timings": timings,
        }

    def fragments(self, text: str, count: int) -> list[dict[str, object]]:
        """The `count` sentences the model would quote as showing a skill or
        experience (mean log-probability per token, one batch)."""
        sentences = split_sentences(text)
        if not sentences:
            return []
        eos = self.lm.tokenizer.eos_token_id
        continuations = [
            self.lm.tokenizer.encode(text[start:end], add_special_tokens=False) + [eos]
            for start, end in sentences
        ]
        messages = [{"role": "user", "content": f"Профиль кандидата: {text}\n{FRAGMENT_REQUEST}"}]
        means = self._sentence_means(messages, continuations)
        order = sorted(range(len(sentences)), key=lambda i: -means[i])[:count]
        return [
            {"text": text[sentences[i][0]:sentences[i][1]], "start": sentences[i][0],
             "end": sentences[i][1], "rank": rank, "mean_log_probability": round(means[i], 3)}
            for rank, i in enumerate(order, start=1)
        ]

    def select_quote_first(
        self, text: str, max_competencies: int | None = None,
    ) -> dict[str, object]:
        """Quotes first, then competencies for each quote, then the yes/no check."""
        config = self.config
        limit = max_competencies or config.max_competencies
        timings: dict[str, float] = {}

        started = perf_counter()
        quotes = self.fragments(text, config.first_quotes)
        timings["quotes_s"] = perf_counter() - started

        started = perf_counter()
        word_config = self.words.config
        # One signal is enough for a short quote; narrow to a group of <= 5.
        self.words.config = replace(word_config, group_size=5, min_signals=1, group_at_cap=5)
        found_by_quote = []
        calls = 0
        try:
            for quote in quotes:
                found, used = self.words.select_paths(quote["text"], config.per_quote, coherent=True)
                calls += used
                found_by_quote.append(found)
        finally:
            self.words.config = word_config
        timings["search_s"] = perf_counter() - started

        started = perf_counter()
        candidates: dict[str, dict[str, object]] = {}
        for quote, found in zip(quotes, found_by_quote):
            quote["candidates"] = []
            for competency_id, path in found:
                quote["candidates"].append(competency_id)
                if competency_id in candidates:
                    candidates[competency_id]["quotes"].append(quote["rank"])
                    continue
                competency = self.competencies[competency_id]
                margin, p_yes = self.check(text, competency)
                candidates[competency_id] = {
                    "id": competency_id, "name": competency.name,
                    "search_rank": len(candidates) + 1, "quote_rank": quote["rank"],
                    "quotes": [quote["rank"]],
                    "path_score": round(path.score, 3),
                    "signals": [sorted(signal) for signal in path.signals],
                    "words": [word.word for word in path.words],
                    "margin": round(margin, 3), "p_yes": round(p_yes, 4),
                    "confirmed": margin > config.min_margin,
                    "quote": {key: quote[key] for key in ("text", "start", "end", "rank")},
                }
        timings["check_s"] = perf_counter() - started
        timings["total_s"] = sum(timings.values())
        for quote in quotes:
            quote["margins"] = {c: candidates[c]["margin"] for c in quote["candidates"]}
        ranked = sorted(candidates.values(), key=lambda item: -item["margin"])
        selected = [dict(item) for item in ranked if item["confirmed"]][:limit]
        return {
            "mode": "quote_first",
            "text": text,
            "catalog": {"framework_id": self.catalog.framework_id,
                        "competencies": len(self.catalog.competencies)},
            "config": asdict(config) | {"max_competencies": limit},
            "quotes": quotes,
            "selected": selected or [{"id": NONE_ID, "name": "Подходящих компетенций не найдено"}],
            "candidates": ranked,
            "search_forward_calls": calls,
            "timings": timings,
        }
