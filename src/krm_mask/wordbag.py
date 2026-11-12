"""Word-bag selection (experiment 5): generate catalog words in any order.

Every catalog phrase becomes a set of lemmas (function words removed). The
model writes one word per line; the mask allows only words of competencies
that are still possible, and after each word only competencies containing all
chosen lemmas remain. Selection ends when one competency is left.

A word is chosen by beam search over the trie of allowed surface forms with
    score = log P(word | text) - alpha * log P(word | empty text) + beta * IDF(lemma)
PMI (the first two terms) removes words the model writes for any text; IDF
removes words shared by many competencies. Priors and IDF are computed once
per catalog, so the per-request cost does not grow with the catalog.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .graph import NONE_ID, PhraseGraph

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9+#\-]*")
STOP_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NPRO", "NUMR", "PRED"}
WORD_DELIMITER = "\n"


def word_messages(text: str) -> list[dict[str, str]]:
    """Selection prompt for the word mode; no competency names are given."""
    return [
        {
            "role": "system",
            "content": (
                "Выбери ключевые слова, которые описывают навыки, подтверждаемые "
                "текстом. Пиши по одному слову на строке, начиная с самого "
                "характерного."
            ),
        },
        {"role": "user", "content": f"Текст кандидата:\n{text}\n\nКлючевые слова:"},
    ]


def pymorphy_lemmatizer() -> Callable[[str], str | None]:
    """Lemma of a word, or None for function words."""
    import pymorphy3

    morph = pymorphy3.MorphAnalyzer()
    cache: dict[str, str | None] = {}

    def lemmatize(word: str) -> str | None:
        if word not in cache:
            # Latin terms and abbreviations ("ИИ", "БЯМ") are kept as is:
            # pymorphy turns "ИИ" into "ия".
            if re.fullmatch(r"[A-Za-z0-9+#\-]+", word) or (word.isupper() and len(word) <= 6):
                cache[word] = word.lower()
            else:
                parse = morph.parse(word)[0]
                cache[word] = None if parse.tag.POS in STOP_POS else parse.normal_form
        return cache[word]

    return lemmatize


def split_words(text: str) -> list[str]:
    words = [w.rstrip("-") for w in WORD_RE.findall(text)]
    return [w for w in words if len(w) >= 2 and not w.isdigit()]


@dataclass
class WordIndex:
    competency_lemmas: dict[str, set[str]]
    phrase_lemmas: list[tuple[set[str], set[str]]]  # (competency ids, lemmas) per phrase
    surfaces: dict[str, str]  # writable surface form -> lemma
    idf: dict[str, float]

    @classmethod
    def build(cls, graph: PhraseGraph, lemmatize: Callable[[str], str | None]) -> "WordIndex":
        competency_lemmas: dict[str, set[str]] = defaultdict(set)
        phrase_lemmas: list[tuple[set[str], set[str]]] = []
        surfaces: dict[str, str] = {}
        for phrase, competency_ids in graph.phrases.items():
            competency_ids = set(competency_ids) - {NONE_ID}
            if not competency_ids:
                continue
            lemmas: set[str] = set()
            for word in split_words(phrase):
                lemma = lemmatize(word)
                if not lemma:
                    continue
                lemmas.add(lemma)
                for surface in (word, word.lower(), lemma):
                    surfaces.setdefault(surface, lemma)
            for competency_id in competency_ids:
                competency_lemmas[competency_id].update(lemmas)
            phrase_lemmas.append((competency_ids, lemmas))
        total = len(competency_lemmas)
        frequency: dict[str, int] = defaultdict(int)
        for lemmas in competency_lemmas.values():
            for lemma in lemmas:
                frequency[lemma] += 1
        idf = {lemma: math.log(total / count) for lemma, count in frequency.items()}
        return cls(dict(competency_lemmas), phrase_lemmas, surfaces, idf)

    def best_phrase_coverage(self, competency_id: str, lemmas: Iterable[str]) -> int:
        wanted = set(lemmas)
        return max(
            (len(wanted & phrase) for ids, phrase in self.phrase_lemmas if competency_id in ids),
            default=0,
        )


@dataclass
class WordBagConfig:
    alpha: float = 1.0
    beta: float = 1.0
    beam: int = 8
    max_words: int = 6


@dataclass
class WordChoice:
    word: str
    lemma: str
    log_probability: float
    log_prior: float
    idf: float
    score: float
    candidates_after: int


@dataclass
class WordBagResult:
    competency_id: str | None
    words: list[WordChoice]
    resolved: bool
    forward_calls: int
    remaining: list[str] = field(default_factory=list)


class WordBagSelector:
    def __init__(
        self,
        lm: object,
        index: WordIndex,
        prompt: Callable[[str], list[int]],
        config: WordBagConfig | None = None,
    ) -> None:
        self.lm = lm
        self.index = index
        self.prompt = prompt
        self.config = config or WordBagConfig()
        tokenizer = lm.tokenizer
        delimiter = tokenizer.encode(WORD_DELIMITER, add_special_tokens=False)
        self.tokens: dict[str, tuple[int, ...]] = {
            surface: tuple(tokenizer.encode(surface, add_special_tokens=False) + delimiter)
            for surface in index.surfaces
        }
        # Prefix priors under the empty-text prompt, computed once per catalog.
        surfaces = list(self.tokens)
        per_token = lm.score_continuations(
            lm_prompt_tensor(lm, prompt("")), [list(self.tokens[s]) for s in surfaces],
            per_token=True,
        )
        self.prior: dict[tuple[int, ...], float] = {(): 0.0}
        for surface, values in zip(surfaces, per_token):
            sequence = self.tokens[surface]
            total = 0.0
            for depth, value in enumerate(values):
                total += value
                self.prior.setdefault(sequence[: depth + 1], total)

    def select(self, text: str, blocked: set[str]) -> WordBagResult:
        config = self.config
        candidates = set(self.index.competency_lemmas) - blocked
        context = list(self.prompt(text))
        chosen: list[WordChoice] = []
        calls = 0
        for _ in range(config.max_words):
            if len(candidates) <= 1:
                break
            allowed = self._narrowing_lemmas(candidates, {c.lemma for c in chosen})
            if not allowed:
                break
            choice, used = self._choose_word(context, allowed)
            calls += used
            if choice is None:
                break
            candidates = {
                c for c in candidates if choice.lemma in self.index.competency_lemmas[c]
            }
            choice.candidates_after = len(candidates)
            chosen.append(choice)
            context += list(self.tokens[choice.word])
        if len(candidates) == 1:
            return WordBagResult(next(iter(candidates)), chosen, True, calls)
        lemmas = [c.lemma for c in chosen]
        ranked = sorted(
            candidates,
            key=lambda c: (-self.index.best_phrase_coverage(c, lemmas),
                           len(self.index.competency_lemmas[c]), c),
        )
        return WordBagResult(ranked[0] if ranked else None, chosen, False, calls, ranked)

    def _narrowing_lemmas(self, candidates: set[str], used: set[str]) -> set[str]:
        """Lemmas of the candidates that would actually split them (not in all of them)."""
        counts: dict[str, int] = defaultdict(int)
        for competency_id in candidates:
            for lemma in self.index.competency_lemmas[competency_id]:
                counts[lemma] += 1
        return {
            lemma for lemma, count in counts.items()
            if count < len(candidates) and lemma not in used
        }

    def _choose_word(self, context: list[int], allowed: set[str]) -> tuple[WordChoice | None, int]:
        config = self.config
        children: dict[tuple[int, ...], set[int]] = defaultdict(set)
        terminal: dict[tuple[int, ...], str] = {}
        best_idf: dict[tuple[int, ...], float] = defaultdict(float)
        for surface, lemma in self.index.surfaces.items():
            if lemma not in allowed:
                continue
            sequence = self.tokens[surface]
            idf = self.index.idf.get(lemma, 0.0)
            for depth in range(len(sequence)):
                children[sequence[:depth]].add(sequence[depth])
                prefix = sequence[: depth + 1]
                best_idf[prefix] = max(best_idf[prefix], idf)
            terminal.setdefault(sequence, surface)

        def score(node: tuple[int, ...]) -> float:
            return (log_prob[node] - config.alpha * self.prior.get(node, 0.0)
                    + config.beta * best_idf[node])

        log_prob: dict[tuple[int, ...], float] = {(): 0.0}
        live: list[tuple[int, ...]] = [()]
        finished: list[tuple[int, ...]] = []
        calls = 0
        while live:
            wanted = [sorted(children[node]) for node in live]
            rows = self.lm.next_token_logprobs(context, live, wanted)
            calls += 1
            expanded: list[tuple[int, ...]] = []
            for node, tokens, values in zip(live, wanted, rows):
                for token, value in zip(tokens, values):
                    child = node + (token,)
                    log_prob[child] = log_prob[node] + value
                    expanded.append(child)
            expanded.sort(key=score, reverse=True)
            live = []
            for node in expanded[: config.beam]:
                if node in terminal:
                    finished.append(node)
                if children.get(node):
                    live.append(node)
        if not finished:
            return None, calls
        best = max(finished, key=score)
        surface = terminal[best]
        lemma = self.index.surfaces[surface]
        return WordChoice(
            word=surface, lemma=lemma, log_probability=log_prob[best],
            log_prior=self.prior.get(best, 0.0), idf=self.index.idf.get(lemma, 0.0),
            score=score(best), candidates_after=0,
        ), calls


def lm_prompt_tensor(lm: object, ids: list[int]) -> object:
    import torch

    return torch.tensor([ids], device=lm.device)
