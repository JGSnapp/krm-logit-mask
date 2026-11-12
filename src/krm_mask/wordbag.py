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
SENTENCE_RE = re.compile(r"\.\s+(?=[A-ZА-ЯЁ])")


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
            # Some topic cells list several topics separated by dots.
            for sentence in SENTENCE_RE.split(phrase):
                lemmas: set[str] = set()
                for word in split_words(sentence):
                    lemma = lemmatize(word)
                    if not lemma:
                        continue
                    lemmas.add(lemma)
                    for surface in (word, word.lower(), lemma):
                        surfaces.setdefault(surface, lemma)
                if not lemmas:
                    continue
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
    # Path search (experiment 6): alternative branches between words.
    path_beam: int = 6  # word paths kept after each step
    word_branches: int = 3  # next words tried from each path
    min_words: int = 1  # words a path needs before it may end on one competency
    # PMI floor (experiment 7): log P(word | empty text) is clipped from below,
    # so very rare words do not get an unbounded bonus. None = no floor.
    prior_floor: float | None = None
    # Phrase coherence (experiment 7): words come in signals; all words of a
    # signal must occur in one catalog phrase; competencies are intersected
    # across signals, as in the phrase-signal mode of the pipeline.
    words_per_signal: int = 2
    min_signals: int = 2
    max_signals: int = 4


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


@dataclass
class WordPath:
    words: list[WordChoice]
    candidates: frozenset[str]
    context: list[int]
    score: float

    @property
    def lemmas(self) -> frozenset[str]:
        return frozenset(word.lemma for word in self.words)


@dataclass
class SignalPath(WordPath):
    signals: tuple[frozenset[str], ...] = ()  # lemmas of closed signals
    current: frozenset[str] = frozenset()  # lemmas of the open signal
    phrases: frozenset[int] = frozenset()  # phrases holding all lemmas of the open signal


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
        # Signals are separated by an empty line (one more delimiter after the word's own).
        self.signal_separator: tuple[int, ...] = tuple(delimiter)
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

    def select_paths(
        self, text: str, count: int, blocked: set[str] | None = None, coherent: bool = False,
    ) -> tuple[list[tuple[str, WordPath]], int]:
        """Up to `count` competencies, ranked by their best finished word path.

        One search usually yields several competencies; it is repeated with the
        found ones blocked only if it ended with fewer than `count`. With
        `coherent`, words are grouped into signals from single phrases.
        """
        blocked = set(blocked or ())
        found: list[tuple[str, WordPath]] = []
        calls = 0
        search = self._signal_search if coherent else self._path_search
        while len(found) < count:
            finished, used = search(text, blocked)
            calls += used
            best: dict[str, WordPath] = {}
            for path in finished:
                (competency_id,) = path.candidates
                if competency_id not in best or path.score > best[competency_id].score:
                    best[competency_id] = path
            ranked = sorted(best.items(), key=lambda item: -item[1].score)
            if not ranked:
                break
            for competency_id, path in ranked[: count - len(found)]:
                found.append((competency_id, path))
                blocked.add(competency_id)
        return found, calls

    def _path_search(self, text: str, blocked: set[str]) -> tuple[list[WordPath], int]:
        """Beam search over word sequences: alternative branches between words.

        Paths are compared by the sum of their word scores, so a first word that
        looks good alone but is poorly confirmed by the next words loses.
        """
        config = self.config
        start = WordPath(
            [], frozenset(set(self.index.competency_lemmas) - blocked),
            list(self.prompt(text)), 0.0,
        )
        live = [start]
        finished: dict[frozenset[str], WordPath] = {}
        calls = 0
        for _ in range(config.max_words):
            children: dict[frozenset[str], WordPath] = {}
            for path in live:
                used = path.lemmas
                if len(path.candidates) > 1:
                    allowed = self._narrowing_lemmas(set(path.candidates), set(used))
                else:  # Confirmation words of the only remaining competency.
                    (only,) = path.candidates
                    allowed = self.index.competency_lemmas[only] - used
                if not allowed:
                    continue
                options, used_calls = self._word_options(path.context, allowed, config.word_branches)
                calls += used_calls
                for option in options:
                    candidates = frozenset(
                        c for c in path.candidates
                        if option.lemma in self.index.competency_lemmas[c]
                    )
                    if not candidates:
                        continue
                    word = WordChoice(**{**option.__dict__, "candidates_after": len(candidates)})
                    child = WordPath(
                        path.words + [word], candidates,
                        path.context + list(self.tokens[option.word]), path.score + option.score,
                    )
                    key = child.lemmas  # Same words in another order = same path.
                    if len(candidates) == 1 and len(child.words) >= config.min_words:
                        if key not in finished or child.score > finished[key].score:
                            finished[key] = child
                    elif key not in children or child.score > children[key].score:
                        children[key] = child
            live = sorted(children.values(), key=lambda p: -p.score)[: config.path_beam]
            if not live:
                break
        return sorted(finished.values(), key=lambda p: -p.score), calls

    def _signal_search(self, text: str, blocked: set[str]) -> tuple[list[WordPath], int]:
        """Path search where words form signals taken from single catalog phrases.

        Inside a signal the words come in any order but must all occur in one
        phrase. A signal closes after `words_per_signal` words (or when its
        phrases have no unused words left); the competencies of its phrases are
        then intersected with the path's candidates. A path finishes on one
        competency with at least `min_signals` signals. Branches exist both
        between words of a signal and between signals.
        """
        config = self.config
        index = self.index
        start = SignalPath(
            [], frozenset(set(index.competency_lemmas) - blocked), list(self.prompt(text)), 0.0,
        )
        live: list[SignalPath] = [start]
        finished: dict[tuple, SignalPath] = {}
        calls = 0
        for _ in range(config.max_signals * config.words_per_signal):
            children: dict[tuple, SignalPath] = {}
            for path in live:
                phrases = path.phrases if path.current else self._open_phrases(path)
                used = path.lemmas
                allowed = {
                    lemma for phrase in phrases for lemma in index.phrase_lemmas[phrase][1]
                } - used
                if not allowed:
                    continue
                options, used_calls = self._word_options(path.context, allowed, config.word_branches)
                calls += used_calls
                for option in options:
                    kept = frozenset(p for p in phrases if option.lemma in index.phrase_lemmas[p][1])
                    current = path.current | {option.lemma}
                    left = {
                        lemma for phrase in kept for lemma in index.phrase_lemmas[phrase][1]
                    } - used - {option.lemma}
                    closes = len(current) >= config.words_per_signal or not left
                    context = path.context + list(self.tokens[option.word])
                    candidates, signals = path.candidates, path.signals
                    if closes:
                        competencies = {c for p in kept for c in index.phrase_lemmas[p][0]}
                        candidates = frozenset(path.candidates & competencies)
                        if not candidates:
                            continue
                        signals = signals + (current,)
                        context += list(self.signal_separator)
                    word = WordChoice(**{**option.__dict__, "candidates_after": len(candidates)})
                    child = SignalPath(
                        path.words + [word], candidates, context, path.score + option.score,
                        signals=signals,
                        current=frozenset() if closes else current,
                        phrases=frozenset() if closes else kept,
                    )
                    key = (frozenset(child.signals), child.current)
                    if closes and len(candidates) == 1 and len(signals) >= config.min_signals:
                        if key not in finished or child.score > finished[key].score:
                            finished[key] = child
                    elif closes and len(signals) >= config.max_signals:
                        continue
                    elif key not in children or child.score > children[key].score:
                        children[key] = child
            live = sorted(children.values(), key=lambda p: -p.score)[: config.path_beam]
            if not live:
                break
        return sorted(finished.values(), key=lambda p: -p.score), calls

    def _open_phrases(self, path: "SignalPath") -> frozenset[int]:
        """Phrases a new signal may come from: phrases of the remaining candidates,
        except phrases that already contain an earlier signal (no repeating it)."""
        return frozenset(
            i for i, (competencies, lemmas) in enumerate(self.index.phrase_lemmas)
            if competencies & path.candidates
            and not any(signal <= lemmas for signal in path.signals)
        )

    def _choose_word(self, context: list[int], allowed: set[str]) -> tuple[WordChoice | None, int]:
        options, calls = self._word_options(context, allowed, 1)
        return (options[0] if options else None), calls

    def _word_options(
        self, context: list[int], allowed: set[str], top: int,
    ) -> tuple[list[WordChoice], int]:
        """Best `top` complete words (distinct lemmas) by token-level beam search."""
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
            prior = self.prior.get(node, 0.0)
            if config.prior_floor is not None:
                prior = max(prior, config.prior_floor)
            return log_prob[node] - config.alpha * prior + config.beta * best_idf[node]

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
        options: list[WordChoice] = []
        seen: set[str] = set()
        for node in sorted(finished, key=score, reverse=True):
            surface = terminal[node]
            lemma = self.index.surfaces[surface]
            if lemma in seen:
                continue
            seen.add(lemma)
            options.append(WordChoice(
                word=surface, lemma=lemma, log_probability=log_prob[node],
                log_prior=self.prior.get(node, 0.0), idf=self.index.idf.get(lemma, 0.0),
                score=score(node), candidates_after=0,
            ))
            if len(options) == top:
                break
        return options, calls


def lm_prompt_tensor(lm: object, ids: list[int]) -> object:
    import torch

    return torch.tensor([ids], device=lm.device)
