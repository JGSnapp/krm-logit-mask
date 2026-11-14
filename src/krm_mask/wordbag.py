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
import random
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
    # Word positions of each lemma inside its phrase (parallel to phrase_lemmas).
    phrase_positions: list[dict[str, list[int]]] = field(default_factory=list)

    @classmethod
    def build(cls, graph: PhraseGraph, lemmatize: Callable[[str], str | None]) -> "WordIndex":
        competency_lemmas: dict[str, set[str]] = defaultdict(set)
        phrase_lemmas: list[tuple[set[str], set[str]]] = []
        phrase_positions: list[dict[str, list[int]]] = []
        surfaces: dict[str, str] = {}
        for phrase, competency_ids in graph.phrases.items():
            competency_ids = set(competency_ids) - {NONE_ID}
            if not competency_ids:
                continue
            # Some topic cells list several topics separated by dots.
            for sentence in SENTENCE_RE.split(phrase):
                lemmas: set[str] = set()
                positions: dict[str, list[int]] = defaultdict(list)
                for position, word in enumerate(split_words(sentence)):
                    lemma = lemmatize(word)
                    if not lemma:
                        continue
                    lemmas.add(lemma)
                    positions[lemma].append(position)
                    for surface in (word, word.lower(), lemma):
                        surfaces.setdefault(surface, lemma)
                if not lemmas:
                    continue
                for competency_id in competency_ids:
                    competency_lemmas[competency_id].update(lemmas)
                phrase_lemmas.append((competency_ids, lemmas))
                phrase_positions.append(dict(positions))
        total = len(competency_lemmas)
        frequency: dict[str, int] = defaultdict(int)
        for lemmas in competency_lemmas.values():
            for lemma in lemmas:
                frequency[lemma] += 1
        idf = {lemma: math.log(total / count) for lemma, count in frequency.items()}
        return cls(dict(competency_lemmas), phrase_lemmas, surfaces, idf, phrase_positions)

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
    # Evidence (experiment 12): `WordBagSelector.evidence` holds lemmas the
    # model named itself in a free keyword list and/or lemmas of the resume.
    # Evidence words get a bonus; with `require_evidence` every signal needs one.
    evidence_bonus: float = 0.0
    require_evidence: bool = False
    # Proximity (experiment 13): words of one signal must stand within this
    # many word positions of each other in the phrase. None = anywhere.
    signal_window: int | None = None
    # Anchors (experiment 14): a path may finish on a competency only after the
    # model wrote one of its key words (lemma in at most `anchor_df`
    # competencies), with PMI = log P(word | text) - log P(word | empty text)
    # of at least `anchor_min_pmi` (None = no threshold). None = no anchors.
    anchor_df: int | None = None
    anchor_min_pmi: float | None = None
    # Corridor search (experiment 17): branches whose mean log-mass per word
    # (share of the model's probability the mask left) falls more than
    # `corridor` below the second-best branch are dropped; free places are
    # refilled with new first words drawn in proportion to exp(score / T)
    # from a pool of `refill_pool` first words, without repeating a tried one.
    corridor_search: bool = False
    corridor: float | None = None
    refill_pool: int = 24
    refill_temperature: float = 1.0
    extra_steps: int = 6
    seed: int = 0
    # How a competency is ranked: its best path ("max") or all its paths ("sum").
    aggregate: str = "max"
    # Speed (experiment 20): all paths of a step share one model call per token
    # step, the resume prompt being a cached prefix (`batch_paths`); one search
    # pass returns all competencies it reached instead of restarting with the
    # found ones blocked (`single_pass`).
    batch_paths: bool = False
    single_pass: bool = False
    # Groups (experiment 24): a path may finish once at most `group_size`
    # competencies remain (all of them become candidates; the yes/no check
    # chooses); with `group_at_cap`, a path that runs out of signals keeps its
    # group of at most that many competencies instead of being dropped.
    group_size: int | None = None
    group_at_cap: int | None = None


@dataclass
class WordChoice:
    word: str
    lemma: str
    log_probability: float
    log_prior: float
    idf: float
    score: float
    candidates_after: int
    # log of the probability mass the model put on allowed first tokens at this
    # choice (how much the mask did NOT cut off); experiment 17.
    log_mass: float = 0.0


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
        self.evidence: set[str] = set()
        # Shared prompt prefix of the current search (for the cached batch mode).
        self.prefix: list[int] | None = None
        # Negative selection (experiment 13): phrases no signal may come from.
        self.blocked_phrases: set[int] = set()
        self.df: dict[str, int] = defaultdict(int)
        for lemmas in index.competency_lemmas.values():
            for lemma in lemmas:
                self.df[lemma] += 1
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
        if not coherent:
            search = self._path_search
        elif self.config.corridor_search:
            search = self._corridor_search
        else:
            search = self._signal_search
        while len(found) < count:
            finished, used = search(text, blocked)
            calls += used
            best: dict[str, WordPath] = {}
            scores: dict[str, list[float]] = defaultdict(list)
            for path in finished:
                # A path may end on a group of competencies (`group_size`):
                # every member gets the path.
                for competency_id in path.candidates:
                    if competency_id in blocked:
                        continue
                    scores[competency_id].append(path.score)
                    if competency_id not in best or path.score > best[competency_id].score:
                        best[competency_id] = path

            def rank(item: tuple[str, WordPath]) -> float:
                if self.config.aggregate == "sum":
                    values = scores[item[0]]
                    top = max(values)
                    return -(top + math.log(sum(math.exp(v - top) for v in values)))
                return -item[1].score

            ranked = sorted(best.items(), key=rank)
            if not ranked:
                break
            for competency_id, path in ranked[: count - len(found)]:
                found.append((competency_id, path))
                blocked.add(competency_id)
            if self.config.single_pass:
                break
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
        self.prefix = list(start.context)
        live: list[SignalPath] = [start]
        finished: dict[tuple, SignalPath] = {}
        calls = 0
        for _ in range(config.max_signals * config.words_per_signal):
            children: dict[tuple, SignalPath] = {}
            calls += self._expand_step(live, children, finished)
            live = sorted(children.values(), key=lambda p: -p.score)[: config.path_beam]
            if not live:
                break
        return sorted(finished.values(), key=lambda p: -p.score), calls

    def _expand_step(self, live, children: dict, finished: dict) -> int:
        """Word options for all live paths at once; children placed in path order."""
        requests, owners = [], []
        for path in live:
            phrases, allowed = self._allowed(path)
            if allowed:
                requests.append((path.context, allowed, self.config.word_branches))
                owners.append((path, phrases))
        if not requests:
            return 0
        results, calls = self._word_options_many(requests)
        for (path, phrases), options in zip(owners, results):
            for option in options:
                self._place(self._extend(path, phrases, option), children, finished)
        return calls

    def _allowed(self, path: "SignalPath") -> tuple[frozenset[int], set[str]]:
        """Phrases the next word may come from and the lemmas allowed next."""
        config = self.config
        phrases = path.phrases if path.current else self._open_phrases(path)
        allowed = {
            lemma for phrase in phrases for lemma in self._near(phrase, path.current)
        } - path.lemmas
        if (config.anchor_df is not None and not path.current
                and len(path.candidates) == 1):
            (only,) = path.candidates
            if not self._anchored(path.words, only):
                allowed &= self._anchor_lemmas(only)  # Write a key word next.
        last_word = len(path.current) + 1 >= config.words_per_signal
        if config.require_evidence and last_word and not path.current & self.evidence:
            allowed &= self.evidence  # The signal's last chance to be evidenced.
        return phrases, allowed

    def _extend(
        self, path: "SignalPath", phrases: frozenset[int], option: WordChoice,
    ) -> tuple["SignalPath", str] | None:
        """The path with one more word: (child, "finished" | "live") or None (dead)."""
        config = self.config
        index = self.index
        used = path.lemmas
        kept = frozenset(p for p in phrases if option.lemma in self._near(p, path.current))
        current = path.current | {option.lemma}
        left = {lemma for phrase in kept for lemma in self._near(phrase, current)} - used
        closes = len(current) >= config.words_per_signal or not left
        context = path.context + list(self.tokens[option.word])
        candidates, signals = path.candidates, path.signals
        if closes and config.require_evidence and not current & self.evidence:
            return None
        if closes:
            competencies = {c for p in kept for c in index.phrase_lemmas[p][0]}
            candidates = frozenset(path.candidates & competencies)
            if not candidates:
                return None
            signals = signals + (current,)
            context += list(self.signal_separator)
        word = WordChoice(**{**option.__dict__, "candidates_after": len(candidates)})
        child = SignalPath(
            path.words + [word], candidates, context, path.score + option.score,
            signals=signals,
            current=frozenset() if closes else current,
            phrases=frozenset() if closes else kept,
        )
        group = config.group_size or 1
        resolved = closes and len(candidates) <= group and len(signals) >= config.min_signals
        if resolved and config.anchor_df is not None:
            resolved = self._anchored(child.words, next(iter(candidates)))
        if resolved:
            return child, "finished"
        if closes and len(signals) >= config.max_signals:
            # Out of signals: similar competencies may be impossible to tell
            # apart by words; keep the path for its group instead of dropping it.
            if config.group_at_cap and len(candidates) <= config.group_at_cap:
                return child, "finished"
            return None
        return child, "live"

    @staticmethod
    def _place(result, live: dict, finished: dict) -> None:
        if result is None:
            return
        child, status = result
        key = (frozenset(child.signals), child.current)
        bucket = finished if status == "finished" else live
        if key not in bucket or child.score > bucket[key].score:
            bucket[key] = child

    @staticmethod
    def _mean_mass(path: "SignalPath") -> float:
        return sum(word.log_mass for word in path.words) / len(path.words)

    def _corridor_search(self, text: str, blocked: set[str]) -> tuple[list[WordPath], int]:
        """Signal search with a corridor on the mass the mask cuts off, and refills.

        Start: a pool of the `refill_pool` best first words. The `word_branches`
        best start paths; free places (up to `path_beam`) are refilled with first
        words drawn from the untried rest of the pool in proportion to
        exp(score / T), so improbable starts ("API", "риск") get a chance.
        After every step, a branch whose mean log-mass per word is more than
        `corridor` below the second-best branch is dropped (the mask keeps
        forcing it away from what the model wants); live paths are ranked by
        mean word score, since refilled paths are shorter.
        """
        config = self.config
        rng = random.Random(f"{config.seed}:{text}")
        start = SignalPath(
            [], frozenset(set(self.index.competency_lemmas) - blocked),
            list(self.prompt(text)), 0.0,
        )
        self.prefix = list(start.context)
        root_phrases, root_allowed = self._allowed(start)
        if not root_allowed:
            return [], 0
        token_beam = config.beam
        config.beam = max(config.beam, config.refill_pool)
        try:
            pool, calls = self._word_options(start.context, root_allowed, config.refill_pool)
        finally:
            config.beam = token_beam
        tried: set[str] = set()
        live: dict[tuple, SignalPath] = {}
        finished: dict[tuple, SignalPath] = {}

        def start_with(option: WordChoice) -> None:
            tried.add(option.lemma)
            self._place(self._extend(start, root_phrases, option), live, finished)

        for option in pool[: config.word_branches]:
            start_with(option)
        for _ in range(config.max_signals * config.words_per_signal + config.extra_steps):
            while len(live) < config.path_beam:
                remaining = [o for o in pool if o.lemma not in tried]
                if not remaining:
                    break
                top = remaining[0].score
                weights = [
                    math.exp((o.score - top) / config.refill_temperature) for o in remaining]
                start_with(rng.choices(remaining, weights)[0])
            if not live:
                break
            children: dict[tuple, SignalPath] = {}
            calls += self._expand_step(list(live.values()), children, finished)
            candidates = list(children.values())
            if config.corridor is not None and len(candidates) >= 2:
                masses = sorted((self._mean_mass(p) for p in candidates), reverse=True)
                floor = masses[1] - config.corridor
                candidates = [p for p in candidates if self._mean_mass(p) >= floor]
            candidates.sort(key=lambda p: -p.score / len(p.words))
            live = {
                (frozenset(p.signals), p.current): p for p in candidates[: config.path_beam]}
        return sorted(finished.values(), key=lambda p: -p.score), calls

    def _open_phrases(self, path: "SignalPath") -> frozenset[int]:
        """Phrases a new signal may come from: phrases of the remaining candidates,
        except phrases that already contain an earlier signal (no repeating it)."""
        return frozenset(
            i for i, (competencies, lemmas) in enumerate(self.index.phrase_lemmas)
            if competencies & path.candidates and i not in self.blocked_phrases
            and not any(signal <= lemmas for signal in path.signals)
        )

    def _anchor_lemmas(self, competency_id: str) -> set[str]:
        return {
            lemma for lemma in self.index.competency_lemmas[competency_id]
            if self.df[lemma] <= self.config.anchor_df
        }

    def _anchored(self, words: list[WordChoice], competency_id: str) -> bool:
        """The path holds a key word of the competency the model wanted enough."""
        threshold = self.config.anchor_min_pmi
        anchors = self._anchor_lemmas(competency_id)
        return any(
            word.lemma in anchors
            and (threshold is None or word.log_probability - word.log_prior >= threshold)
            for word in words
        )

    def _near(self, phrase: int, current: frozenset[str] | set[str]) -> set[str]:
        """Lemmas of the phrase that may join the signal `current`: with a window,
        each must stand within `signal_window` positions of every current word."""
        lemmas = self.index.phrase_lemmas[phrase][1]
        window = self.config.signal_window
        if window is None or not current:
            return lemmas - set(current)
        positions = self.index.phrase_positions[phrase]
        return {
            lemma for lemma in lemmas - set(current)
            if all(
                any(abs(a - b) <= window for a in positions[lemma] for b in positions[other])
                for other in current
            )
        }

    def _choose_word(self, context: list[int], allowed: set[str]) -> tuple[WordChoice | None, int]:
        options, calls = self._word_options(context, allowed, 1)
        return (options[0] if options else None), calls

    def _word_options(
        self, context: list[int], allowed: set[str], top: int,
    ) -> tuple[list[WordChoice], int]:
        """Best `top` complete words (distinct lemmas) by token-level beam search."""
        results, calls = self._word_options_many([(context, allowed, top)])
        return results[0], calls

    def _word_options_many(
        self, requests: list[tuple[list[int], set[str], int]],
    ) -> tuple[list[list[WordChoice]], int]:
        """Token-level beams of several contexts, run in lockstep.

        With `batch_paths` (and a model that caches the resume prompt) all live
        prefixes of all requests go to the model in one call per token step;
        otherwise every request makes its own calls (the original behaviour).
        """
        beams = [_WordBeam(self, context, allowed, top) for context, allowed, top in requests]
        prefix = self.prefix if self.config.batch_paths else None
        batched = (
            prefix is not None and hasattr(self.lm, "tail_logprobs")
            and all(beam.context[: len(prefix)] == prefix for beam in beams)
        )
        calls = 0
        while any(beam.live for beam in beams):
            active = [beam for beam in beams if beam.live]
            if batched:
                tails: list[tuple[int, ...]] = []
                wanted: list[list[int]] = []
                for beam in active:
                    tail = tuple(beam.context[len(prefix):])
                    for node in beam.live:
                        tails.append(tail + node)
                        wanted.append(sorted(beam.children[node]))
                rows = self.lm.tail_logprobs(prefix, tails, wanted)
                calls += 1
                offset = 0
                for beam in active:
                    size = len(beam.live)
                    beam.advance(wanted[offset : offset + size], rows[offset : offset + size])
                    offset += size
            else:
                for beam in active:
                    wanted = [sorted(beam.children[node]) for node in beam.live]
                    rows = self.lm.next_token_logprobs(beam.context, beam.live, wanted)
                    calls += 1
                    beam.advance(wanted, rows)
        return [beam.options() for beam in beams], calls


class _WordBeam:
    """Token-level beam search for one context: best complete allowed words."""

    def __init__(
        self, selector: "WordBagSelector", context: list[int], allowed: set[str], top: int,
    ) -> None:
        self.selector = selector
        self.config = config = selector.config
        self.context = context
        self.top = top
        self.children: dict[tuple[int, ...], set[int]] = defaultdict(set)
        self.terminal: dict[tuple[int, ...], str] = {}
        self.best_extra: dict[tuple[int, ...], float] = defaultdict(lambda: -math.inf)
        for surface, lemma in selector.index.surfaces.items():
            if lemma not in allowed:
                continue
            sequence = selector.tokens[surface]
            idf = selector.index.idf.get(lemma, 0.0)
            extra = config.beta * idf + config.evidence_bonus * (lemma in selector.evidence)
            for depth in range(len(sequence)):
                self.children[sequence[:depth]].add(sequence[depth])
                prefix = sequence[: depth + 1]
                self.best_extra[prefix] = max(self.best_extra[prefix], extra)
            self.terminal.setdefault(sequence, surface)
        self.log_prob: dict[tuple[int, ...], float] = {(): 0.0}
        self.live: list[tuple[int, ...]] = [()] if self.children else []
        self.finished: list[tuple[int, ...]] = []
        self.steps = 0
        self.step_mass = 0.0

    def score(self, node: tuple[int, ...]) -> float:
        config = self.config
        prior = self.selector.prior.get(node, 0.0)
        if config.prior_floor is not None:
            prior = max(prior, config.prior_floor)
        return self.log_prob[node] - config.alpha * prior + self.best_extra[node]

    def advance(self, wanted: list[list[int]], rows: list[list[float]]) -> None:
        if self.steps == 0 and rows and rows[0]:
            top_value = max(rows[0])
            self.step_mass = top_value + math.log(sum(math.exp(v - top_value) for v in rows[0]))
        self.steps += 1
        expanded: list[tuple[int, ...]] = []
        for node, tokens, values in zip(self.live, wanted, rows):
            for token, value in zip(tokens, values):
                child = node + (token,)
                self.log_prob[child] = self.log_prob[node] + value
                expanded.append(child)
        expanded.sort(key=self.score, reverse=True)
        self.live = []
        for node in expanded[: self.config.beam]:
            if node in self.terminal:
                self.finished.append(node)
            if self.children.get(node):
                self.live.append(node)

    def options(self) -> list[WordChoice]:
        selector = self.selector
        options: list[WordChoice] = []
        seen: set[str] = set()
        for node in sorted(self.finished, key=self.score, reverse=True):
            surface = self.terminal[node]
            lemma = selector.index.surfaces[surface]
            if lemma in seen:
                continue
            seen.add(lemma)
            options.append(WordChoice(
                word=surface, lemma=lemma, log_probability=self.log_prob[node],
                log_prior=selector.prior.get(node, 0.0), idf=selector.index.idf.get(lemma, 0.0),
                score=self.score(node), candidates_after=0, log_mass=self.step_mass,
            ))
            if len(options) == self.top:
                break
        return options

def lm_prompt_tensor(lm: object, ids: list[int]) -> object:
    import torch

    return torch.tensor([ids], device=lm.device)
