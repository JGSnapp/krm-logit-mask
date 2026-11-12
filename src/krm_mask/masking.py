from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Sequence

from .graph import PhraseGraph


TokenSeq = tuple[int, ...]


def _starts_with(sequence: Sequence[int], prefix: Sequence[int]) -> bool:
    return len(prefix) <= len(sequence) and list(sequence[: len(prefix)]) == list(prefix)


@dataclass
class PhraseMaskState:
    selected: list[str]
    candidates: set[str]
    tail: list[int]


class PhraseGraphMask:
    """Order-independent phrase grammar used as a Transformers logits processor.

    A generated line is one phrase from the graph. After every complete line the
    candidate set is intersected with the competencies attached to that phrase.
    EOS is legal only when exactly one competency remains.
    """

    def __init__(
        self,
        tokenizer: object,
        graph: PhraseGraph,
        prompt_length: int,
        delimiter: str = "\n",
        blocked_ids: set[str] | None = None,
        allow_abort_token: bool = False,
        resolution_extra_chars: int = 0,
        complete_phrase_before_eos: bool = False,
        min_complete_phrases: int = 1,
        resolve_on_unique_phrase: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.graph = graph
        self.prompt_length = prompt_length
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.delimiter_ids: TokenSeq = tuple(
            tokenizer.encode(delimiter, add_special_tokens=False)
        )
        if not self.delimiter_ids:
            raise ValueError("delimiter must produce at least one token")
        blocked = blocked_ids or set()
        self.phrase_ids: dict[str, TokenSeq] = {
            phrase: tuple(tokenizer.encode(phrase, add_special_tokens=False))
            for phrase, ids in graph.phrases.items()
            if ids - blocked
        }
        self.blocked_ids = blocked
        self.resolution_extra_chars = max(0, resolution_extra_chars)
        self.complete_phrase_before_eos = complete_phrase_before_eos
        self.min_complete_phrases = max(1, min_complete_phrases)
        self.resolve_on_unique_phrase = resolve_on_unique_phrase
        self.abort_token_ids: set[int] = set()
        if allow_abort_token:
            for value in (".", " ."):
                encoded = tokenizer.encode(value, add_special_tokens=False)
                if len(encoded) == 1:
                    self.abort_token_ids.add(int(encoded[0]))

    def is_aborted(self, generated: Sequence[int]) -> bool:
        return any(token in self.abort_token_ids for token in generated)

    def _first_resolution_length(self, generated: Sequence[int]) -> int | None:
        for length in range(1, len(generated) + 1):
            resolved_count = (
                len(self.resolved_phrases(generated[:length]))
                if self.resolve_on_unique_phrase
                else len(self.resolved_candidates(generated[:length]))
            )
            if resolved_count == 1:
                return length
        return None

    def _extra_char_count(self, generated: Sequence[int]) -> int:
        first = self._first_resolution_length(generated)
        if first is None:
            return 0
        return len(
            self.tokenizer.decode(
                list(generated[first:]), skip_special_tokens=True
            )
        )

    def parse(self, generated: Sequence[int]) -> PhraseMaskState:
        rest = list(generated)
        selected: list[str] = []
        while rest:
            match = None
            for phrase, ids in self.phrase_ids.items():
                full = list(ids + self.delimiter_ids)
                if _starts_with(rest, full):
                    match = (phrase, len(full))
                    break
            if match is None:
                break
            selected.append(match[0])
            rest = rest[match[1] :]
        candidates = self.graph.candidates(selected) - self.blocked_ids
        return PhraseMaskState(selected=selected, candidates=candidates, tail=rest)

    def resolved_candidates(self, generated: Sequence[int]) -> set[str]:
        """Candidates compatible with completed phrases and the current prefix.

        This permits early class resolution before a unique phrase is fully
        generated. The canonical class name is supplied by the graph afterward.
        """
        if generated and generated[-1] == self.eos_token_id:
            return self.resolved_candidates(generated[:-1])
        state = self.parse(generated)
        if not state.tail:
            return state.candidates
        compatible: set[str] = set()
        selected_set = set(state.selected)
        for phrase, ids in self.phrase_ids.items():
            if phrase in selected_set:
                continue
            full = ids + self.delimiter_ids
            if _starts_with(full, state.tail):
                compatible.update(self.graph.phrases[phrase] - self.blocked_ids)
        return state.candidates & compatible

    def resolved_phrases(self, generated: Sequence[int]) -> set[str]:
        """Phrases compatible with the current unfinished prefix."""
        if generated and generated[-1] == self.eos_token_id:
            return self.resolved_phrases(generated[:-1])
        state = self.parse(generated)
        if state.selected:
            return {state.selected[-1]} if not state.tail else set()
        return {
            phrase
            for phrase, ids in self.phrase_ids.items()
            if _starts_with(ids + self.delimiter_ids, state.tail)
        }

    def completion_tails(self, generated: Sequence[int]) -> list[TokenSeq]:
        """Unwritten token suffixes when the mask stopped before the phrase end.

        Each tail is the rest of one compatible phrase plus the delimiter. An
        empty list means the last phrase was written in full (or aborted).
        """
        content = list(generated)
        if content and content[-1] == self.eos_token_id:
            content = content[:-1]
        if self.is_aborted(content):
            return []
        state = self.parse(content)
        if not state.tail:
            return []
        if self.resolve_on_unique_phrase:
            phrases = self.resolved_phrases(content)
        else:
            resolved = self.resolved_candidates(content)
            phrases = {
                phrase
                for phrase in self.phrase_ids
                if phrase not in state.selected
                and (self.graph.phrases[phrase] - self.blocked_ids) & resolved
            }
        tails: list[TokenSeq] = []
        for phrase in phrases:
            full = self.phrase_ids[phrase] + self.delimiter_ids
            if _starts_with(full, state.tail) and len(state.tail) < len(full):
                tails.append(full[len(state.tail) :])
        return tails

    def allowed_tokens(self, generated: Sequence[int]) -> set[int]:
        if self.is_aborted(generated):
            return {self.eos_token_id}
        state = self.parse(generated)
        resolved = self.resolved_candidates(generated)
        if not resolved:
            return {self.eos_token_id}
        resolved_count = (
            len(self.resolved_phrases(generated))
            if self.resolve_on_unique_phrase
            else len(resolved)
        )
        # The hypothesis defines class selection as complete at the first point
        # where only one permissible competency remains.
        if (
            resolved_count == 1
            and (
                not self.complete_phrase_before_eos
                or (
                    len(state.selected) >= self.min_complete_phrases
                    and not state.tail
                )
            )
            and (
                self.resolution_extra_chars == 0
                or self._extra_char_count(generated) >= self.resolution_extra_chars
            )
        ):
            return {self.eos_token_id} | self.abort_token_ids

        allowed: set[int] = set()
        selected_set = set(state.selected)
        for phrase, ids in self.phrase_ids.items():
            if phrase in selected_set:
                continue
            phrase_candidates = self.graph.phrases[phrase] - self.blocked_ids
            if not (phrase_candidates & resolved):
                continue
            full = ids + self.delimiter_ids
            if _starts_with(full, state.tail) and len(state.tail) < len(full):
                allowed.add(full[len(state.tail)])

        if self.abort_token_ids:
            allowed.update(self.abort_token_ids)
        return allowed or {self.eos_token_id}

    def __call__(self, input_ids: object, scores: object) -> object:
        import torch

        for batch_index in range(input_ids.shape[0]):
            generated = input_ids[batch_index, self.prompt_length :].tolist()
            allowed = self.allowed_tokens(generated)
            mask = torch.full_like(scores[batch_index], -inf)
            indices = torch.tensor(list(allowed), device=scores.device, dtype=torch.long)
            mask[indices] = scores[batch_index, indices]
            scores[batch_index] = mask
        return scores


class SourceSpanMask:
    """Restrict output to one contiguous token span copied from source text."""

    def __init__(
        self,
        source_token_ids: Sequence[int],
        prompt_length: int,
        eos_token_id: int,
        min_tokens: int = 2,
        max_tokens: int = 96,
        blocked_token_positions: set[int] | None = None,
        blocked_start_positions: set[int] | None = None,
        abort_token_ids: set[int] | None = None,
    ) -> None:
        self.source = list(source_token_ids)
        self.prompt_length = prompt_length
        self.eos_token_id = eos_token_id
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.blocked_token_positions = blocked_token_positions or set()
        self.blocked_start_positions = blocked_start_positions or set()
        self.abort_token_ids = abort_token_ids or set()

    def is_aborted(self, generated: Sequence[int]) -> bool:
        return any(token in self.abort_token_ids for token in generated)

    def occurrences(self, prefix: Sequence[int]) -> list[int]:
        if not prefix:
            return [
                position
                for position in range(len(self.source))
                if position not in self.blocked_token_positions
                and position not in self.blocked_start_positions
            ]
        size = len(prefix)
        return [
            i
            for i in range(len(self.source) - size + 1)
            if self.source[i : i + size] == list(prefix)
            and i not in self.blocked_start_positions
            and not any(
                position in self.blocked_token_positions
                for position in range(i, i + size)
            )
        ]

    def allowed_tokens(self, generated: Sequence[int]) -> set[int]:
        generated = list(generated)
        if self.is_aborted(generated):
            return {self.eos_token_id}
        allowed: set[int] = set()
        occurrences = self.occurrences(generated)
        for start in occurrences:
            next_index = start + len(generated)
            if next_index < len(self.source):
                allowed.add(self.source[next_index])
        if len(generated) >= self.min_tokens:
            allowed.add(self.eos_token_id)
        if len(generated) >= self.max_tokens:
            return {self.eos_token_id}
        allowed.update(self.abort_token_ids)
        return allowed or {self.eos_token_id}

    def __call__(self, input_ids: object, scores: object) -> object:
        import torch

        for batch_index in range(input_ids.shape[0]):
            generated = input_ids[batch_index, self.prompt_length :].tolist()
            allowed = self.allowed_tokens(generated)
            mask = torch.full_like(scores[batch_index], -inf)
            indices = torch.tensor(list(allowed), device=scores.device, dtype=torch.long)
            mask[indices] = scores[batch_index, indices]
            scores[batch_index] = mask
        return scores

