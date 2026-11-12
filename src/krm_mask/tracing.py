from __future__ import annotations

from typing import Sequence

from .masking import PhraseGraphMask, SourceSpanMask


def _token_text(tokenizer: object, token_id: int) -> str:
    if token_id == tokenizer.eos_token_id:
        return "<EOS>"
    value = tokenizer.decode([token_id], skip_special_tokens=False)
    return value.replace("\n", "↵\n")


def trace_phrase_generation(
    mask: PhraseGraphMask,
    generated: Sequence[int],
    tokenizer: object,
    step_scores: Sequence[dict[str, float | int]] | None = None,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    prefix: list[int] = []
    names = mask.graph.competency_names
    for index, token_id in enumerate(generated):
        before = mask.parse(prefix)
        candidates_before = mask.resolved_candidates(prefix)
        allowed = mask.allowed_tokens(prefix)
        after_prefix = prefix + [token_id]
        after = mask.parse(after_prefix)
        candidates_after = mask.resolved_candidates(after_prefix)
        event: dict[str, object] = {
                "step": index,
                "token_id": token_id,
                "token_text": _token_text(tokenizer, token_id),
                "allowed_count": len(allowed),
                "allowed_tokens": [
                    {"id": value, "text": _token_text(tokenizer, value)}
                    for value in sorted(allowed)[:40]
                ],
                "candidates_before": [
                    {"id": value, "name": names.get(value, value)}
                    for value in sorted(candidates_before)
                ],
                "candidates_after": [
                    {"id": value, "name": names.get(value, value)}
                    for value in sorted(candidates_after)
                ],
                "selected_before": before.selected,
                "selected_after": after.selected,
                "completed_phrase": (
                    after.selected[-1]
                    if len(after.selected) > len(before.selected)
                    else None
                ),
                "resolved_by_prefix": (
                    len(candidates_after) == 1
                    and len(after.candidates) > 1
                    and bool(after.tail)
                    and token_id != mask.eos_token_id
                ),
            }
        if step_scores and index < len(step_scores):
            event.update(step_scores[index])
        events.append(event)
        prefix = after_prefix
    return events


def trace_source_generation(
    mask: SourceSpanMask,
    generated: Sequence[int],
    tokenizer: object,
    offsets: Sequence[Sequence[int]],
    step_scores: Sequence[dict[str, float | int]] | None = None,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    prefix: list[int] = []
    for index, token_id in enumerate(generated):
        allowed = mask.allowed_tokens(prefix)
        after_prefix = prefix + [token_id]
        occurrences = mask.occurrences(after_prefix)
        spans: list[dict[str, int]] = []
        if after_prefix:
            for start in occurrences:
                end_token = start + len(after_prefix) - 1
                if start < len(offsets) and end_token < len(offsets):
                    spans.append(
                        {"start": int(offsets[start][0]), "end": int(offsets[end_token][1])}
                    )
        event: dict[str, object] = {
                "step": index,
                "token_id": token_id,
                "token_text": _token_text(tokenizer, token_id),
                "allowed_count": len(allowed),
                "allowed_tokens": [
                    {"id": value, "text": _token_text(tokenizer, value)}
                    for value in sorted(allowed)[:40]
                ],
                "matching_spans": spans[:80],
                "occurrence_count": len(occurrences),
            }
        if step_scores and index < len(step_scores):
            event.update(step_scores[index])
        events.append(event)
        prefix = after_prefix
    return events

