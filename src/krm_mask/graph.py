from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .catalog import Catalog, iter_competency_phrases


NONE_ID = "__NONE__"
NONE_PHRASE = "НЕТ ПОДХОДЯЩЕЙ КОМПЕТЕНЦИИ"


def normalize_phrase(value: str) -> str:
    return " ".join(value.casefold().split())


@dataclass
class PhraseGraph:
    phrases: dict[str, set[str]]
    competency_names: dict[str, str]

    @classmethod
    def from_catalog(
        cls,
        catalog: Catalog,
        competency_ids: Iterable[str] | None = None,
        phrase_sources: Iterable[str] | None = None,
    ) -> "PhraseGraph":
        sources = set(phrase_sources or ("names", "keywords"))
        unknown = sources - {"names", "indicators", "keywords"}
        if unknown:
            raise ValueError(f"Unknown phrase sources: {sorted(unknown)}")
        allowed = set(competency_ids) if competency_ids is not None else None
        phrases: dict[str, set[str]] = {}
        names: dict[str, str] = {NONE_ID: NONE_PHRASE}
        for competency in catalog.competencies:
            if allowed is not None and competency.id not in allowed:
                continue
            names[competency.id] = competency.name
            phrases_for_competency: list[str] = []
            if "names" in sources:
                phrases_for_competency.append(competency.name)
            if "indicators" in sources:
                phrases_for_competency.extend(
                    indicator.text for indicator in competency.indicators
                )
            if "keywords" in sources:
                phrases_for_competency.extend(competency.keywords)
            for phrase in phrases_for_competency:
                phrase = " ".join(phrase.split()).strip()
                if phrase and normalize_phrase(phrase):
                    phrases.setdefault(phrase, set()).add(competency.id)
        phrases[NONE_PHRASE] = {NONE_ID}
        return cls(phrases=phrases, competency_names=names)

    def candidates(self, selected_phrases: Iterable[str]) -> set[str]:
        selected = list(selected_phrases)
        if not selected:
            return set(self.competency_names)
        result: set[str] | None = None
        lookup = {normalize_phrase(k): v for k, v in self.phrases.items()}
        for phrase in selected:
            current = lookup.get(normalize_phrase(phrase), set())
            result = set(current) if result is None else result & current
        return result or set()

