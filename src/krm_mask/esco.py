"""ESCO skills as a catalog, and the Skill-XL benchmark (sentence-level ESCO labels).

Catalog: the 13 826 ESCO skill names (preferred labels) listed in TechWolf's
Synthetic-ESCO-skill-sentences (CC-BY-4.0; the official ESCO API was not
reachable). Only the names are used: no alternative labels or descriptions
(not available here), and never the synthetic sentences themselves (they are
training data). A skill's id is its name, as in Skill-XL.

Skill-XL (TechWolf, CC-BY-4.0): job ads split into sentences; every relevant
sentence lists the ESCO skills it requires, grouped into clusters (one cluster
= one mention, several near-equivalent skills).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import Catalog, Competency


def esco_catalog(synthetic_csv: str | Path) -> Catalog:
    names = sorted({row["skill"].strip() for row in csv.DictReader(open(synthetic_csv, encoding="utf-8"))
                    if row["skill"].strip()})
    return Catalog(
        framework_id="esco-skills-en",
        framework_version="names from TechWolf/Synthetic-ESCO-skill-sentences",
        source_file=str(synthetic_csv),
        competencies=[Competency(id=name, name=name, formulation="", keywords=[name]) for name in names],
        roles=[],
    )


def add_skill_labels(catalog: Catalog, labels: set[str]) -> Catalog:
    """Add version-mismatched benchmark labels as name-only ESCO entries.

    The downloaded Synthetic-ESCO names omit a handful of labels present in
    Skill-XL.  Dropping those gold labels would silently inflate evaluation
    scores; keeping the label text is safe because it contributes no example or
    annotation beyond the catalog name.  The returned catalog is a copy.
    """
    present = {competency.id for competency in catalog.competencies}
    additions = sorted(label.strip() for label in labels if label.strip() and label not in present)
    if not additions:
        return catalog
    return Catalog(
        framework_id=catalog.framework_id,
        framework_version=catalog.framework_version + "; supplemented with Skill-XL-only labels",
        source_file=catalog.source_file,
        competencies=[*catalog.competencies, *[
            Competency(id=label, name=label, formulation="", keywords=[label])
            for label in additions
        ]],
        roles=list(catalog.roles),
    )


@dataclass
class SkillSentence:
    ad: str
    title: str
    sentence: str
    skills: set[str] = field(default_factory=set)
    clusters: list[set[str]] = field(default_factory=list)


def load_skill_xl(path: str | Path, relevant_only: bool = True) -> list[SkillSentence]:
    """Sentences of a Skill-XL split in file order, with their gold skills."""
    order: list[tuple[str, str]] = []
    items: dict[tuple[str, str], SkillSentence] = {}
    clusters: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in csv.DictReader(open(path, encoding="utf-8")):
        key = (row["ID"], row["sentence"])
        if key not in items:
            items[key] = SkillSentence(row["ID"], row["title"], row["sentence"])
            order.append(key)
        skill = row["skill"].strip()  # some labels carry a trailing space
        if row["relevant"] == "True" and skill:
            items[key].skills.add(skill)
            clusters[key][row["cluster"]].add(skill)
    result = []
    for key in order:
        item = items[key]
        item.clusters = [set(group) for _, group in sorted(clusters[key].items())]
        if item.skills or not relevant_only:
            result.append(item)
    return result
