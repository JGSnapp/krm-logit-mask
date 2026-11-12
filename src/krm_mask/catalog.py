from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


LEVELS = ("Б", "С", "П")
DETAIL_SHEETS = (
    "Профессиональные",
    "Опережающие профессиональные",
    "Отраслевые",
    "Универсальные (надпроф.)",
)


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", str(value))
    return re.sub(r"\s+", " ", text).strip()


def _code(value: object) -> str:
    text = _clean(value)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-")


@dataclass
class Indicator:
    id: str
    text: str
    levels: dict[str, str] = field(default_factory=dict)


@dataclass
class Competency:
    id: str
    name: str
    formulation: str
    domain: str = ""
    block: str = ""
    keywords: list[str] = field(default_factory=list)
    indicators: list[Indicator] = field(default_factory=list)
    source_sheet: str = ""
    source_row: int = 0


@dataclass
class Role:
    id: str
    name: str
    definition: str = ""
    competencies: dict[str, str] = field(default_factory=dict)


@dataclass
class Catalog:
    framework_id: str
    framework_version: str
    source_file: str
    competencies: list[Competency]
    roles: list[Role]

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Catalog":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["competencies"] = [
            Competency(
                **{
                    **item,
                    "indicators": [Indicator(**i) for i in item["indicators"]],
                }
            )
            for item in raw["competencies"]
        ]
        raw["roles"] = [Role(**item) for item in raw["roles"]]
        return cls(**raw)


def _split_keywords(value: object) -> list[str]:
    """Split the source column without corrupting compound technical terms.

    Newlines, bullets, commas and semicolons delimit recommended topics, except
    inside parentheses. A slash is part of terms such as CI/CD and must not be
    treated as a delimiter. This function deliberately receives the raw cell
    value because `_clean` collapses the newlines we need here.
    """
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for character in text:
        if character in "([{" :
            depth += 1
        elif character in ")]}":
            depth = max(0, depth - 1)
        if depth == 0 and character in ",;\n•":
            parts.append("".join(current))
            current = []
        else:
            current.append(character)
    parts.append("".join(current))

    topics: list[str] = []
    for part in parts:
        topic = _clean(part).strip(" .:-")
        if 3 <= len(topic) <= 180 and topic not in topics:
            topics.append(topic)
    return topics


def _sheet_offset(title: str) -> int:
    # The industry sheet has one leading empty column.
    return 1 if title == "Отраслевые" else 0


def extract_catalog(path: str | Path) -> Catalog:
    """Extract ITMO KRM 3.0 without silently inventing missing semantics."""
    from openpyxl import load_workbook

    source = Path(path)
    workbook = load_workbook(source, read_only=True, data_only=True)
    competencies: dict[str, Competency] = {}

    for sheet_name in DETAIL_SHEETS:
        if sheet_name not in workbook.sheetnames:
            continue
        sheet = workbook[sheet_name]
        offset = _sheet_offset(sheet_name)
        current_competency: Competency | None = None
        current_indicator: Indicator | None = None

        for row_number, row in enumerate(
            sheet.iter_rows(min_row=3, values_only=True), start=3
        ):
            values = list(row)
            values += [None] * max(0, 13 + offset - len(values))
            domain = _clean(values[0 + offset])
            block = _clean(values[1 + offset])
            code = _code(values[2 + offset])
            name = _clean(values[3 + offset])
            formulation = _clean(values[4 + offset])
            indicator_text = _clean(values[6 + offset])
            anchors = [_clean(values[i + offset]) for i in (7, 8, 9)]
            keyword_value = values[10 + offset]

            if code:
                current_competency = competencies.get(code)
                if current_competency is None:
                    current_competency = Competency(
                        id=code,
                        name=name or formulation,
                        formulation=formulation,
                        domain=domain,
                        block=block,
                        source_sheet=sheet_name,
                        source_row=row_number,
                    )
                    competencies[code] = current_competency
                current_indicator = None

            if current_competency is None:
                continue

            if keyword_value:
                for keyword in _split_keywords(keyword_value):
                    if keyword not in current_competency.keywords:
                        current_competency.keywords.append(keyword)

            if indicator_text:
                match = re.match(r"^([A-Za-zА-Яа-я0-9 -]+\.\d+)\.?\s*(.*)$", indicator_text)
                indicator_id = (
                    _code(match.group(1))
                    if match
                    else f"{current_competency.id}.row-{row_number}"
                )
                text = match.group(2).strip() if match and match.group(2) else indicator_text
                current_indicator = Indicator(id=indicator_id, text=text)
                current_competency.indicators.append(current_indicator)

            # Blank indicator cells frequently continue anchors from the previous row.
            if current_indicator is not None:
                for level, anchor in zip(LEVELS, anchors):
                    if not anchor or anchor == "-":
                        continue
                    previous = current_indicator.levels.get(level, "")
                    current_indicator.levels[level] = " ".join(
                        part for part in (previous, anchor) if part
                    )

    roles = _extract_roles(workbook)
    return Catalog(
        framework_id="itmo-krm",
        framework_version="3.0",
        source_file=str(source),
        competencies=list(competencies.values()),
        roles=roles,
    )


def _extract_roles(workbook: object) -> list[Role]:
    matrix = workbook["Роли-компетенции наглядно"]
    rows = list(matrix.iter_rows(values_only=True))
    headers = [_clean(v) for v in rows[0]]
    role_columns = [(i, name) for i, name in enumerate(headers[4:], 4) if name]
    roles = {name: Role(id=_slug(name), name=name) for _, name in role_columns}

    for row in rows[2:]:
        code = _code(row[2] if len(row) > 2 else "")
        if not code:
            continue
        for index, name in role_columns:
            level = _clean(row[index] if len(row) > index else "")
            if level in LEVELS:
                roles[name].competencies[code] = level

    if "Роли - определения и функции" in workbook.sheetnames:
        sheet = workbook["Роли - определения и функции"]
        for row in sheet.iter_rows(min_row=3, values_only=True):
            name, definition = _clean(row[0]), _clean(row[1])
            if name in roles:
                roles[name].definition = definition
    return list(roles.values())


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zа-я0-9]+", "-", text.lower(), flags=re.IGNORECASE)
    return value.strip("-")


def iter_competency_phrases(competency: Competency) -> Iterable[str]:
    """Phrases for the decoding graph, ordered from canonical to noisy."""
    yield competency.name
    for keyword in competency.keywords:
        yield keyword

