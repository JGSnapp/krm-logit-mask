"""Loader for the MIPT competency-role models (one XLSX per role).

Each role workbook has the columns: competency ("ИП-1. Способен ..."),
indicator ("ИП-1.1. ..."), indicator level, learning outcomes, technology
examples, source. Two shared workbooks hold cross-professional ("ППК-...")
and universal competencies; the sheet "Компетенции по ролям" maps
cross-professional competencies to roles.

Mapped to the ITMO data model: indicators are indicators, "Примеры технологий"
are split into topics (keywords) like ITMO's "Рекомендуемые темы". Learning
outcomes are not used. A competency has no short name in MIPT, so its name is
the formulation.
"""
from __future__ import annotations

import re
from pathlib import Path

from .catalog import Catalog, Competency, Indicator, Role, _clean, _slug, _split_keywords

CODE_RE = re.compile(r"^([A-ZА-ЯЁ]{1,6}(?:-[A-ZА-ЯЁ]{0,4})?\d+(?:\.\d+)?)\.?\s+(.*)$")
ROLE_MATRIX = "Компетенции по ролям"
LEVEL_CODES = {"базовый": "Б", "средний": "С", "продвинутый": "П"}


def _code(text: str) -> tuple[str, str] | None:
    match = CODE_RE.match(text.replace("–", "-"))
    return (match.group(1), match.group(2).lstrip("– -").strip()) if match else None


def _read_sheet(sheet, competencies: dict[str, Competency], source: str) -> list[str]:
    """Parse one sheet; return competency codes in their order of appearance."""
    order: list[str] = []
    current: Competency | None = None
    block = ""
    for row_number, row in enumerate(sheet.iter_rows(values_only=True), start=1):
        cells = [_clean(value) for value in (list(row) + [None] * 6)[:6]]
        first, indicator, level, _outcomes, technologies, _ = cells
        if first == "Компетенция":
            continue
        parsed = _code(first) if first else None
        if first and not parsed and not indicator:
            block = first  # Section title, e.g. "Профессиональные базовые компетенции".
            current = None
            continue
        if parsed:
            code, formulation = parsed
            current = competencies.get(code)
            if current is None:
                current = Competency(
                    id=code, name=formulation, formulation=formulation, domain="МФТИ",
                    block=block, source_sheet=f"{source}:{sheet.title}", source_row=row_number,
                )
                competencies[code] = current
            if code not in order:
                order.append(code)
        if current is None:
            continue
        if indicator:
            parsed_indicator = _code(indicator)
            indicator_id, text = parsed_indicator or (f"{current.id}.?", indicator)
            if not any(item.id == indicator_id for item in current.indicators):
                levels = {}
                if level.lower() in LEVEL_CODES:
                    levels[LEVEL_CODES[level.lower()]] = text
                current.indicators.append(Indicator(id=indicator_id, text=text, levels=levels))
        raw_technologies = (list(row) + [None] * 6)[4]
        for keyword in _split_keywords(raw_technologies):
            if keyword not in current.keywords:
                current.keywords.append(keyword)
    return order


def extract_mipt_catalog(folder: str | Path) -> Catalog:
    import openpyxl

    folder = Path(folder)
    competencies: dict[str, Competency] = {}
    roles: list[Role] = []
    cross_by_role: dict[str, dict[str, str]] = {}
    for path in sorted(folder.glob("*.xlsx")):
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        codes: list[str] = []
        for sheet in workbook.worksheets:
            if sheet.title == ROLE_MATRIX:
                rows = list(sheet.iter_rows(values_only=True))
                header = [_clean(value) for value in rows[1]]
                for row in rows[2:]:
                    parsed = _code(_clean(row[0])) if row[0] else None
                    if not parsed:
                        continue
                    for column, role_name in enumerate(header[1:], start=1):
                        mark = _clean(row[column]) if column < len(row) else ""
                        if role_name and mark:
                            cross_by_role.setdefault(role_name, {})[parsed[0]] = mark[:1]
                continue
            codes += _read_sheet(sheet, competencies, path.name)
        title = re.sub(r"^\d+\.\s*(КРМ\s*)?", "", path.stem).strip()
        if not path.stem.startswith("0."):
            roles.append(Role(id=_slug(title), name=title,
                              competencies={code: "" for code in dict.fromkeys(codes)}))
    for role in roles:
        for role_name, marks in cross_by_role.items():
            if _slug(role_name) in role.id or role.id in _slug(role_name):
                role.competencies.update(marks)
    return Catalog(
        framework_id="mipt-it-krm", framework_version="github-c13764b",
        source_file=str(folder), competencies=list(competencies.values()), roles=roles,
    )
