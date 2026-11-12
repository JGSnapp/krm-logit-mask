from pathlib import Path

from openpyxl import Workbook

from krm_mask.catalog import _split_keywords, extract_catalog


def test_splits_recommended_topics_without_breaking_compound_terms():
    assert _split_keywords(
        "DevSecOps\nCI/CD; Vision-Language-Action Model (VLA, embodied AI), CUDA"
    ) == [
        "DevSecOps",
        "CI/CD",
        "Vision-Language-Action Model (VLA, embodied AI)",
        "CUDA",
    ]


def test_extracts_continuation_anchors_and_role_matrix(tmp_path: Path):
    workbook = Workbook()
    workbook.remove(workbook.active)
    for title in (
        "Профессиональные",
        "Опережающие профессиональные",
        "Отраслевые",
        "Универсальные (надпроф.)",
    ):
        sheet = workbook.create_sheet(title)
        if title == "Отраслевые":
            sheet.append([None, "Домен", "Блок", "Код", "Имя", "Формулировка", None, "Индикатор", "Б", "С", "П", "Темы"])
            sheet.append([None] * 12)
        else:
            sheet.append(["Домен", "Блок", "Код", "Имя", "Формулировка", None, "Индикатор", "Б", "С", "П", "Темы"])
            sheet.append([None] * 11)
    professional = workbook["Профессиональные"]
    professional.append(["D", "B", "X-1", "SQL", "Работает с SQL", None, "X-1.1. Пишет запросы", "select", "window", "optimizer", "SQL, оконные функции"])
    professional.append([None, None, None, None, None, None, None, "join", "CTE", "plan", None])

    matrix = workbook.create_sheet("Роли-компетенции наглядно")
    matrix.append(["Блок", "Направление", "Код", "Формулировка", "Data Analyst"])
    matrix.append([None] * 5)
    matrix.append([None, None, "X-1", None, "С"])
    definitions = workbook.create_sheet("Роли - определения и функции")
    definitions.append([None])
    definitions.append([None])
    definitions.append(["Data Analyst", "Аналитик"])
    path = tmp_path / "krm.xlsx"
    workbook.save(path)

    catalog = extract_catalog(path)
    item = next(c for c in catalog.competencies if c.id == "X-1")
    assert item.indicators[0].levels["Б"] == "select join"
    assert item.indicators[0].levels["С"] == "window CTE"
    assert catalog.roles[0].competencies == {"X-1": "С"}


def test_normalizes_inconsistent_source_codes(tmp_path: Path):
    workbook = Workbook()
    workbook.remove(workbook.active)
    for title in (
        "Профессиональные",
        "Опережающие профессиональные",
        "Отраслевые",
        "Универсальные (надпроф.)",
    ):
        sheet = workbook.create_sheet(title)
        sheet.append(["Домен", "Блок", "Код", "Имя", "Формулировка", None, "Индикатор", "Б", "С", "П", "Темы"])
        sheet.append([None] * 11)
    workbook["Профессиональные"].append(["D", "B", "AI S - 3", "Security", "F", None, "AI S - 3.1. Does", "b", "s", "p", "security"])
    matrix = workbook.create_sheet("Роли-компетенции наглядно")
    matrix.append(["Блок", "Направление", "Код", "Формулировка", "Role"])
    matrix.append([None] * 5)
    matrix.append([None, None, "AI-S--3", None, "Б"])
    definitions = workbook.create_sheet("Роли - определения и функции")
    definitions.append([None])
    definitions.append([None])
    path = tmp_path / "codes.xlsx"
    workbook.save(path)

    catalog = extract_catalog(path)
    assert catalog.competencies[0].id == "AI-S-3"
    assert catalog.roles[0].competencies == {"AI-S-3": "Б"}

