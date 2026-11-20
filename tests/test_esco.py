from pathlib import Path

from krm_mask.esco import add_skill_labels, esco_catalog, load_skill_xl


ROOT = Path(__file__).resolve().parents[1]


def test_esco_catalog_and_skill_xl_are_consistent():
    catalog = esco_catalog(ROOT / "data/esco/synthetic.csv")
    development = load_skill_xl(ROOT / "data/esco/skill-xl/development.csv")
    test = load_skill_xl(ROOT / "data/esco/skill-xl/test.csv")
    names = {item.id for item in catalog.competencies}
    assert len(names) == 13_826
    assert development and test
    assert not all(sentence.skills <= names for sentence in development + test)
    supplemented = add_skill_labels(catalog, set().union(*(sentence.skills for sentence in development + test)))
    assert all(sentence.skills <= {item.id for item in supplemented.competencies}
               for sentence in development + test)
    assert all(set().union(*sentence.clusters) == sentence.skills for sentence in development + test)
