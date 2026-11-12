import math

import pytest

from krm_mask.graph import PhraseGraph
from krm_mask.wordbag import WordBagConfig, WordBagSelector, WordIndex, split_words

STOP = {"на", "и"}


def lemmatize(word):
    word = word.lower()
    return None if word in STOP else {"данных": "данные"}.get(word, word)


def graph():
    return PhraseGraph(
        {
            "Осуществляет выбор инструментов на Python": {"PY"},
            "Осуществляет выбор инструментов на Java": {"JV"},
            "Обработка данных и Python": {"DATA"},
        },
        {"PY": "Python", "JV": "Java", "DATA": "Data"},
    )


def test_index_lemmatizes_drops_function_words_and_computes_idf():
    index = WordIndex.build(graph(), lemmatize)
    assert "на" not in index.competency_lemmas["PY"]
    assert index.competency_lemmas["DATA"] == {"обработка", "данные", "python"}
    assert index.surfaces["данных"] == "данные"
    assert index.idf["осуществляет"] == pytest.approx(math.log(3 / 2))
    assert index.idf["java"] == pytest.approx(math.log(3))
    # One-letter words and bare numbers are dropped before lemmatization.
    assert split_words("C++ и ИИ-моделей, 2024") == ["C++", "ИИ-моделей"]


def test_selector_starts_with_distinctive_word_and_skips_shared_ones():
    torch = pytest.importorskip("torch")
    index = WordIndex.build(graph(), lemmatize)
    vocabulary = {}

    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            if text == "\n":
                return [1]
            return [vocabulary.setdefault(text.lower(), len(vocabulary) + 10)]

    likes = {"python": -0.5, "java": -3.0, "инструментов": -0.1}

    class LM:
        tokenizer = Tokenizer()
        device = "cpu"
        calls = 0

        def score_continuations(self, prompt, continuations, per_token=False):
            return [[-2.0] * len(c) for c in continuations]

        def next_token_logprobs(self, context, suffixes, wanted):
            LM.calls += 1
            names = {v: k for k, v in vocabulary.items()}
            return [[-0.01 if t == 1 else likes.get(names.get(t), -6.0) for t in row]
                    for row in wanted]

    lm = LM()
    selector = WordBagSelector(lm, index, lambda text: [5, 6], WordBagConfig(beam=4))
    result = selector.select("Писал сервисы на Python", blocked=set())
    # "инструментов" is liked most but is shared by PY and JV: allowed (it splits DATA
    # off), yet "python" is chosen first because IDF and PMI favour it as well.
    assert result.words[0].lemma in {"python", "инструментов"}
    assert result.competency_id == "PY" and result.resolved
    assert all(w.lemma not in {"осуществляет", "выбор"} or w.candidates_after < 3
               for w in result.words)
    assert result.forward_calls == LM.calls
    blocked = selector.select("Писал сервисы на Python", blocked={"PY"})
    assert blocked.competency_id in {"DATA", "JV"}


def _trap_selector(min_words):
    """'контракты' is the best first word, but its competency is not confirmed later."""
    pytest.importorskip("torch")
    trap = PhraseGraph(
        {"контракты право договор": {"LAW"}, "dataops данные интеграция": {"DATA"}},
        {"LAW": "Law", "DATA": "Data"},
    )
    index = WordIndex.build(trap, lemmatize)
    vocabulary = {}

    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            if text == "\n":
                return [1]
            return [vocabulary.setdefault(text.lower(), len(vocabulary) + 10)]

    likes = {"контракты": -0.3, "dataops": -0.6, "данные": -0.5, "интеграция": -0.5,
             "право": -9.0, "договор": -9.0}

    class LM:
        tokenizer = Tokenizer()
        device = "cpu"

        def score_continuations(self, prompt, continuations, per_token=False):
            return [[-2.0] * len(c) for c in continuations]

        def next_token_logprobs(self, context, suffixes, wanted):
            names = {v: k for k, v in vocabulary.items()}
            return [[-0.01 if t == 1 else likes.get(names.get(t), -12.0) for t in row]
                    for row in wanted]

    config = WordBagConfig(alpha=0.0, beta=0.0, beam=4, path_beam=2, word_branches=2,
                           min_words=min_words)
    return WordBagSelector(LM(), index, lambda text: [5], config)


def test_word_greedy_falls_into_single_word_trap():
    assert _trap_selector(1).select("text", set()).competency_id == "LAW"


def test_path_branches_reject_word_that_is_not_confirmed_later():
    found, calls = _trap_selector(2).select_paths("text", count=2)
    assert [competency for competency, _ in found] == ["DATA", "LAW"]
    data_path = found[0][1]
    assert "контракты" not in data_path.lemmas
    assert len(data_path.words) == 2 and data_path.score > found[1][1].score
    assert calls > 0


def test_path_branches_need_more_than_one_word_to_help():
    found, _ = _trap_selector(1).select_paths("text", count=1)
    assert found[0][0] == "LAW"  # One word ends the path: nothing later to compare.
