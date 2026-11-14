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


def _table_selector(phrases, likes, priors, config):
    """Selector with one token per word; log-probs and priors looked up by word."""
    pytest.importorskip("torch")
    names = {}
    for phrase, ids in phrases.items():
        names.update({c: c for c in ids})
    index = WordIndex.build(PhraseGraph(phrases, names), lemmatize)
    vocabulary = {}

    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            if text == "\n":
                return [1]
            return [vocabulary.setdefault(text.lower(), len(vocabulary) + 10)]

    def word(token):
        return {v: k for k, v in vocabulary.items()}.get(token)

    class LM:
        tokenizer = Tokenizer()
        device = "cpu"

        def score_continuations(self, prompt, continuations, per_token=False):
            return [[-0.1 if t == 1 else priors.get(word(t), -2.0) for t in c]
                    for c in continuations]

        def next_token_logprobs(self, context, suffixes, wanted):
            LM.calls += 1
            return [[-0.01 if t == 1 else likes.get(word(t), -12.0) for t in row]
                    for row in wanted]

        def tail_logprobs(self, prefix, tails, wanted):
            LM.batched_calls += 1
            return [[-0.01 if t == 1 else likes.get(word(t), -12.0) for t in row]
                    for row in wanted]

    LM.calls = 0
    LM.batched_calls = 0
    return WordBagSelector(LM(), index, lambda text: [5], config)


def test_coherent_signal_words_must_share_one_phrase():
    # "прогнозирование" and "обучение" both occur in ART, but in different phrases.
    phrases = {
        "прогнозирование спроса": {"ART"},
        "обучение художников": {"ART"},
        "обучение прогнозирование моделей": {"ML"},
    }
    likes = {"прогнозирование": -0.2, "обучение": -0.3}
    config = WordBagConfig(alpha=0.0, beta=0.0, beam=4, path_beam=4, word_branches=3,
                           words_per_signal=2, min_signals=1)
    selector = _table_selector(phrases, likes, {}, config)
    found, _ = selector.select_paths("text", count=1, coherent=True)
    competency, path = found[0]
    assert competency == "ML"
    assert path.signals == (frozenset({"прогнозирование", "обучение"}),)


def test_coherent_search_intersects_competencies_across_signals():
    phrases = {
        "python инструменты": {"PY", "SEC"},
        "git тестирование": {"PY", "QA"},
        "угрозы аудит": {"SEC"},
    }
    likes = {"python": -0.1, "инструменты": -0.2, "git": -0.3, "тестирование": -0.3}
    config = WordBagConfig(alpha=0.0, beta=0.0, beam=4, path_beam=4, word_branches=3,
                           words_per_signal=2, min_signals=2)
    found, _ = _table_selector(phrases, likes, {}, config).select_paths(
        "text", count=1, coherent=True)
    competency, path = found[0]
    assert competency == "PY" and len(path.signals) == 2


def test_prior_floor_stops_rare_word_bonus():
    phrases = {"onnx": {"RARE"}, "python": {"PY"}}
    likes = {"onnx": -6.0, "python": -1.0}
    priors = {"onnx": -20.0, "python": -2.0}

    def first_word(floor):
        config = WordBagConfig(alpha=1.0, beta=0.0, beam=4, prior_floor=floor)
        selector = _table_selector(phrases, likes, priors, config)
        options, _ = selector._word_options([5], {"onnx", "python"}, 1)
        return options[0].lemma

    assert first_word(None) == "onnx"  # PMI +14 for a word the model barely wants.
    assert first_word(-5.0) == "python"


def test_evidence_requirement_and_bonus_beat_generic_pairs():
    # "команда работа" is the model's favourite pair, but neither word is evidenced.
    phrases = {
        "команда работа": {"CI"},
        "сборка тестирование": {"CI"},
        "витрины данные": {"DATA"},
        "схемы данные": {"DATA"},
    }
    likes = {"команда": -0.1, "работа": -0.2, "сборка": -0.5, "тестирование": -0.5,
             "витрины": -2.0, "данные": -1.0, "схемы": -2.0}
    base = dict(alpha=0.0, beta=0.0, beam=8, path_beam=6, word_branches=5,
                words_per_signal=2, min_signals=2)

    def pick(**extra):
        selector = _table_selector(phrases, likes, {}, WordBagConfig(**base, **extra))
        selector.evidence = {"витрины", "данные", "схемы"}
        found, _ = selector.select_paths("text", count=1, coherent=True)
        return found[0][0]

    assert pick() == "CI"
    assert pick(require_evidence=True) == "DATA"
    assert pick(evidence_bonus=2.0) == "DATA"


def test_signal_window_and_phrase_veto_stop_template_matches():
    # BLD: two generic words far apart in a long template phrase; SIAM: adjacent words.
    phrases = {
        "разработка методов моделей для задач в процессах архитектурного проектирования": {"BLD"},
        "проектирования городов": {"BLD"},
        "сравнения поиска объектов": {"SIAM"},
        "разработка сравнения": {"SIAM"},
    }
    likes = {"разработка": -0.1, "проектирования": -0.2, "сравнения": -0.6, "поиска": -0.6,
             "объектов": -3.0, "городов": -9.0}
    base = dict(alpha=0.0, beta=0.0, beam=8, path_beam=6, word_branches=4,
                words_per_signal=2, min_signals=1)

    def pick(blocked_phrases=(), **extra):
        selector = _table_selector(phrases, likes, {}, WordBagConfig(**base, **extra))
        selector.blocked_phrases = set(blocked_phrases)
        found, _ = selector.select_paths("text", count=1, coherent=True)
        return found[0]

    competency, path = pick()
    assert competency == "BLD"  # "разработка ... проектирования" are 8 words apart.
    assert pick(signal_window=2)[0] == "SIAM"
    assert pick(blocked_phrases={0})[0] == "SIAM"
    index = _table_selector(phrases, likes, {}, WordBagConfig(**base)).index
    assert index.phrase_positions[2] == {"сравнения": [0], "поиска": [1], "объектов": [2]}


def test_anchor_word_makes_generic_paths_pay_for_the_domain_word():
    # Generic words lead to BLD; its key words ("градостроительный", "город") are
    # unlikely for this text. PY has a key word the model wants: "python".
    phrases = {
        "разработка проектирование": {"BLD"},
        "ии модель": {"BLD", "PY"},
        "градостроительный город": {"BLD"},
        "python разработка код": {"PY"},
        "проектирование код": {"PY"},
    }
    likes = {"разработка": -0.1, "проектирование": -0.2, "ии": -0.1, "модель": -0.2,
             "python": -1.0, "код": -1.5, "градостроительный": -12.0, "город": -12.0}
    priors = {"python": -6.0, "код": -6.0, "градостроительный": -10.0, "город": -10.0}
    base = dict(alpha=0.0, beta=0.0, beam=8, path_beam=6, word_branches=4,
                words_per_signal=2, min_signals=2)

    def found(count=1, **extra):
        selector = _table_selector(phrases, likes, priors, WordBagConfig(**base, **extra))
        return selector.select_paths("text", count=count, coherent=True)[0]

    assert found()[0][0] == "BLD"
    with_anchor = found(anchor_df=1)
    assert with_anchor[0][0] == "PY"
    assert "python" in with_anchor[0][1].lemmas
    # With a PMI threshold BLD cannot finish at all: its key word is not wanted.
    assert [c for c, _ in found(count=2, anchor_df=1, anchor_min_pmi=0.0)] == ["PY"]


def test_corridor_search_refills_with_improbable_first_word():
    # "api" and "rest" are not among the two likeliest first words, yet their path wins.
    phrases = {
        "разработка зебра": {"X"},
        "работа жираф": {"Y"},
        "api rest": {"INT"},
    }
    likes = {"разработка": -1.0, "работа": -1.2, "api": -3.0, "rest": -2.0,
             "зебра": -9.0, "жираф": -9.0}
    base = dict(alpha=0.0, beta=0.0, beam=8, path_beam=3, word_branches=2,
                words_per_signal=2, min_signals=1)

    def best(**extra):
        selector = _table_selector(phrases, likes, {}, WordBagConfig(**base, **extra))
        found, _ = selector.select_paths("text", count=1, coherent=True)
        return found[0]

    assert best()[0] in {"X", "Y"}  # Beam keeps the two likeliest first words only.
    competency, path = best(corridor_search=True, refill_pool=4)
    assert competency == "INT" and {"api", "rest"} <= path.lemmas
    assert all(word.log_mass < 0 for word in path.words)


def test_batched_paths_give_the_same_paths_with_fewer_calls():
    phrases = {
        "python инструменты": {"PY", "SEC"},
        "git тестирование": {"PY", "QA"},
        "угрозы аудит": {"SEC"},
        "тестирование качество": {"QA"},
    }
    likes = {"python": -0.1, "инструменты": -0.2, "git": -0.3, "тестирование": -0.3,
             "угрозы": -2.0, "аудит": -2.5, "качество": -1.0}
    base = dict(alpha=0.0, beta=0.3, beam=4, path_beam=4, word_branches=3,
                words_per_signal=2, min_signals=2)

    def run(**extra):
        selector = _table_selector(phrases, likes, {}, WordBagConfig(**base, **extra))
        found, calls = selector.select_paths("text", count=3, coherent=True)
        return [(c, round(p.score, 6), p.signals) for c, p in found], calls, selector.lm

    plain, plain_calls, plain_lm = run()
    batched, batched_calls, batched_lm = run(batch_paths=True)
    assert batched == plain
    assert batched_lm.batched_calls == batched_calls and batched_lm.calls == 0
    assert batched_calls < plain_calls
    single, _, _ = run(batch_paths=True, single_pass=True)
    assert single[0] == plain[0] and len(single) <= len(plain)
