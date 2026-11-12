import math
import random
from types import SimpleNamespace

import pytest

from krm_mask.disc import Candidate, DiscDecodingError, select_disc
from krm_mask.model import LocalCausalLM


class RejectRng:
    def random(self):
        return 0.99

    def choices(self, pool, weights, k):
        self.weights = weights
        return [pool[-1]]


def test_fallback_draws_fresh_candidates_and_preserves_tiny_weights():
    values = iter(range(4))
    rng = RejectRng()

    def draw():
        value = next(values)
        return Candidate(value, -1000.0 - value, {"value": value})

    value, summary = select_disc(draw, budget=2, rng=rng)
    assert value == 3  # Not either rejected value, 0 or 1.
    assert summary["attempt_count"] == 4
    assert summary["selected_attempt"] == 4
    assert rng.weights == pytest.approx([1.0, math.exp(-1)])


def test_unit_weight_accepts_immediately():
    calls = []

    def draw():
        calls.append(True)
        return Candidate("A", 0.0, {})

    assert select_disc(draw, rng=random.Random(1))[0] == "A"
    assert len(calls) == 1


def test_invalid_passes_are_bounded():
    calls = []

    def draw():
        calls.append(True)
        raise DiscDecodingError("token limit")

    with pytest.raises(DiscDecodingError, match="all bounded passes"):
        select_disc(draw, budget=2)
    assert len(calls) == 4


class TableCache:
    """Minimal KV cache stand-in: remembers the fed tokens."""

    def __init__(self):
        self.tokens = []

    def get_seq_length(self):
        return len(self.tokens)

    def crop(self, drop):
        assert drop < 0  # transformers 5 accepts only "remove N tokens".
        del self.tokens[drop:]


class TableModel:
    """Next-token distribution looked up by the full context; leftover mass on token 8."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, input_ids, past_key_values=None, use_cache=True, logits_to_keep=1):
        import torch

        cache = past_key_values or TableCache()
        self.calls.append(input_ids.tolist()[0])
        rows = []
        for token in input_ids.tolist()[0]:
            cache.tokens.append(token)
            probabilities = torch.full((10,), 1e-12)
            spec = self.table.get(tuple(cache.tokens), {})
            for key, value in spec.items():
                probabilities[key] = value
            probabilities[8] += max(0.0, 1.0 - sum(spec.values()))
            rows.append(probabilities.log())
        logits = torch.stack(rows)[-logits_to_keep:][None]
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_forced_tokens_skip_forward_but_are_scored_from_next_forward(monkeypatch):
    torch = pytest.importorskip("torch")
    model = TableModel({
        (9,): {1: 0.2, 2: 0.3},
        (9, 1): {3: 0.5},
        (9, 1, 3): {4: 0.4},
        (9, 1, 3, 4): {5: 0.08, 6: 0.12},
    })

    class Mask:
        eos_token_id = 0
        abort_token_ids = set()

        def allowed_tokens(self, prefix):
            return {(): {1, 2}, (1,): {3}, (1, 3): {4},
                    (1, 3, 4): {5, 6}, (1, 3, 4, 5): {0}}[tuple(prefix)]

    monkeypatch.setattr(torch, "multinomial", lambda p, n: torch.tensor([0]))
    lm = LocalCausalLM(model, SimpleNamespace(eos_token_id=0))
    generated, scores = lm.sample_decision_path(torch.tensor([[9]]), Mask(), 10)
    assert generated == [1, 3, 4, 5, 0]
    assert model.calls == [[9], [1, 3, 4]]  # No forward of their own for 3 and 4.
    # Legal mass .5 at the first fork, P(3)=.5 and P(4)=.4 forced, .2 at the second fork.
    assert sum(s["disc_log_mass"] for s in scores) == pytest.approx(math.log(0.02))
    assert scores[1]["model_probability"] == pytest.approx(0.5)
    assert scores[2]["model_probability"] == pytest.approx(0.4)
    assert [s["forced"] for s in scores] == [False, True, True, False, True]
    assert "probability" not in scores[1]


def test_path_limit_is_failure_not_partial_candidate():
    torch = pytest.importorskip("torch")

    class Mask:
        eos_token_id = 0
        abort_token_ids = set()

        def allowed_tokens(self, prefix):
            return {1}

    lm = LocalCausalLM(None, None)
    with pytest.raises(DiscDecodingError, match="Token limit"):
        lm.sample_decision_path(torch.tensor([[9]]), Mask(), 3)


def _phrase_tokenizer(vocabulary):
    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return vocabulary[text]

    return Tokenizer()


def test_unique_phrase_stops_early_but_scores_unwritten_suffix(monkeypatch):
    torch = pytest.importorskip("torch")
    from krm_mask.graph import PhraseGraph
    from krm_mask.masking import PhraseGraphMask

    tokenizer = _phrase_tokenizer({"long A": [1, 3, 4], "long B": [2, 3, 4], "\n": [9]})
    graph = PhraseGraph({"long A": {"A"}, "long B": {"B"}}, {"A": "A", "B": "B"})
    mask = PhraseGraphMask(tokenizer, graph, 1, resolve_on_unique_phrase=True)
    model = TableModel({
        (9,): {1: 0.6, 2: 0.4},
        (9, 1): {3: 0.5},
        (9, 1, 3): {4: 0.5},
        (9, 1, 3, 4): {9: 0.5},
    })
    monkeypatch.setattr(torch, "multinomial", lambda p, n: torch.tensor([0]))
    generated, scores = LocalCausalLM(model, tokenizer).sample_decision_path(
        torch.tensor([[9]]), mask, 10
    )
    assert generated == [1, 0]  # Generation still stops once the phrase is unique.
    assert mask.resolved_phrases(generated) == {"long A"}
    # Fork mass is 1 (both phrases legal); the unwritten "3 4 \\n" has .5^3.
    assert scores[-1]["tail_probability"] == pytest.approx(0.125)
    assert sum(s["disc_log_mass"] for s in scores) == pytest.approx(math.log(0.125))
    assert model.calls == [[9], [1], [3, 4]]


def test_paper_figure_1_trap_gets_low_weight(monkeypatch):
    """Ye et al. (2025), Fig. 1: "soccer" is likely only because of "soccer shoes",
    which is not in the set; the only legal tail "gloves" is unlikely."""
    torch = pytest.importorskip("torch")
    from krm_mask.graph import PhraseGraph
    from krm_mask.masking import PhraseGraphMask

    # soccer=1 gloves=2 used=3 shoes=4 shirts=5, delimiter=9.
    tokenizer = _phrase_tokenizer({
        "soccer gloves": [1, 2], "used shirts": [3, 5],
        "used soccer shoes": [3, 1, 4], "\n": [9],
    })
    graph = PhraseGraph(
        {"soccer gloves": {"G"}, "used shirts": {"S"}, "used soccer shoes": {"U"}},
        {"G": "G", "S": "S", "U": "U"},
    )
    mask = PhraseGraphMask(tokenizer, graph, 1, resolve_on_unique_phrase=True)
    model = TableModel({
        (7,): {1: 0.6, 3: 0.4},
        (7, 1): {2: 0.1},  # The model wanted "shoes" (mass on 8).
        (7, 1, 2): {9: 1.0},
    })
    monkeypatch.setattr(torch, "multinomial", lambda p, n: torch.tensor([0]))
    generated, scores = LocalCausalLM(model, tokenizer).sample_decision_path(
        torch.tensor([[7]]), mask, 10
    )
    assert generated == [1, 0]
    # Before the fix this path had weight 1 and was always accepted.
    assert math.exp(sum(s["disc_log_mass"] for s in scores)) == pytest.approx(0.1)


def test_completion_tails_cover_every_phrase_of_resolved_class():
    from krm_mask.graph import PhraseGraph
    from krm_mask.masking import PhraseGraphMask

    tokenizer = _phrase_tokenizer({"alpha one": [1, 2, 3], "alpha two": [1, 2, 4],
                                   "beta": [5], "\n": [9]})
    graph = PhraseGraph({"alpha one": {"A"}, "alpha two": {"A"}, "beta": {"B"}},
                        {"A": "A", "B": "B"})
    mask = PhraseGraphMask(tokenizer, graph, 1)
    assert mask.allowed_tokens([1]) == {0}  # Class A is already unique.
    assert sorted(mask.completion_tails([1, 0])) == [(2, 3, 9), (2, 4, 9)]
    assert mask.completion_tails([1, 2, 3, 9, 0]) == []  # Phrase written in full.


def test_disc_corrects_distribution_on_branching_graph():
    # P(a)=.4*.2=.08, P(b)=.6*.8=.48; masked proposal picks .4/.6.
    rng = random.Random(42)

    def sample():
        a = rng.random() < 0.4
        return Candidate("a" if a else "b", math.log(0.2 if a else 0.8), {})

    results = [select_disc(sample, budget=8, rng=rng)[0] for _ in range(3000)]
    assert results.count("a") / len(results) == pytest.approx(1 / 7, abs=0.025)


def test_controller_wraps_all_signals_and_autocompletes_each(monkeypatch):
    torch = pytest.importorskip("torch")
    from krm_mask.catalog import Catalog
    from krm_mask.graph import PhraseGraph
    from krm_mask.pipeline import ExtractionPipeline

    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return {"python": [1], "analytics": [2], "sql": [3], "\n": [9]}[text]

        def decode(self, ids, skip_special_tokens=True):
            return "".join({1: "p", 2: "a", 3: "s"}.get(i, "") for i in ids)

    class LM:
        tokenizer = Tokenizer()

        def __init__(self):
            self.prompts = []

        def prompt_ids(self, messages):
            self.prompts.append(messages[-1]["content"])
            return torch.tensor([[9]])

        def sample_decision_path(self, prompt, mask, max_new_tokens):
            token = 1 if len(self.prompts) % 2 else 2
            return [token, 0], [{"forced": False, "disc_log_mass": math.log(0.5)},
                                {"forced": True, "disc_log_mass": 0.0}]

    class AcceptRng:
        def random(self):
            return 0.0

    monkeypatch.setattr("krm_mask.disc.random.Random", AcceptRng)
    lm = LM()
    graph = PhraseGraph({"python": {"A", "B"}, "analytics": {"A"}, "sql": {"B"}},
                        {"A": "A", "B": "B"})
    pipeline = ExtractionPipeline(Catalog("test", "1", "test", [], []), lm)
    competency, phrases, trace, aborted = pipeline._select_competency(
        "Python analytics", graph, set(), [], False, 0,
        min_complete_phrases=2, use_disc=True,
    )
    assert competency == "A" and not aborted
    assert phrases == ["python", "analytics"]
    assert "- python" in lm.prompts[1]  # Full completion enters next signal prompt.
    assert trace[-1]["disc"]["weight"] == pytest.approx(0.25)
    assert trace[-1]["disc"]["attempts"][0]["decision_count"] == 2
