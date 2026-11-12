from krm_mask.graph import PhraseGraph
from krm_mask.masking import PhraseGraphMask, SourceSpanMask


class Tokenizer:
    eos_token_id = 0
    vocab = {"python": [1], "analytics": [2], "sql": [3], "\n": [9], ".": [8], " .": [8]}
    decoded = {0: "", 1: "p", 2: "a", 3: "s", 8: ".", 9: "\n"}

    def encode(self, text, add_special_tokens=False):
        return self.vocab[text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.decoded.get(i, "x") for i in ids)


def test_phrase_mask_is_order_independent_and_stops_when_unique():
    graph = PhraseGraph(
        phrases={
            "python": {"A", "B"},
            "analytics": {"A"},
            "sql": {"B"},
        },
        competency_names={"A": "Python analytics", "B": "SQL/Python"},
    )
    mask = PhraseGraphMask(Tokenizer(), graph, prompt_length=0)

    assert mask.allowed_tokens([]) == {1, 2, 3}
    assert mask.allowed_tokens([1]) == {9}
    assert mask.allowed_tokens([1, 9]) == {2, 3}
    assert mask.allowed_tokens([1, 9, 2]) == {0}
    assert mask.allowed_tokens([2]) == {0}
    assert mask.resolved_candidates([2]) == {"A"}
    assert mask.resolved_candidates([2, 0]) == {"A"}


def test_catalog_graph_can_use_indicators_keywords_or_both():
    from krm_mask.catalog import Catalog, Competency, Indicator

    catalog = Catalog(
        framework_id="test",
        framework_version="1",
        source_file="test.xlsx",
        competencies=[
            Competency(
                id="A",
                name="Canonical name",
                formulation="",
                keywords=["CUDA"],
                indicators=[Indicator(id="A.1", text="Ускоряет обучение")],
            )
        ],
        roles=[],
    )
    indicator_graph = PhraseGraph.from_catalog(
        catalog, phrase_sources={"indicators"}
    )
    keyword_graph = PhraseGraph.from_catalog(catalog, phrase_sources={"keywords"})
    both_graph = PhraseGraph.from_catalog(
        catalog, phrase_sources={"indicators", "keywords"}
    )
    competency_graph = PhraseGraph.from_catalog(catalog, phrase_sources={"names"})

    assert "Ускоряет обучение" in indicator_graph.phrases
    assert "CUDA" not in indicator_graph.phrases
    assert "Canonical name" not in indicator_graph.phrases
    assert "CUDA" in keyword_graph.phrases
    assert {"Ускоряет обучение", "CUDA"} <= set(both_graph.phrases)
    assert "Canonical name" in competency_graph.phrases
    assert "CUDA" not in competency_graph.phrases


def test_source_span_mask_never_jumps_between_spans():
    mask = SourceSpanMask([4, 5, 6, 4, 7], prompt_length=0, eos_token_id=0)
    assert mask.allowed_tokens([]) == {4, 5, 6, 7}
    assert mask.allowed_tokens([4]) == {5, 7}
    assert mask.allowed_tokens([4, 5]) == {6, 0}
    assert mask.allowed_tokens([4, 5, 7]) == {0}


def test_source_span_mask_can_exclude_previous_evidence():
    mask = SourceSpanMask(
        [4, 5, 6, 4, 7],
        prompt_length=0,
        eos_token_id=0,
        blocked_token_positions={0, 1, 2},
    )
    assert mask.allowed_tokens([]) == {4, 7}
    assert mask.allowed_tokens([4]) == {7}


def test_source_span_mask_can_block_only_previous_start():
    mask = SourceSpanMask(
        [4, 5, 4, 6],
        prompt_length=0,
        eos_token_id=0,
        blocked_start_positions={0},
    )
    assert mask.occurrences([4]) == [2]
    assert mask.allowed_tokens([4]) == {6}


def test_abort_token_is_always_available_and_invalidates_path():
    graph = PhraseGraph(
        phrases={"analytics": {"A"}}, competency_names={"A": "Analytics"}
    )
    mask = PhraseGraphMask(
        Tokenizer(), graph, prompt_length=0, allow_abort_token=True
    )
    assert 8 in mask.allowed_tokens([])
    assert mask.allowed_tokens([8]) == {0}
    assert mask.is_aborted([8])


def test_unique_prefix_can_require_two_extra_characters():
    tokenizer = Tokenizer()
    tokenizer.vocab = {**tokenizer.vocab, "long": [4, 5, 6]}
    tokenizer.decoded = {**tokenizer.decoded, 4: "a", 5: "b", 6: "c"}
    graph = PhraseGraph(phrases={"long": {"A"}}, competency_names={"A": "A"})
    mask = PhraseGraphMask(
        tokenizer, graph, prompt_length=0, resolution_extra_chars=2
    )
    assert mask.allowed_tokens([4]) == {5}
    assert mask.allowed_tokens([4, 5]) == {6}
    assert mask.allowed_tokens([4, 5, 6]) == {0}


def test_new_mode_finishes_the_full_phrase_before_eos():
    tokenizer = Tokenizer()
    tokenizer.vocab = {**tokenizer.vocab, "long": [4, 5, 6]}
    tokenizer.decoded = {**tokenizer.decoded, 4: "a", 5: "b", 6: "c"}
    graph = PhraseGraph(phrases={"long": {"A"}}, competency_names={"A": "A"})
    mask = PhraseGraphMask(
        tokenizer,
        graph,
        prompt_length=0,
        complete_phrase_before_eos=True,
    )

    assert mask.allowed_tokens([4]) == {5}
    assert mask.allowed_tokens([4, 5]) == {6}
    assert mask.allowed_tokens([4, 5, 6]) == {9}
    assert mask.allowed_tokens([4, 5, 6, 9]) == {0}


def test_consensus_mode_requires_two_compatible_phrases():
    graph = PhraseGraph(
        phrases={
            "python": {"A", "B"},
            "analytics": {"A"},
            "sql": {"B"},
        },
        competency_names={"A": "Python analytics", "B": "SQL/Python"},
    )
    mask = PhraseGraphMask(
        Tokenizer(),
        graph,
        prompt_length=0,
        complete_phrase_before_eos=True,
        min_complete_phrases=2,
    )

    # `analytics` alone resolves A, but the new mode requires another full,
    # compatible signal before it may finish.
    assert mask.allowed_tokens([2, 9]) == {1}
    assert mask.allowed_tokens([2, 9, 1]) == {9}
    assert mask.allowed_tokens([2, 9, 1, 9]) == {0}


def test_phrase_resolution_can_autocomplete_before_class_is_unique():
    graph = PhraseGraph(
        phrases={"python": {"A", "B"}, "analytics": {"A"}, "sql": {"B"}},
        competency_names={"A": "A", "B": "B"},
    )
    mask = PhraseGraphMask(
        Tokenizer(), graph, prompt_length=0, resolve_on_unique_phrase=True
    )

    assert mask.allowed_tokens([]) == {1, 2, 3}
    assert mask.resolved_phrases([2]) == {"analytics"}
    assert mask.allowed_tokens([2]) == {0}
    assert mask.resolved_candidates([2]) == {"A"}

