from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

from .catalog import Catalog, Competency
from .graph import NONE_ID, PhraseGraph
from .disc import Candidate, DiscDecodingError, select_disc
from .masking import PhraseGraphMask, SourceSpanMask
from .model import LocalCausalLM
from .tracing import trace_phrase_generation, trace_source_generation


_COMPLETION_WEIGHT_KEYS = (
    "disc_log_mass", "tail_log_probability", "tail_probability", "tail_count",
)


def _completion_weight(terminal: dict[str, object]) -> dict[str, object]:
    """Move the scored phrase tail from the dropped EOS event to the autocompletion."""
    return {key: terminal[key] for key in _COMPLETION_WEIGHT_KEYS if key in terminal}


EMPTY_HISTORY = "пока ничего не выбрано"
EMPTY_SIGNALS = "- пока нет"
MAX_SIGNAL_CYCLES = 8


def _direct_messages(text: str, history: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Выбери одну компетенцию, подтверждаемую текстом. Пиши только её "
                "допустимое название. Полное название будет автоматически подставлено, "
                "когда класс станет однозначным."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Текст кандидата:\n{text}\n\n"
                f"Уже обработано:\n{history}\n\n"
                "Найди следующую, ещё не обработанную компетенцию."
            ),
        },
    ]


def _signal_messages(text: str, history: str, rendered_signals: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Выбери следующий подтверждаемый текстом признак. "
                "Пиши только начало одного допустимого признака: когда он "
                "станет однозначным, система автоматически дополнит его."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Текст кандидата:\n{text}\n\n"
                f"Уже обработано:\n{history}\n\n"
                f"Уже выбранные признаки:\n{rendered_signals}\n\n"
                "Следующий совместимый признак:"
            ),
        },
    ]


@dataclass
class Extraction:
    competency_id: str
    competency_name: str
    selected_phrases: list[str]
    quote: str
    quotes: list[str]
    support_scores: dict[str, float]
    level: str
    level_scores: dict[str, float]


class ExtractionPipeline:
    def __init__(self, catalog: Catalog, lm: LocalCausalLM) -> None:
        self.catalog = catalog
        self.lm = lm
        self.by_id = {item.id: item for item in catalog.competencies}
        # (direct_mode, phrase) -> log P(phrase + delimiter | prompt without candidate text)
        self._phrase_prior: dict[tuple[bool, str], float] = {}

    def phrase_log_prior(self, graph: PhraseGraph, direct_mode: bool) -> dict[str, float]:
        """Prior of every graph phrase under the selection prompt with an empty text.

        It does not depend on the candidate, so each phrase is scored once per
        model and cached; the cost grows with the catalog, not with requests.
        """
        missing = [
            phrase for phrase in graph.phrases if (direct_mode, phrase) not in self._phrase_prior
        ]
        if missing:
            messages = (
                _direct_messages("", EMPTY_HISTORY) if direct_mode
                else _signal_messages("", EMPTY_HISTORY, EMPTY_SIGNALS)
            )
            tokenizer = self.lm.tokenizer
            delimiter = tokenizer.encode("\n", add_special_tokens=False)
            continuations = [
                tokenizer.encode(phrase, add_special_tokens=False) + delimiter
                for phrase in missing
            ]
            scores = self.lm.score_continuations(self.lm.prompt_ids(messages), continuations)
            for phrase, score in zip(missing, scores):
                self._phrase_prior[(direct_mode, phrase)] = score
        return {phrase: self._phrase_prior[(direct_mode, phrase)] for phrase in graph.phrases}

    def run(
        self,
        text: str,
        role_id: str | None = None,
        max_competencies: int = 8,
        max_quotes: int = 3,
        allow_abort_token: bool = False,
        resolution_extra_chars: int = 0,
        matching_basis: str = "both",
        use_disc: bool = False,
        disc_budget: int = 4,
        disc_prior_alpha: float = 0.0,
    ) -> list[dict[str, object]]:
        return self.run_detailed(
            text,
            role_id,
            max_competencies,
            max_quotes,
            allow_abort_token,
            resolution_extra_chars,
            matching_basis,
            use_disc,
            disc_budget,
            disc_prior_alpha,
        )["extractions"]

    def run_detailed(
        self,
        text: str,
        role_id: str | None = None,
        max_competencies: int = 8,
        max_quotes: int = 3,
        allow_abort_token: bool = False,
        resolution_extra_chars: int = 0,
        matching_basis: str = "both",
        use_disc: bool = False,
        disc_budget: int = 4,
        disc_prior_alpha: float = 0.0,
    ) -> dict[str, object]:
        total_started = perf_counter()
        allowed_ids = None
        if role_id:
            role = next(role for role in self.catalog.roles if role.id == role_id)
            allowed_ids = set(role.competencies)
        phrase_sources = {
            "competencies": {"names"},
            "indicators": {"indicators"},
            "keywords": {"keywords"},
            "both": {"indicators", "keywords"},
        }
        if matching_basis not in phrase_sources:
            raise ValueError(f"Unknown matching basis: {matching_basis}")
        graph = PhraseGraph.from_catalog(
            self.catalog, allowed_ids, phrase_sources[matching_basis]
        )
        found: set[str] = set()
        results: list[Extraction] = []
        attempts: list[dict[str, object]] = []

        for _ in range(max_competencies):
            selection_context = {
                "previous_competencies": [
                    str(item.get("competency_id"))
                    for item in attempts
                    if item.get("competency_id") != NONE_ID
                ],
                "previous_supported_quotes": sum(
                    1
                    for item in attempts
                    for quote in item.get("quotes", [])
                    if quote.get("supported")
                ),
            }
            selection_started = perf_counter()
            competency_id, phrases, selection_trace, selection_aborted = (
                self._select_competency(
                    text,
                    graph,
                    found,
                    attempts,
                    allow_abort_token,
                    resolution_extra_chars,
                    matching_basis != "competencies",
                    2 if matching_basis != "competencies" else 1,
                    matching_basis == "competencies",
                    use_disc=use_disc,
                    disc_budget=disc_budget,
                    disc_prior_alpha=disc_prior_alpha,
                )
            )
            selection_seconds = perf_counter() - selection_started
            if selection_aborted:
                attempts.append(
                    {
                        "status": "aborted_generation",
                        "competency_id": "__ABORT__",
                        "selected_phrases": phrases,
                        "selection_trace": selection_trace,
                        "context": selection_context,
                        "timings": {"selection_s": selection_seconds},
                    }
                )
                break
            if competency_id == NONE_ID:
                attempts.append(
                    {
                        "status": "no_competency",
                        "competency_id": NONE_ID,
                        "selected_phrases": phrases,
                        "selection_trace": selection_trace,
                        "context": selection_context,
                        "timings": {"selection_s": selection_seconds},
                    }
                )
                break
            competency = self.by_id[competency_id]
            blocked_positions: set[int] = set()
            quote_items: list[dict[str, object]] = []
            accepted_quotes: list[str] = []
            first_support_scores: dict[str, float] = {}

            for quote_index in range(max_quotes):
                quote_started = perf_counter()
                quote, quote_trace, used_positions, quote_aborted = self._copy_quote(
                    text,
                    competency,
                    blocked_positions,
                    quote_index,
                    accepted_quotes,
                    allow_abort_token,
                )
                quote_seconds = perf_counter() - quote_started
                if not quote:
                    break
                support_started = perf_counter()
                if quote_aborted:
                    supported, support_scores = False, {}
                else:
                    supported, support_scores = self._score_support(quote, competency)
                support_seconds = perf_counter() - support_started
                if not first_support_scores:
                    first_support_scores = support_scores
                quote_item: dict[str, object] = {
                        "index": quote_index,
                        "text": quote,
                        "supported": supported,
                        "aborted": quote_aborted,
                        "support_scores": support_scores,
                        "context_previous_quotes": len(accepted_quotes),
                        "trace": quote_trace,
                        "timings": {
                            "generation_s": quote_seconds,
                            "support_s": support_seconds,
                        },
                    }
                quote_items.append(quote_item)
                if not supported:
                    quote_item["continuation_decision"] = "ЗАКОНЧИТЬ"
                    quote_item["continuation_reason"] = (
                        "abort_token" if quote_aborted else "support_gate_rejected"
                    )
                    break
                accepted_quotes.append(quote)
                blocked_positions.update(used_positions)
                if quote_index + 1 >= max_quotes:
                    quote_item["continuation_decision"] = "ЗАКОНЧИТЬ"
                    quote_item["continuation_reason"] = "safety_limit"
                    break
                decision_started = perf_counter()
                should_continue, decision_scores = self._should_continue_quotes(
                    text, competency, accepted_quotes
                )
                decision_seconds = perf_counter() - decision_started
                quote_item["continuation_decision"] = (
                    "ЕСТЬ ЕЩЁ" if should_continue else "ЗАКОНЧИТЬ"
                )
                quote_item["continuation_scores"] = decision_scores
                quote_item["continuation_reason"] = "model_decision"
                quote_item["timings"]["continuation_s"] = decision_seconds
                if not should_continue:
                    break

            supported = bool(accepted_quotes)
            quote = accepted_quotes[0] if accepted_quotes else (
                str(quote_items[0]["text"]) if quote_items else ""
            )
            found.add(competency_id)
            quote_generation_aborted = any(
                bool(item.get("aborted")) for item in quote_items
            )
            attempt: dict[str, object] = {
                "status": (
                    "accepted"
                    if supported
                    else (
                        "aborted_generation"
                        if quote_generation_aborted
                        else "rejected_by_support_gate"
                    )
                ),
                "competency_id": competency.id,
                "competency_name": competency.name,
                "selected_phrases": phrases,
                "selection_trace": selection_trace,
                "context": selection_context,
                "quote": quote,
                "quotes": quote_items,
                "quote_trace": quote_items[0]["trace"] if quote_items else [],
                "quote_traces": [item["trace"] for item in quote_items],
                "support_scores": first_support_scores,
                "timings": {
                    "selection_s": selection_seconds,
                    "quotes_s": sum(
                        item["timings"]["generation_s"]
                        + item["timings"]["support_s"]
                        + item["timings"].get("continuation_s", 0.0)
                        for item in quote_items
                    ),
                },
            }
            attempts.append(attempt)
            if not supported:
                continue
            level_started = perf_counter()
            level, scores = self._score_level("\n".join(accepted_quotes), competency)
            level_seconds = perf_counter() - level_started
            attempt["level"] = level
            attempt["level_scores"] = scores
            attempt["timings"]["level_s"] = level_seconds
            attempt["timings"]["total_s"] = (
                selection_seconds + attempt["timings"]["quotes_s"] + level_seconds
            )
            results.append(
                Extraction(
                    competency_id=competency.id,
                    competency_name=competency.name,
                    selected_phrases=phrases,
                    quote=quote,
                    quotes=accepted_quotes,
                    support_scores=first_support_scores,
                    level=level,
                    level_scores=scores,
                )
            )
        validation_started = perf_counter()
        final_validation = self._validate_final_answer(text, attempts)
        final_validation["timing_s"] = perf_counter() - validation_started
        return {
            "text": text,
            "role_id": role_id,
            "matching_basis": matching_basis,
            "use_disc": use_disc,
            "disc_budget": disc_budget,
            "disc_prior_alpha": disc_prior_alpha,
            "extractions": [asdict(item) for item in results],
            "attempts": attempts,
            "final_validation": final_validation,
            "timings": {"analysis_s": perf_counter() - total_started},
        }

    def run_quote_first_detailed(
        self,
        text: str,
        role_id: str | None = None,
        max_quotes: int = 5,
        max_competencies_per_quote: int = 4,
        allow_abort_token: bool = False,
        resolution_extra_chars: int = 0,
        matching_basis: str = "both",
        use_disc: bool = False,
        disc_budget: int = 4,
        disc_prior_alpha: float = 0.0,
    ) -> dict[str, object]:
        total_started = perf_counter()
        allowed_ids = None
        if role_id:
            role = next(role for role in self.catalog.roles if role.id == role_id)
            allowed_ids = set(role.competencies)
        phrase_sources = {
            "competencies": {"names"},
            "indicators": {"indicators"},
            "keywords": {"keywords"},
            "both": {"indicators", "keywords"},
        }
        if matching_basis not in phrase_sources:
            raise ValueError(f"Unknown matching basis: {matching_basis}")
        graph = PhraseGraph.from_catalog(
            self.catalog, allowed_ids, phrase_sources[matching_basis]
        )
        blocked_starts: set[int] = set()
        selected_quote_texts: list[str] = []
        quote_results: list[dict[str, object]] = []
        validation_attempts: list[dict[str, object]] = []

        for quote_index in range(max_quotes):
            quote_started = perf_counter()
            quote, trace, start_position, aborted = self._copy_general_quote(
                text,
                selected_quote_texts,
                blocked_starts,
                allow_abort_token,
            )
            quote_seconds = perf_counter() - quote_started
            if not quote:
                break
            quote_result: dict[str, object] = {
                "index": quote_index,
                "text": quote,
                "start_token": start_position,
                "trace": trace,
                "aborted": aborted,
                "competencies": [],
                "timings": {"generation_s": quote_seconds},
            }
            quote_results.append(quote_result)
            if aborted:
                quote_result["continuation_decision"] = "ЗАКОНЧИТЬ"
                quote_result["continuation_reason"] = "abort_token"
                break
            if start_position is not None:
                blocked_starts.add(start_position)
            selected_quote_texts.append(quote)

            found_for_quote: set[str] = set()
            competency_attempts: list[dict[str, object]] = []
            for _ in range(max_competencies_per_quote):
                selection_started = perf_counter()
                competency_id, phrases, selection_trace, selection_aborted = (
                    self._select_competency(
                        quote,
                        graph,
                        found_for_quote,
                        competency_attempts,
                        allow_abort_token,
                        resolution_extra_chars,
                        matching_basis != "competencies",
                        2 if matching_basis != "competencies" else 1,
                        matching_basis == "competencies",
                        use_disc=use_disc,
                        disc_budget=disc_budget,
                        disc_prior_alpha=disc_prior_alpha,
                    )
                )
                selection_seconds = perf_counter() - selection_started
                if selection_aborted or competency_id == NONE_ID:
                    competency_attempts.append(
                        {
                            "status": (
                                "aborted_generation"
                                if selection_aborted
                                else "no_competency"
                            ),
                            "competency_id": (
                                "__ABORT__" if selection_aborted else NONE_ID
                            ),
                            "selected_phrases": phrases,
                            "selection_trace": selection_trace,
                            "timings": {"selection_s": selection_seconds},
                        }
                    )
                    break
                competency = self.by_id[competency_id]
                found_for_quote.add(competency_id)
                support_started = perf_counter()
                supported, support_scores = self._score_support(quote, competency)
                support_seconds = perf_counter() - support_started
                item: dict[str, object] = {
                    "status": (
                        "accepted" if supported else "rejected_by_support_gate"
                    ),
                    "competency_id": competency.id,
                    "competency_name": competency.name,
                    "selected_phrases": phrases,
                    "selection_trace": selection_trace,
                    "support_scores": support_scores,
                    "quote": quote,
                    "quotes": [{"text": quote, "supported": supported}],
                    "timings": {
                        "selection_s": selection_seconds,
                        "support_s": support_seconds,
                    },
                }
                competency_attempts.append(item)
                if supported:
                    level_started = perf_counter()
                    level, level_scores = self._score_level(quote, competency)
                    level_seconds = perf_counter() - level_started
                    item["level"] = level
                    item["level_scores"] = level_scores
                    item["timings"]["level_s"] = level_seconds
                    validation_attempts.append(item)

            quote_result["competencies"] = competency_attempts
            quote_result["timings"]["competencies_s"] = sum(
                sum(float(value) for value in item.get("timings", {}).values())
                for item in competency_attempts
            )
            if quote_index + 1 >= max_quotes:
                quote_result["continuation_decision"] = "ЗАКОНЧИТЬ"
                quote_result["continuation_reason"] = "safety_limit"
                break
            decision_started = perf_counter()
            should_continue, decision_scores = self._should_continue_general_quotes(
                text, selected_quote_texts
            )
            decision_seconds = perf_counter() - decision_started
            quote_result["continuation_decision"] = (
                "ЕСТЬ ЕЩЁ" if should_continue else "ЗАКОНЧИТЬ"
            )
            quote_result["continuation_scores"] = decision_scores
            quote_result["continuation_reason"] = "model_decision"
            quote_result["timings"]["continuation_s"] = decision_seconds
            if not should_continue:
                break

        validation_started = perf_counter()
        final_validation = self._validate_final_answer(text, validation_attempts)
        final_validation["timing_s"] = perf_counter() - validation_started
        return {
            "mode": "quote_first",
            "use_disc": use_disc,
            "disc_budget": disc_budget,
            "disc_prior_alpha": disc_prior_alpha,
            "text": text,
            "role_id": role_id,
            "matching_basis": matching_basis,
            "quotes": quote_results,
            "final_validation": final_validation,
            "timings": {"analysis_s": perf_counter() - total_started},
        }

    def _copy_general_quote(
        self,
        text: str,
        previous_quotes: list[str],
        blocked_starts: set[int],
        allow_abort_token: bool,
    ) -> tuple[str, list[dict[str, object]], int | None, bool]:
        previous = "\n".join(
            f"- {value}" for value in previous_quotes
        ) or "- пока нет"
        prompt = self.lm.prompt_ids(
            [
                {
                    "role": "system",
                    "content": (
                        "Скопируй один непрерывный дословный фрагмент резюме, "
                        "который описывает опыт работы, выполненную задачу, "
                        "применённый навык или достигнутый результат."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Исходное резюме:\n{text}\n\n"
                        f"Уже выбранные цитаты:\n{previous}\n\n"
                        "Следующая независимая цитата:"
                    ),
                },
            ]
        )
        encoded = self.lm.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        abort_ids: set[int] = set()
        if allow_abort_token:
            for value in (".", " ."):
                token_ids = self.lm.tokenizer.encode(value, add_special_tokens=False)
                if len(token_ids) == 1:
                    abort_ids.add(int(token_ids[0]))
        mask = SourceSpanMask(
            encoded["input_ids"],
            prompt_length=prompt.shape[1],
            eos_token_id=self.lm.tokenizer.eos_token_id,
            blocked_start_positions=blocked_starts,
            abort_token_ids=abort_ids,
        )
        generated, step_scores = self.lm.generate(
            prompt, [mask], max_new_tokens=96, return_scores=True
        )
        quote_ids = (
            generated[:-1]
            if generated and generated[-1] == self.lm.tokenizer.eos_token_id
            else generated
        )
        aborted = mask.is_aborted(generated)
        quote = self.lm.tokenizer.decode(
            quote_ids, skip_special_tokens=True
        ).strip()
        trace = trace_source_generation(
            mask,
            generated,
            self.lm.tokenizer,
            encoded["offset_mapping"],
            step_scores,
        )
        occurrences = mask.occurrences(quote_ids)
        return quote, trace, (occurrences[0] if occurrences else None), aborted

    def _should_continue_general_quotes(
        self, text: str, selected_quotes: list[str]
    ) -> tuple[bool, dict[str, float]]:
        rendered = "\n".join(
            f"{index + 1}. {quote}" for index, quote in enumerate(selected_quotes)
        )
        prompt = (
            "Определи, остался ли в резюме ещё один содержательно новый фрагмент "
            "с описанием опыта работы, выполненной задачи, применённого навыка "
            "или достигнутого результата, который стоит извлечь отдельной цитатой. "
            "Ответь ЕСТЬ ЕЩЁ или ЗАКОНЧИТЬ.\n"
            f"Резюме:\n{text}\nВыбранные цитаты:\n{rendered}\nОтвет: "
        )
        scores = self.lm.score_options(prompt, ["ЕСТЬ ЕЩЁ", "ЗАКОНЧИТЬ"])
        return scores["ЕСТЬ ЕЩЁ"] > scores["ЗАКОНЧИТЬ"], scores

    def _select_competency(
        self,
        text: str,
        graph: PhraseGraph,
        blocked: set[str],
        previous_attempts: list[dict[str, object]],
        allow_abort_token: bool,
        resolution_extra_chars: int,
        complete_phrase_before_eos: bool = False,
        min_complete_phrases: int = 1,
        direct_competency_mode: bool = False,
        use_disc: bool = False,
        disc_budget: int = 4,
        disc_prior_alpha: float = 0.0,
        disc_sampling: bool = False,
    ) -> tuple[str, list[str], list[dict[str, object]], bool]:
        if use_disc:
            # PMI target (experiment 1): divide each phrase's probability by its
            # prior without the candidate text, P(phrase | resume) / P(phrase | ∅)^α.
            prior = (
                self.phrase_log_prior(graph, direct_competency_mode)
                if disc_prior_alpha > 0 else {}
            )
            max_phrases = 1 if direct_competency_mode else MAX_SIGNAL_CYCLES
            log_bound = disc_prior_alpha * max_phrases * max(
                [-value for value in prior.values()] + [0.0]
            )

            def sample():
                value = self._select_competency(
                    text, graph, blocked, previous_attempts, allow_abort_token,
                    resolution_extra_chars, complete_phrase_before_eos,
                    min_complete_phrases, direct_competency_mode,
                    disc_sampling=True,
                )
                competency_id, phrases, trace, aborted = value
                # NONE is valid only when the explicit NONE phrase was selected.
                if competency_id == NONE_ID and not aborted and not any(
                    graph.phrases.get(phrase) == {NONE_ID} for phrase in phrases
                ):
                    raise DiscDecodingError("Signals did not resolve a unique competency")
                model_log_weight = sum(float(event.get("disc_log_mass", 0.0)) for event in trace)
                prior_log = sum(prior.get(phrase, 0.0) for phrase in phrases)
                return Candidate(
                    value=value,
                    log_weight=model_log_weight - disc_prior_alpha * prior_log,
                    details={
                        "competency_id": competency_id, "selected_phrases": phrases,
                        "aborted": aborted,
                        "decision_count": sum(event.get("forced") is False for event in trace),
                        "skipped_token_count": sum(event.get("forced") is True for event in trace),
                        "model_log_weight": model_log_weight,
                        "prior_log_probability": prior_log,
                    },
                )
            value, summary = select_disc(sample, budget=disc_budget, log_bound=log_bound)
            if value[2]:
                value[2][-1]["disc"] = summary
            return value
        history_lines: list[str] = []
        for attempt in previous_attempts:
            status = str(attempt.get("status", ""))
            competency = str(attempt.get("competency_name") or attempt.get("competency_id"))
            quotes = [
                str(item.get("text", ""))[:400]
                for item in attempt.get("quotes", [])
                if item.get("supported")
            ]
            history_lines.append(
                f"- {competency}: {status}; уже выбранные цитаты: "
                + (" | ".join(quotes) if quotes else "нет")
            )
        history = "\n".join(history_lines) or EMPTY_HISTORY
        if not direct_competency_mode:
            return self._select_competency_by_signals(
                text,
                graph,
                blocked,
                history,
                allow_abort_token,
                resolution_extra_chars,
                min_complete_phrases,
                disc_sampling,
            )
        prompt = self.lm.prompt_ids(_direct_messages(text, history))
        mask = PhraseGraphMask(
            self.lm.tokenizer,
            graph,
            prompt.shape[1],
            blocked_ids=blocked,
            allow_abort_token=allow_abort_token,
            resolution_extra_chars=resolution_extra_chars,
            complete_phrase_before_eos=complete_phrase_before_eos,
            min_complete_phrases=min_complete_phrases,
        )
        if disc_sampling:
            generated, step_scores = self.lm.sample_decision_path(
                prompt, mask, max_new_tokens=256 if complete_phrase_before_eos else 80,
            )
        else:
            generated, step_scores = self.lm.generate(
                prompt, [mask],
                max_new_tokens=256 if complete_phrase_before_eos else 80,
                return_scores=True,
            )
        content_ids = (
            generated[:-1]
            if generated and generated[-1] == self.lm.tokenizer.eos_token_id
            else generated
        )
        state = mask.parse(content_ids)
        resolved_candidates = mask.resolved_candidates(content_ids)
        trace = trace_phrase_generation(
            mask, generated, self.lm.tokenizer, step_scores
        )
        aborted = mask.is_aborted(generated)
        if aborted:
            return NONE_ID, ["."], trace, True
        selected = state.selected
        if not selected and content_ids:
            selected = [
                self.lm.tokenizer.decode(content_ids, skip_special_tokens=True).strip()
            ]
        if complete_phrase_before_eos and len(selected) < min_complete_phrases:
            return NONE_ID, selected, trace, False
        if len(resolved_candidates) != 1:
            return NONE_ID, state.selected, trace, False
        competency_id = next(iter(resolved_candidates))
        if direct_competency_mode:
            selected = [graph.competency_names[competency_id]]
            terminal: dict[str, object] = {}
            if trace and trace[-1].get("token_id") == self.lm.tokenizer.eos_token_id:
                terminal = trace.pop()
            trace.append(
                _completion_weight(terminal) |
                {
                    "step": len(trace),
                    "token_id": None,
                    "token_text": (
                        f"⇒ {competency_id} · "
                        f"{graph.competency_names[competency_id]}"
                    ),
                    "allowed_count": 0,
                    "allowed_tokens": [],
                    "candidates_before": [
                        {
                            "id": competency_id,
                            "name": graph.competency_names[competency_id],
                        }
                    ],
                    "candidates_after": [
                        {
                            "id": competency_id,
                            "name": graph.competency_names[competency_id],
                        }
                    ],
                    "selected_before": [],
                    "selected_after": list(selected),
                    "completed_phrase": graph.competency_names[competency_id],
                    "resolved_competency": competency_id,
                    "autocompleted": True,
                }
            )
        return competency_id, selected, trace, False

    def _select_competency_by_signals(
        self,
        text: str,
        graph: PhraseGraph,
        blocked: set[str],
        history: str,
        allow_abort_token: bool,
        resolution_extra_chars: int,
        min_signals: int,
        disc_sampling: bool = False,
    ) -> tuple[str, list[str], list[dict[str, object]], bool]:
        selected: list[str] = []
        combined_trace: list[dict[str, object]] = []
        candidates = set(graph.competency_names) - blocked

        for _ in range(MAX_SIGNAL_CYCLES):
            cycle_phrases = {
                phrase: (ids & candidates)
                for phrase, ids in graph.phrases.items()
                if phrase not in selected and (ids & candidates)
            }
            if selected:
                cycle_phrases.pop("НЕТ ПОДХОДЯЩЕЙ КОМПЕТЕНЦИИ", None)
            if not cycle_phrases:
                return NONE_ID, selected, combined_trace, False
            cycle_graph = PhraseGraph(
                phrases=cycle_phrases,
                competency_names=graph.competency_names,
            )
            rendered_signals = "\n".join(f"- {value}" for value in selected) or EMPTY_SIGNALS
            prompt = self.lm.prompt_ids(_signal_messages(text, history, rendered_signals))
            mask = PhraseGraphMask(
                self.lm.tokenizer,
                cycle_graph,
                prompt.shape[1],
                allow_abort_token=allow_abort_token,
                resolution_extra_chars=resolution_extra_chars,
                resolve_on_unique_phrase=True,
            )
            if disc_sampling:
                generated, step_scores = self.lm.sample_decision_path(
                    prompt, mask, max_new_tokens=80,
                )
            else:
                generated, step_scores = self.lm.generate(
                    prompt, [mask], max_new_tokens=80, return_scores=True
                )
            content_ids = (
                generated[:-1]
                if generated and generated[-1] == self.lm.tokenizer.eos_token_id
                else generated
            )
            cycle_trace = trace_phrase_generation(
                mask, generated, self.lm.tokenizer, step_scores
            )
            terminal: dict[str, object] = {}
            if generated and generated[-1] == self.lm.tokenizer.eos_token_id:
                terminal = cycle_trace[-1]
                cycle_trace = cycle_trace[:-1]
            if mask.is_aborted(generated):
                return NONE_ID, ["."], combined_trace + cycle_trace, True
            phrases = mask.resolved_phrases(content_ids)
            if len(phrases) != 1:
                return NONE_ID, selected, combined_trace + cycle_trace, False
            chosen = next(iter(phrases))
            selected.append(chosen)
            candidates = graph.candidates(selected) - blocked
            generated_prefix = self.lm.tokenizer.decode(
                content_ids, skip_special_tokens=True
            ).strip()
            completion = (
                chosen[len(generated_prefix) :]
                if chosen.startswith(generated_prefix)
                else chosen
            )
            names = graph.competency_names
            if cycle_trace:
                cycle_trace[-1]["completed_phrase"] = chosen
                cycle_trace[-1]["autocompleted_phrase"] = chosen
                cycle_trace[-1]["selected_after"] = list(selected)
                cycle_trace[-1]["candidates_after"] = [
                    {"id": value, "name": names.get(value, value)}
                    for value in sorted(candidates)
                ]
            combined_trace.extend(cycle_trace)
            combined_trace.append(
                _completion_weight(terminal) | {
                    "step": len(combined_trace),
                    "token_id": None,
                    "token_text": completion + "\n",
                    "allowed_count": 0,
                    "allowed_tokens": [],
                    "candidates_before": [
                        {"id": value, "name": names.get(value, value)}
                        for value in sorted(candidates)
                    ],
                    "candidates_after": [
                        {"id": value, "name": names.get(value, value)}
                        for value in sorted(candidates)
                    ],
                    "selected_before": list(selected),
                    "selected_after": list(selected),
                    "completed_phrase": chosen,
                    "autocompleted": True,
                }
            )
            if len(selected) >= min_signals and len(candidates) == 1:
                competency_id = next(iter(candidates))
                combined_trace.append(
                    {
                        "step": len(combined_trace),
                        "token_id": None,
                        "token_text": (
                            f"⇒ {competency_id} · "
                            f"{names.get(competency_id, competency_id)}"
                        ),
                        "allowed_count": 0,
                        "allowed_tokens": [],
                        "candidates_before": [
                            {
                                "id": competency_id,
                                "name": names.get(competency_id, competency_id),
                            }
                        ],
                        "candidates_after": [
                            {
                                "id": competency_id,
                                "name": names.get(competency_id, competency_id),
                            }
                        ],
                        "selected_before": list(selected),
                        "selected_after": list(selected),
                        "completed_phrase": None,
                        "resolved_competency": competency_id,
                        "autocompleted": True,
                    }
                )
                return competency_id, selected, combined_trace, False
        return NONE_ID, selected, combined_trace, False

    def _copy_quote(
        self,
        text: str,
        competency: Competency,
        blocked_positions: set[int] | None = None,
        quote_index: int = 0,
        previous_quotes: list[str] | None = None,
        allow_abort_token: bool = False,
    ) -> tuple[str, list[dict[str, object]], set[int], bool]:
        previous_quote_text = "\n".join(
            f"- {value}" for value in (previous_quotes or [])
        ) or "- пока нет"
        prompt = self.lm.prompt_ids(
            [
                {
                    "role": "system",
                    "content": "Скопируй один непрерывный дословный фрагмент исходного текста.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Компетенция: {competency.name}\n"
                        f"Исходный текст:\n{text}\n\n"
                        f"Уже выбранные цитаты, которые нельзя повторять:\n"
                        f"{previous_quote_text}\n\n"
                        f"Независимая цитата №{quote_index + 1}:"
                    ),
                },
            ]
        )
        encoded_source = self.lm.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        source_ids = encoded_source["input_ids"]
        offsets = encoded_source["offset_mapping"]
        abort_token_ids: set[int] = set()
        if allow_abort_token:
            for value in (".", " ."):
                encoded_abort = self.lm.tokenizer.encode(
                    value, add_special_tokens=False
                )
                if len(encoded_abort) == 1:
                    abort_token_ids.add(int(encoded_abort[0]))
        mask = SourceSpanMask(
            source_ids,
            prompt_length=prompt.shape[1],
            eos_token_id=self.lm.tokenizer.eos_token_id,
            blocked_token_positions=blocked_positions,
            abort_token_ids=abort_token_ids,
        )
        generated, step_scores = self.lm.generate(
            prompt, [mask], max_new_tokens=96, return_scores=True
        )
        if generated and generated[-1] == self.lm.tokenizer.eos_token_id:
            quote_ids = generated[:-1]
        else:
            quote_ids = generated
        quote = self.lm.tokenizer.decode(quote_ids, skip_special_tokens=True).strip()
        trace = trace_source_generation(
            mask, generated, self.lm.tokenizer, offsets, step_scores
        )
        occurrences = mask.occurrences(quote_ids)
        used_positions: set[int] = set()
        if occurrences:
            used_positions.update(
                range(occurrences[0], occurrences[0] + len(quote_ids))
            )
        return quote, trace, used_positions, mask.is_aborted(generated)

    def _score_level(
        self, quote: str, competency: Competency
    ) -> tuple[str, dict[str, float]]:
        anchors: dict[str, list[str]] = {"Б": [], "С": [], "П": []}
        for indicator in competency.indicators:
            for level in anchors:
                if indicator.levels.get(level):
                    anchors[level].append(indicator.levels[level])
        anchor_text = "\n".join(
            f"{level}: {' '.join(values)[:2500]}" for level, values in anchors.items()
        )
        prompt = (
            "Оцени только достаточность приведённой цитаты по якорям КРМ. "
            "UNKNOWN означает, что данных недостаточно.\n"
            f"Компетенция: {competency.name}\nЯкоря:\n{anchor_text}\n"
            f"Цитата: {quote}\nОтвет (одна метка): "
        )
        scores = self.lm.score_options(prompt, ["Б", "С", "П", "UNKNOWN"])
        return max(scores, key=scores.get), scores

    def _should_continue_quotes(
        self, text: str, competency: Competency, selected_quotes: list[str]
    ) -> tuple[bool, dict[str, float]]:
        rendered_quotes = "\n".join(
            f"{index + 1}. {quote}" for index, quote in enumerate(selected_quotes)
        )
        prompt = (
            "Определи, осталось ли в исходном тексте ещё одно независимое, "
            "непересекающееся и содержательно новое свидетельство этой компетенции. "
            "Не считай перефразирование или продолжение уже выбранной цитаты новым "
            "свидетельством. Ответь ЕСТЬ ЕЩЁ только при наличии конкретного нового "
            "фрагмента; иначе ответь ЗАКОНЧИТЬ.\n"
            f"Компетенция: {competency.name}\n"
            f"Исходный текст:\n{text}\n"
            f"Уже выбранные цитаты:\n{rendered_quotes}\nОтвет: "
        )
        scores = self.lm.score_options(prompt, ["ЕСТЬ ЕЩЁ", "ЗАКОНЧИТЬ"])
        return scores["ЕСТЬ ЕЩЁ"] > scores["ЗАКОНЧИТЬ"], scores

    def _validate_final_answer(
        self, text: str, attempts: list[dict[str, object]]
    ) -> dict[str, object]:
        accepted = [item for item in attempts if item.get("status") == "accepted"]
        if accepted:
            answer_parts: list[str] = []
            for item in accepted:
                quotes = " | ".join(
                    str(quote.get("text", ""))
                    for quote in item.get("quotes", [])
                    if quote.get("supported")
                )
                answer_parts.append(
                    f"- {item.get('competency_id')} {item.get('competency_name')}; "
                    f"уровень {item.get('level')}; цитаты: {quotes}"
                )
            rendered_answer = "\n".join(answer_parts)
        else:
            rendered_answer = "Подтверждённых компетенций не найдено."
        prompt = (
            "Проверь итог извлечения относительно только исходного резюме. Ответь ДА, "
            "если каждая принятая компетенция подтверждается указанными дословными "
            "цитатами, цитаты присутствуют в резюме, а присвоенные уровни не "
            "противоречат объёму свидетельств. Если есть хотя бы одно неподтверждённое "
            "утверждение, ответь НЕТ. Не добавляй новые компетенции.\n"
            f"Исходное резюме:\n{text}\n\n"
            f"Сгенерированный итог:\n{rendered_answer}\nОтвет: "
        )
        scores = self.lm.score_options(prompt, ["ДА", "НЕТ"])
        label = "ДА" if scores["ДА"] > scores["НЕТ"] else "НЕТ"
        return {"label": label, "valid": label == "ДА", "scores": scores}

    def _score_support(
        self, quote: str, competency: Competency
    ) -> tuple[bool, dict[str, float]]:
        indicators = "\n".join(
            f"- {indicator.text}" for indicator in competency.indicators
        )[:5000]
        prompt = (
            "Проверь наличие свидетельства, не оценивая уровень. Ответ ДА, только если "
            "цитата прямо упоминает компетенцию либо описывает конкретное действие, "
            "соответствующее хотя бы одному индикатору. Общее отсутствие технического "
            "опыта — это НЕТ.\n"
            f"Компетенция: {competency.name}\nИндикаторы:\n{indicators}\n"
            f"Цитата: {quote}\nОтвет: "
        )
        scores = self.lm.score_options(prompt, ["ДА", "НЕТ"])
        return scores["ДА"] > scores["НЕТ"], scores

