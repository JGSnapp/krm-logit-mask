"""Experiment 18: word-signal search proposes candidates, the model confirms them (ДА/НЕТ).

The search of experiments 7/9 (coherent signals, IDF 0.3) ranks up to 10
competencies. They are checked one by one, in that order, with a yes/no
question in the style of the pipeline's support gate (ДА vs НЕТ by mean token
log-likelihood, `LocalCausalLM.score_options`), until `max_competencies` are
confirmed. If none is confirmed the answer is __NONE__. The check sees one
candidate's name (and optionally its indicators) at a time, so the prompt does
not grow with the catalog.

Usage: .venv/bin/python experiments/run_verify_eval.py [--eval-set ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_selection_eval import load_cases, summarize  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import NONE_ID, PhraseGraph  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.wordbag import (  # noqa: E402
    WordBagConfig, WordBagSelector, WordIndex, pymorphy_lemmatizer, word_messages,
)

MAX_CANDIDATES = 10
# name -> (candidates checked, show indicators, required margin ДА - НЕТ)
CONFIGS = {
    "search_only": (3, None, None),
    "k6_indicators": (6, True, 0.0),
    "k10_indicators": (10, True, 0.0),
    "k10_indicators_margin1": (10, True, 1.0),
    "k10_name_only": (10, False, 0.0),
}


def gate_prompt(text: str, name: str, indicators: list[str] | None) -> str:
    listed = ""
    if indicators is not None:
        listed = "Индикаторы:\n" + "\n".join(f"- {item}" for item in indicators)[:3000] + "\n"
    return (
        "Проверь наличие свидетельства, не оценивая уровень. Ответ ДА, только если "
        "текст кандидата прямо упоминает компетенцию либо описывает конкретное действие, "
        "соответствующее хотя бы одному индикатору. Общее отсутствие подходящего "
        "опыта — это НЕТ.\n"
        f"Компетенция: {name}\n{listed}"
        f"Текст кандидата: {text}\nОтвет: "
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp18_verify")
    args = parser.parse_args()
    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    catalog_path = spec.get("catalog", "data/itmo_krm_3.json")
    catalog = Catalog.from_json(ROOT / catalog_path)
    competencies = {item.id: item for item in catalog.competencies}
    lm = LocalCausalLM.load(args.model)
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    index = WordIndex.build(graph, pymorphy_lemmatizer())

    def prompt(text: str) -> list[int]:
        return lm.prompt_ids(word_messages(text))[0].tolist()

    selector = WordBagSelector(lm, index, prompt, WordBagConfig(alpha=0.0, beta=0.3))
    report: dict[str, object] = {
        "experiment": args.out, "model": args.model, "eval_set": args.eval_set,
        "catalog": catalog_path, "max_candidates": MAX_CANDIDATES,
        "configs": {name: {"params": params, "rows": []} for name, params in CONFIGS.items()},
        "gate": {},
    }
    for case in cases:
        started = time.perf_counter()
        found, calls = selector.select_paths(case["text"], MAX_CANDIDATES, coherent=True)
        search_seconds = time.perf_counter() - started
        candidates = [competency_id for competency_id, _ in found]
        gate: dict[tuple[str, bool], float] = {}
        gate_seconds: dict[bool, float] = {True: 0.0, False: 0.0}
        for with_indicators in (True, False):
            for competency_id in candidates:
                item = competencies[competency_id]
                started = time.perf_counter()
                scores = lm.score_options(
                    gate_prompt(case["text"], item.name,
                                [i.text for i in item.indicators] if with_indicators else None),
                    ["ДА", "НЕТ"],
                )
                gate_seconds[with_indicators] += time.perf_counter() - started
                gate[(competency_id, with_indicators)] = scores["ДА"] - scores["НЕТ"]
        report["gate"][case["id"]] = [
            {"id": c, "name": competencies[c].name, "acceptable": c in case["acceptable"],
             "margin_indicators": round(gate[(c, True)], 2),
             "margin_name": round(gate[(c, False)], 2)}
            for c in candidates
        ]
        print(case["id"], f"search {search_seconds:.1f}s calls={calls}",
              [(c, round(gate[(c, True)], 2), c in case["acceptable"]) for c in candidates],
              flush=True)
        for name, (k, with_indicators, margin) in CONFIGS.items():
            limit = case["max_competencies"]
            if with_indicators is None:
                selected = candidates[:limit]
                checks = 0
            else:
                selected, checks = [], 0
                for competency_id in candidates[:k]:
                    checks += 1
                    if gate[(competency_id, with_indicators)] > margin:
                        selected.append(competency_id)
                        if len(selected) == limit:
                            break
            per_check = gate_seconds[bool(with_indicators)] / max(len(candidates), 1)
            report["configs"][name]["rows"].append({
                "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                "selected": selected or [NONE_ID], "checks": checks,
                "seconds": round(search_seconds + checks * per_check, 2),
            })
    for name, entry in report["configs"].items():
        rows = entry["rows"]
        web = [r for r in rows if r["kind"] == "web"]
        summary = summarize(rows)
        summary["hits_per_profile"] = [
            len([c for c in r["selected"] if c in r["acceptable"]]) for r in web]
        summary["predicted_per_profile"] = [
            len([c for c in r["selected"] if c != NONE_ID]) for r in web]
        summary["synthetic_selected"] = [r["selected"] for r in rows if r["kind"] == "synthetic"]
        entry["summary"] = summary
        print(name, json.dumps(summary, ensure_ascii=False), flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
