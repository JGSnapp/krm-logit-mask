"""Experiment 7: word signals from single phrases (phrase coherence), IDF and floored PMI.

Usage: .venv/bin/python experiments/run_signals_eval.py
Whole catalog, selection only, path search with coherent signals
(2 words per signal, at least 2 signals, competencies intersected).
PMI floors are set from the distribution of word priors: the median and the
25th percentile of log P(word | empty text) over all writable word forms.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_selection_eval import load_cases, summarize  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import PhraseGraph  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.wordbag import (  # noqa: E402
    WordBagConfig, WordBagSelector, WordIndex, pymorphy_lemmatizer, word_messages,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--only", nargs="*", help="run only these configs")
    parser.add_argument("--out", default="exp07_signals")
    parser.add_argument("--cases", nargs="*", help="run only these profile ids")
    args = parser.parse_args()
    catalog = Catalog.from_json(ROOT / "data/itmo_krm_3.json")
    lm = LocalCausalLM.load(args.model)
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    index = WordIndex.build(graph, pymorphy_lemmatizer())

    def prompt(text: str) -> list[int]:
        return lm.prompt_ids(word_messages(text))[0].tolist()

    selector = WordBagSelector(lm, index, prompt, WordBagConfig())
    priors = sorted(selector.prior[selector.tokens[s]] for s in index.surfaces)
    quartiles = statistics.quantiles(priors, n=4)
    floors = {"p25": round(quartiles[0], 1), "median": round(quartiles[1], 1)}
    print(f"phrases {len(index.phrase_lemmas)}; word prior quartiles {quartiles}", flush=True)

    configs = [
        ("raw", 0.0, 0.0, None),
        ("idf0.3", 0.0, 0.3, None),
        ("idf1", 0.0, 1.0, None),
        ("pmi_floor_median", 1.0, 0.0, floors["median"]),
        ("pmi_floor_p25", 1.0, 0.0, floors["p25"]),
        ("pmi_floor_median+idf0.3", 1.0, 0.3, floors["median"]),
    ]
    if args.only:
        configs = [config for config in configs if config[0] in args.only]
    cases, _ = load_cases(role_filter=False)
    if args.cases:
        cases = [case for case in cases if case["id"] in args.cases]
    base = WordBagConfig()
    report: dict[str, object] = {
        "experiment": args.out, "model": args.model, "role_filter": False,
        "phrases": len(index.phrase_lemmas), "word_prior_quartiles": quartiles,
        "floors": floors, "words_per_signal": base.words_per_signal,
        "min_signals": base.min_signals, "max_signals": base.max_signals,
        "path_beam": base.path_beam, "word_branches": base.word_branches, "configs": {},
    }
    for name, alpha, beta, floor in configs:
        selector.config = WordBagConfig(alpha=alpha, beta=beta, prior_floor=floor)
        rows = []
        for case in cases:
            started = time.perf_counter()
            found, calls = selector.select_paths(
                case["text"], case["max_competencies"], coherent=True)
            picks = [
                {"competency_id": competency_id, "score": round(path.score, 2),
                 "signals": [sorted(signal) for signal in path.signals],
                 "words": [w.word for w in path.words]}
                for competency_id, path in found
            ]
            rows.append({
                "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                "selected": [p["competency_id"] for p in picks], "picks": picks,
                "forward_calls": calls, "seconds": round(time.perf_counter() - started, 2),
            })
            print(name, case["id"], [(p["competency_id"], p["signals"]) for p in picks],
                  f"{rows[-1]['seconds']}s calls={calls}", flush=True)
        summary = summarize(rows)
        summary["mean_forward_calls_web"] = sum(
            r["forward_calls"] for r in rows if r["kind"] == "web") / 5
        report["configs"][name] = {
            "alpha": alpha, "beta": beta, "prior_floor": floor,
            "summary": summary, "rows": rows,
        }
        print(name, json.dumps(summary), flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
