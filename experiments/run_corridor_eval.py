"""Experiment 17: corridor on the cut-off mass, DISC-like refills, summed paths.

Diagnosis (experiment 16): the beam drops good paths on their first word
("API" -8.4 vs "разработка" -3). Corridor search keeps the beam, but
  - a branch is dropped when its mean share of probability left by the mask
    (log-mass per word) falls more than `corridor` below the second-best branch;
  - free places are refilled with new first words drawn from a pool of 24 in
    proportion to exp(score / T), without repeating a tried first word;
  - optionally a competency is ranked by all its finished paths (log-sum-exp)
    instead of its best one.
The search rules (coherent signals, IDF 0.3) are those of experiments 7/9.

Usage: .venv/bin/python experiments/run_corridor_eval.py [--eval-set ...] [--out ...] [--only ...]
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

BASE = {"alpha": 0.0, "beta": 0.3}
CONFIGS = {
    "base": {},
    "base_sum": {"aggregate": "sum"},
    "refill_sum_s0": {"corridor_search": True, "aggregate": "sum", "seed": 0},
    "corridor3_s0": {"corridor_search": True, "corridor": 3.0, "seed": 0},
    "corridor3_s1": {"corridor_search": True, "corridor": 3.0, "seed": 1},
    "corridor3_sum_s0": {"corridor_search": True, "corridor": 3.0, "aggregate": "sum", "seed": 0},
    "corridor3_sum_s1": {"corridor_search": True, "corridor": 3.0, "aggregate": "sum", "seed": 1},
    "corridor3_T2_sum_s0": {"corridor_search": True, "corridor": 3.0, "aggregate": "sum",
                            "refill_temperature": 2.0, "seed": 0},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp17_corridor")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    catalog_path = spec.get("catalog", "data/itmo_krm_3.json")
    catalog = Catalog.from_json(ROOT / catalog_path)
    lm = LocalCausalLM.load(args.model)
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    index = WordIndex.build(graph, pymorphy_lemmatizer())

    def prompt(text: str) -> list[int]:
        return lm.prompt_ids(word_messages(text))[0].tolist()

    selector = WordBagSelector(lm, index, prompt)
    configs = {k: v for k, v in CONFIGS.items() if not args.only or k in args.only}
    report: dict[str, object] = {
        "experiment": args.out, "model": args.model, "eval_set": args.eval_set,
        "catalog": catalog_path, "base": BASE,
        "configs": {name: {"params": params, "rows": []} for name, params in configs.items()},
    }
    for name, params in configs.items():
        selector.config = WordBagConfig(**BASE, **params)
        for case in cases:
            started = time.perf_counter()
            found, calls = selector.select_paths(case["text"], case["max_competencies"], coherent=True)
            row = {
                "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                "selected": [c for c, _ in found] or [NONE_ID],
                "signals": [[sorted(s) for s in path.signals] for _, path in found],
                "first_words": [path.words[0].word for _, path in found],
                "forward_calls": calls,
                "seconds": round(time.perf_counter() - started, 2),
            }
            report["configs"][name]["rows"].append(row)
            print(name, case["id"], row["selected"], row["signals"],
                  f"{row['seconds']}s calls={calls}", flush=True)
        rows = report["configs"][name]["rows"]
        web = [r for r in rows if r["kind"] == "web"]
        summary = summarize(rows)
        summary["mean_forward_calls_web"] = sum(r["forward_calls"] for r in web) / len(web)
        summary["hits_per_profile"] = [
            len([c for c in r["selected"] if c in r["acceptable"]]) for r in web]
        report["configs"][name]["summary"] = summary
        print(name, json.dumps(summary), flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
