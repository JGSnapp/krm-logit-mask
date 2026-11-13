"""Experiment 11: first-pass shortlist of competencies, then the signal search of experiment 7.

The first pass (krm_mask.firstpass) ranks all competencies from the next-token
distributions of one free keyword list; the K best stay, the rest are blocked,
and the phrase-coherent signal search (IDF 0.3, as frozen in experiment 9) runs
only over them.

Usage: .venv/bin/python experiments/run_shortlist_eval.py [--eval-set ...] [--out ...] [--only ...]
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
from krm_mask.firstpass import FirstPassScorer  # noqa: E402
from krm_mask.graph import PhraseGraph  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.wordbag import (  # noqa: E402
    WordBagConfig, WordBagSelector, WordIndex, pymorphy_lemmatizer, word_messages,
)

# name -> (first-pass alpha, beta, depth, aggregate, K); K=None = no shortlist.
CONFIGS = {
    "full_search": None,
    "bag_k10": (1.0, 0.3, 2, "bag", 10),
    "bag_k20": (1.0, 0.3, 2, "bag", 20),
    "phrases_k10": (1.0, 0.3, 2, "phrases", 10),
    "phrases_k20": (1.0, 0.3, 2, "phrases", 20),
    "raw_bag_k10": (0.0, 0.0, 1, "bag", 10),
    "raw_bag_k20": (0.0, 0.0, 1, "bag", 20),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp11_shortlist")
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

    selector = WordBagSelector(lm, index, prompt, WordBagConfig(alpha=0.0, beta=0.3))
    scorer = FirstPassScorer(selector)
    configs = {k: v for k, v in CONFIGS.items() if not args.only or k in args.only}
    everything = set(index.competency_lemmas)
    report: dict[str, object] = {
        "experiment": args.out, "model": args.model, "eval_set": args.eval_set,
        "catalog": catalog_path, "competencies": len(everything),
        "search": "coherent signals, alpha 0, beta 0.3 (experiment 7/9 frozen)",
        "configs": {name: {"params": params, "rows": []} for name, params in configs.items()},
    }
    for case in cases:
        started = time.perf_counter()
        evidence = scorer.evidence(case["text"])
        first_seconds = time.perf_counter() - started
        print(case["id"], f"first pass {first_seconds:.1f}s", repr(evidence.free_list[:120]),
              flush=True)
        for name, params in configs.items():
            started = time.perf_counter()
            shortlist, rank = None, None
            if params is not None:
                alpha, beta, depth, aggregate, k = params
                scores = scorer.competency_scores(
                    scorer.word_values(evidence, alpha, beta, depth), aggregate)
                ranking = sorted(scores, key=scores.get, reverse=True)
                shortlist = ranking[:k]
                rank = min((ranking.index(c) + 1 for c in case["acceptable"] if c in ranking),
                           default=None)
            blocked = everything - set(shortlist) if shortlist is not None else set()
            found, calls = selector.select_paths(
                case["text"], case["max_competencies"], blocked=blocked, coherent=True)
            seconds = time.perf_counter() - started + (first_seconds if params else 0.0)
            row = {
                "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                "selected": [c for c, _ in found],
                "signals": [[sorted(s) for s in path.signals] for _, path in found],
                "shortlist": shortlist,
                "shortlist_hits": (len(set(shortlist) & set(case["acceptable"]))
                                   if shortlist is not None else None),
                "best_acceptable_rank": rank,
                "forward_calls": calls + (evidence.forward_calls if params else 0),
                "seconds": round(seconds, 2),
            }
            report["configs"][name]["rows"].append(row)
            print(name, case["id"], row["selected"], row["signals"],
                  f"{row['seconds']}s calls={row['forward_calls']}", flush=True)
    for name, entry in report["configs"].items():
        rows = entry["rows"]
        web = [r for r in rows if r["kind"] == "web"]
        summary = summarize(rows)
        summary["mean_forward_calls_web"] = sum(r["forward_calls"] for r in web) / len(web)
        summary["shortlist_hits_web"] = [r["shortlist_hits"] for r in web]
        entry["summary"] = summary
        print(name, json.dumps(summary), flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
