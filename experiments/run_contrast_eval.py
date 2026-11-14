"""Experiment 23: contrastive quote score (how much the competency name raises a sentence).

For every selected competency (selections of experiment 22 for ITMO; a fresh
selection without quotes for other sets) all sentences of the resume are
scored under the quote request with the competency and under the same request
without a competency. The quote is the sentence with the best
    mean + w * (mean - neutral mean)
for w in WEIGHTS ("pure" = only the lift). Reported: the quotes and the number
of distinct quotes per profile (experiment 22: often one quote for all).

Usage: .venv/bin/python experiments/run_contrast_eval.py [--eval-set ...] [--reuse ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_selection_eval import load_cases  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import NONE_ID  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.selection import CompetencySelector, SelectionConfig  # noqa: E402

WEIGHTS = {"w0": 0.0, "w0.5": 0.5, "w1": 1.0, "w2": 2.0, "w4": 4.0, "pure": None,
           "sib1": ("sibling", 1.0), "sib2": ("sibling", 2.0), "sib4": ("sibling", 4.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--reuse", help="results JSON of experiment 22 with the selections")
    parser.add_argument("--out", default="exp23_contrast")
    args = parser.parse_args()
    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    catalog = Catalog.from_json(ROOT / spec.get("catalog", "data/itmo_krm_3.json"))
    selector = CompetencySelector(LocalCausalLM.load(args.model), catalog, SelectionConfig(quotes=False))
    reused = {}
    if args.reuse:
        reused = {row["id"]: row for row in json.loads((ROOT / args.reuse).read_text("utf-8"))["rows"]}
    report: dict[str, object] = {"experiment": args.out, "eval_set": args.eval_set,
                                 "weights": WEIGHTS, "rows": []}
    distinct: dict[str, list[int]] = {name: [] for name in WEIGHTS}
    for case in cases:
        if case["id"] in reused:
            picks = [(p["id"], p["name"]) for p in reused[case["id"]]["picks"] if p["id"] != NONE_ID]
        else:
            result = selector.select(case["text"], case["max_competencies"])
            picks = [(p["id"], p["name"]) for p in result["selected"] if p["id"] != NONE_ID]
        text = case["text"]
        row = {"id": case["id"], "picks": []}
        chosen: dict[str, list[str]] = {name: [] for name in WEIGHTS}
        print(f"\n## {case['id']}", flush=True)
        all_scores = {cid: selector.quote_scores(text, selector.competencies[cid]) for cid, _ in picks}
        for competency_id, name in picks:
            scores = all_scores[competency_id]
            sentences, means, neutral = scores["sentences"], scores["means"], scores["neutral"]
            siblings = [sum(all_scores[c]["means"][i] for c, _ in picks) / len(picks)
                        for i in range(len(sentences))]
            entry = {"id": competency_id, "name": name, "quotes": {}}
            for label, weight in WEIGHTS.items():
                if isinstance(weight, tuple):
                    values = [m + weight[1] * (m - b) for m, b in zip(means, siblings)]
                else:
                    values = [(m - n) if weight is None else m + weight * (m - n)
                              for m, n in zip(means, neutral)]
                best = max(range(len(sentences)), key=lambda i: values[i])
                quote = text[sentences[best][0]:sentences[best][1]]
                entry["quotes"][label] = {"text": quote, "lift": round(means[best] - neutral[best], 3),
                                          "mean": round(means[best], 3)}
                chosen[label].append(quote)
            row["picks"].append(entry)
            print(f"   {competency_id} {name[:60]}", flush=True)
            shown = set()
            for label in WEIGHTS:
                quote = entry["quotes"][label]
                if quote["text"] in shown:
                    continue
                shown.add(quote["text"])
                labels = [k for k in WEIGHTS if entry["quotes"][k]["text"] == quote["text"]]
                print(f"      [{', '.join(labels)}] (+{quote['lift']}) «{quote['text'][:170]}»", flush=True)
        for label in WEIGHTS:
            if chosen[label]:
                distinct[label].append(len(set(chosen[label])))
        row["distinct"] = {label: len(set(chosen[label])) for label in WEIGHTS}
        row["picked"] = len(picks)
        report["rows"].append(row)
    multi = [r for r in report["rows"] if r["picked"] > 1]
    report["summary"] = {
        label: {"distinct_quotes": sum(r["distinct"][label] for r in multi),
                "of_picks": sum(r["picked"] for r in multi)}
        for label in WEIGHTS
    }
    print("\ndistinct quotes per profile (profiles with >1 competency):",
          json.dumps(report["summary"], ensure_ascii=False), flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
