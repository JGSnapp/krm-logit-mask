"""Experiment 22: the updated system end to end, with quotes, for a qualitative review.

krm_mask.selection.CompetencySelector: word-signal search (pool 15) -> simple
yes/no check (re-rank by margin) -> for each selected competency a verbatim quote
("Выведи из текста цитату, подтверждающую её больше всего. Пиши сразу дословно,
как в тексте.", copied under a source-span mask).

Usage: .venv/bin/python experiments/run_quotes_eval.py [--eval-set ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_selection_eval import load_cases, summarize  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import NONE_ID  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.selection import CompetencySelector, SelectionConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp22_quotes")
    parser.add_argument("--pool", type=int, default=15)
    parser.add_argument("--reuse", help="results JSON: keep its selections, recompute quotes only")
    args = parser.parse_args()
    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    catalog_path = spec.get("catalog", "data/itmo_krm_3.json")
    catalog = Catalog.from_json(ROOT / catalog_path)
    selector = CompetencySelector(LocalCausalLM.load(args.model), catalog,
                                  SelectionConfig(pool=args.pool))
    report: dict[str, object] = {
        "experiment": args.out, "model": args.model, "eval_set": args.eval_set,
        "catalog": catalog_path, "pool": args.pool, "rows": [],
    }
    reused = {}
    if args.reuse:
        reused = {row["id"]: row for row in json.loads((ROOT / args.reuse).read_text("utf-8"))["rows"]}
    for case in cases:
        if case["id"] in reused:
            old = reused[case["id"]]
            picks = [dict(p) for p in old["picks"]]
            for pick in picks:
                if pick["id"] != NONE_ID:
                    pick["quote"] = selector.quote(case["text"], selector.competencies[pick["id"]])
            result = {"selected": picks, "candidates": [{"id": c, "margin": m} for c, m in old["candidates"]],
                      "timings": dict(old["timings"])}
        else:
            result = selector.select(case["text"], case["max_competencies"])
        selected = [item["id"] for item in result["selected"]]
        row = {
            "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
            "selected": selected, "seconds": round(result["timings"]["total_s"], 2),
            "timings": {k: round(v, 2) for k, v in result["timings"].items()},
            "picks": [
                {"id": item["id"], "name": item["name"], "acceptable": item["id"] in case["acceptable"],
                 "margin": item.get("margin"), "search_rank": item.get("search_rank"),
                 "signals": item.get("signals"), "quote": item.get("quote")}
                for item in result["selected"]
            ],
            "candidates": [(c["id"], c["margin"]) for c in result["candidates"]],
        }
        report["rows"].append(row)
        print(f"\n## {case['id']}  ({row['seconds']} s: {row['timings']})", flush=True)
        for pick in row["picks"]:
            if pick["id"] == NONE_ID:
                print("   — ничего не подходит", flush=True)
                continue
            quote = pick["quote"] or {}
            print(f"   {'+' if pick['acceptable'] else '-'} {pick['id']} {pick['name'][:60]}"
                  f"  [ДА−НЕТ {pick['margin']}, место в поиске {pick['search_rank']}]", flush=True)
            print(f"       «{quote.get('text')}»  (доля {round(quote.get('share') or 0, 2)})", flush=True)
            if quote.get("by_sum") and quote["by_sum"]["start"] != quote.get("start"):
                print(f"       по сумме: «{quote['by_sum']['text']}»", flush=True)
    summary = summarize(report["rows"])
    report["summary"] = summary
    print("\nsummary", json.dumps(summary, ensure_ascii=False), flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
