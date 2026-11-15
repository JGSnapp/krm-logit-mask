"""Experiment 26: both service modes on one model (quote last vs quote first).

Runs `CompetencySelector.select` ("quote last": search on the whole resume, pool
15, yes/no check, quote for each selected competency) and
`CompetencySelector.select_quote_first` ("quote first": 3 quotes, search on
each, yes/no check on the whole resume) - the code the service uses.

Usage: .venv/bin/python experiments/run_modes_eval.py --model Qwen/Qwen2.5-3B-Instruct [--eval-set ...] [--out ...]
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


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp26_modes")
    parser.add_argument("--modes", nargs="*", default=["quote_last", "quote_first"])
    args = parser.parse_args()
    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    catalog = Catalog.from_json(ROOT / spec.get("catalog", "data/itmo_krm_3.json"))
    selector = CompetencySelector(LocalCausalLM.load(args.model), catalog, SelectionConfig())
    report: dict[str, object] = {"experiment": args.out, "model": args.model,
                                 "eval_set": args.eval_set, "modes": {}}
    for mode in args.modes:
        run = selector.select if mode == "quote_last" else selector.select_quote_first
        rows = []
        for case in cases:
            torch.cuda.synchronize()
            result = run(case["text"], case["max_competencies"])
            torch.cuda.synchronize()
            selected = [item for item in result["selected"] if item["id"] != NONE_ID]
            rows.append({
                "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                "selected": [item["id"] for item in selected] or [NONE_ID],
                "margins": [item["margin"] for item in selected],
                "quotes": [(item.get("quote") or {}).get("text") for item in selected],
                "pool": [item["id"] for item in result["candidates"]],
                "acceptable_in_pool": len({c["id"] for c in result["candidates"]} & set(case["acceptable"])),
                "timings": {k: round(v, 2) for k, v in result["timings"].items()},
            })
            print(mode, case["id"], list(zip(rows[-1]["selected"], rows[-1]["margins"])),
                  rows[-1]["timings"], flush=True)
        web = [r for r in rows if r["kind"] == "web"]
        syn = [r for r in rows if r["kind"] == "synthetic"]
        predicted = sum(len([c for c in r["selected"] if c != NONE_ID]) for r in web)
        hits = sum(len([c for c in r["selected"] if c in r["acceptable"]]) for r in web)
        summary = {
            "web_precision": round(hits / predicted, 3) if predicted else 0.0,
            "web_hits": hits, "web_predicted": predicted,
            "synthetic_hit_at_1": sum(r["selected"][0] in r["acceptable"] for r in syn) / len(syn),
            "synthetic_selected": [r["selected"] for r in syn],
            "acceptable_in_pool_web": [r["acceptable_in_pool"] for r in web],
            "distinct_quotes_web": [len(set(q for q in r["quotes"] if q)) for r in web],
            "seconds_web": round(sum(r["timings"]["total_s"] for r in web) / len(web), 2),
        }
        report["modes"][mode] = {"summary": summary, "rows": rows}
        print(mode, json.dumps(summary, ensure_ascii=False), flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
