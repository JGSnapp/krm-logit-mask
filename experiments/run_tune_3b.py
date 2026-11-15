"""Experiment 28: search settings retuned for Qwen 3B (quote-first mode).

The search was tuned on 1.5B (IDF weight 0.3, 6 paths, 3 next words per path);
on 3B it found far fewer acceptable candidates (experiment 26). Grid on ITMO
(validation): IDF weight `beta`, `path_beam`, `word_branches`; the best setting
by acceptable candidates (then precision) is checked on MIPT (test). Yes/no
answers do not depend on the search, so they are cached across settings.

Usage: .venv/bin/python experiments/run_tune_3b.py [--eval-set ...] [--only NAME ...] [--out ...]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_selection_eval import load_cases  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import NONE_ID  # noqa: E402
from krm_mask.model import LocalCausalLM  # noqa: E402
from krm_mask.selection import CompetencySelector, SelectionConfig  # noqa: E402

MODEL = "Qwen/Qwen2.5-3B-Instruct"
GRID = {
    f"b{beta}_p{beam}_w{branches}": {"beta": beta, "path_beam": beam, "word_branches": branches}
    for beta, beam, branches in itertools.product([0.0, 0.3, 0.6, 1.0], [6, 12], [3, 5])
}


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--modes", nargs="*", default=["quote_first"])
    parser.add_argument("--out", default="exp28_tune_3b")
    args = parser.parse_args()
    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    catalog = Catalog.from_json(ROOT / spec.get("catalog", "data/itmo_krm_3.json"))
    selector = CompetencySelector(LocalCausalLM.load(MODEL), catalog, SelectionConfig())
    base_words = selector.words.config
    checked: dict[tuple[str, str], tuple[float, float]] = {}
    original_check = selector.check

    def cached_check(text, competency):
        key = (text, competency.id)
        if key not in checked:
            checked[key] = original_check(text, competency)
        return checked[key]

    selector.check = cached_check
    grid = {k: v for k, v in GRID.items() if not args.only or k in args.only}
    report: dict[str, object] = {"experiment": args.out, "model": MODEL, "eval_set": args.eval_set,
                                 "configs": {}}
    for name, overrides in grid.items():
        for key, value in overrides.items():
            setattr(base_words, key, value)
        for mode in args.modes:
            run = selector.select_quote_first if mode == "quote_first" else selector.select
            rows = []
            for case in cases:
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = run(case["text"], case["max_competencies"])
                seconds = time.perf_counter() - started
                selected = [i["id"] for i in result["selected"] if i["id"] != NONE_ID]
                rows.append({
                    "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                    "selected": selected or [NONE_ID],
                    "acceptable_in_pool": len({c["id"] for c in result["candidates"]} & set(case["acceptable"])),
                    "pool_size": len(result["candidates"]),
                    "search_s": round(result["timings"]["search_s"], 2), "seconds": round(seconds, 2),
                })
            web = [r for r in rows if r["kind"] == "web"]
            syn = [r for r in rows if r["kind"] == "synthetic"]
            predicted = sum(len([c for c in r["selected"] if c != NONE_ID]) for r in web)
            hits = sum(len([c for c in r["selected"] if c in r["acceptable"]]) for r in web)
            summary = {
                "web_precision": round(hits / predicted, 3) if predicted else 0.0,
                "web_hits": hits, "web_predicted": predicted,
                "synthetic_hit_at_1": sum(r["selected"][0] in r["acceptable"] for r in syn) / len(syn),
                "acceptable_in_pool_total": sum(r["acceptable_in_pool"] for r in web),
                "acceptable_in_pool_web": [r["acceptable_in_pool"] for r in web],
                "pool_size_web": [r["pool_size"] for r in web],
                "seconds_web": round(sum(r["seconds"] for r in web) / len(web), 2),
            }
            report["configs"][f"{name}:{mode}"] = {"overrides": overrides, "mode": mode,
                                                    "summary": summary, "rows": rows}
            print(name, mode, json.dumps(summary, ensure_ascii=False), flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
