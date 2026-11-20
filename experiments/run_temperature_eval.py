"""Experiment 29: temperature and sampled paths in the word search (quote-first mode).

Hypothesis (user): the search takes the most likely words, so it keeps reaching
the same competencies; a larger model is surer still (experiments 26/28: 3B
puts 6-8 acceptable competencies in the pool, 1.5B puts 16). Drawing paths
from the model's own distribution at a higher temperature should give a more
varied pool, and the yes/no check then chooses from it.

Variants (only the search changes; quotes and the yes/no check stay at T = 1,
check answers are cached across variants):
  beam            - the current deterministic search (baseline), optionally at
                    other temperatures (log softmax(logits / T) in the beams);
  sample N, T     - N paths per quote drawn under the mask, every token in
                    proportion to P^(1/T) among the allowed ones, no IDF;
                    competencies ranked by how many paths reached them;
                    several seeds per setting (it is random).
Pool per quote = `per_quote` (5 in experiments 25-28).

Usage (GPU):
  python experiments/run_temperature_eval.py --model Qwen/Qwen2.5-3B-Instruct --beta 0
  python experiments/run_temperature_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --beta 0.3
  add --eval-set experiments/eval_set_mipt.json for MIPT (test: only the best ITMO setting)
Smoke test (CPU): --model Qwen/Qwen2.5-0.5B-Instruct --only syn-001 --samples 4 --temperatures 1.5 --seeds 0
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

from run_selection_eval import load_cases  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import NONE_ID  # noqa: E402
from krm_mask.model import LocalCausalLM  # noqa: E402
from krm_mask.selection import CompetencySelector, SelectionConfig  # noqa: E402


def sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(rows: list[dict]) -> dict:
    web = [r for r in rows if r["kind"] == "web"]
    syn = [r for r in rows if r["kind"] == "synthetic"]
    predicted = sum(len([c for c in r["selected"] if c != NONE_ID]) for r in web)
    hits = sum(len([c for c in r["selected"] if c in r["acceptable"]]) for r in web)
    return {
        "web_precision": round(hits / predicted, 3) if predicted else 0.0,
        "web_hits": hits, "web_predicted": predicted,
        "synthetic_hit_at_1": round(
            sum(r["selected"][0] in r["acceptable"] for r in syn) / max(len(syn), 1), 3),
        "acceptable_in_pool_total": sum(r["acceptable_in_pool"] for r in web),
        "acceptable_in_pool_web": [r["acceptable_in_pool"] for r in web],
        "pool_size_web": [r["pool_size"] for r in web],
        "seconds_web": round(sum(r["seconds"] for r in web) / max(len(web), 1), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--beta", type=float, default=None,
                        help="IDF weight of the beam search (exp. 28: 0 for 3B; service 0.3)")
    parser.add_argument("--beam-temperatures", nargs="*", type=float, default=[1.0])
    parser.add_argument("--samples", nargs="*", type=int, default=[8, 16, 32])
    parser.add_argument("--temperatures", nargs="*", type=float, default=[0.7, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--per-quote", nargs="*", type=int, default=[5, 10])
    parser.add_argument("--only", nargs="*", help="case ids to run (smoke tests)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    catalog = Catalog.from_json(ROOT / spec.get("catalog", "data/itmo_krm_3.json"))
    selector = CompetencySelector(LocalCausalLM.load(args.model), catalog, SelectionConfig())
    words = selector.words.config
    if args.beta is not None:
        words.beta = args.beta
    beam_defaults = {"beta": words.beta, "aggregate": words.aggregate}

    checked: dict[tuple[str, str], tuple[float, float]] = {}
    original_check = selector.check

    def cached_check(text, competency):
        key = (text, competency.id)
        if key not in checked:
            checked[key] = original_check(text, competency)
        return checked[key]

    selector.check = cached_check

    variants: list[tuple[str, dict]] = []
    for per_quote in args.per_quote:
        for t in args.beam_temperatures:
            variants.append((f"beam_T{t}_q{per_quote}", {
                "per_quote": per_quote, "samples": None, "temperature": t, "seed": 0,
                **beam_defaults}))
        for n in args.samples:
            for t in args.temperatures:
                for seed in args.seeds:
                    variants.append((f"sample{n}_T{t}_q{per_quote}_s{seed}", {
                        "per_quote": per_quote, "samples": n, "temperature": t, "seed": seed,
                        "beta": 0.0, "aggregate": "count"}))

    name = args.out or "exp29_temperature_" + args.model.split("/")[-1].lower() + (
        "_mipt" if "mipt" in args.eval_set else "")
    out = ROOT / "experiments/results" / f"{name}.json"
    report: dict[str, object] = {
        "experiment": name, "model": args.model, "eval_set": args.eval_set,
        "beam_beta": beam_defaults["beta"], "configs": {}, "grouped": {},
    }
    for label, settings in variants:
        selector.config.per_quote = settings["per_quote"]
        for key in ("samples", "temperature", "seed", "beta", "aggregate"):
            setattr(words, key, settings[key])
        rows = []
        for case in cases:
            sync()
            started = time.perf_counter()
            result = selector.select_quote_first(case["text"], case["max_competencies"])
            sync()
            seconds = time.perf_counter() - started
            selected = [i["id"] for i in result["selected"] if i["id"] != NONE_ID]
            pool = [c["id"] for c in result["candidates"]]
            rows.append({
                "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                "selected": selected or [NONE_ID], "pool": pool,
                "acceptable_in_pool": len(set(pool) & set(case["acceptable"])),
                "pool_size": len(pool),
                "candidates_by_quote": [q["candidates"] for q in result["quotes"]],
                "search_calls": result["search_forward_calls"],
                "search_s": round(result["timings"]["search_s"], 2), "seconds": round(seconds, 2),
            })
        summary = summarize(rows)
        report["configs"][label] = {"settings": settings, "summary": summary, "rows": rows}
        print(label, json.dumps(summary, ensure_ascii=False), flush=True)

        # Mean over seeds for the sampled settings.
        group = label.rsplit("_s", 1)[0] if settings["samples"] else label
        members = [c["summary"] for k, c in report["configs"].items()
                   if (k.rsplit("_s", 1)[0] if c["settings"]["samples"] else k) == group]
        report["grouped"][group] = {
            "runs": len(members),
            "precision_mean": round(statistics.mean(m["web_precision"] for m in members), 3),
            "acceptable_in_pool_mean": round(
                statistics.mean(m["acceptable_in_pool_total"] for m in members), 2),
            "pool_size_mean": round(statistics.mean(sum(m["pool_size_web"]) for m in members), 1),
            "synthetic_mean": round(statistics.mean(m["synthetic_hit_at_1"] for m in members), 3),
        }
        # Saved after every variant: a CUDA crash (exp. 28) must not lose finished runs.
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)
    for group, values in report["grouped"].items():
        print(group, values)


if __name__ == "__main__":
    main()
