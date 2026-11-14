"""Experiment 24: narrow to a group of competencies instead of exactly one.

Before: a path finishes only when exactly one competency remains and is dropped
after `max_signals` (4) signals otherwise, so two similar competencies that
words cannot tell apart lose the evidence of the path. Variants:
  cap5      - a path that runs out of signals keeps its group of <= 5;
  group3    - a path finishes once <= 3 competencies remain (+ cap5);
  group5_1s - <= 5 competencies after a single signal (+ cap5).
All members of the group become candidates; the simple yes/no question of
experiment 19 (same 1.5B model, as in experiment 21) re-ranks the pool.

Usage: .venv/bin/python experiments/run_group_eval.py [--eval-set ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_pool_eval import pick  # noqa: E402
from run_selection_eval import load_cases  # noqa: E402
from run_simple_verify_eval import NO, YES, first_tokens, question  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import NONE_ID, PhraseGraph  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.wordbag import (  # noqa: E402
    WordBagConfig, WordBagSelector, WordIndex, pymorphy_lemmatizer, word_messages,
)

BASE = {"alpha": 0.0, "beta": 0.3, "batch_paths": True}
CONFIGS = {
    "base": {},
    "cap5": {"group_at_cap": 5},
    "group3": {"group_size": 3, "group_at_cap": 5},
    "group5_1s": {"group_size": 5, "min_signals": 1, "group_at_cap": 5},
}
POOLS = [10, 15]


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp24_group")
    args = parser.parse_args()
    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    catalog = Catalog.from_json(ROOT / spec.get("catalog", "data/itmo_krm_3.json"))
    names = {item.id: item.name for item in catalog.competencies}
    lm = LocalCausalLM.load(args.model)
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    index = WordIndex.build(graph, pymorphy_lemmatizer())
    everything = sorted(index.competency_lemmas)
    yes_ids, no_ids = first_tokens(lm.tokenizer, YES), first_tokens(lm.tokenizer, NO)

    def prompt(text: str) -> list[int]:
        return lm.prompt_ids(word_messages(text))[0].tolist()

    def margin(text: str, name: str) -> float:
        ids = lm.prompt_ids(question(text, name))
        with torch.inference_mode():
            logits = lm.model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        return float(torch.logsumexp(logp[yes_ids], 0) - torch.logsumexp(logp[no_ids], 0))

    selector = WordBagSelector(lm, index, prompt)
    report: dict[str, object] = {"experiment": args.out, "eval_set": args.eval_set,
                                 "base": BASE, "configs": {}}
    margins_by_case = {case["id"]: {c: margin(case["text"], names[c]) for c in everything}
                       for case in cases}
    for name, overrides in CONFIGS.items():
        selector.config = WordBagConfig(**BASE, **overrides)
        entry: dict[str, object] = {"overrides": overrides, "pools": {}}
        for size in POOLS:
            rows = []
            for case in cases:
                torch.cuda.synchronize()
                started = time.perf_counter()
                found, calls = selector.select_paths(case["text"], size, coherent=True)
                torch.cuda.synchronize()
                pool = [c for c, _ in found]
                margins = margins_by_case[case["id"]]
                selected = pick(pool, margins, case["max_competencies"]) or [NONE_ID]
                rows.append({
                    "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                    "pool": pool, "selected": selected,
                    "acceptable_in_pool": len(set(pool) & set(case["acceptable"])),
                    "group_paths": sum(len(p.candidates) > 1 for _, p in found),
                    "search_seconds": round(time.perf_counter() - started, 2), "calls": calls,
                })
            web = [r for r in rows if r["kind"] == "web"]
            syn = [r for r in rows if r["kind"] == "synthetic"]
            predicted = sum(len([c for c in r["selected"] if c != NONE_ID]) for r in web)
            hits = sum(len([c for c in r["selected"] if c in r["acceptable"]]) for r in web)
            summary = {
                "web_precision": round(hits / predicted, 3) if predicted else 0.0,
                "synthetic_hit_at_1": sum(r["selected"][0] in r["acceptable"] for r in syn) / len(syn),
                "acceptable_in_pool_web": [r["acceptable_in_pool"] for r in web],
                "acceptable_in_pool_total": sum(r["acceptable_in_pool"] for r in web),
                "pool_from_groups_web": [r["group_paths"] for r in web],
                "search_seconds_web": round(sum(r["search_seconds"] for r in web) / len(web), 2),
            }
            entry["pools"][size] = {"summary": summary, "rows": rows}
            print(name, size, json.dumps(summary, ensure_ascii=False), flush=True)
        report["configs"][name] = entry
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
