"""Experiment 20: speed of the search (cached resume prefix, batched paths, one pass).

1. Numeric check: `tail_logprobs` (resume prompt as a cached prefix, tails only)
   against `next_token_logprobs` (full recompute) on the same rows.
2. Same search, old vs batched: selected competencies must match; time and calls.
3. Candidates for the yes/no check (experiment 19): 10 by restarts (as before)
   vs one wide pass (path_beam 16 / 32, up to 20 candidates). Re-ranking by the
   simple question; candidate recall (acceptable among candidates) and precision.

Usage: .venv/bin/python experiments/run_speed_eval.py [--eval-set ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_selection_eval import load_cases, summarize  # noqa: E402
from run_simple_verify_eval import NO, YES, first_tokens, question  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import NONE_ID, PhraseGraph  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.wordbag import (  # noqa: E402
    WordBagConfig, WordBagSelector, WordIndex, pymorphy_lemmatizer, word_messages,
)

BASE = {"alpha": 0.0, "beta": 0.3}
# name -> (config overrides, number of candidates, verified by the question)
CONFIGS = {
    "top3_old": ({}, 3, False),
    "top3_batched": ({"batch_paths": True}, 3, False),
    "restarts10_batched": ({"batch_paths": True}, 10, True),
    "pass16_batched": ({"batch_paths": True, "single_pass": True, "path_beam": 16}, 20, True),
    "pass32_batched": ({"batch_paths": True, "single_pass": True, "path_beam": 32}, 20, True),
}


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp20_speed")
    args = parser.parse_args()
    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    catalog_path = spec.get("catalog", "data/itmo_krm_3.json")
    catalog = Catalog.from_json(ROOT / catalog_path)
    competencies = {item.id: item for item in catalog.competencies}
    lm = LocalCausalLM.load(args.model)
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    index = WordIndex.build(graph, pymorphy_lemmatizer())
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
    report: dict[str, object] = {
        "experiment": args.out, "model": args.model, "eval_set": args.eval_set,
        "catalog": catalog_path, "base": BASE, "numeric_check": {}, "configs": {},
    }

    # 1. Numeric check on random rows of the first profile.
    context = prompt(cases[0]["text"])
    rng = random.Random(0)
    surfaces = list(selector.tokens)
    tails = []
    for _ in range(48):
        words = [selector.tokens[rng.choice(surfaces)] for _ in range(rng.randint(0, 3))]
        tails.append(tuple(t for word in words for t in word)
                     + selector.tokens[rng.choice(surfaces)][: rng.randint(0, 2)])
    wanted = [[rng.randrange(1000, 100000) for _ in range(5)] for _ in tails]
    started = time.perf_counter()
    cached = lm.tail_logprobs(context, tails, wanted)
    cached_seconds = time.perf_counter() - started
    started = time.perf_counter()
    full = [lm.next_token_logprobs(context, [tail], [w])[0] for tail, w in zip(tails, wanted)]
    full_seconds = time.perf_counter() - started
    diffs = [abs(a - b) for x, y in zip(cached, full) for a, b in zip(x, y)]
    report["numeric_check"] = {
        "rows": len(tails), "max_abs_diff": round(max(diffs), 4),
        "mean_abs_diff": round(sum(diffs) / len(diffs), 5),
        "cached_seconds": round(cached_seconds, 3), "full_seconds": round(full_seconds, 3),
    }
    print("numeric", report["numeric_check"], flush=True)

    for name, (overrides, count, verified) in CONFIGS.items():
        selector.config = WordBagConfig(**BASE, **overrides)
        rows = []
        for case in cases:
            torch.cuda.synchronize()
            started = time.perf_counter()
            found, calls = selector.select_paths(case["text"], count, coherent=True)
            torch.cuda.synchronize()
            search_seconds = time.perf_counter() - started
            candidates = [c for c, _ in found]
            limit = case["max_competencies"]
            check_seconds = 0.0
            margins: dict[str, float] = {}
            if verified:
                started = time.perf_counter()
                margins = {c: margin(case["text"], competencies[c].name) for c in candidates}
                check_seconds = time.perf_counter() - started
                selected = [c for c in sorted(candidates, key=lambda c: -margins[c])
                            if margins[c] > 0][:limit]
            else:
                selected = candidates[:limit]
            rows.append({
                "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                "selected": selected or [NONE_ID], "candidates": candidates,
                "acceptable_in_candidates": len(set(candidates) & set(case["acceptable"])),
                "margins": {c: round(m, 2) for c, m in margins.items()},
                "forward_calls": calls, "search_seconds": round(search_seconds, 2),
                "check_seconds": round(check_seconds, 2),
                "seconds": round(search_seconds + check_seconds, 2),
            })
            print(name, case["id"], selected, f"cands={len(candidates)}",
                  f"acc_in={rows[-1]['acceptable_in_candidates']}",
                  f"search={search_seconds:.1f}s check={check_seconds:.1f}s calls={calls}",
                  flush=True)
        web = [r for r in rows if r["kind"] == "web"]
        summary = summarize(rows)
        summary.update({
            "hits_per_profile": [len([c for c in r["selected"] if c in r["acceptable"]]) for r in web],
            "acceptable_in_candidates_web": [r["acceptable_in_candidates"] for r in web],
            "candidates_web": [len(r["candidates"]) for r in web],
            "mean_search_seconds_web": round(sum(r["search_seconds"] for r in web) / len(web), 2),
            "mean_check_seconds_web": round(sum(r["check_seconds"] for r in web) / len(web), 2),
            "mean_forward_calls_web": sum(r["forward_calls"] for r in web) / len(web),
        })
        report["configs"][name] = {"overrides": overrides, "count": count,
                                   "verified": verified, "summary": summary, "rows": rows}
        print(name, json.dumps(summary, ensure_ascii=False), flush=True)
    old = {r["id"]: r["selected"] for r in report["configs"]["top3_old"]["rows"]}
    new = {r["id"]: r["selected"] for r in report["configs"]["top3_batched"]["rows"]}
    report["batched_matches_old"] = {case_id: old[case_id] == new[case_id] for case_id in old}
    print("batched matches old:", report["batched_matches_old"], flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
