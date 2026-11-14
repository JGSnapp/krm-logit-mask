"""Experiment 21: how many search candidates the yes/no check needs, with controls.

For every profile the simple question of experiment 19 is asked once for every
competency of the catalog (margins kept). Then any pool can be scored offline:
  search K  - the first K competencies of the word-signal search (restarts with
              the found ones blocked, cached batched search of experiment 20);
  random K  - K random competencies (mean over 200 draws) - what the check alone
              does with a pool of the same size;
  all       - the whole catalog (brute force; does not scale, a ceiling only).
Selection: candidates re-ranked by margin, top `max_competencies` with margin > 0.

Usage: .venv/bin/python experiments/run_pool_eval.py [--eval-set ...] [--out ...]
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

from run_selection_eval import load_cases  # noqa: E402
from run_simple_verify_eval import NO, YES, first_tokens, question  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import NONE_ID, PhraseGraph  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.wordbag import (  # noqa: E402
    WordBagConfig, WordBagSelector, WordIndex, pymorphy_lemmatizer, word_messages,
)

POOLS = [3, 5, 10, 15, 20, 30]
DRAWS = 200


def pick(pool: list[str], margins: dict[str, float], limit: int) -> list[str]:
    return [c for c in sorted(pool, key=lambda c: -margins[c]) if margins[c] > 0][:limit]


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp21_pool")
    args = parser.parse_args()
    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    catalog_path = spec.get("catalog", "data/itmo_krm_3.json")
    catalog = Catalog.from_json(ROOT / catalog_path)
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

    selector = WordBagSelector(lm, index, prompt, WordBagConfig(alpha=0.0, beta=0.3, batch_paths=True))
    rng = random.Random(0)
    report: dict[str, object] = {
        "experiment": args.out, "model": args.model, "eval_set": args.eval_set,
        "catalog": catalog_path, "catalog_size": len(everything), "pools": POOLS,
        "profiles": {},
    }
    for case in cases:
        started = time.perf_counter()
        margins = {c: margin(case["text"], names[c]) for c in everything}
        all_seconds = time.perf_counter() - started
        limit = case["max_competencies"]
        acceptable = set(case["acceptable"])
        entry: dict[str, object] = {
            "kind": case["kind"], "acceptable": case["acceptable"],
            "check_all_seconds": round(all_seconds, 2),
            "margins": {c: round(m, 2) for c, m in margins.items()},
            "all": pick(everything, margins, limit) or [NONE_ID],
            "search": {}, "random": {},
        }
        for size in POOLS:
            torch.cuda.synchronize()
            started = time.perf_counter()
            found, calls = selector.select_paths(case["text"], size, coherent=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            pool = [c for c, _ in found]
            entry["search"][size] = {
                "pool": pool, "selected": pick(pool, margins, limit) or [NONE_ID],
                "acceptable_in_pool": len(set(pool) & acceptable),
                "search_seconds": round(seconds, 2), "calls": calls,
                "check_seconds": round(all_seconds / len(everything) * len(pool), 2),
            }
            hits = []
            for _ in range(DRAWS):
                drawn = pick(rng.sample(everything, size), margins, limit) or [NONE_ID]
                hits.append(len([c for c in drawn if c in acceptable])
                            if case["kind"] == "web" else int(drawn[0] in acceptable))
            entry["random"][size] = sum(hits) / DRAWS
        report["profiles"][case["id"]] = entry
        print(case["id"], f"all={entry['all']}",
              {k: (v["acceptable_in_pool"], v["selected"], v["search_seconds"])
               for k, v in entry["search"].items()}, flush=True)

    web = [p for p in report["profiles"].values() if p["kind"] == "web"]
    syn = [p for p in report["profiles"].values() if p["kind"] == "synthetic"]

    def precision(selections: list[list[str]], profiles) -> float:
        predicted = sum(len([c for c in s if c != NONE_ID]) for s in selections)
        hits = sum(len([c for c in s if c in p["acceptable"]]) for s, p in zip(selections, profiles))
        return round(hits / predicted, 3) if predicted else 0.0

    summary: dict[str, object] = {
        "all": {"web_precision": precision([p["all"] for p in web], web),
                "synthetic_hit_at_1": sum(p["all"][0] in p["acceptable"] for p in syn) / len(syn),
                "check_seconds_web": round(sum(p["check_all_seconds"] for p in web) / len(web), 2)},
    }
    for size in POOLS:
        key = str(size) if str(size) in web[0]["search"] else size
        summary[f"search_{size}"] = {
            "web_precision": precision([p["search"][key]["selected"] for p in web], web),
            "synthetic_hit_at_1": sum(p["search"][key]["selected"][0] in p["acceptable"]
                                      for p in syn) / len(syn),
            "acceptable_in_pool_web": [p["search"][key]["acceptable_in_pool"] for p in web],
            "seconds_web": round(sum(p["search"][key]["search_seconds"]
                                     + p["search"][key]["check_seconds"] for p in web) / len(web), 2),
        }
        summary[f"random_{size}"] = {
            "web_precision": round(sum(p["random"][key] for p in web) / (3 * len(web)), 3),
            "synthetic_hit_at_1": round(sum(p["random"][key] for p in syn) / len(syn), 3),
        }
    report["summary"] = summary
    for name, values in summary.items():
        print(name, json.dumps(values, ensure_ascii=False), flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
