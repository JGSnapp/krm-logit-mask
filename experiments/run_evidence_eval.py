"""Experiment 12: signals must rest on evidenced words (requirement or bonus).

Evidence lemmas of a profile:
  list - words the model writes itself in a free keyword list (greedy, one per line),
  text - words of the resume itself (lemmatized),
  both - union.
Search: phrase-coherent signals, alpha 0, beta 0.3 (experiment 7/9 frozen), plus
  require  - every signal needs at least one evidence word,
  bonus B  - evidence words get +B to their score.

Usage: .venv/bin/python experiments/run_evidence_eval.py [--eval-set ...] [--out ...] [--only ...]
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
    WordBagConfig, WordBagSelector, WordIndex, pymorphy_lemmatizer, split_words, word_messages,
)

# name -> (evidence source, require, bonus); source None = no evidence.
CONFIGS = {
    "base": (None, False, 0.0),
    "require_list": ("list", True, 0.0),
    "require_text": ("text", True, 0.0),
    "require_both": ("both", True, 0.0),
    "bonus1_list": ("list", False, 1.0),
    "bonus2_list": ("list", False, 2.0),
    "bonus1_text": ("text", False, 1.0),
    "bonus2_text": ("text", False, 2.0),
    "bonus2_both": ("both", False, 2.0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp12_evidence")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    catalog_path = spec.get("catalog", "data/itmo_krm_3.json")
    catalog = Catalog.from_json(ROOT / catalog_path)
    lm = LocalCausalLM.load(args.model)
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    lemmatize = pymorphy_lemmatizer()
    index = WordIndex.build(graph, lemmatize)
    catalog_lemmas = set(index.idf)

    def prompt(text: str) -> list[int]:
        return lm.prompt_ids(word_messages(text))[0].tolist()

    def lemmas_of(text: str) -> set[str]:
        return {lemma for word in split_words(text) if (lemma := lemmatize(word))} & catalog_lemmas

    selector = WordBagSelector(lm, index, prompt)
    lister = FirstPassScorer(selector, expand=0)  # Only the free keyword list is used.
    configs = {k: v for k, v in CONFIGS.items() if not args.only or k in args.only}
    report: dict[str, object] = {
        "experiment": args.out, "model": args.model, "eval_set": args.eval_set,
        "catalog": catalog_path, "search": "coherent signals, alpha 0, beta 0.3",
        "configs": {name: {"params": params, "rows": []} for name, params in configs.items()},
        "evidence": {},
    }
    for case in cases:
        started = time.perf_counter()
        listed = lister.evidence(case["text"])
        list_seconds = time.perf_counter() - started
        evidence = {"list": lemmas_of(listed.free_list), "text": lemmas_of(case["text"])}
        evidence["both"] = evidence["list"] | evidence["text"]
        report["evidence"][case["id"]] = {
            "free_list": listed.free_list, **{k: sorted(v) for k, v in evidence.items()},
        }
        print(case["id"], f"list {list_seconds:.1f}s", sorted(evidence["list"]), flush=True)
        for name, (source, require, bonus) in configs.items():
            selector.config = WordBagConfig(
                alpha=0.0, beta=0.3, require_evidence=require, evidence_bonus=bonus)
            selector.evidence = evidence[source] if source else set()
            started = time.perf_counter()
            found, calls = selector.select_paths(case["text"], case["max_competencies"], coherent=True)
            uses_list = source in ("list", "both")
            row = {
                "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                "selected": [c for c, _ in found],
                "signals": [[sorted(s) for s in path.signals] for _, path in found],
                "forward_calls": calls + (listed.forward_calls if uses_list else 0),
                "seconds": round(time.perf_counter() - started
                                 + (list_seconds if uses_list else 0.0), 2),
            }
            report["configs"][name]["rows"].append(row)
            print(name, case["id"], row["selected"], row["signals"],
                  f"{row['seconds']}s calls={row['forward_calls']}", flush=True)
    for name, entry in report["configs"].items():
        rows = entry["rows"]
        web = [r for r in rows if r["kind"] == "web"]
        summary = summarize(rows)
        summary["mean_forward_calls_web"] = sum(r["forward_calls"] for r in web) / len(web)
        summary["hits_per_profile"] = [
            len([c for c in r["selected"] if c in r["acceptable"]]) for r in web]
        entry["summary"] = summary
        print(name, json.dumps(summary), flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
