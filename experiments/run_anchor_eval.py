"""Experiment 14: a path finishes on a competency only after one of its key words.

Key (anchor) words of a competency: its lemmas found in at most `df` competencies
of the catalog (from the catalog alone). The search keeps writing until the path
holds such a word; its log-probability is part of the path score, so a domain
word the model does not want ("градостроительный" for a backend resume) makes
the path lose. With a PMI threshold the key word also has to be more likely with
the resume than without it (priors are computed once per catalog); a profile
with no such path gets __NONE__.

Usage: .venv/bin/python experiments/run_anchor_eval.py [--eval-set ...] [--out ...] [--only ...]
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
from krm_mask.graph import NONE_ID, PhraseGraph  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.wordbag import (  # noqa: E402
    WordBagConfig, WordBagSelector, WordIndex, pymorphy_lemmatizer, word_messages,
)

# name -> (anchor_df, anchor_min_pmi)
CONFIGS = {
    "base": (None, None),
    "anchor_df1": (1, None),
    "anchor_df2": (2, None),
    "anchor_df3": (3, None),
    "anchor_df1_pmi0": (1, 0.0),
    "anchor_df2_pmi0": (2, 0.0),
    "anchor_df3_pmi0": (3, 0.0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp14_anchor")
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

    selector = WordBagSelector(lm, index, prompt)
    configs = {k: v for k, v in CONFIGS.items() if not args.only or k in args.only}
    report: dict[str, object] = {
        "experiment": args.out, "model": args.model, "eval_set": args.eval_set,
        "catalog": catalog_path, "search": "coherent signals, alpha 0, beta 0.3",
        "configs": {name: {"params": params, "rows": []} for name, params in configs.items()},
    }
    for name, (anchor_df, min_pmi) in configs.items():
        selector.config = WordBagConfig(
            alpha=0.0, beta=0.3, anchor_df=anchor_df, anchor_min_pmi=min_pmi)
        for case in cases:
            started = time.perf_counter()
            found, calls = selector.select_paths(case["text"], case["max_competencies"], coherent=True)
            row = {
                "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                "selected": [c for c, _ in found] or [NONE_ID],
                "signals": [[sorted(s) for s in path.signals] for _, path in found],
                "anchors": [
                    [(w.word, round(w.log_probability - w.log_prior, 1)) for w in path.words
                     if anchor_df and selector.df[w.lemma] <= anchor_df]
                    for _, path in found
                ],
                "forward_calls": calls,
                "seconds": round(time.perf_counter() - started, 2),
            }
            report["configs"][name]["rows"].append(row)
            print(name, case["id"], row["selected"], row["signals"], row["anchors"],
                  f"{row['seconds']}s calls={calls}", flush=True)
        rows = report["configs"][name]["rows"]
        web = [r for r in rows if r["kind"] == "web"]
        summary = summarize(rows)
        summary["mean_forward_calls_web"] = sum(r["forward_calls"] for r in web) / len(web)
        summary["hits_per_profile"] = [
            len([c for c in r["selected"] if c in r["acceptable"]]) for r in web]
        report["configs"][name]["summary"] = summary
        print(name, json.dumps(summary), flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
