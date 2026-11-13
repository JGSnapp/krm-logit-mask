"""Experiment 13: proximity window inside signals and negative selection of phrases.

Mechanism found in experiment 12: a signal takes any two words of a phrase, so
two generic words far apart in a long template ("адаптация и разработка методов
и моделей ИИ в процессах архитектурного проектирования") lead to a domain
competency whose domain word was never written.

Window W: words of one signal must stand within W word positions in the phrase.
Veto (negative selection) from the first pass (krm_mask.firstpass, PMI of the
first one or two tokens, read for every catalog word at once):
  distinct q - a phrase is closed if its most distinctive words (max IDF in the
               phrase) all score below the q-quantile of this resume's word values;
  min q      - a phrase is closed if any of its words scores below the q-quantile.

Usage: .venv/bin/python experiments/run_window_veto_eval.py [--eval-set ...] [--out ...] [--only ...]
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
    WordBagConfig, WordBagSelector, WordIndex, pymorphy_lemmatizer, word_messages,
)

# name -> (window, veto kind, quantile)
CONFIGS = {
    "base": (None, None, None),
    "window2": (2, None, None),
    "window3": (3, None, None),
    "window5": (5, None, None),
    "veto_distinct30": (None, "distinct", 0.30),
    "veto_distinct50": (None, "distinct", 0.50),
    "veto_min10": (None, "min", 0.10),
    "veto_min25": (None, "min", 0.25),
    "window3+veto_distinct30": (3, "distinct", 0.30),
}


def vetoed_phrases(index: WordIndex, values: dict[str, float], kind: str, q: float) -> set[int]:
    ordered = sorted(values.values())
    threshold = ordered[int(q * (len(ordered) - 1))]
    blocked = set()
    for i, (_, lemmas) in enumerate(index.phrase_lemmas):
        scored = [l for l in lemmas if l in values]
        if not scored:
            continue
        if kind == "min":
            bad = min(values[l] for l in scored) < threshold
        else:
            top = max(index.idf.get(l, 0.0) for l in scored)
            bad = all(values[l] < threshold for l in scored if index.idf.get(l, 0.0) == top)
        if bad:
            blocked.add(i)
    return blocked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp13_window_veto")
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
    scorer = FirstPassScorer(selector)
    configs = {k: v for k, v in CONFIGS.items() if not args.only or k in args.only}
    report: dict[str, object] = {
        "experiment": args.out, "model": args.model, "eval_set": args.eval_set,
        "catalog": catalog_path, "phrases": len(index.phrase_lemmas),
        "search": "coherent signals, alpha 0, beta 0.3",
        "configs": {name: {"params": params, "rows": []} for name, params in configs.items()},
    }
    for case in cases:
        started = time.perf_counter()
        evidence = scorer.evidence(case["text"])
        values = scorer.word_values(evidence, alpha=1.0, beta=0.0, depth=2)
        first_seconds = time.perf_counter() - started
        print(case["id"], f"first pass {first_seconds:.1f}s", flush=True)
        for name, (window, kind, q) in configs.items():
            selector.config = WordBagConfig(alpha=0.0, beta=0.3, signal_window=window)
            selector.blocked_phrases = vetoed_phrases(index, values, kind, q) if kind else set()
            started = time.perf_counter()
            found, calls = selector.select_paths(case["text"], case["max_competencies"], coherent=True)
            row = {
                "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                "selected": [c for c, _ in found],
                "signals": [[sorted(s) for s in path.signals] for _, path in found],
                "blocked_phrases": len(selector.blocked_phrases),
                "forward_calls": calls + (evidence.forward_calls if kind else 0),
                "seconds": round(time.perf_counter() - started
                                 + (first_seconds if kind else 0.0), 2),
            }
            report["configs"][name]["rows"].append(row)
            print(name, case["id"], row["selected"], row["signals"],
                  f"blocked={row['blocked_phrases']} {row['seconds']}s calls={row['forward_calls']}",
                  flush=True)
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
