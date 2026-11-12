"""Experiment 6: word selection with alternative branches between words.

Usage: .venv/bin/python experiments/run_wordpaths_eval.py
Whole catalog, selection only. Top-3 (web) / top-1 (synthetic) competencies are
taken from the finished paths of one path search (repeated with blocking only
if fewer competencies were found).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_selection_eval import load_cases, summarize  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import PhraseGraph  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.wordbag import (  # noqa: E402
    WordBagConfig, WordBagSelector, WordIndex, pymorphy_lemmatizer, word_messages,
)

SCORINGS = [("raw", 0.0, 0.0), ("idf", 0.0, 1.0), ("pmi+idf", 1.0, 1.0)]
MIN_WORDS = [1, 2]


def main() -> None:
    catalog = Catalog.from_json(ROOT / "data/itmo_krm_3.json")
    lm = LocalCausalLM.load(DEFAULT_MODEL)
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    index = WordIndex.build(graph, pymorphy_lemmatizer())

    def prompt(text: str) -> list[int]:
        return lm.prompt_ids(word_messages(text))[0].tolist()

    selector = WordBagSelector(lm, index, prompt, WordBagConfig())
    cases, _ = load_cases(role_filter=False)
    base = WordBagConfig()
    report: dict[str, object] = {
        "experiment": "exp06_wordpaths", "model": DEFAULT_MODEL, "role_filter": False,
        "path_beam": base.path_beam, "word_branches": base.word_branches,
        "token_beam": base.beam, "max_words": base.max_words, "configs": {},
    }
    for scoring, alpha, beta in SCORINGS:
        for min_words in MIN_WORDS:
            name = f"{scoring}/min{min_words}"
            selector.config = WordBagConfig(alpha=alpha, beta=beta, min_words=min_words)
            rows = []
            for case in cases:
                started = time.perf_counter()
                found, calls = selector.select_paths(case["text"], case["max_competencies"])
                picks = [
                    {"competency_id": competency_id, "score": round(path.score, 2),
                     "words": [w.word for w in path.words]}
                    for competency_id, path in found
                ]
                rows.append({
                    "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                    "selected": [p["competency_id"] for p in picks], "picks": picks,
                    "forward_calls": calls, "seconds": round(time.perf_counter() - started, 2),
                })
                print(name, case["id"], [(p["competency_id"], p["words"]) for p in picks],
                      f"{rows[-1]['seconds']}s calls={calls}", flush=True)
            summary = summarize(rows)
            summary["mean_forward_calls_web"] = sum(
                r["forward_calls"] for r in rows if r["kind"] == "web") / 5
            report["configs"][name] = {
                "alpha": alpha, "beta": beta, "min_words": min_words,
                "summary": summary, "rows": rows,
            }
            print(name, json.dumps(summary), flush=True)
    out = ROOT / "experiments/results/exp06_wordpaths.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
