"""Experiment 5: word-bag selection from the whole catalog.

Usage: .venv/bin/python experiments/run_wordbag_eval.py
Selection only (no quotes/levels): top-3 competencies for web profiles, top-1 for
synthetic ones, chosen one after another with earlier picks blocked.
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

CONFIGS = [
    ("raw", 0.0, 0.0),
    ("pmi", 1.0, 0.0),
    ("idf", 0.0, 1.0),
    ("pmi+idf", 1.0, 1.0),
]


def main() -> None:
    catalog = Catalog.from_json(ROOT / "data/itmo_krm_3.json")
    lm = LocalCausalLM.load(DEFAULT_MODEL)
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    started = time.perf_counter()
    index = WordIndex.build(graph, pymorphy_lemmatizer())

    def prompt(text: str) -> list[int]:
        return lm.prompt_ids(word_messages(text))[0].tolist()

    selector = WordBagSelector(lm, index, prompt, WordBagConfig())
    setup_seconds = time.perf_counter() - started
    print(f"setup {setup_seconds:.1f}s: {len(index.idf)} lemmas, {len(index.surfaces)} surfaces",
          flush=True)

    cases, spec = load_cases(role_filter=False)
    report: dict[str, object] = {
        "experiment": "exp05_wordbag", "model": DEFAULT_MODEL, "role_filter": False,
        "setup_seconds": round(setup_seconds, 1), "lemmas": len(index.idf),
        "surfaces": len(index.surfaces), "beam": selector.config.beam, "configs": {},
    }
    for name, alpha, beta in CONFIGS:
        selector.config = WordBagConfig(alpha=alpha, beta=beta)
        rows = []
        for case in cases:
            started = time.perf_counter()
            blocked: set[str] = set()
            picks = []
            calls = 0
            for _ in range(case["max_competencies"]):
                result = selector.select(case["text"], blocked)
                calls += result.forward_calls
                if result.competency_id is None:
                    break
                blocked.add(result.competency_id)
                picks.append({
                    "competency_id": result.competency_id, "resolved": result.resolved,
                    "words": [w.word for w in result.words],
                    "candidates_after": [w.candidates_after for w in result.words],
                })
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
        report["configs"][name] = {"alpha": alpha, "beta": beta, "summary": summary, "rows": rows}
        print(name, json.dumps(summary), flush=True)
    out = ROOT / "experiments/results/exp05_wordbag.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
