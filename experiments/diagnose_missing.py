"""Experiment 16 (diagnostic): why acceptable competencies are not selected.

For every web profile: the normal search (coherent signals, IDF 0.3, as frozen in
experiment 9) gives the selected competencies and their path scores. Then, for
each acceptable competency, the same search runs with all other competencies
blocked: its best path (score, signals, words) or "unreachable".

  forced score > score of a selected one -> search error (the beam missed it);
  forced score < scores of selected ones -> scoring error (the model prefers others);
  unreachable                            -> no path under the signal rules.

Usage: .venv/bin/python experiments/diagnose_missing.py [--eval-set ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_selection_eval import load_cases  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import PhraseGraph  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.wordbag import (  # noqa: E402
    WordBagConfig, WordBagSelector, WordIndex, pymorphy_lemmatizer, word_messages,
)


def describe(path) -> dict[str, object]:
    return {
        "score": round(path.score, 2),
        "signals": [sorted(s) for s in path.signals],
        "words": [
            {"word": w.word, "logp": round(w.log_probability, 2), "idf": round(w.idf, 2),
             "score": round(w.score, 2)}
            for w in path.words
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp16_diagnose_itmo")
    args = parser.parse_args()
    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    cases = [case for case in cases if case["kind"] == "web"]
    catalog_path = spec.get("catalog", "data/itmo_krm_3.json")
    catalog = Catalog.from_json(ROOT / catalog_path)
    names = {item.id: item.name for item in catalog.competencies}
    lm = LocalCausalLM.load(args.model)
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    index = WordIndex.build(graph, pymorphy_lemmatizer())

    def prompt(text: str) -> list[int]:
        return lm.prompt_ids(word_messages(text))[0].tolist()

    selector = WordBagSelector(lm, index, prompt, WordBagConfig(alpha=0.0, beta=0.3))
    everything = set(index.competency_lemmas)
    report: dict[str, object] = {
        "experiment": args.out, "model": args.model, "catalog": catalog_path,
        "search": "coherent signals, alpha 0, beta 0.3", "profiles": {},
    }
    for case in cases:
        found, _ = selector.select_paths(case["text"], case["max_competencies"], coherent=True)
        selected = [
            {"id": c, "name": names[c], "acceptable": c in case["acceptable"], **describe(p)}
            for c, p in found
        ]
        worst_selected = min(item["score"] for item in selected)
        forced = []
        for competency_id in case["acceptable"]:
            if competency_id not in everything:
                continue
            paths, _ = selector.select_paths(
                case["text"], 1, blocked=everything - {competency_id}, coherent=True)
            entry: dict[str, object] = {"id": competency_id, "name": names[competency_id]}
            if paths:
                entry.update(describe(paths[0][1]))
                entry["verdict"] = (
                    "selected" if competency_id in [s["id"] for s in selected]
                    else "search_error" if entry["score"] > worst_selected
                    else "scoring_error"
                )
            else:
                entry["verdict"] = "unreachable"
            forced.append(entry)
            print(case["id"], competency_id, entry["verdict"], entry.get("score"),
                  entry.get("signals"), flush=True)
        report["profiles"][case["id"]] = {"selected": selected, "acceptable": forced}
        print(case["id"], "selected", [(s["id"], s["score"], s["signals"]) for s in selected],
              flush=True)
    verdicts: dict[str, int] = {}
    for profile in report["profiles"].values():
        for entry in profile["acceptable"]:
            verdicts[entry["verdict"]] = verdicts.get(entry["verdict"], 0) + 1
    report["verdicts"] = verdicts
    print("verdicts", verdicts, flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
