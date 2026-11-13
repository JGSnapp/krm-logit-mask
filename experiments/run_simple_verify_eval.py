"""Experiment 19: the simple yes/no question, as a chat, one forward per candidate.

  "Имеет ли кандидат указанную компетенцию? Ответь ДА или НЕТ.
   Профиль кандидата: <резюме>
   Компетенция: <название>"

The chat template of the model is applied; the answer is read from the first
generated token: log of the summed probability of "ДА"/"Да"/"да" minus that of
"НЕТ"/"Нет"/"нет". The word-signal search (experiments 7/9) proposes up to 10
candidates; they are confirmed in order until `max_competencies` say ДА
(otherwise __NONE__), or re-ranked by the margin.

Usage: .venv/bin/python experiments/run_simple_verify_eval.py [--eval-set ...] [--out ...]
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

from run_selection_eval import load_cases, summarize  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import NONE_ID, PhraseGraph  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.wordbag import (  # noqa: E402
    WordBagConfig, WordBagSelector, WordIndex, pymorphy_lemmatizer, word_messages,
)

MAX_CANDIDATES = 10
YES = ["ДА", "Да", "да"]
NO = ["НЕТ", "Нет", "нет"]


def question(text: str, competency: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": (
        "Имеет ли кандидат указанную компетенцию? Ответь ДА или НЕТ.\n"
        f"Профиль кандидата: {text}\n"
        f"Компетенция: {competency}"
    )}]


def first_tokens(tokenizer, words: list[str]) -> list[int]:
    return sorted({tokenizer.encode(word, add_special_tokens=False)[0] for word in words})


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--verifier", default=None, help="model for the yes/no check")
    parser.add_argument("--eval-set", default="experiments/eval_set.json")
    parser.add_argument("--out", default="exp19_simple_verify")
    args = parser.parse_args()
    cases, spec = load_cases(role_filter=False, eval_set=args.eval_set)
    catalog_path = spec.get("catalog", "data/itmo_krm_3.json")
    catalog = Catalog.from_json(ROOT / catalog_path)
    competencies = {item.id: item for item in catalog.competencies}
    lm = LocalCausalLM.load(args.model)
    verifier = LocalCausalLM.load(args.verifier) if args.verifier else lm
    yes_ids = first_tokens(verifier.tokenizer, YES)
    no_ids = first_tokens(verifier.tokenizer, NO)
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    index = WordIndex.build(graph, pymorphy_lemmatizer())

    def prompt(text: str) -> list[int]:
        return lm.prompt_ids(word_messages(text))[0].tolist()

    def margin(text: str, competency: str) -> float:
        ids = verifier.prompt_ids(question(text, competency))
        with torch.inference_mode():
            logits = verifier.model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        return float(torch.logsumexp(logp[yes_ids], 0) - torch.logsumexp(logp[no_ids], 0))

    selector = WordBagSelector(lm, index, prompt, WordBagConfig(alpha=0.0, beta=0.3))
    report: dict[str, object] = {
        "experiment": args.out, "model": args.model, "verifier": args.verifier or args.model,
        "eval_set": args.eval_set, "catalog": catalog_path,
        "question": question("<профиль>", "<название>")[0]["content"],
        "configs": {"search_only": {"rows": []}, "confirm_in_order": {"rows": []},
                    "rerank_by_margin": {"rows": []}},
        "gate": {},
    }
    for case in cases:
        started = time.perf_counter()
        found, calls = selector.select_paths(case["text"], MAX_CANDIDATES, coherent=True)
        search_seconds = time.perf_counter() - started
        candidates = [c for c, _ in found]
        started = time.perf_counter()
        margins = {c: margin(case["text"], competencies[c].name) for c in candidates}
        check_seconds = (time.perf_counter() - started) / max(len(candidates), 1)
        report["gate"][case["id"]] = [
            {"id": c, "name": competencies[c].name, "acceptable": c in case["acceptable"],
             "margin": round(margins[c], 2)} for c in candidates]
        print(case["id"], f"search {search_seconds:.1f}s check {check_seconds:.2f}s/cand",
              [(c, round(margins[c], 2), c in case["acceptable"]) for c in candidates], flush=True)
        limit = case["max_competencies"]
        confirmed, checks = [], 0
        for c in candidates:
            checks += 1
            if margins[c] > 0:
                confirmed.append(c)
                if len(confirmed) == limit:
                    break
        reranked = [c for c in sorted(candidates, key=lambda c: -margins[c])
                    if margins[c] > 0][:limit]
        for name, selected, used in (
            ("search_only", candidates[:limit], 0),
            ("confirm_in_order", confirmed, checks),
            ("rerank_by_margin", reranked, len(candidates)),
        ):
            report["configs"][name]["rows"].append({
                "id": case["id"], "kind": case["kind"], "acceptable": case["acceptable"],
                "selected": selected or [NONE_ID], "checks": used,
                "seconds": round(search_seconds + used * check_seconds, 2),
            })
    for name, entry in report["configs"].items():
        rows = entry["rows"]
        web = [r for r in rows if r["kind"] == "web"]
        summary = summarize(rows)
        summary["hits_per_profile"] = [
            len([c for c in r["selected"] if c in r["acceptable"]]) for r in web]
        summary["predicted_per_profile"] = [
            len([c for c in r["selected"] if c != NONE_ID]) for r in web]
        summary["synthetic_selected"] = [r["selected"] for r in rows if r["kind"] == "synthetic"]
        entry["summary"] = summary
        print(name, json.dumps(summary, ensure_ascii=False), flush=True)
    gates = [g for v in report["gate"].values() for g in v]
    good = [g["margin"] for g in gates if g["acceptable"]]
    bad = [g["margin"] for g in gates if not g["acceptable"]]
    report["gate_summary"] = {
        "acceptable": {"n": len(good), "yes": sum(m > 0 for m in good),
                       "mean_margin": round(statistics.mean(good), 2) if good else None},
        "other": {"n": len(bad), "yes": sum(m > 0 for m in bad),
                  "mean_margin": round(statistics.mean(bad), 2) if bad else None},
    }
    print("gate", report["gate_summary"], flush=True)
    out = ROOT / "experiments/results" / f"{args.out}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
