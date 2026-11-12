"""Experiment 3: score every catalog phrase for each profile, no search, no sampling.

Separates "is the scoring right?" from "does the search find it?". For every
profile, all graph phrases are scored by teacher forcing under the first
selection prompt (whole catalog, no role filter), then ranked by:
  cond      log P(phrase | text)
  mean      log P(phrase | text) / tokens        (length normalisation)
  pmi_a0.5  log P(phrase | text) - 0.5 log P(phrase | empty text)
  pmi_a1.0  log P(phrase | text) - 1.0 log P(phrase | empty text)
Competency score = best phrase (max) or log-sum-exp over its phrases (lse).

Usage: .venv/bin/python experiments/rank_all_phrases.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_selection_eval import load_cases  # noqa: E402

from krm_mask.catalog import Catalog  # noqa: E402
from krm_mask.graph import PhraseGraph  # noqa: E402
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM  # noqa: E402
from krm_mask.pipeline import (  # noqa: E402
    EMPTY_HISTORY, EMPTY_SIGNALS, ExtractionPipeline, _signal_messages,
)


def logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(v - peak) for v in values))


def main() -> None:
    catalog = Catalog.from_json(ROOT / "data/itmo_krm_3.json")
    lm = LocalCausalLM.load(DEFAULT_MODEL)
    pipeline = ExtractionPipeline(catalog, lm)
    graph = PhraseGraph.from_catalog(catalog, None, {"indicators", "keywords"})
    phrases = list(graph.phrases)
    delimiter = lm.tokenizer.encode("\n", add_special_tokens=False)
    continuations = [
        lm.tokenizer.encode(p, add_special_tokens=False) + delimiter for p in phrases
    ]
    lengths = [len(c) for c in continuations]
    started = time.perf_counter()
    prior_map = pipeline.phrase_log_prior(graph, False)
    prior = [prior_map[p] for p in phrases]
    prior_seconds = time.perf_counter() - started

    cases, _ = load_cases(role_filter=False)
    methods = ["cond", "mean", "pmi_a0.5", "pmi_a1.0"]
    report: dict[str, object] = {
        "experiment": "exp03_phrase_ranking", "model": DEFAULT_MODEL,
        "phrases": len(phrases), "competencies": len(graph.competency_names),
        "prior_seconds": round(prior_seconds, 1), "profiles": {},
    }
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for case in cases:
        started = time.perf_counter()
        prompt = lm.prompt_ids(_signal_messages(case["text"], EMPTY_HISTORY, EMPTY_SIGNALS))
        cond = lm.score_continuations(prompt, continuations)
        seconds = time.perf_counter() - started
        phrase_scores = {
            "cond": cond,
            "mean": [c / n for c, n in zip(cond, lengths)],
            "pmi_a0.5": [c - 0.5 * p for c, p in zip(cond, prior)],
            "pmi_a1.0": [c - 1.0 * p for c, p in zip(cond, prior)],
        }
        acceptable = set(case["acceptable"])
        profile: dict[str, object] = {
            "kind": case["kind"], "acceptable": sorted(acceptable),
            "seconds": round(seconds, 1), "methods": {},
        }
        for method in methods:
            scores = phrase_scores[method]
            by_competency: dict[str, list[float]] = defaultdict(list)
            for phrase, score in zip(phrases, scores):
                for competency_id in graph.phrases[phrase]:
                    by_competency[competency_id].append(score)
            top_phrases = sorted(range(len(phrases)), key=lambda i: -scores[i])[:20]
            entry: dict[str, object] = {
                "top_phrases": [
                    {"phrase": phrases[i], "score": round(scores[i], 2),
                     "competencies": sorted(graph.phrases[phrases[i]])}
                    for i in top_phrases
                ],
            }
            for aggregate in ("max", "lse"):
                ranked = sorted(
                    by_competency,
                    key=lambda c: -(max(by_competency[c]) if aggregate == "max"
                                    else logsumexp(by_competency[c])),
                )
                ranks = {c: ranked.index(c) + 1 for c in acceptable if c in ranked}
                key = f"{method}/{aggregate}"
                entry[aggregate] = {
                    "top10": ranked[:10],
                    "acceptable_ranks": dict(sorted(ranks.items(), key=lambda kv: kv[1])),
                    "hits_top3": len(set(ranked[:3]) & acceptable),
                    "hits_top10": len(set(ranked[:10]) & acceptable),
                }
                if case["kind"] == "web":
                    totals[key]["web_hits_top3"] += entry[aggregate]["hits_top3"]
                    totals[key]["web_hits_top10"] += entry[aggregate]["hits_top10"]
                    totals[key]["web_best_rank_sum"] += min(ranks.values()) if ranks else 84
                else:
                    totals[key]["synthetic_hit_at_1"] += ranked[0] in acceptable
            profile["methods"][method] = entry
        report["profiles"][case["id"]] = profile
        print(case["id"], f"{seconds:.0f}s",
              {m: profile["methods"][m]["max"]["top10"][:3] for m in methods}, flush=True)

    web_count = sum(case["kind"] == "web" for case in cases)
    report["summary"] = {
        key: {
            "web_precision_top3": round(values["web_hits_top3"] / (3 * web_count), 3),
            "web_hits_top10": int(values["web_hits_top10"]),
            "web_mean_best_rank": round(values["web_best_rank_sum"] / web_count, 1),
            "synthetic_hit_at_1": int(values["synthetic_hit_at_1"]),
        }
        for key, values in totals.items()
    }
    for key, values in report["summary"].items():
        print(key, values, flush=True)
    out = ROOT / "experiments/results/exp03_phrase_ranking.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
