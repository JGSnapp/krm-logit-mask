"""Small-to-large evaluation of masked search on ESCO / Skill-XL.

Each already-segmented Skill-XL sentence is the evidence quote. The masked word
search proposes a pool without seeing ESCO names; the mandatory English YES/NO
gate accepts or rejects every candidate. Metrics below score *only accepted*
skills, never the pre-gate pool. Development is the tuning split and is
deliberately the default. Use ``--split test`` only for a frozen configuration.

Examples (GPU environment)::

    .venv/bin/python experiments/run_skill_xl_eval.py --limit 20
    .venv/bin/python experiments/run_skill_xl_eval.py --limit 0 --scales 13826 \\
        --split test --out exp30_skill_xl_test
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from krm_mask.catalog import Catalog
from krm_mask.esco import add_skill_labels, load_skill_xl
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM
from krm_mask.selection import CompetencySelector, SelectionConfig


def subset_catalog(catalog: Catalog, required: set[str], size: int, seed: int) -> Catalog:
    """Deterministic catalog subset that always includes labels being scored."""
    if size < len(required):
        raise ValueError(f"scale {size} is below {len(required)} gold skills in this run")
    by_id = {skill.id: skill for skill in catalog.competencies}
    missing = required - set(by_id)
    if missing:
        raise ValueError(f"Skill-XL labels absent from ESCO catalog: {sorted(missing)[:5]}")
    if size >= len(by_id):
        selected = list(catalog.competencies)
    else:
        others = sorted(set(by_id) - required)
        rng = random.Random(seed)
        selected = [by_id[item] for item in required]
        selected += [by_id[item] for item in rng.sample(others, size - len(selected))]
        selected.sort(key=lambda item: item.id)
    return replace(catalog, competencies=selected, roles=[])


def selected_metrics(rows: list[dict]) -> dict[str, float | int]:
    """Exact-label metrics over final YES/NO-confirmed skills only."""
    total_gold = sum(len(row["gold"]) for row in rows)
    predicted = sum(len(row["selected"]) for row in rows)
    hits = sum(len(set(row["selected"]) & set(row["gold"])) for row in rows)
    precision = hits / predicted if predicted else 0.0
    recall = hits / total_gold if total_gold else 0.0
    sentence_recall = [
        len(set(row["selected"]) & set(row["gold"])) / len(row["gold"])
        for row in rows
    ]
    return {
        "sentences": len(rows), "gold_labels": total_gold, "predicted": predicted, "hits": hits,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
        if precision + recall else 0.0,
        "sentence_hit_rate": round(sum(bool(set(row["selected"]) & set(row["gold"]))
                                         for row in rows) / max(len(rows), 1), 4),
        "mean_sentence_recall": round(sum(sentence_recall) / max(len(rows), 1), 4),
        "mean_selected": round(predicted / max(len(rows), 1), 3),
    }


def parse_scale_pools(values: list[str]) -> dict[int, list[int]]:
    """Parse e.g. ``100:10,15,20,30 1000:10,15,20,30,50,75,100``."""
    result: dict[int, list[int]] = {}
    for value in values:
        scale, separator, pools = value.partition(":")
        if not separator:
            raise ValueError(f"Invalid scale/pool specification: {value}")
        result[int(scale)] = sorted({int(pool) for pool in pools.split(",") if int(pool) > 0})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--catalog", default="data/esco/esco_skills_en.json")
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--limit", type=int, default=20,
                        help="relevant sentences; 0 means all (default: 20 tuning smoke run)")
    parser.add_argument(
        "--scale-pools", nargs="+",
        default=["100:10,15,20,30", "1000:10,15,20,30,50,75,100"],
        help="catalog scale and staged pools, e.g. 100:10,15,20,30 1000:10,20,50,100",
    )
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.7, 1.5, 2.0],
                        help="temperature variants in addition to the explicit no-temperature baseline")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--out", default="exp30_skill_xl_dev_smoke")
    args = parser.parse_args()

    scale_pools = parse_scale_pools(args.scale_pools)
    if any(temperature <= 0 for temperature in args.temperatures):
        raise ValueError("temperatures must be positive")
    dataset = ROOT / "data/esco/skill-xl" / f"{args.split}.csv"
    items = load_skill_xl(dataset)
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise ValueError("no relevant Skill-XL sentences selected")
    gold = set().union(*(item.skills for item in items))
    catalog = add_skill_labels(Catalog.from_json(ROOT / args.catalog), gold)
    print(f"{args.split}: {len(items)} sentences, {len(gold)} gold skills; catalog: {len(catalog.competencies)}")

    lm = LocalCausalLM.load(args.model)
    report: dict[str, object] = {
        "experiment": args.out, "model": args.model, "split": args.split,
        "limit": args.limit, "scale_pools": scale_pools, "temperatures": args.temperatures,
        "baseline": "deterministic beam search without temperature transform",
        "yes_no_min_margin": 0.0, "seed": args.seed,
        "catalog_total": len(catalog.competencies), "configs": {},
    }
    for scale, pools in sorted(scale_pools.items()):
        trial_catalog = subset_catalog(catalog, gold, scale, args.seed)
        started = time.perf_counter()
        selector = CompetencySelector(
            lm, trial_catalog,
            SelectionConfig(quotes=False, language="en_jobs"),
        )
        index_seconds = time.perf_counter() - started
        # A Skill-XL sentence is already the evidence quote. One coherent signal
        # narrows to a small indistinguishable group; the gate resolves its members.
        selector.words.config = replace(
            selector.words.config, group_size=5, group_at_cap=5, min_signals=1,
            max_signals=2, single_pass=True,
        )
        for pool in pools:
            # The baseline deliberately keeps temperature at 1.0, which is
            # mathematically the unmodified model distribution; it is logged
            # separately so it cannot be confused with a temperature trial.
            variants = [("baseline", 1.0), *[(f"T{temperature}", temperature)
                                               for temperature in args.temperatures]]
            for variant, temperature in variants:
                selector.config.pool = selector.config.max_competencies = pool
                selector.words.config = replace(selector.words.config, temperature=temperature, samples=None)
                rows = []
                for position, item in enumerate(items, 1):
                    started = time.perf_counter()
                    found, calls = selector.words.select_paths(item.sentence, pool, coherent=True)
                    search_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    scored = []
                    for search_rank, (skill_id, path) in enumerate(found, 1):
                        margin, probability = selector.check(item.sentence, selector.competencies[skill_id])
                        scored.append({"id": skill_id, "search_rank": search_rank,
                                       "margin": round(margin, 5), "p_yes": round(probability, 5),
                                       "confirmed": margin > selector.config.min_margin,
                                       "words": [word.word for word in path.words]})
                    check_seconds = time.perf_counter() - started
                    selected = [entry["id"] for entry in scored if entry["confirmed"]]
                    rows.append({"ad": item.ad, "title": item.title, "sentence": item.sentence,
                                 "gold": sorted(item.skills),
                                 "clusters": [sorted(group) for group in item.clusters],
                                 "selected": selected, "candidates": scored, "search_forward_calls": calls,
                                 "search_s": round(search_seconds, 4), "check_s": round(check_seconds, 4)})
                    print(f"scale={scale} pool={pool} {variant} {position}/{len(items)} "
                          f"selected={len(selected)}", flush=True)
                metrics = selected_metrics(rows)
                metrics["index_s"] = round(index_seconds, 3)
                metrics["mean_search_s"] = round(sum(row["search_s"] for row in rows) / len(rows), 4)
                metrics["mean_check_s"] = round(sum(row["check_s"] for row in rows) / len(rows), 4)
                metrics["mean_forward_calls"] = round(sum(row["search_forward_calls"] for row in rows) / len(rows), 2)
                label = f"scale{scale}_pool{pool}_{variant}"
                report["configs"][label] = {
                    "scale": scale, "pool": pool, "variant": variant, "temperature": temperature,
                    "metrics": metrics, "rows": rows,
                }
                print(label, json.dumps(metrics), flush=True)
                out = ROOT / "experiments/results" / f"{args.out}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
