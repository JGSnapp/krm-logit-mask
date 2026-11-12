"""Competency-selection benchmark used by EXPERIMENTS.md.

Runs the full pipeline (selection, quotes, support, level) and scores only the
selected competencies, taken from `attempts` (support gate does not hide them).

Usage (from the project root, GPU environment):
    .venv/bin/python experiments/run_selection_eval.py EXP_ID CONFIG_JSON [--repeats N]
CONFIG_JSON is a list of {"name": ..., "kwargs": {...run_detailed kwargs...}}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from krm_mask.catalog import Catalog
from krm_mask.evaluation import read_jsonl
from krm_mask.graph import NONE_ID
from krm_mask.model import DEFAULT_MODEL, LocalCausalLM
from krm_mask.pipeline import ExtractionPipeline

ROOT = Path(__file__).resolve().parents[1]


def load_cases() -> tuple[list[dict[str, object]], dict[str, object]]:
    spec = json.loads((ROOT / "experiments/eval_set.json").read_text("utf-8"))
    profiles = {p["id"]: p for p in json.loads((ROOT / "web/profiles.json").read_text("utf-8"))}
    cases = []
    for profile_id, acceptable in spec["web_profiles"].items():
        profile = profiles[profile_id]
        cases.append({
            "id": profile_id, "kind": "web", "text": profile["text"],
            "role_id": profile["role_hint"], "acceptable": acceptable,
            "max_competencies": spec["web_max_competencies"],
        })
    for record in read_jsonl(ROOT / "data/synthetic_profiles.jsonl"):
        gold = [item["competency_id"] for item in record.get("gold", [])] or [NONE_ID]
        cases.append({
            "id": record["id"], "kind": "synthetic", "text": record["text"],
            "role_id": record.get("role_id"), "acceptable": gold,
            "max_competencies": spec["synthetic_max_competencies"],
        })
    return cases, spec


def selected_ids(result: dict[str, object]) -> list[str]:
    return [
        str(item["competency_id"]) for item in result["attempts"]
        if item.get("competency_id") not in (None, "__ABORT__")
    ]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    web = [row for row in rows if row["kind"] == "web"]
    synthetic = [row for row in rows if row["kind"] == "synthetic"]
    web_predicted = sum(len([c for c in row["selected"] if c != NONE_ID]) for row in web)
    web_hits = sum(len([c for c in row["selected"] if c in row["acceptable"]]) for row in web)
    return {
        "web_precision": web_hits / web_predicted if web_predicted else 0.0,
        "web_hits": web_hits,
        "web_predicted": web_predicted,
        "synthetic_hit_at_1": (
            sum(bool(row["selected"]) and row["selected"][0] in row["acceptable"] for row in synthetic)
            / len(synthetic) if synthetic else 0.0
        ),
        "mean_seconds_web": sum(row["seconds"] for row in web) / len(web) if web else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("configs")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    cases, spec = load_cases()
    catalog = Catalog.from_json(ROOT / "data/itmo_krm_3.json")
    roles = {role.id: set(role.competencies) for role in catalog.roles}
    pipeline = ExtractionPipeline(catalog, LocalCausalLM.load(args.model))
    configs = json.loads(args.configs)
    report: dict[str, object] = {
        "experiment": args.experiment, "model": args.model, "eval_set_note": spec["note"],
        "base_rate": {
            case["id"]: round(len(set(case["acceptable"]) & roles[case["role_id"]])
                              / len(roles[case["role_id"]]), 3)
            for case in cases if case["kind"] == "web"
        },
        "configs": {},
    }
    for config in configs:
        repeats = 1 if not config["kwargs"].get("use_disc") else args.repeats
        rows: list[dict[str, object]] = []
        for repeat in range(repeats):
            for case in cases:
                started = time.perf_counter()
                result = pipeline.run_detailed(
                    case["text"], role_id=case["role_id"],
                    max_competencies=case["max_competencies"],
                    max_quotes=spec["max_quotes"], **config["kwargs"],
                )
                rows.append({
                    "repeat": repeat, "id": case["id"], "kind": case["kind"],
                    "acceptable": case["acceptable"], "selected": selected_ids(result),
                    "seconds": round(time.perf_counter() - started, 2),
                })
                print(config["name"], repeat, case["id"], rows[-1]["selected"],
                      rows[-1]["seconds"], flush=True)
        report["configs"][config["name"]] = {
            "kwargs": config["kwargs"], "repeats": repeats,
            "summary": summarize(rows), "rows": rows,
        }
        print(config["name"], json.dumps(report["configs"][config["name"]]["summary"]), flush=True)
    out = ROOT / "experiments/results" / f"{args.experiment}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
