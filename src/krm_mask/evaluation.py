from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score_profiles(records: Iterable[dict[str, object]]) -> dict[str, float | int]:
    true_positive = false_positive = false_negative = 0
    valid_quotes = exact_quotes = level_correct = paired = 0
    predicted_count = 0

    for record in records:
        text = str(record["text"])
        gold = {item["competency_id"]: item for item in record.get("gold", [])}
        predicted = {
            item["competency_id"]: item for item in record.get("predictions", [])
        }
        gold_ids, predicted_ids = set(gold), set(predicted)
        true_positive += len(gold_ids & predicted_ids)
        false_positive += len(predicted_ids - gold_ids)
        false_negative += len(gold_ids - predicted_ids)
        predicted_count += len(predicted)
        valid_quotes += sum(str(item.get("quote", "")) in text for item in predicted.values())
        for competency_id in gold_ids & predicted_ids:
            paired += 1
            exact_quotes += predicted[competency_id].get("quote") == gold[competency_id].get("quote")
            level_correct += predicted[competency_id].get("level") == gold[competency_id].get("level")

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "competency_precision": precision,
        "competency_recall": recall,
        "competency_f1": f1,
        "quote_substring_rate": valid_quotes / predicted_count if predicted_count else 1.0,
        "quote_exact_rate": exact_quotes / paired if paired else 0.0,
        "level_accuracy_on_matched": level_correct / paired if paired else 0.0,
    }
