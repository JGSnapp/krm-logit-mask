from krm_mask.evaluation import score_profiles


def test_evaluation_separates_grounding_from_classification():
    records = [
        {
            "text": "used Python",
            "gold": [{"competency_id": "PL-1", "quote": "Python", "level": "Б"}],
            "predictions": [
                {"competency_id": "PL-1", "quote": "Python", "level": "Б"},
                {"competency_id": "BD-1", "quote": "invented", "level": "С"},
            ],
        }
    ]
    result = score_profiles(records)
    assert result["competency_precision"] == 0.5
    assert result["competency_recall"] == 1.0
    assert result["quote_substring_rate"] == 0.5
    assert result["quote_exact_rate"] == 1.0
