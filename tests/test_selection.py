from krm_mask.selection import split_sentences


def test_split_sentences_keeps_offsets_and_joins_short_pieces():
    text = "Знаю Python. Ок.\nПишу сервисы на FastAPI и Django!  Итог: да."
    spans = split_sentences(text)
    pieces = [text[a:b] for a, b in spans]
    assert pieces == ["Знаю Python.", "Ок.\nПишу сервисы на FastAPI и Django!  Итог: да."]
    assert all(text[a:b] == text[a:b].strip() for a, b in spans)
