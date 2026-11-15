from krm_mask.catalog import Catalog
from krm_mask.server import Runtime
import pytest


def test_runtime_reuses_and_switches_models(monkeypatch):
    catalog = Catalog(
        framework_id="test",
        framework_version="1",
        source_file="test.xlsx",
        competencies=[],
        roles=[],
    )
    loaded: list[str] = []

    def fake_load(model_name: str):
        loaded.append(model_name)
        return object()

    monkeypatch.setattr("krm_mask.server.LocalCausalLM.load", fake_load)
    runtime = Runtime(catalog, "Qwen/Qwen2.5-1.5B-Instruct")

    first = runtime.get_lm("qwen2.5-1.5b")
    assert runtime.get_lm("qwen2.5-1.5b") is first
    second = runtime.get_lm("qwen2.5-3b")

    assert second is not first
    assert runtime.active_model_id == "qwen2.5-3b"
    assert loaded == [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ]


def test_select_runs_the_requested_mode(monkeypatch, tmp_path):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from krm_mask.server import create_app

    catalog_path = tmp_path / "catalog.json"
    Catalog("test", "1", "test.xlsx", [], []).to_json(catalog_path)
    calls = []

    class Selector:
        class config:
            pool = 15
            first_quotes = 3

        def select(self, text, limit):
            calls.append(("quote_last", limit, Selector.config.pool))
            return {"timings": {}}

        def select_quote_first(self, text, limit):
            calls.append(("quote_first", limit, Selector.config.first_quotes))
            return {"timings": {}}

    monkeypatch.setattr(Runtime, "get_selector", lambda *args: Selector())
    monkeypatch.setattr(Runtime, "model_public", lambda *args: {})
    monkeypatch.setattr(Runtime, "catalog_public", lambda *args: {})
    with TestClient(create_app(str(catalog_path))) as client:
        assert client.post("/api/select", json={"text": "Python"}).status_code == 200
        assert client.post("/api/select", json={"text": "Python", "mode": "quote_last",
                                                "pool": 10}).status_code == 200
        assert client.post("/api/select", json={"text": "Python", "mode": "other"}).status_code == 422
        assert client.post("/api/analyze", json={"text": "Python"}).status_code in (404, 405)
    assert calls == [("quote_first", 3, 3), ("quote_last", 3, 10)]
