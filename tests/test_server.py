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

    first = runtime.get_pipeline("qwen2.5-1.5b")
    assert runtime.get_pipeline("qwen2.5-1.5b") is first
    second = runtime.get_pipeline("qwen2.5-3b")

    assert second is not first
    assert runtime.active_model_id == "qwen2.5-3b"
    assert loaded == [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ]


@pytest.mark.parametrize("route", ["/api/analyze", "/api/analyze-quote-first"])
def test_api_forwards_disc_and_validates_budget(monkeypatch, tmp_path, route):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from krm_mask.server import create_app

    catalog_path = tmp_path / "catalog.json"
    Catalog("test", "1", "test.xlsx", [], []).to_json(catalog_path)
    received = []

    class Pipeline:
        def run_detailed(self, *args):
            received.append(args)
            return {"timings": {}}

        run_quote_first_detailed = run_detailed

    monkeypatch.setattr(Runtime, "get_pipeline", lambda *args: Pipeline())
    with TestClient(create_app(str(catalog_path))) as client:
        assert client.post(route, json={"text": "Python", "use_disc": True, "disc_budget": 2}).status_code == 200
        assert received[-1][-2:] == (True, 2)
        assert client.post(route, json={"text": "Python"}).status_code == 200
        assert received[-1][-2:] == (False, 4)
        assert client.post(route, json={"text": "Python", "disc_budget": 0}).status_code == 422


def test_api_reports_disc_exhaustion(monkeypatch, tmp_path):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from krm_mask.disc import DiscDecodingError
    from krm_mask.server import create_app

    path = tmp_path / "catalog.json"
    Catalog("test", "1", "test.xlsx", [], []).to_json(path)

    class Pipeline:
        def run_detailed(self, *args):
            raise DiscDecodingError("DISC budget exhausted")

    monkeypatch.setattr(Runtime, "get_pipeline", lambda *args: Pipeline())
    with TestClient(create_app(str(path))) as client:
        response = client.post("/api/analyze", json={"text": "Python", "use_disc": True})
        assert response.status_code == 422
        assert response.json()["detail"] == "DISC budget exhausted"
