"""Local web service: competency selection from a resume (experiments 19-26).

Two modes of `selection.CompetencySelector`:
  quote_last  - word search on the whole resume -> yes/no check -> a quote for
                each selected competency;
  quote_first - the model picks quotes that show a skill or experience -> word
                search on each quote -> yes/no check -> each competency keeps
                the quote that proposed it.
The page `/` runs both; `/catalog.html` browses the server catalog.
"""
import gc
import threading
from pathlib import Path
from time import perf_counter

from .catalog import Catalog
from .model import DEFAULT_MODEL, LocalCausalLM
from .selection import CompetencySelector, SelectionConfig

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CATALOG_NAMES = {"itmo-krm": "КРМ ИТМО", "mipt-it-krm": "КРМ МФТИ (ИТ-роли)"}
MODES = ("quote_last", "quote_first")

MODEL_OPTIONS = {
    "qwen2.5-1.5b": {
        "name": "Qwen 2.5 · 1.5B",
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "Быстрее · около 3 ГБ весов FP16",
    },
    "qwen2.5-3b": {
        "name": "Qwen 2.5 · 3B",
        "hf_name": "Qwen/Qwen2.5-3B-Instruct",
        "description": "Точнее · около 6 ГБ весов FP16",
    },
}


class Runtime:
    """One model in GPU memory at a time and one selector per catalog on it."""

    def __init__(self, catalog: Catalog, model_name: str) -> None:
        self.catalog = catalog
        self.models = dict(MODEL_OPTIONS)
        matching = next(
            (key for key, value in self.models.items() if value["hf_name"] == model_name),
            None,
        )
        if matching is None:
            matching = "server-default"
            self.models[matching] = {
                "name": model_name,
                "hf_name": model_name,
                "description": "Модель, переданная при запуске сервера",
            }
        self.default_model_id = matching
        self.active_model_id: str | None = None
        self.lm: LocalCausalLM | None = None
        self.lock = threading.RLock()
        # The server catalog first, plus the other known catalogs found in data/.
        self.catalogs: dict[str, Catalog] = {catalog.framework_id: catalog}
        for extra in ("itmo_krm_3.json", "mipt_krm.json"):
            extra_path = DATA_DIR / extra
            if extra_path.exists():
                loaded = Catalog.from_json(extra_path)
                self.catalogs.setdefault(loaded.framework_id, loaded)
        self.selectors: dict[str, CompetencySelector] = {}

    def get_lm(self, model_id: str | None = None) -> LocalCausalLM:
        with self.lock:
            requested = model_id or self.default_model_id
            if requested not in self.models:
                raise KeyError(requested)
            if self.lm is None or self.active_model_id != requested:
                self.lm = None
                self.active_model_id = None
                self.selectors = {}
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                self.lm = LocalCausalLM.load(str(self.models[requested]["hf_name"]))
                self.active_model_id = requested
            return self.lm

    def get_selector(self, model_id: str | None, catalog_id: str) -> CompetencySelector:
        """Selector of the catalog on the active model (built once: word index and priors)."""
        with self.lock:
            lm = self.get_lm(model_id)
            if catalog_id not in self.selectors:
                self.selectors[catalog_id] = CompetencySelector(
                    lm, self.catalogs[catalog_id], SelectionConfig())
            return self.selectors[catalog_id]

    def catalog_public(self, catalog_id: str) -> dict[str, object]:
        catalog = self.catalogs[catalog_id]
        return {
            "id": catalog_id,
            "name": CATALOG_NAMES.get(catalog_id, catalog_id),
            "version": catalog.framework_version,
            "competencies": len(catalog.competencies),
            "indicators": sum(len(item.indicators) for item in catalog.competencies),
            "ready": catalog_id in self.selectors,
        }

    def model_public(self, model_id: str) -> dict[str, object]:
        item = self.models[model_id]
        return {
            "id": model_id,
            "name": item["name"],
            "hf_name": item["hf_name"],
            "description": item["description"],
            "loaded": self.active_model_id == model_id and self.lm is not None,
        }


def create_app(catalog_path: str, model_name: str = DEFAULT_MODEL):
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    catalog = Catalog.from_json(catalog_path)
    runtime = Runtime(catalog, model_name)
    app = FastAPI(title="KRM Trace Lab", version="0.2.0")

    class SelectRequest(BaseModel):
        text: str = Field(min_length=3, max_length=30_000)
        mode: str = Field(default="quote_first", pattern="^(quote_last|quote_first)$")
        catalog_id: str | None = None
        model_id: str | None = None
        max_competencies: int = Field(default=3, ge=1, le=6)
        pool: int = Field(default=15, ge=3, le=30)  # quote_last: candidates to check
        quotes: int = Field(default=3, ge=1, le=6)  # quote_first: quotes chosen first

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model": runtime.models[runtime.active_model_id or runtime.default_model_id]["hf_name"],
            "model_loaded": runtime.lm is not None,
            "active_model_id": runtime.active_model_id,
            "default_model_id": runtime.default_model_id,
            "models": [runtime.model_public(key) for key in runtime.models],
            "modes": list(MODES),
        }

    @app.get("/api/catalogs")
    def catalogs() -> dict[str, object]:
        return {
            "default_catalog_id": catalog.framework_id,
            "catalogs": [runtime.catalog_public(key) for key in runtime.catalogs],
        }

    @app.get("/api/catalogs/{catalog_id}")
    def catalog_details(catalog_id: str) -> dict[str, object]:
        if catalog_id not in runtime.catalogs:
            raise HTTPException(status_code=404, detail="Unknown catalog_id")
        chosen = runtime.catalogs[catalog_id]
        return {
            "id": catalog_id,
            "competencies": {
                item.id: {
                    "name": item.name, "formulation": item.formulation,
                    "indicators": [i.text for i in item.indicators],
                    "keywords": item.keywords,
                }
                for item in chosen.competencies
            },
        }

    @app.post("/api/select")
    def select(request: SelectRequest) -> dict[str, object]:
        catalog_id = request.catalog_id or catalog.framework_id
        if catalog_id not in runtime.catalogs:
            raise HTTPException(status_code=400, detail="Unknown catalog_id")
        requested_model = request.model_id or runtime.default_model_id
        if requested_model not in runtime.models:
            raise HTTPException(status_code=400, detail="Unknown model_id")
        request_started = perf_counter()
        with runtime.lock:
            was_ready = (runtime.lm is not None and runtime.active_model_id == requested_model
                         and catalog_id in runtime.selectors)
            load_started = perf_counter()
            selector = runtime.get_selector(requested_model, catalog_id)
            load_seconds = perf_counter() - load_started if not was_ready else 0.0
            selector.config.pool = request.pool
            selector.config.first_quotes = request.quotes
            if request.mode == "quote_first":
                result = selector.select_quote_first(request.text, request.max_competencies)
            else:
                result = selector.select(request.text, request.max_competencies)
        result["model"] = runtime.model_public(requested_model)
        result["catalog"] = runtime.catalog_public(catalog_id)
        result["timings"].update({
            "model_load_s": load_seconds,
            "request_total_s": perf_counter() - request_started,
        })
        return result

    @app.get("/api/catalog")
    def get_catalog() -> dict[str, object]:
        """The server catalog with indicators and roles (for /catalog.html)."""
        return {
            "framework_id": catalog.framework_id,
            "framework_version": catalog.framework_version,
            "counts": {
                "competencies": len(catalog.competencies),
                "indicators": sum(len(c.indicators) for c in catalog.competencies),
                "roles": len(catalog.roles),
            },
            "competencies": [
                {
                    "id": item.id,
                    "name": item.name,
                    "formulation": item.formulation,
                    "domain": item.domain,
                    "block": item.block,
                    "keyword_count": len(item.keywords),
                    "keywords": item.keywords,
                    "indicators": [
                        {"id": indicator.id, "text": indicator.text, "levels": indicator.levels}
                        for indicator in item.indicators
                    ],
                }
                for item in catalog.competencies
            ],
            "roles": [
                {
                    "id": role.id,
                    "name": role.name,
                    "definition": role.definition,
                    "competencies": role.competencies,
                }
                for role in catalog.roles
            ],
        }

    web_root = Path(__file__).resolve().parents[2] / "web"
    app.mount("/", StaticFiles(directory=web_root, html=True), name="web")
    return app


def serve(
    catalog_path: str,
    model_name: str = DEFAULT_MODEL,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    import uvicorn

    uvicorn.run(create_app(catalog_path, model_name), host=host, port=port)
