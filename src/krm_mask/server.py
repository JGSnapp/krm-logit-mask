import gc
import threading
from pathlib import Path
from time import perf_counter

from .catalog import Catalog
from .model import DEFAULT_MODEL, LocalCausalLM
from .pipeline import ExtractionPipeline
from .disc import DiscDecodingError


MODEL_OPTIONS = {
    "qwen2.5-1.5b": {
        "name": "Qwen 2.5 · 1.5B",
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "Быстрее · около 3 ГБ весов FP16",
    },
    "qwen2.5-3b": {
        "name": "Qwen 2.5 · 3B",
        "hf_name": "Qwen/Qwen2.5-3B-Instruct",
        "description": "Сильнее · около 6 ГБ весов FP16",
    },
}


class Runtime:
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
        self.pipeline: ExtractionPipeline | None = None
        self.lock = threading.RLock()

    def get_pipeline(self, model_id: str | None = None) -> ExtractionPipeline:
        with self.lock:
            requested = model_id or self.default_model_id
            if requested not in self.models:
                raise KeyError(requested)
            if self.pipeline is None or self.active_model_id != requested:
                self.pipeline = None
                self.active_model_id = None
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                model_name = str(self.models[requested]["hf_name"])
                self.pipeline = ExtractionPipeline(
                    self.catalog, LocalCausalLM.load(model_name)
                )
                self.active_model_id = requested
            return self.pipeline

    def model_public(self, model_id: str) -> dict[str, object]:
        item = self.models[model_id]
        return {
            "id": model_id,
            "name": item["name"],
            "hf_name": item["hf_name"],
            "description": item["description"],
            "loaded": self.active_model_id == model_id and self.pipeline is not None,
        }


def create_app(catalog_path: str, model_name: str = DEFAULT_MODEL):
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    catalog = Catalog.from_json(catalog_path)
    runtime = Runtime(catalog, model_name)
    app = FastAPI(title="KRM Trace Lab", version="0.1.0")

    @app.exception_handler(DiscDecodingError)
    async def disc_error_handler(request, error):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=422, content={"detail": str(error)})

    class AnalyzeRequest(BaseModel):
        text: str = Field(min_length=3, max_length=30_000)
        role_id: str | None = None
        max_competencies: int = Field(default=1, ge=1, le=8)
        max_quotes: int = Field(default=3, ge=1, le=5)
        allow_abort_token: bool = False
        resolution_extra_chars: int = Field(default=0, ge=0, le=10)
        matching_basis: str = Field(default="both", pattern="^(competencies|indicators|keywords|both)$")
        model_id: str | None = None
        use_disc: bool = False
        disc_budget: int = Field(default=4, ge=1, le=8)
        disc_prior_alpha: float = Field(default=0.0, ge=0.0, le=1.0)

    class QuoteFirstRequest(BaseModel):
        text: str = Field(min_length=3, max_length=30_000)
        role_id: str | None = None
        max_quotes: int = Field(default=5, ge=1, le=8)
        max_competencies_per_quote: int = Field(default=4, ge=1, le=8)
        allow_abort_token: bool = False
        resolution_extra_chars: int = Field(default=0, ge=0, le=10)
        matching_basis: str = Field(default="both", pattern="^(competencies|indicators|keywords|both)$")
        model_id: str | None = None
        use_disc: bool = False
        disc_budget: int = Field(default=4, ge=1, le=8)
        disc_prior_alpha: float = Field(default=0.0, ge=0.0, le=1.0)

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model": (
                runtime.models[runtime.active_model_id]["hf_name"]
                if runtime.active_model_id
                else runtime.models[runtime.default_model_id]["hf_name"]
            ),
            "model_loaded": runtime.pipeline is not None,
            "active_model_id": runtime.active_model_id,
            "default_model_id": runtime.default_model_id,
            "models": [runtime.model_public(key) for key in runtime.models],
        }

    @app.get("/api/models")
    def models() -> dict[str, object]:
        return {
            "active_model_id": runtime.active_model_id,
            "default_model_id": runtime.default_model_id,
            "models": [runtime.model_public(key) for key in runtime.models],
        }

    @app.get("/api/catalog")
    def get_catalog() -> dict[str, object]:
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
                        {
                            "id": indicator.id,
                            "text": indicator.text,
                            "levels": indicator.levels,
                        }
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

    @app.post("/api/analyze")
    def analyze(request: AnalyzeRequest) -> dict[str, object]:
        if request.role_id and request.role_id not in {r.id for r in catalog.roles}:
            raise HTTPException(status_code=400, detail="Unknown role_id")
        requested_model = request.model_id or runtime.default_model_id
        if requested_model not in runtime.models:
            raise HTTPException(status_code=400, detail="Unknown model_id")
        request_started = perf_counter()
        with runtime.lock:
            was_loaded = runtime.pipeline is not None and runtime.active_model_id == requested_model
            load_started = perf_counter()
            pipeline = runtime.get_pipeline(requested_model)
            model_load_seconds = perf_counter() - load_started if not was_loaded else 0.0
            result = pipeline.run_detailed(
                request.text,
                request.role_id,
                request.max_competencies,
                request.max_quotes,
                request.allow_abort_token,
                request.resolution_extra_chars,
                request.matching_basis,
                request.use_disc,
                request.disc_budget,
                request.disc_prior_alpha,
            )
        result["model"] = runtime.model_public(requested_model)
        result["timings"].update(
            {
                "model_load_s": model_load_seconds,
                "request_total_s": perf_counter() - request_started,
            }
        )
        return result

    @app.post("/api/analyze-quote-first")
    def analyze_quote_first(request: QuoteFirstRequest) -> dict[str, object]:
        if request.role_id and request.role_id not in {r.id for r in catalog.roles}:
            raise HTTPException(status_code=400, detail="Unknown role_id")
        requested_model = request.model_id or runtime.default_model_id
        if requested_model not in runtime.models:
            raise HTTPException(status_code=400, detail="Unknown model_id")
        request_started = perf_counter()
        with runtime.lock:
            was_loaded = runtime.pipeline is not None and runtime.active_model_id == requested_model
            load_started = perf_counter()
            pipeline = runtime.get_pipeline(requested_model)
            model_load_seconds = perf_counter() - load_started if not was_loaded else 0.0
            result = pipeline.run_quote_first_detailed(
                request.text,
                request.role_id,
                request.max_quotes,
                request.max_competencies_per_quote,
                request.allow_abort_token,
                request.resolution_extra_chars,
                request.matching_basis,
                request.use_disc,
                request.disc_budget,
                request.disc_prior_alpha,
            )
        result["model"] = runtime.model_public(requested_model)
        result["timings"].update(
            {
                "model_load_s": model_load_seconds,
                "request_total_s": perf_counter() - request_started,
            }
        )
        return result

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

