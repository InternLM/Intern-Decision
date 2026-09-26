"""HTTP adapters for the shared inference pipeline and streaming demo."""

import json
import logging
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from src.inference.config import ROOT, InferenceConfig
from src.inference.engine import DecisionEngine
from src.inference.pipeline import DecisionPipeline
from src.inference.thinking import ThinkingModel
from src.service.examples import examples
from src.service.uploads import MAX_UPLOAD_BYTES, MAX_UPLOAD_TOTAL, materialize_images

STATIC = Path(__file__).with_name("static")


def create_app(pipeline=None):
    @asynccontextmanager
    async def lifespan(app):
        if pipeline is None:
            config = InferenceConfig.load(os.environ.get("INFERENCE_CONFIG", ROOT / "configs/inference/default.json"))
            thinker = ThinkingModel(config.thinking)
            try:
                engine = DecisionEngine.from_config(config)
                app.state.pipeline = DecisionPipeline(engine, config, thinker)
                yield
            finally:
                await thinker.close()
                app.state.pipeline = None
        else:
            app.state.pipeline = pipeline
            yield

    app = FastAPI(title="InternDecision", lifespan=lifespan)
    app.state.pipeline = pipeline

    def current():
        if app.state.pipeline is None:
            raise HTTPException(503, "Inference is not ready.")
        return app.state.pipeline

    @app.get("/health")
    def health():
        return {"status": "ok" if app.state.pipeline else "unavailable", "model_loaded": app.state.pipeline is not None}

    @app.get("/demo-meta")
    def metadata():
        inference = current()
        return {
            "thinking": {
                "available": inference.thinker.available,
                "enabled": inference.config.thinking.enabled,
                "confidence_threshold": inference.config.thinking.confidence_threshold,
            },
            "uploads": {
                "enabled": True,
                "max_images": 8,
                "max_bytes": MAX_UPLOAD_BYTES,
                "max_total_bytes": MAX_UPLOAD_TOTAL,
            },
        }

    @app.post("/v1/jev")
    @app.post("/v1/decisions")
    def predict(request: dict):
        upload_dir = None
        try:
            request, upload_dir = materialize_images(request)
            result = current().predict(request, allow_images=bool(upload_dir))
            if log_path := os.environ.get("INFERENCE_LOG"):
                record = {
                    "request_id": result["request_id"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": result["model"],
                    **result["usage"],
                    **result["timing"],
                }
                try:
                    with Path(log_path).open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record) + "\n")
                except OSError:
                    logging.getLogger(__name__).warning("Unable to write inference timing log")
            return result
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)

    @app.post("/v1/decisions/{request_id}/thinking")
    async def thinking(request_id: str, request: dict):
        if set(request) != {"field", "token"}:
            raise HTTPException(422, "Supply field and token from the initial response.")
        inference = current()
        try:
            session, cached = inference.claim(request_id, request["field"], request["token"])
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

        async def events():
            async for event in inference.stream_claimed(session, request["field"], cached):
                yield (
                    "event: "
                    + event["event"]
                    + "\ndata: "
                    + json.dumps(event, ensure_ascii=False, allow_nan=False)
                    + "\n\n"
                )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/demo.js")
    def javascript():
        return FileResponse(STATIC / "demo.js", media_type="application/javascript")

    @app.get("/examples")
    def example_requests():
        return examples()

    return app


app = create_app()
