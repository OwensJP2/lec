"""FastAPI interface for the LEC codec."""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from queue import Queue
from threading import RLock, Thread
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .codec import LECCodec
from .config import LECConfig
from .crypto import CryptoError
from .model import LanguageModel, ModelError, ToyLanguageModel, TransformersModel
from .probabilities import SamplingPolicy
from .range_coder import CodingError

DEFAULT_CAPACITY = LECConfig().capacity
HOSTED_DEMO = os.environ.get("VERCEL") == "1"
_cache_lock = RLock()
_loaded_modes: set[bool] = set()


class CodecOptions(BaseModel):
    key: str = Field(min_length=1)
    capacity: int = Field(default=DEFAULT_CAPACITY, ge=1, le=256)
    toy: bool = False
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=2, le=100)
    top_p: float = Field(default=0.95, ge=0.1, le=1.0)
    precision_bits: int = Field(default=16, ge=8, le=24)
    repetition_penalty: float = Field(default=1.25, ge=1.0, le=2.0)
    no_repeat_ngram: int = Field(default=6, ge=0, le=16)


class EncodeRequest(CodecOptions):
    message: str


class DecodeRequest(CodecOptions):
    text: str


class MetricsResponse(BaseModel):
    tokens: int
    payload_bits: int
    bits_per_token: float
    model_entropy_bits: float
    entropy_utilization: float


class EncodeResponse(BaseModel):
    text: str
    metrics: MetricsResponse


class DecodeResponse(BaseModel):
    message: str
    metrics: MetricsResponse


@lru_cache(maxsize=2)
def _model(toy: bool) -> LanguageModel:
    model: LanguageModel
    if toy:
        model = ToyLanguageModel()
    else:
        if HOSTED_DEMO:
            raise ModelError(
                "The hosted Vercel demo uses the lightweight model. "
                "Run LEC locally for DistilGPT-2."
            )
        config = LECConfig()
        model = TransformersModel(config.model_id, config.model_revision)
    _loaded_modes.add(toy)
    return model


@lru_cache(maxsize=64)
def _codec(
    toy: bool,
    capacity: int,
    temperature: float,
    top_k: int,
    top_p: float,
    precision_bits: int,
    repetition_penalty: float,
    no_repeat_ngram: int,
) -> LECCodec:
    policy = SamplingPolicy(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        precision_bits=precision_bits,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram=no_repeat_ngram,
    )
    return LECCodec(LECConfig(capacity=capacity, policy=policy), model=_model(toy))


def _get_codec(options: CodecOptions) -> LECCodec:
    # lru_cache can invoke a missing value more than once under concurrent calls.
    with _cache_lock:
        return _codec(
            options.toy,
            options.capacity,
            options.temperature,
            options.top_k,
            options.top_p,
            options.precision_bits,
            options.repetition_penalty,
            options.no_repeat_ngram,
        )


def _metrics(value: Any) -> MetricsResponse:
    return MetricsResponse(**value.to_dict())


app = FastAPI(
    title="Linguistic Entropy Coding",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url=None,
)


@app.get("/api/status")
def status() -> dict[str, Any]:
    config = LECConfig()
    return {
        "ok": True,
        "default_capacity": config.capacity,
        "model_id": config.model_id,
        "real_model_loaded": False in _loaded_modes,
        "real_model_available": not HOSTED_DEMO,
        "hosted_demo": HOSTED_DEMO,
        "toy_model_loaded": True in _loaded_modes,
        "toy_available": True,
    }


@app.post("/api/encode", response_model=EncodeResponse)
def encode(request: EncodeRequest) -> EncodeResponse:
    try:
        result = _get_codec(request).encode(
            request.message,
            request.key,
        )
        return EncodeResponse(text=result.text, metrics=_metrics(result.metrics))
    except (ValueError, CodingError, CryptoError, ModelError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.post("/api/encode/stream")
def encode_stream(request: EncodeRequest) -> StreamingResponse:
    """Stream cumulative covertext snapshots as newline-delimited JSON."""
    events: Queue[dict[str, Any] | None] = Queue()

    def on_token(text: str, tokens: int) -> None:
        events.put({"type": "token", "text": text, "tokens": tokens})

    def run() -> None:
        try:
            result = _get_codec(request).encode(
                request.message,
                request.key,
                on_token=on_token,
            )
            events.put(
                {
                    "type": "done",
                    "text": result.text,
                    "metrics": result.metrics.to_dict(),
                }
            )
        except (ValueError, CodingError, CryptoError, ModelError) as exc:
            events.put({"type": "error", "detail": str(exc)})
        except Exception:
            events.put({"type": "error", "detail": "Encoding failed unexpectedly."})
        finally:
            events.put(None)

    def generate() -> Iterator[str]:
        yield json.dumps({"type": "start"}) + "\n"
        Thread(target=run, daemon=True).start()
        while True:
            event = events.get()
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/decode", response_model=DecodeResponse)
def decode(request: DecodeRequest) -> DecodeResponse:
    try:
        message, metrics = _get_codec(request).decode(
            request.text,
            request.key,
        )
        return DecodeResponse(message=message, metrics=_metrics(metrics))
    except (ValueError, CodingError, CryptoError, ModelError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


def _static_directory() -> Path:
    source = Path(__file__).resolve().parent.parent / "web"
    if source.is_dir():
        return source
    installed = Path(sys.prefix) / "share" / "lec" / "web"
    if installed.is_dir():
        return installed
    raise RuntimeError("LEC web assets are not installed")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(
        _static_directory() / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


app.mount("/", StaticFiles(directory=_static_directory(), html=True), name="web")
