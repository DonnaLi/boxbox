"""FastAPI inference server.

    uvicorn boxbox.server:app --port 8000
    BOXBOX_MAX_BATCH=1 uvicorn boxbox.server:app   # batching off, for comparison
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel

from .batcher import MicroBatcher
from .features import LapState

MODEL_PATH = Path(os.environ.get("BOXBOX_MODEL", "models/lap_time.onnx"))
MAX_BATCH = int(os.environ.get("BOXBOX_MAX_BATCH", "20"))
MAX_WAIT_MS = float(os.environ.get("BOXBOX_MAX_WAIT_MS", "5"))


class LapIn(BaseModel):
    driver: str
    lap_number: int
    total_laps: int
    tyre_life: float
    stint: int
    position: int
    compound: str
    last_lap_s: float
    rolling3_lap_s: float


class PredictionOut(BaseModel):
    driver: str
    lap_number: int
    predicted_next_lap_s: float
    pace_drop_s: float    # predicted minus recent average: a crude "box soon?" signal


def make_predict_fn(path: Path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return lambda x: sess.run(None, {name: x})[0].reshape(-1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.batcher = MicroBatcher(make_predict_fn(MODEL_PATH), MAX_BATCH, MAX_WAIT_MS)
    app.state.batcher.start()
    yield
    await app.state.batcher.stop()


app = FastAPI(title="boxbox", lifespan=lifespan)


@app.get("/health")
async def health():
    b = app.state.batcher
    avg = float(np.mean(b.batch_sizes)) if b.batch_sizes else 0.0
    return {"ok": True, "max_batch": b.max_batch, "max_wait_ms": b.max_wait * 1000,
            "avg_batch_size": round(avg, 2), "batches_run": len(b.batch_sizes)}


@app.post("/predict", response_model=PredictionOut)
async def predict(lap: LapIn):
    state = LapState(**lap.model_dump())
    pred = await app.state.batcher.submit(state.to_vector())
    return PredictionOut(driver=lap.driver, lap_number=lap.lap_number,
                         predicted_next_lap_s=round(pred, 3),
                         pace_drop_s=round(pred - lap.rolling3_lap_s, 3))
