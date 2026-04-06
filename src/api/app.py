from __future__ import annotations

import csv
import json
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.common.config import load_params, resolve_path
from src.common.features import FEATURE_COLUMNS

REQUEST_COUNT = Counter("api_requests_total", "Total de requests recibidos", ["endpoint", "method"])
PREDICTION_COUNT = Counter("predictions_total", "Total de predicciones emitidas", ["prediction"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latencia de inferencia")
MODEL_LOADED = Gauge("model_loaded", "Indica si el modelo está cargado")
MODEL_VERSION = Gauge("model_version_info", "Versión del modelo local cargado", ["version"])

WRITE_LOCK = threading.Lock()
MODEL = None


class PredictionRequest(BaseModel):
    tenure_months: int = Field(..., ge=0)
    monthly_charge: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    support_tickets: int = Field(..., ge=0)
    late_payments: int = Field(..., ge=0)
    avg_monthly_usage_gb: float = Field(..., ge=0)
    contract_type: str
    payment_method: str
    internet_service: str
    has_streaming: int = Field(..., ge=0, le=1)
    has_security_pack: int = Field(..., ge=0, le=1)
    num_products: int = Field(..., ge=1)
    region: str
    customer_age: int = Field(..., ge=18)
    is_promo: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    churn_probability: float
    predicted_class: int
    model_path: str


def get_paths() -> Dict[str, Path]:
    params = load_params()
    return {
        "model_path": resolve_path(params["artifacts"]["model_path"]),
        "current_data_path": resolve_path(params["monitoring"]["current_data_path"]),
    }


def load_model() -> None:
    global MODEL
    model_path = get_paths()["model_path"]
    if not model_path.exists():
        MODEL_LOADED.set(0)
        raise FileNotFoundError(
            f"No existe el modelo en {model_path}. Ejecuta primero el entrenamiento."
        )
    MODEL = joblib.load(model_path)
    MODEL_LOADED.set(1)
    MODEL_VERSION.labels(version=str(int(model_path.stat().st_mtime))).set(1)


def ensure_monitoring_csv() -> None:
    current_data_path = get_paths()["current_data_path"]
    current_data_path.parent.mkdir(parents=True, exist_ok=True)
    if not current_data_path.exists():
        header = FEATURE_COLUMNS + ["prediction", "prediction_proba", "timestamp"]
        pd.DataFrame(columns=header).to_csv(current_data_path, index=False, encoding="utf-8")


def append_monitoring_row(payload: PredictionRequest, prediction: int, probability: float) -> None:
    current_data_path = get_paths()["current_data_path"]
    row = payload.model_dump()
    row["prediction"] = prediction
    row["prediction_proba"] = probability
    row["timestamp"] = datetime.now(timezone.utc).isoformat()

    with WRITE_LOCK:
        file_exists = current_data_path.exists()
        with open(current_data_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists or current_data_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_monitoring_csv()
    load_model()
    yield


app = FastAPI(
    title="API local de churn",
    version="1.0.0",
    description=(
        "Servicio local de inferencia para el demo MLOps. "
        "Expone scoring, healthcheck y endpoint de métricas para Prometheus."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> Dict[str, str]:
    REQUEST_COUNT.labels(endpoint="/health", method="GET").inc()
    return {"status": "ok", "model_loaded": "yes" if MODEL is not None else "no"}


@app.get("/sample-payload")
def sample_payload() -> Dict[str, object]:
    REQUEST_COUNT.labels(endpoint="/sample-payload", method="GET").inc()
    sample = {
        "tenure_months": 6,
        "monthly_charge": 92.5,
        "total_charges": 555.0,
        "support_tickets": 3,
        "late_payments": 2,
        "avg_monthly_usage_gb": 180.0,
        "contract_type": "mensual",
        "payment_method": "credito",
        "internet_service": "fibra",
        "has_streaming": 1,
        "has_security_pack": 0,
        "num_products": 1,
        "region": "centro",
        "customer_age": 29,
        "is_promo": 1,
    }
    return sample


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()

    if MODEL is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")

    start = time.perf_counter()
    payload = pd.DataFrame([request.model_dump()])[FEATURE_COLUMNS]
    probability = float(MODEL.predict_proba(payload)[0, 1])
    predicted_class = int(probability >= load_params()["model"]["threshold"])
    elapsed = time.perf_counter() - start
    PREDICTION_LATENCY.observe(elapsed)
    PREDICTION_COUNT.labels(prediction=str(predicted_class)).inc()

    append_monitoring_row(request, predicted_class, probability)

    return PredictionResponse(
        churn_probability=round(probability, 6),
        predicted_class=predicted_class,
        model_path=str(get_paths()["model_path"]),
    )


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    REQUEST_COUNT.labels(endpoint="/metrics", method="GET").inc()
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
