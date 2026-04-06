from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import requests

from src.common.config import resolve_path
from src.common.features import FEATURE_COLUMNS


def apply_drift(df: pd.DataFrame) -> pd.DataFrame:
    drifted = df.copy()
    drifted["monthly_charge"] = (drifted["monthly_charge"] * 1.25).round(2)
    drifted["support_tickets"] = (drifted["support_tickets"] + 2).clip(lower=0)
    drifted["late_payments"] = (drifted["late_payments"] + 1).clip(lower=0)
    drifted["contract_type"] = "mensual"
    drifted["payment_method"] = "efectivo"
    drifted["internet_service"] = "movil"
    return drifted


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera tráfico de scoring contra la API local.")
    parser.add_argument("--n", type=int, default=50, help="Cantidad de requests a enviar.")
    parser.add_argument("--url", type=str, default="http://localhost:8000/predict", help="URL del endpoint.")
    parser.add_argument("--drift", action="store_true", help="Si se activa, altera la distribución para generar drift.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Pausa entre requests.")
    args = parser.parse_args()

    scoring_path = resolve_path("data/scoring/scoring_samples.csv")
    df = pd.read_csv(scoring_path)[FEATURE_COLUMNS]

    if args.drift:
        df = apply_drift(df)

    rows = []
    for i in range(args.n):
        row = df.iloc[i % len(df)].to_dict()
        response = requests.post(args.url, json=row, timeout=30)
        response.raise_for_status()
        rows.append(response.json())
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(json.dumps({"requests_sent": len(rows), "drift_mode": args.drift, "last_response": rows[-1]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
