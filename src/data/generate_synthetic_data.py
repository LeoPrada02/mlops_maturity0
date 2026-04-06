from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import PROJECT_ROOT

OUTPUT_PATH = PROJECT_ROOT / "data/raw/churn_sintetico.csv"


def main() -> None:
    rng = np.random.default_rng(42)
    n = 5000

    tenure = rng.integers(1, 73, size=n)
    monthly_charge = np.round(rng.normal(65, 18, size=n).clip(15, 160), 2)
    support_tickets = rng.poisson(1.7, size=n).clip(0, 10)
    late_payments = rng.poisson(0.7, size=n).clip(0, 8)
    usage_gb = np.round(rng.normal(120, 50, size=n).clip(5, 350), 2)
    contract = rng.choice(["mensual", "anual", "bianual"], size=n, p=[0.56, 0.28, 0.16])
    payment = rng.choice(["debito", "credito", "transferencia", "efectivo"], size=n, p=[0.35, 0.32, 0.21, 0.12])
    internet = rng.choice(["fibra", "cable", "movil", "ninguno"], size=n, p=[0.45, 0.28, 0.20, 0.07])
    streaming = rng.choice([0, 1], size=n, p=[0.33, 0.67])
    security_pack = rng.choice([0, 1], size=n, p=[0.46, 0.54])
    num_products = rng.integers(1, 5, size=n)
    region = rng.choice(["norte", "sur", "centro", "oeste"], size=n, p=[0.24, 0.21, 0.37, 0.18])
    customer_age = rng.integers(18, 79, size=n)
    is_promo = rng.choice([0, 1], size=n, p=[0.62, 0.38])
    total_charges = np.round((monthly_charge * tenure + rng.normal(0, 120, size=n)).clip(50, 12000), 2)

    score = (
        -1.2
        + 1.4 * (contract == "mensual").astype(float)
        - 0.9 * (contract == "bianual").astype(float)
        + 0.8 * (internet == "movil").astype(float)
        - 0.4 * (internet == "fibra").astype(float)
        + 0.35 * (payment == "efectivo").astype(float)
        + 0.18 * support_tickets
        + 0.28 * late_payments
        + 0.007 * (monthly_charge - 65)
        - 0.018 * tenure
        - 0.25 * security_pack
        + 0.12 * (num_products == 1).astype(float)
        - 0.08 * (num_products >= 3).astype(float)
        + 0.22 * is_promo
        + 0.05 * (customer_age < 25).astype(float)
    )
    proba = 1 / (1 + np.exp(-score))
    churn = rng.binomial(1, proba)

    df = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charge": monthly_charge,
            "total_charges": total_charges,
            "support_tickets": support_tickets,
            "late_payments": late_payments,
            "avg_monthly_usage_gb": usage_gb,
            "contract_type": contract,
            "payment_method": payment,
            "internet_service": internet,
            "has_streaming": streaming,
            "has_security_pack": security_pack,
            "num_products": num_products,
            "region": region,
            "customer_age": customer_age,
            "is_promo": is_promo,
            "churn": churn,
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Dataset sintético generado en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
