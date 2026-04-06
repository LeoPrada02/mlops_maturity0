NUMERIC_COLUMNS = [
    "tenure_months",
    "monthly_charge",
    "total_charges",
    "support_tickets",
    "late_payments",
    "avg_monthly_usage_gb",
    "has_streaming",
    "has_security_pack",
    "num_products",
    "customer_age",
    "is_promo",
]

CATEGORICAL_COLUMNS = [
    "contract_type",
    "payment_method",
    "internet_service",
    "region",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET_COLUMN = "churn"
