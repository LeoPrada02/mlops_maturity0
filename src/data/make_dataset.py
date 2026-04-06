from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.config import ensure_parent, load_params, resolve_path
from src.common.features import FEATURE_COLUMNS, TARGET_COLUMN


def main() -> None:
    params = load_params()
    dataset_cfg = params["dataset"]

    raw_path = resolve_path(dataset_cfg["raw_path"])
    train_path = resolve_path(dataset_cfg["train_path"])
    test_path = resolve_path(dataset_cfg["test_path"])
    reference_path = resolve_path(dataset_cfg["reference_path"])
    scoring_path = resolve_path(dataset_cfg["scoring_path"])

    random_state = int(dataset_cfg["random_state"])
    test_size = float(dataset_cfg["test_size"])

    df = pd.read_csv(raw_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    reference_df = X_train.sample(n=min(1000, len(X_train)), random_state=random_state).copy()
    scoring_df = X_test.sample(n=min(100, len(X_test)), random_state=random_state).copy()

    for path in [train_path, test_path, reference_path, scoring_path]:
        ensure_parent(path)

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    reference_df.to_csv(reference_path, index=False, encoding="utf-8")
    scoring_df.to_csv(scoring_path, index=False, encoding="utf-8")

    print(f"Train guardado en: {train_path}")
    print(f"Test guardado en: {test_path}")
    print(f"Reference guardado en: {reference_path}")
    print(f"Scoring guardado en: {scoring_path}")


if __name__ == "__main__":
    main()
