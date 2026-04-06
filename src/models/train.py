from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.config import ensure_parent, load_params, resolve_path
from src.common.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False


def build_pipeline(max_iter: int, class_weight: str) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )

    model = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def log_to_mlflow(
    model: Pipeline,
    metrics: Dict[str, float],
    params: Dict[str, str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    if not MLFLOW_AVAILABLE:
        print("MLflow no está instalado en este entorno. Se omite tracking/registry.")
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", params["tracking_uri"])
    experiment_name = params["experiment_name"]
    registered_model_name = params["registered_model_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="train_logistic_regression_local") as run:
        mlflow.log_params(
            {
                "model_type": "logistic_regression",
                "max_iter": params["max_iter"],
                "class_weight": params["class_weight"],
                "threshold": params["threshold"],
                "train_rows": len(train_df),
                "test_rows": len(test_df),
            }
        )
        mlflow.log_metrics(metrics)

        input_example = test_df[FEATURE_COLUMNS].head(2)
        signature = mlflow.models.infer_signature(input_example, model.predict(input_example))

        try:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
            )

            client = mlflow.MlflowClient()
            latest_versions = client.search_model_versions(f"name='{registered_model_name}'")
            if latest_versions:
                latest = sorted(latest_versions, key=lambda v: int(v.version))[-1]
                client.set_registered_model_alias(registered_model_name, "champion", latest.version)
                print(
                    f"Modelo registrado en MLflow: {registered_model_name} "
                    f"versión {latest.version} con alias @champion"
                )
            print(f"Run MLflow: {run.info.run_id}")
            print(f"Model URI: {model_info.model_uri}")
        except Exception as exc:
            print(f"No se pudo registrar el modelo en MLflow Registry. Detalle: {exc}")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )


def main() -> None:
    params = load_params()
    dataset_cfg = params["dataset"]
    model_cfg = params["model"]
    artifact_cfg = params["artifacts"]
    mlflow_cfg = params["mlflow"]

    train_path = resolve_path(dataset_cfg["train_path"])
    test_path = resolve_path(dataset_cfg["test_path"])
    model_path = resolve_path(artifact_cfg["model_path"])
    metrics_path = resolve_path(artifact_cfg["metrics_path"])
    confusion_matrix_path = resolve_path(artifact_cfg["confusion_matrix_path"])

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    pipeline = build_pipeline(
        max_iter=int(model_cfg["max_iter"]),
        class_weight=str(model_cfg["class_weight"]),
    )
    pipeline.fit(X_train, y_train)

    prediction_threshold = float(model_cfg["threshold"])
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= prediction_threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }

    for path in [model_path, metrics_path, confusion_matrix_path]:
        ensure_parent(path)

    joblib.dump(pipeline, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(confusion_matrix_path, dpi=150)
    plt.close(fig)

    # enrich reference file with prediction for later Evidently comparison
    reference_path = resolve_path(dataset_cfg["reference_path"])
    reference_df = pd.read_csv(reference_path)
    reference_df["prediction"] = pipeline.predict(reference_df[FEATURE_COLUMNS])
    reference_df["prediction_proba"] = pipeline.predict_proba(reference_df[FEATURE_COLUMNS])[:, 1]
    reference_df.to_csv(reference_path, index=False, encoding="utf-8")

    merged_mlflow_cfg = {
        "tracking_uri": mlflow_cfg["tracking_uri"],
        "experiment_name": mlflow_cfg["experiment_name"],
        "registered_model_name": mlflow_cfg["registered_model_name"],
        "max_iter": model_cfg["max_iter"],
        "class_weight": model_cfg["class_weight"],
        "threshold": model_cfg["threshold"],
    }
    log_to_mlflow(
        model=pipeline,
        metrics=metrics,
        params=merged_mlflow_cfg,
        train_df=train_df,
        test_df=test_df,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Modelo guardado en: {model_path}")
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Matriz de confusión guardada en: {confusion_matrix_path}")


if __name__ == "__main__":
    main()
