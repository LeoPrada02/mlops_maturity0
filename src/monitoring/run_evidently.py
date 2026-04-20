from __future__ import annotations

import json

import pandas as pd

from src.common.config import ensure_parent, load_params, resolve_path
from src.common.features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


def main() -> None:
    params = load_params()
    dataset_cfg = params["dataset"]
    monitoring_cfg = params["monitoring"]

    reference_path = resolve_path(dataset_cfg["reference_path"])
    current_path = resolve_path(monitoring_cfg["current_data_path"])
    report_path = resolve_path(monitoring_cfg["evidently_report_path"])
    summary_path = resolve_path(monitoring_cfg["evidently_summary_path"])

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    if current_df.empty:
        raise ValueError(
            "monitoring/current_data.csv está vacío. Ejecuta predicciones antes de correr Evidently."
        )

    cols_to_keep = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ["prediction", "prediction_proba"]
    reference_df = reference_df[cols_to_keep].copy()
    current_df = current_df[cols_to_keep].copy()

    schema = DataDefinition(
        numerical_columns=NUMERIC_COLUMNS + ["prediction_proba"],
        categorical_columns=CATEGORICAL_COLUMNS + ["prediction"],
    )

    current_dataset = Dataset.from_pandas(current_df, data_definition=schema)
    reference_dataset = Dataset.from_pandas(reference_df, data_definition=schema)

    report = Report([
        DataDriftPreset(),
        DataSummaryPreset(),
    ])

    snapshot = report.run(current_data=current_dataset, reference_data=reference_dataset)

    ensure_parent(report_path)
    ensure_parent(summary_path)

    snapshot.save_html(str(report_path))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(snapshot.json())   #CORECCION
#with open(summary_path, "w", encoding="utf-8") as f:
#        json.dump(snapshot.dict(), f, indent=2, ensure_ascii=False) // ANTES ESTABA ASI, DABA ERROR POR OBJETOS NO SERIALIZABLES, USAMOS EL MÉTODO .json() DE SNAPSHOT QUE YA DEVUELVE UN STRING JSON SERIALIZABLE

    print(f"Reporte Evidently HTML: {report_path}")
    print(f"Resumen JSON: {summary_path}")


if __name__ == "__main__":
    main()
