"""Patient-level differential checks between H.1-R6 and Mesa outputs."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


H1_MARK_COLUMNS = (
    "arrival_minute", "registration_start", "registration_end",
    "triage_start", "triage_end", "attention_start", "doctor_release",
    "box_release", "departure_minute",
)
H1_KPI_COLUMNS = (
    "wait_triage_sim", "wait_attention_sim", "door2doctor_sim",
    "attention_time_sim", "doctor_contact_sim", "box_occupation_sim",
    "los_sim",
)
H1_CATEGORICAL_COLUMNS = ("triage_observed", "triage_assigned")


def _category_number(value: object) -> int:
    text = str(value).strip().upper()
    if text.startswith("C"):
        text = text[1:]
    if text.endswith(".0"):
        text = text[:-2]
    number = int(text)
    if number not in range(1, 6):
        raise ValueError(f"Categoría de triage inválida: {value!r}")
    return number


def compare_h1_patient_outputs(
    reference_h1: pd.DataFrame,
    candidate_mesa: pd.DataFrame,
    *,
    key_columns: Iterable[str] = ("patient_id", "window_id"),
) -> dict[str, object]:
    """Compare event marks, assigned categories, and KPIs by patient-window.

    Returns aggregate discrepancy statistics only; the result does not contain
    patient identifiers or patient-level rows.
    """
    keys = tuple(key_columns)
    ref = reference_h1.copy()
    mesa = candidate_mesa.copy()
    if "departure_minute" not in mesa and "departure" in mesa:
        mesa = mesa.rename(columns={"departure": "departure_minute"})

    required = set(keys) | set(H1_MARK_COLUMNS) | set(H1_KPI_COLUMNS) | set(H1_CATEGORICAL_COLUMNS)
    for label, frame in (("H.1-R6", ref), ("Mesa", mesa)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"La salida {label} no contiene {sorted(missing)}.")
        if frame.duplicated(list(keys)).any():
            raise ValueError(f"La salida {label} tiene claves paciente–ventana duplicadas.")
    ref["patient_id"] = ref["patient_id"].astype(str)
    mesa["patient_id"] = mesa["patient_id"].astype(str)
    ref["window_id"] = ref["window_id"].astype(str)
    mesa["window_id"] = mesa["window_id"].astype(str)
    joined = ref.merge(
        mesa,
        on=list(keys),
        how="outer",
        suffixes=("_h1", "_mesa"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = int(joined["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"Las salidas H.1-R6 y Mesa difieren en {unmatched} claves paciente–ventana.")

    categorical = {}
    for column in H1_CATEGORICAL_COLUMNS:
        a = joined[f"{column}_h1"].map(_category_number)
        b = joined[f"{column}_mesa"].map(_category_number)
        categorical[column] = {
            "equal_rows": int(a.eq(b).sum()),
            "rows": int(len(joined)),
        }

    metrics: dict[str, object] = {}
    for column in (*H1_MARK_COLUMNS, *H1_KPI_COLUMNS):
        a = pd.to_numeric(joined[f"{column}_h1"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(joined[f"{column}_mesa"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"La columna {column} contiene valores faltantes o no finitos.")
        delta = b - a
        abs_delta = np.abs(delta)
        metrics[column] = {
            "equal_rows": int(np.count_nonzero(delta == 0)),
            "rows": int(len(delta)),
            "mean_absolute_error": float(abs_delta.mean()),
            "p95_absolute_error": float(np.quantile(abs_delta, 0.95)),
            "max_absolute_error": float(abs_delta.max()),
        }

    return {
        "patient_window_rows": int(len(joined)),
        "unmatched_patient_window_rows": 0,
        "categories": categorical,
        "metrics": metrics,
        "exact_equivalence": all(
            item["equal_rows"] == item["rows"]
            for item in [*categorical.values(), *metrics.values()]
        ),
    }
