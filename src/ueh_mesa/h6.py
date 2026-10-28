"""Puente de entradas H.6 hacia el motor Mesa.

H.6 publica asignaciones de severidad por paciente; las horas de llegada se
recuperan de la cohorte bloqueada H.1-R6. Cada ventana se simula por separado
para no introducir periodos vacíos entre ventanas en el reloj del modelo.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .calibration import EmpiricalInputModel, load_restricted_cache
from .model import EmergencyDepartmentModelMesa, ModelConfig


H6_WINDOW_SIZES = {
    "W1_20180101_20180106": 950,
    "W2_20180501_20180506": 989,
    "W3_20180829_20180903": 990,
    "W4_20181227_20190101": 1019,
}
H6_TOTAL_PATIENTS = sum(H6_WINDOW_SIZES.values())
_SCENARIO_ALIASES = {
    "observada": "observed", "observed": "observed",
    "empirica": "empirical", "empirical": "empirical",
    "jerarquica": "hierarchical", "hierarchical": "hierarchical",
}
_TRIAGE_ALIASES = {
    **{str(i): f"C{i}" for i in range(1, 6)},
    **{f"{i}.0": f"C{i}" for i in range(1, 6)},
    **{f"C{i}": f"C{i}" for i in range(1, 6)},
    **{f"ESI {i}": f"C{i}" for i in range(1, 6)},
    **{f"ESI{i}": f"C{i}" for i in range(1, 6)},
}


def _normalise_scenario(value: object) -> str:
    key = str(value).strip().lower()
    if key not in _SCENARIO_ALIASES:
        raise ValueError(f"Escenario H.6 no reconocido: {value!r}")
    return _SCENARIO_ALIASES[key]


def _normalise_triage(value: object) -> str:
    if pd.isna(value):
        raise ValueError("La categoría de triage contiene valores faltantes.")
    key = str(value).strip().upper()
    if key not in _TRIAGE_ALIASES:
        raise ValueError(f"Categoría de triage inválida: {value!r}")
    return _TRIAGE_ALIASES[key]


def read_h6_assignments(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    replication_id: int | None = None,
    scenario: str | None = None,
) -> pd.DataFrame:
    """Read a locked H.6 assignment CSV, optionally enforcing its SHA manifest."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest_path is not None:
        manifest = pd.read_csv(manifest_path)
        required = {"replica", "scenario", "rows", "file", "sha256"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"El manifiesto H.6 no contiene {sorted(missing)}.")
        match = manifest.loc[manifest["file"].astype(str).eq(path.name)]
        if replication_id is not None:
            match = match.loc[pd.to_numeric(match["replica"], errors="coerce").eq(int(replication_id))]
        if scenario is not None:
            expected_scenario = _normalise_scenario(scenario)
            match = match.loc[match["scenario"].map(_normalise_scenario).eq(expected_scenario)]
        if len(match) != 1:
            raise ValueError(f"El manifiesto debe identificar una sola entrada para {path.name}.")
        expected_hash = str(match.iloc[0]["sha256"]).lower()
        if digest != expected_hash:
            raise ValueError(f"El SHA-256 de {path.name} no coincide con el manifiesto H.6.")
        declared_rows = int(match.iloc[0]["rows"])
    else:
        declared_rows = None

    frame = pd.read_csv(path, compression="infer")
    if declared_rows is not None and len(frame) != declared_rows:
        raise ValueError(f"{path.name}: manifiesto declara {declared_rows} filas y contiene {len(frame)}.")
    return frame


def read_h6_reference_cohort(
    cache_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> pd.DataFrame:
    """Read the three H.1-R6 cohort fields through the restricted cache loader."""
    frame, _ = load_restricted_cache(cache_path, expected_sha256)
    required = {"patient_id", "arrival_time", "triage_observed"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"La caché H.1-R6 no contiene {sorted(missing)}.")
    return frame[["patient_id", "arrival_time", "triage_observed"]].copy()


def prepare_h6_windows(
    assignments: pd.DataFrame,
    reference_cohort: pd.DataFrame,
    *,
    replication_id: int | None = None,
    scenario: str | None = None,
    expected_window_sizes: Mapping[str, int] | None = H6_WINDOW_SIZES,
) -> dict[str, list[dict[str, object]]]:
    """Join fixed H.6 severities to the locked H.1-R6 arrivals and validate them.

    The returned groups are ready for separate ``EmergencyDepartmentModelMesa``
    runs, one per H.6 window. Patient identifiers remain local input keys.
    """
    required_assignments = {
        "replica", "scenario", "patient_id", "window_id",
        "triage_observed", "triage_assigned",
    }
    missing = required_assignments - set(assignments.columns)
    if missing:
        raise ValueError(f"El insumo H.6 no contiene {sorted(missing)}.")
    required_reference = {"patient_id", "arrival_time", "triage_observed"}
    missing = required_reference - set(reference_cohort.columns)
    if missing:
        raise ValueError(f"La cohorte H.1-R6 no contiene {sorted(missing)}.")

    left = assignments[list(required_assignments)].copy()
    if left[list(required_assignments)].isna().any().any():
        raise ValueError("El insumo H.6 contiene campos requeridos faltantes.")
    left["patient_id"] = left["patient_id"].astype(str)
    left["window_id"] = left["window_id"].astype(str)
    left["scenario_normalized"] = left["scenario"].map(_normalise_scenario)
    left["triage_observed_normalized"] = left["triage_observed"].map(_normalise_triage)
    left["triage_assigned_normalized"] = left["triage_assigned"].map(_normalise_triage)

    replicas = pd.to_numeric(left["replica"], errors="coerce")
    if replicas.isna().any() or replicas.nunique() != 1:
        raise ValueError("Cada archivo de entrada H.6 debe contener una sola réplica.")
    detected_replication = int(replicas.iloc[0])
    if replication_id is not None and detected_replication != int(replication_id):
        raise ValueError(f"La réplica del archivo ({detected_replication}) no coincide con {replication_id}.")
    detected_scenarios = set(left["scenario_normalized"])
    if len(detected_scenarios) != 1:
        raise ValueError("Cada archivo H.6 debe contener un único escenario.")
    detected_scenario = next(iter(detected_scenarios))
    if scenario is not None and detected_scenario != _normalise_scenario(scenario):
        raise ValueError(f"El escenario del archivo ({detected_scenario}) no coincide con {scenario!r}.")
    if left.duplicated(["patient_id", "window_id"]).any():
        raise ValueError("El insumo H.6 tiene claves paciente–ventana duplicadas.")
    if left["patient_id"].duplicated().any():
        raise ValueError("El insumo H.6 repite pacientes entre ventanas.")

    right = reference_cohort[["patient_id", "arrival_time", "triage_observed"]].copy()
    if right["patient_id"].isna().any():
        raise ValueError("La cohorte H.1-R6 contiene patient_id faltantes.")
    right["patient_id"] = right["patient_id"].astype(str)
    right["arrival_time"] = pd.to_datetime(right["arrival_time"], errors="coerce")
    right["triage_observed_normalized"] = right["triage_observed"].map(_normalise_triage)
    if right["patient_id"].duplicated().any():
        matched = right["patient_id"].isin(set(left["patient_id"]))
        if right.loc[matched, "patient_id"].duplicated().any():
            raise ValueError("La cohorte H.1-R6 tiene patient_id duplicados para H.6.")

    joined = left.merge(
        right[["patient_id", "arrival_time", "triage_observed_normalized"]],
        on="patient_id", how="left", validate="many_to_one", indicator=True,
        suffixes=("_assignment", "_reference"),
    )
    unmatched = int(joined["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"{unmatched} pacientes H.6 no aparecen en la cohorte H.1-R6.")
    mismatch = joined["triage_observed_normalized_assignment"].ne(
        joined["triage_observed_normalized_reference"]
    )
    if mismatch.any():
        raise ValueError(f"{int(mismatch.sum())} categorías observadas H.6 difieren de H.1-R6.")
    if joined["arrival_time"].isna().any():
        raise ValueError("La cohorte bloqueada contiene horas de llegada faltantes.")

    expected = dict(expected_window_sizes) if expected_window_sizes is not None else None
    observed_sizes = joined.groupby("window_id").size().to_dict()
    if expected is not None and observed_sizes != expected:
        raise ValueError(f"Las ventanas o tamaños no coinciden con H.6: recibidos={observed_sizes}.")
    if expected is not None and set(joined["window_id"]) != set(expected):
        raise ValueError("Los identificadores de ventana no coinciden con H.6.")

    results: dict[str, list[dict[str, object]]] = {}
    for window_id, group in joined.groupby("window_id", sort=True):
        rows = []
        for row in group.itertuples(index=False):
            rows.append({
                "patient_id": str(row.patient_id),
                "arrival_time": row.arrival_time,
                "triage_observed": row.triage_observed_normalized_reference,
                "triage_assigned": row.triage_assigned_normalized,
                "window_id": str(row.window_id),
                "severity_source": detected_scenario,
                "replication_id": detected_replication,
            })
        results[str(window_id)] = rows
    return results


def build_observed_h6_assignments(
    template_assignments: pd.DataFrame,
    reference_cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Build the observed control branch from the locked IDs and H.1-R6 labels.

    The confirmatory H.6 input manifest contains the empirical and hierarchical
    predictive branches. The observed control is retained from the reference
    cohort and does not need a separately manifested predictive input file.
    """
    required = {"replica", "patient_id", "window_id"}
    missing = required - set(template_assignments.columns)
    if missing:
        raise ValueError(f"La plantilla de cohortes H.6 no contiene {sorted(missing)}.")
    if not {"patient_id", "triage_observed"}.issubset(reference_cohort.columns):
        raise ValueError("La cohorte H.1-R6 requiere patient_id y triage_observed.")
    template = template_assignments[list(required)].copy()
    if template.isna().any().any() or pd.to_numeric(template["replica"], errors="coerce").nunique() != 1:
        raise ValueError("La plantilla H.6 debe tener una sola réplica y campos completos.")
    template["replica"] = pd.to_numeric(template["replica"], errors="raise").astype(int)
    template["patient_id"] = template["patient_id"].astype(str)
    template["window_id"] = template["window_id"].astype(str)
    if template.duplicated(["patient_id", "window_id"]).any():
        raise ValueError("La plantilla H.6 contiene paciente–ventana duplicados.")

    reference = reference_cohort[["patient_id", "triage_observed"]].copy()
    reference["patient_id"] = reference["patient_id"].astype(str)
    if reference["patient_id"].duplicated().any():
        matched = reference["patient_id"].isin(set(template["patient_id"]))
        if reference.loc[matched, "patient_id"].duplicated().any():
            raise ValueError("La cohorte H.1-R6 tiene patient_id duplicados para H.6.")
    reference["triage_observed"] = reference["triage_observed"].map(_normalise_triage)
    joined = template.merge(reference, on="patient_id", how="left", validate="many_to_one", indicator=True)
    if joined["_merge"].ne("both").any():
        raise ValueError("Faltan pacientes de la plantilla H.6 en la cohorte H.1-R6.")
    joined["triage_assigned"] = joined["triage_observed"]
    joined["scenario"] = "observed"
    return joined[[
        "replica", "scenario", "patient_id", "window_id",
        "triage_observed", "triage_assigned",
    ]]


def run_h6_mesa_scenario(
    assignments: pd.DataFrame,
    reference_cohort: pd.DataFrame,
    service_samples: Mapping[str, Sequence[float]] | EmpiricalInputModel,
    *,
    config: ModelConfig | None = None,
    replication_id: int | None = None,
    scenario: str | None = None,
    expected_window_sizes: Mapping[str, int] | None = H6_WINDOW_SIZES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one H.6 fixed-assignment scenario in Mesa, separately by window."""
    windows = prepare_h6_windows(
        assignments, reference_cohort,
        replication_id=replication_id,
        scenario=scenario,
        expected_window_sizes=expected_window_sizes,
    )
    replica = int(pd.to_numeric(assignments["replica"], errors="raise").iloc[0])
    scenario_name = _normalise_scenario(assignments["scenario"].iloc[0])
    base_config = config or ModelConfig()
    patient_frames: list[pd.DataFrame] = []
    minute_frames: list[pd.DataFrame] = []
    for window_id, cohort in windows.items():
        model = EmergencyDepartmentModelMesa(
            cohort=cohort,
            service_samples=service_samples,
            config=replace(base_config, replication_id=replica),
        )
        patients, minutes = model.run_to_completion()
        patient_frame = pd.DataFrame(patients)
        patient_frame["window_id"] = window_id
        patient_frame["replication_id"] = replica
        patient_frame["scenario"] = scenario_name
        minute_frame = pd.DataFrame(minutes)
        minute_frame["window_id"] = window_id
        minute_frame["replication_id"] = replica
        minute_frame["scenario"] = scenario_name
        patient_frames.append(patient_frame)
        minute_frames.append(minute_frame)

    patient_results = pd.concat(patient_frames, ignore_index=True)
    minute_results = pd.concat(minute_frames, ignore_index=True)
    if len(patient_results) != len(assignments):
        raise RuntimeError("La salida Mesa H.6 no conserva el total de pacientes de entrada.")
    if patient_results.duplicated(["patient_id", "window_id"]).any():
        raise RuntimeError("La salida Mesa H.6 contiene pacientes–ventana duplicados.")
    return patient_results, minute_results
