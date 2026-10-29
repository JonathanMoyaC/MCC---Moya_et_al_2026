"""Reproduce the empirical duration inputs used by F3.1/H.1-R6.

The cache is loaded through a restricted unpickler. Only pandas/numpy
reconstruction globals needed by the audited DataFrame format are permitted.
The original cache is never copied into this package.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from hashlib import sha256
from pathlib import Path
import pickle
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from hashlib import blake2b
from math import floor


EXPECTED_F3_CACHE_SHA256 = "5edef6b238a0723954b3c678d6b8192a3af672afb227245332daa58ad2a74cea"
F3_REFERENCE_NOTEBOOK_ID = "1cpm1IFL7ZwpGpJk8J0Tq8Godx89cy9WO"
F3_REFERENCE_NOTEBOOK_SHA256 = "b64cb3392be0808217cd0c03e2d0ccad7a0a2f488cb21a5369194c1ea43ce561"
SIM_CALIBRATION_START = pd.Timestamp("2017-09-18 00:00:00")
SIM_CALIBRATION_END = pd.Timestamp("2017-09-20 23:59:59")
TRIAGE_LEVELS = ("C1", "C2", "C3", "C4", "C5")

# Exact empirical vectors frozen in the F3.1 notebook. The per-patient
# sampler applies the same deterministic discrete empirical quantile to them.
OBS_ADMISSION_MIN = (
    9,14,26,29,21,24,12,18,21,16,18,11,2,5,5,3,5,4,4,6,
    8,9,8,4,4,5,15,12,3,5,3,1,2,11,7,10,13,21,21,16,
    18,1,1,4,4,20,14,13,11,8,10,9,10,6,7,6,1,6,8,7,
    6,6,5,12,8,5,15,17,10,7,5,11,5,6,3,
)
OBS_TRIAGE_MIN = (
    1,4,3,4,1,3,3,5,1,3,1,2,2,2,2,1,1,3,3,2,
    4,2,3,2,2,2,2,2,1,2,2,3,2,3,8,3,1,2,2,2,
    3,2,1,1,2,3,3,12,8,9,1,2,1,1,1,1,2,10,3,3,
    1,1,2,6,1,3,1,1,1,1,1,2,1,2,1,2,1,1,1,
)


class _RestrictedCacheUnpickler(pickle.Unpickler):
    """Allow only the NumPy/pandas globals observed in the audited cache."""

    _ALLOWED = {
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.internals.managers", "BlockManager"),
        ("pandas._libs.internals", "_unpickle_block"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.indexes.datetimes", "DatetimeIndex"),
        ("pandas.core.arrays.datetimes", "DatetimeArray"),
        ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
        ("pandas._libs.tslibs.timestamps", "_unpickle_timestamp"),
        ("pandas._libs.tslibs.nattype", "__nat_unpickle"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.numeric", "_frombuffer"),
        ("builtins", "slice"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "complex"),
        ("datetime", "datetime"),
        ("datetime", "date"),
        ("datetime", "timedelta"),
    }

    def find_class(self, module: str, name: str):  # noqa: ANN201
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"Global no permitido en caché: {module}.{name}")
        return super().find_class(module, name)


def _cache_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_uniform(master_seed: int, replication_id: int, entity_id: str, process: str) -> float:
    """Exact BLAKE2b key and midpoint conversion from the F3.1 notebook."""
    payload = f"{int(master_seed)}|{int(replication_id)}|{str(entity_id)}|{str(process)}".encode("utf-8")
    digest = blake2b(payload, digest_size=8, person=b"UEH_VE").digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return (integer + 0.5) / (2**64)


def empirical_quantile_sample(values: Sequence[float], u: float) -> int:
    """Exact discrete empirical quantile used by F3.1/H.1-R6."""
    cleaned = np.asarray([float(v) for v in values if 1 <= float(v) <= 720], dtype=float)
    if cleaned.size == 0:
        raise ValueError("Distribución empírica vacía tras aplicar límites 1–720 min.")
    cleaned.sort()
    index = min(int(floor(u * len(cleaned))), len(cleaned) - 1)
    return max(1, int(round(cleaned[index])))


def empirical_quantile_sample_sorted(values: Sequence[float], u: float) -> int:
    """Same operation for a validated, already sorted input vector."""
    if not values:
        raise ValueError("Distribución empírica vacía.")
    index = min(int(floor(u * len(values))), len(values) - 1)
    return max(1, int(round(float(values[index]))))


def _normalize_triage(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper().replace(" ", "")
    if text in TRIAGE_LEVELS:
        return text
    if text in {"1", "2", "3", "4", "5"}:
        return f"C{text}"
    return None


def _build_attention_by_class(
    selected: pd.DataFrame,
    class_nonmissing_counts: Mapping[str, int],
    *,
    threshold: int = 30,
) -> dict[str, tuple[float, ...]]:
    """Match H.1-R6: gate on nonmissing rows, then clean values to 1–720."""
    result: dict[str, tuple[float, ...]] = {}
    for category in TRIAGE_LEVELS:
        values = selected.loc[
            selected["triage_observed"].eq(category), "attention_time_real"
        ].to_numpy(dtype=float)
        if int(class_nonmissing_counts.get(category, 0)) >= threshold and len(values):
            result[category] = tuple(float(value) for value in np.sort(values))
    return result


def load_restricted_cache(path: str | Path, expected_sha256: str | None = EXPECTED_F3_CACHE_SHA256) -> tuple[pd.DataFrame, str]:
    """Load the audited F1_R1_v2 DataFrame and return it with its file hash."""
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    digest = _cache_sha256(cache_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"La caché no coincide con el SHA-256 esperado: {digest}")
    with cache_path.open("rb") as stream:
        frame = _RestrictedCacheUnpickler(stream).load()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Se esperaba pandas.DataFrame; se encontró {type(frame).__name__}")
    return frame, digest


def select_f3_calibration(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply F3.1 temporal and quality rules, then retain attention inputs."""
    required = {
        "patient_id", "arrival_time", "triage_time", "medical_start_time",
        "departure_time", "triage_observed", "wait_triage_real",
        "door2doctor_real", "los_real", "attention_time_real",
    }
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas en la caché: {sorted(missing_columns)}")

    d = frame.copy()
    for column in ("arrival_time", "triage_time", "medical_start_time", "departure_time"):
        d[column] = pd.to_datetime(d[column], errors="coerce")
    d["triage_observed"] = d["triage_observed"].map(_normalize_triage)
    in_window = d["arrival_time"].between(SIM_CALIBRATION_START, SIM_CALIBRATION_END, inclusive="both")
    d = d.loc[in_window].copy()

    required_values = ["arrival_time", "triage_time", "medical_start_time", "departure_time", "triage_observed"]
    missing = d[required_values].isna().any(axis=1)
    ordered = (
        (d["arrival_time"] <= d["triage_time"])
        & (d["triage_time"] <= d["medical_start_time"])
        & (d["medical_start_time"] <= d["departure_time"])
    )
    extreme = (
        (d["wait_triage_real"] > 1440)
        | (d["door2doctor_real"] > 1440)
        | (d["los_real"] > 10080)
    )
    quality = pd.Series("valid", index=d.index, dtype="object")
    quality.loc[extreme & ~missing & ordered] = "review_extreme"
    quality.loc[missing] = "exclude_missing"
    quality.loc[~missing & ~ordered] = "exclude_order"
    quality_eligible = quality.isin(("valid", "review_extreme"))
    attention = pd.to_numeric(d["attention_time_real"], errors="coerce")
    class_nonmissing_counts = (
        d.loc[quality_eligible]
        .assign(_attention_numeric=attention.loc[quality_eligible])
        .dropna(subset=["_attention_numeric"])
        .groupby("triage_observed")
        .size()
        .to_dict()
    )
    eligible = quality_eligible & attention.between(1, 720, inclusive="both")
    selected = d.loc[eligible, ["patient_id", "arrival_time", "triage_observed"]].copy()
    selected["attention_time_real"] = attention.loc[eligible].astype(float)
    quality_counts = quality.value_counts().to_dict()
    audit = {
        "calibration_window_rows": int(len(d)),
        "quality_valid": int(quality_counts.get("valid", 0)),
        "quality_review_extreme": int(quality_counts.get("review_extreme", 0)),
        "quality_exclude_missing": int(quality_counts.get("exclude_missing", 0)),
        "quality_exclude_order": int(quality_counts.get("exclude_order", 0)),
        "eligible_attention_global": int(len(selected)),
        # H.1-R6 tests the class threshold after dropping missing values but
        # before clean_empirical applies its inclusive 1–720 minute limits.
        "attention_class_nonmissing_counts": {
            category: int(class_nonmissing_counts.get(category, 0))
            for category in TRIAGE_LEVELS
        },
    }
    return selected.reset_index(drop=True), audit


@dataclass(frozen=True)
class EmpiricalInputModel:
    registration: tuple[float, ...]
    triage: tuple[float, ...]
    attention_global: tuple[float, ...]
    attention_by_class: Mapping[str, tuple[float, ...]]
    metadata: Mapping[str, object]

    def as_service_samples(self) -> dict[str, Sequence[float]]:
        samples: dict[str, Sequence[float]] = {
            "registration": self.registration,
            "triage": self.triage,
            "attention": self.attention_global,
        }
        samples.update({f"attention_{category}": values for category, values in self.attention_by_class.items()})
        return samples

    @classmethod
    def from_f3_cache(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = EXPECTED_F3_CACHE_SHA256,
    ) -> "EmpiricalInputModel":
        frame, digest = load_restricted_cache(path, expected_sha256)
        selected, audit = select_f3_calibration(frame)
        global_values = np.sort(selected["attention_time_real"].to_numpy(dtype=float))
        if len(global_values) < 100:
            raise ValueError(f"F3/H.1-R6 requiere al menos 100 duraciones de atención; se encontraron {len(global_values)}")
        class_counts = selected["triage_observed"].value_counts().to_dict()
        class_nonmissing_counts = audit["attention_class_nonmissing_counts"]
        by_class = _build_attention_by_class(selected, class_nonmissing_counts)
        metadata: dict[str, object] = {
            "source": "F3.1/H.1-R6 empirical input model",
            "reference_notebook_drive_id": F3_REFERENCE_NOTEBOOK_ID,
            "reference_notebook_sha256": F3_REFERENCE_NOTEBOOK_SHA256,
            "cache_sha256": digest,
            "calibration_start": SIM_CALIBRATION_START.isoformat(),
            "calibration_end": SIM_CALIBRATION_END.isoformat(),
            **audit,
            "attention_by_class_counts": {c: int(class_counts.get(c, 0)) for c in TRIAGE_LEVELS},
            "attention_class_threshold_counts": {
                c: int(class_nonmissing_counts.get(c, 0)) for c in TRIAGE_LEVELS
            },
            "attention_class_threshold": 30,
            "registration_sample_count": len(OBS_ADMISSION_MIN),
            "triage_sample_count": len(OBS_TRIAGE_MIN),
            "attention_global_min": float(global_values.min()),
            "attention_global_max": float(global_values.max()),
        }
        return cls(
            registration=tuple(sorted(float(x) for x in OBS_ADMISSION_MIN if 1 <= x <= 720)),
            triage=tuple(sorted(float(x) for x in OBS_TRIAGE_MIN if 1 <= x <= 720)),
            attention_global=tuple(float(x) for x in global_values),
            attention_by_class=by_class,
            metadata=metadata,
        )


EXPECTED_ADULT_SERVICE_SAMPLES_SHA256 = "340f27d51b7638ab105b65708a98757141eacf4c1094fea23f46cbd796ae94c5"


def with_adult_triage_samples(
    calibration: EmpiricalInputModel,
    path: str | Path,
    *,
    expected_sha256: str | None = EXPECTED_ADULT_SERVICE_SAMPLES_SHA256,
) -> EmpiricalInputModel:
    """Use the frozen adult-only triage and attention inputs for sensitivity runs.

    Registration is a deterministic one-minute placeholder unless overridden
    per simulation request; the 17 registration-to-triage intervals are not
    treated as verified registration service durations. Attention vectors come
    from the same adult-filtered 2017 preparation used by the H6 experiment,
    rather than the broader pre-adult-filter F3 cache vectors.
    """
    samples_path = Path(path)
    if not samples_path.is_file():
        raise FileNotFoundError(samples_path)
    digest = _cache_sha256(samples_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"La muestra externa de triage no coincide con el SHA-256 esperado: {digest}")
    payload = json.loads(samples_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "adult-service-samples-v1":
        raise ValueError("Versión de esquema no válida para las muestras de servicio")
    if payload.get("population") != "confirmed age >=18 years only":
        raise ValueError("Las muestras externas deben corresponder solo a adultos confirmados")
    registration_observations = tuple(float(value) for value in payload.get("registration_minutes", ()))
    if len(registration_observations) != 17 or any(
        not 1 <= value <= 720 for value in registration_observations
    ):
        raise ValueError("Se esperan los 17 intervalos de admisión–triage auditados; no se usan como servicio")
    triage_values = tuple(sorted(float(value) for value in payload.get("triage_minutes", ())))
    if len(triage_values) != 38 or any(not 1 <= value <= 720 for value in triage_values):
        raise ValueError("Se requieren las 38 mediciones adultas de triage dentro de 1–720 min")
    attention_global = tuple(sorted(float(value) for value in payload.get("attention_global_minutes", ())))
    if len(attention_global) != 457 or any(not 1 <= value <= 720 for value in attention_global):
        raise ValueError("Se requieren las 457 duraciones adultas de atención de 2017 dentro de 1–720 min")
    raw_by_class = payload.get("attention_by_class_minutes", {})
    if not isinstance(raw_by_class, dict):
        raise ValueError("Las duraciones de atención por categoría deben ser un objeto")
    attention_by_class = {
        str(category): tuple(sorted(float(value) for value in values))
        for category, values in raw_by_class.items()
    }
    expected_class_counts = {"C2": 200, "C3": 220, "C4": 29}
    if set(attention_by_class) != set(expected_class_counts) or any(
        len(attention_by_class[category]) != expected_count
        or any(not 1 <= value <= 720 for value in attention_by_class[category])
        for category, expected_count in expected_class_counts.items()
    ):
        raise ValueError("Se requieren duraciones adultas de atención para C2=200, C3=220 y C4=29")
    metadata = {
        **calibration.metadata,
        "registration_service_source": "fixed one-minute placeholder; unvalidated; 17 admission-to-triage intervals are not used as service times",
        "registration_interval_observation_count_not_used": len(registration_observations),
        "triage_service_source": "adult-only in-situ measurements, 2025; external exploratory input",
        "triage_service_measurement_year": payload.get("triage_measurement_year"),
        "triage_service_sample_count": len(triage_values),
        "attention_service_source": "adult-only 2017 calibration vectors frozen for H6 triage sensitivity experiment",
        "attention_service_sample_count": len(attention_global),
        "attention_service_class_counts": {category: len(values) for category, values in attention_by_class.items()},
        "adult_service_samples_sha256": digest,
    }
    return EmpiricalInputModel(
        registration=(1.0,),
        triage=triage_values,
        attention_global=attention_global,
        attention_by_class=attention_by_class,
        metadata=metadata,
    )
