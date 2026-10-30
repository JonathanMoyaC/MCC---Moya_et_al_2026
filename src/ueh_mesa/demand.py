"""Generate synthetic arrival cohorts from aggregate Phase 2 profiles.

This module is deliberately separate from the fixed-cohort H.1-R6/H.6 adapter.
Hourly Poisson counts and minute-uniform arrival placement are explicit model
assumptions; they are not empirical claims made by the aggregate profiles.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .calibration import stable_uniform


CATEGORIES = ("C1", "C2", "C3", "C4", "C5")
DAY_TYPES = ("dia_habil", "fin_de_semana")
PROFILE_KEYS = ("month", "day_type", "hour")


def load_arrival_profile(path: str | Path) -> pd.DataFrame:
    """Read and validate the 576-stratum Phase 2 arrival profile."""
    frame = pd.read_csv(path)
    required = {
        *PROFILE_KEYS,
        "mean_arrivals_per_comparable_day_hour",
        *(f"p_{category}_smoothed" for category in CATEGORIES),
        *(f"p_{category}_raw" for category in CATEGORIES),
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"El perfil de demanda no contiene {sorted(missing)}.")
    return validate_arrival_profile(frame)


def validate_arrival_profile(profile: pd.DataFrame) -> pd.DataFrame:
    """Validate grid, rates and both severity probability vectors."""
    required = {
        *PROFILE_KEYS,
        "mean_arrivals_per_comparable_day_hour",
        *(f"p_{category}_smoothed" for category in CATEGORIES),
        *(f"p_{category}_raw" for category in CATEGORIES),
    }
    missing = required - set(profile.columns)
    if missing:
        raise ValueError(f"El perfil de demanda no contiene {sorted(missing)}.")
    frame = profile.copy()
    frame["month"] = pd.to_numeric(frame["month"], errors="raise").astype(int)
    frame["hour"] = pd.to_numeric(frame["hour"], errors="raise").astype(int)
    if not frame["day_type"].isin(DAY_TYPES).all():
        raise ValueError("day_type debe ser dia_habil o fin_de_semana.")
    if frame.duplicated(list(PROFILE_KEYS)).any():
        raise ValueError("El perfil contiene estratos mes–tipo de día–hora duplicados.")
    expected = {(m, d, h) for m in range(1, 13) for d in DAY_TYPES for h in range(24)}
    actual = set(map(tuple, frame[list(PROFILE_KEYS)].itertuples(index=False, name=None)))
    if actual != expected:
        raise ValueError("El perfil debe cubrir exactamente los 576 estratos mes–tipo de día–hora.")

    rate_col = "mean_arrivals_per_comparable_day_hour"
    rates = pd.to_numeric(frame[rate_col], errors="coerce")
    if rates.isna().any() or not np.isfinite(rates.to_numpy()).all() or (rates < 0).any():
        raise ValueError("Las tasas horarias deben ser finitas y no negativas.")
    frame[rate_col] = rates.astype(float)
    for mode in ("raw", "smoothed"):
        columns = [f"p_{category}_{mode}" for category in CATEGORIES]
        probabilities = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError(f"Las probabilidades {mode} deben ser finitas y estar entre 0 y 1.")
        sums = probabilities.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-6, rtol=0):
            raise ValueError(f"Las probabilidades {mode} deben sumar 1 en cada estrato.")
        frame[columns] = probabilities
    return frame.sort_values(list(PROFILE_KEYS), kind="stable").reset_index(drop=True)


def _hour_rng(master_seed: int, replication_id: int, hour_key: str, process: str) -> np.random.Generator:
    uniform = stable_uniform(master_seed, replication_id, hour_key, f"demand_{process}")
    seed = min(int(uniform * (2**64)), 2**64 - 1)
    return np.random.Generator(np.random.PCG64(seed))


def generate_synthetic_cohort(
    profile: pd.DataFrame,
    start: datetime,
    horizon_hours: int,
    *,
    master_seed: int = 42,
    replication_id: int = 0,
    severity_probabilities: Literal["raw", "smoothed"] = "smoothed",
    demand_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Sample a reproducible synthetic cohort and aggregate generation metadata.

    Counts are Poisson with each selected profile cell's mean arrivals per
    comparable calendar day-hour. Arrival times are uniform over integer
    minutes 0–59 within each hour. Triage categories use raw or smoothed
    historical proportions. No source patient rows or identifiers are used.
    """
    if not isinstance(start, datetime):
        raise TypeError("start debe ser datetime.")
    if any((start.minute, start.second, start.microsecond)):
        raise ValueError("start debe estar alineado al inicio de una hora.")
    if isinstance(horizon_hours, bool) or int(horizon_hours) != horizon_hours or horizon_hours <= 0:
        raise ValueError("horizon_hours debe ser un entero positivo.")
    if severity_probabilities not in ("raw", "smoothed"):
        raise ValueError("severity_probabilities debe ser 'raw' o 'smoothed'.")
    if not np.isfinite(demand_multiplier) or not 0.1 <= float(demand_multiplier) <= 3.0:
        raise ValueError("demand_multiplier debe estar entre 0.1 y 3.0.")

    frame = validate_arrival_profile(profile)
    lookup = frame.set_index(list(PROFILE_KEYS))
    sampled: list[tuple[datetime, str]] = []
    hourly_arrivals: list[dict[str, object]] = []
    for offset in range(int(horizon_hours)):
        hour_start = start + timedelta(hours=offset)
        day_type = "fin_de_semana" if hour_start.weekday() >= 5 else "dia_habil"
        cell = lookup.loc[(hour_start.month, day_type, hour_start.hour)]
        hour_key = hour_start.isoformat(timespec="hours")
        count_rng = _hour_rng(master_seed, replication_id, hour_key, "count")
        time_rng = _hour_rng(master_seed, replication_id, hour_key, "arrival_minute")
        severity_rng = _hour_rng(master_seed, replication_id, hour_key, "severity")
        expected_count = float(cell["mean_arrivals_per_comparable_day_hour"]) * float(demand_multiplier)
        count = int(count_rng.poisson(expected_count))
        hourly_arrivals.append({
            "time": hour_start.isoformat(timespec="hours"),
            "expected_arrivals": expected_count,
            "arrivals": count,
        })
        if count == 0:
            continue
        minutes = time_rng.integers(0, 60, size=count)
        probability_columns = [f"p_{category}_{severity_probabilities}" for category in CATEGORIES]
        probabilities = cell[probability_columns].to_numpy(dtype=float)
        severities = severity_rng.choice(CATEGORIES, size=count, p=probabilities)
        sampled.extend(
            (hour_start + timedelta(minutes=int(minute)), str(category))
            for minute, category in zip(minutes, severities)
        )

    sampled.sort(key=lambda row: (row[0], row[1]))
    rows = [
        {
            "patient_id": f"syn-r{int(replication_id):03d}-{index:07d}",
            "arrival_time": arrival,
            "triage_observed": severity,
        }
        for index, (arrival, severity) in enumerate(sampled, start=1)
    ]
    cohort = pd.DataFrame(rows, columns=["patient_id", "arrival_time", "triage_observed"])
    metadata: dict[str, object] = {
        "source_profile": "Fase 2 aggregate arrival/severity profile",
        "profile_year_scope": "2016-2018 pooled historical",
        "start": start.isoformat(),
        "horizon_hours": int(horizon_hours),
        "demand_multiplier": float(demand_multiplier),
        "master_seed": int(master_seed),
        "replication_id": int(replication_id),
        "arrival_count_model": "Poisson per comparable calendar-day hour (assumption)",
        "within_hour_arrival_model": "Uniform over integer minutes 0-59 (assumption)",
        "severity_probability_source": severity_probabilities,
        "synthetic_patient_count": len(cohort),
        "hourly_arrivals": hourly_arrivals,
        "severity_counts": {
            category: int(cohort["triage_observed"].eq(category).sum())
            for category in CATEGORIES
        },
    }
    return cohort, metadata
