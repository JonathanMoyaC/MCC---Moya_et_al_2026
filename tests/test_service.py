from datetime import datetime

import pandas as pd

from ueh_mesa.calibration import EmpiricalInputModel
from ueh_mesa.demand import CATEGORIES, DAY_TYPES
from ueh_mesa.service import SimulationRequest, SimulationService


def profile(rate):
    rows = []
    for month in range(1, 13):
        for day_type in DAY_TYPES:
            for hour in range(24):
                row = {
                    "month": month,
                    "day_type": day_type,
                    "hour": hour,
                    "mean_arrivals_per_comparable_day_hour": rate if (month, day_type, hour) == (1, "dia_habil", 0) else 0,
                }
                row.update({f"p_{category}_{kind}": 0.2 for category in CATEGORIES for kind in ("raw", "smoothed")})
                rows.append(row)
    return pd.DataFrame(rows)


def calibration():
    return EmpiricalInputModel(
        registration=(1,), triage=(1,), attention_global=(4,), attention_by_class={}, metadata={"source": "test"}
    )


def test_service_runs_mesa_and_returns_only_aggregate_indicators():
    service = SimulationService(profile(40), calibration())
    result = service.run(SimulationRequest(start=datetime(2018, 1, 1), horizon_hours=1, master_seed=9))
    assert result["status"] == "completed"
    assert result["indicators"]["arrivals"] == result["indicators"]["completed"] > 0
    assert set(result["indicators"]["severity_counts"]) == set(CATEGORIES)
    assert result["indicators"]["length_of_stay"]["p95"] is not None
    assert result["method"]["engine"] == "Mesa"
    assert result["method"]["esi_model_applied"] is False
    assert "patient_id" not in str(result)
    assert "arrival_time" not in str(result)


def test_service_handles_a_zero_arrival_horizon_without_calling_empty_mesa_cohort():
    service = SimulationService(profile(0), calibration())
    result = service.run(SimulationRequest(start=datetime(2018, 1, 1), horizon_hours=1))
    assert result["indicators"]["arrivals"] == 0
    assert result["indicators"]["completed"] == 0
    assert result["indicators"]["length_of_stay"]["mean"] is None


def test_service_applies_demand_multiplier_and_returns_hourly_aggregates():
    service = SimulationService(profile(40), calibration())
    result = service.run(SimulationRequest(
        start=datetime(2018, 1, 1), horizon_hours=1, master_seed=9, demand_multiplier=1.25
    ))
    assert result["run"]["demand_multiplier"] == 1.25
    assert result["method"]["demand_multiplier"] == 1.25
    assert result["demand"]["hourly_arrivals"][0]["expected_arrivals"] == 50
    assert isinstance(result["demand"]["hourly_arrivals"][0]["arrivals"], int)


def test_service_is_reproducible_for_same_seed_and_varies_with_new_seed():
    service = SimulationService(profile(40), calibration())
    base = dict(start=datetime(2018, 1, 1), horizon_hours=1, master_seed=9)

    first = service.run(SimulationRequest(**base))
    repeated = service.run(SimulationRequest(**base))
    changed = service.run(SimulationRequest(**{**base, "master_seed": 10}))

    assert first == repeated
    assert first["indicators"]["arrivals"] != changed["indicators"]["arrivals"]
