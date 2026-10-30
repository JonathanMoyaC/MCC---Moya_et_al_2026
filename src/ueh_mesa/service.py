"""Application service for aggregate, synthetic-demand Mesa runs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from .calibration import EmpiricalInputModel
from .demand import generate_synthetic_cohort
from .model import EmergencyDepartmentModelMesa, ModelConfig


@dataclass(frozen=True)
class SimulationRequest:
    start: datetime
    horizon_hours: int
    master_seed: int = 42
    replication_id: int = 0
    severity_probabilities: Literal["raw", "smoothed"] = "smoothed"
    demand_multiplier: float = 1.0
    registration_capacity: int = 2
    triage_capacity: int = 1
    doctor_capacity: int = 12
    box_capacity: int = 18


class SimulationService:
    """Runs Mesa and returns summary indicators without patient-level rows."""

    def __init__(self, profile: pd.DataFrame, calibration: EmpiricalInputModel):
        self.profile = profile.copy()
        self.calibration = calibration

    def run(self, request: SimulationRequest) -> dict[str, object]:
        cohort, generation = generate_synthetic_cohort(
            self.profile,
            request.start,
            request.horizon_hours,
            master_seed=request.master_seed,
            replication_id=request.replication_id,
            severity_probabilities=request.severity_probabilities,
            demand_multiplier=request.demand_multiplier,
        )
        capacities = {
            "registration": request.registration_capacity,
            "triage": request.triage_capacity,
            "doctors": request.doctor_capacity,
            "boxes": request.box_capacity,
        }
        if cohort.empty:
            return self._empty_result(request, generation, capacities)

        model = EmergencyDepartmentModelMesa(
            cohort.to_dict("records"),
            self.calibration,
            config=ModelConfig(
                master_seed=request.master_seed,
                replication_id=request.replication_id,
                registration_capacity=request.registration_capacity,
                triage_capacity=request.triage_capacity,
                doctor_capacity=request.doctor_capacity,
                box_capacity=request.box_capacity,
            ),
        )
        patient_rows, minute_rows = model.run_to_completion()
        patient_frame = pd.DataFrame(patient_rows)
        minute_frame = pd.DataFrame(minute_rows)
        kpi_columns = {
            "wait_triage": "wait_triage_sim",
            "wait_attention": "wait_attention_sim",
            "door_to_doctor": "door2doctor_sim",
            "attention_time": "attention_time_sim",
            "length_of_stay": "los_sim",
        }
        indicators = {
            name: self._distribution(patient_frame[column])
            for name, column in kpi_columns.items()
        }
        indicators["queues"] = {
            key: {"mean": float(minute_frame[key].mean()), "max": int(minute_frame[key].max())}
            for key in ("queue_registration", "queue_triage", "queue_attention")
        }
        indicators["resource_utilization"] = {
            key.removesuffix("_utilization"): float(minute_frame[key].mean())
            for key in (
                "registration_utilization", "triage_utilization",
                "doctor_utilization", "box_utilization",
            )
        }
        return {
            "status": "completed",
            "run": self._run_metadata(request, generation, capacities),
            "indicators": {
                "arrivals": len(cohort),
                "completed": len(patient_rows),
                "severity_counts": generation["severity_counts"],
                "simulation_minutes": len(minute_rows),
                **indicators,
            },
            "method": self._method_metadata(request),
            "demand": {"hourly_arrivals": generation["hourly_arrivals"]},
        }

    @staticmethod
    def _distribution(values: pd.Series) -> dict[str, float | None]:
        numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        if len(numeric) == 0:
            return {"mean": None, "median": None, "p90": None, "p95": None}
        return {
            "mean": float(np.mean(numeric)),
            "median": float(np.median(numeric)),
            "p90": float(np.percentile(numeric, 90)),
            "p95": float(np.percentile(numeric, 95)),
        }

    def _empty_result(self, request, generation, capacities) -> dict[str, object]:
        return {
            "status": "completed",
            "run": self._run_metadata(request, generation, capacities),
            "indicators": {
                "arrivals": 0,
                "completed": 0,
                "severity_counts": generation["severity_counts"],
                "simulation_minutes": 0,
                **{
                    name: {"mean": None, "median": None, "p90": None, "p95": None}
                    for name in ("wait_triage", "wait_attention", "door_to_doctor", "attention_time", "length_of_stay")
                },
                "queues": {
                    key: {"mean": 0.0, "max": 0}
                    for key in ("queue_registration", "queue_triage", "queue_attention")
                },
                "resource_utilization": {key: 0.0 for key in ("registration", "triage", "doctor", "box")},
            },
            "method": self._method_metadata(request),
            "demand": {"hourly_arrivals": generation["hourly_arrivals"]},
        }

    @staticmethod
    def _run_metadata(request, generation, capacities) -> dict[str, object]:
        return {
            "start": request.start.isoformat(),
            "horizon_hours": request.horizon_hours,
            "demand_multiplier": request.demand_multiplier,
            "master_seed": request.master_seed,
            "replication_id": request.replication_id,
            "capacities": capacities,
            "synthetic_patient_count": generation["synthetic_patient_count"],
        }

    @staticmethod
    def _method_metadata(request) -> dict[str, object]:
        return {
            "engine": "Mesa",
            "severity_source": "historical_aggregate_probabilities",
            "severity_probabilities": request.severity_probabilities,
            "demand_multiplier": request.demand_multiplier,
            "esi_model_applied": False,
            "arrival_count_assumption": "Poisson per comparable calendar-day hour",
            "within_hour_arrival_assumption": "Uniform over integer minutes 0-59",
            "service_duration_source": "F3.1/H.1-R6 empirical calibration cache",
            "indicators_are_simulated": True,
        }
