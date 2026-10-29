from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import ceil
from numbers import Integral
from typing import Iterable, Mapping, Sequence

from mesa import Agent, Model
from .calibration import EmpiricalInputModel, empirical_quantile_sample_sorted, stable_uniform


SEVERITY_TO_PRIORITY = {f"C{i}": i for i in range(1, 6)}
PROCESS_NAMES = ("registration", "triage", "attention")


@dataclass(frozen=True)
class ModelConfig:
    master_seed: int = 42
    replication_id: int = 0
    registration_capacity: int = 2
    triage_capacity: int = 1
    doctor_capacity: int = 12
    box_capacity: int = 18
    registration_factor: float = 1.0
    registration_service_minutes: int | None = None
    triage_factor: float = 1.0
    attention_factor: float = 1.0
    attention_factor_critical: float = 1.0
    attention_factor_noncritical: float = 1.0
    doctor_occupancy_fraction: float = 0.25
    box_occupancy_fraction: float = 0.60
    box_release_delay_min: int = 0
    aging_interval_min: int = 90
    aging_floor_priority: int = 2
    drain_days: int = 7

    def __post_init__(self) -> None:
        for name in ("registration_capacity", "triage_capacity", "doctor_capacity", "box_capacity", "aging_interval_min", "drain_days"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} debe ser positivo")
        for name in ("registration_factor", "triage_factor", "attention_factor", "attention_factor_critical", "attention_factor_noncritical"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} debe ser positivo")
        if not 0 < self.doctor_occupancy_fraction <= 1:
            raise ValueError("doctor_occupancy_fraction debe estar en (0, 1]")
        if not 0 < self.box_occupancy_fraction <= 1:
            raise ValueError("box_occupancy_fraction debe estar en (0, 1]")
        if self.registration_service_minutes is not None and not 0 <= self.registration_service_minutes <= 2:
            raise ValueError("registration_service_minutes debe estar entre 0 y 2 min")
        if self.box_release_delay_min < 0:
            raise ValueError("box_release_delay_min no puede ser negativo")


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, bool):
        raise ValueError("arrival_minute debe ser entero o datetime-like")
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise ValueError(f"arrival_time debe ser entero o datetime-like: {value!r}")


def normalize_cohort(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Validate and normalize an exogenous cohort; timestamps become relative minutes."""
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("La cohorte no puede estar vacía")
    for index, row in enumerate(materialized):
        if "patient_id" not in row or "arrival_time" not in row:
            raise ValueError("Cada paciente requiere patient_id y arrival_time")
        row["patient_id"] = str(row["patient_id"])
        observed = str(row.get("triage_observed", "")).upper()
        if observed not in SEVERITY_TO_PRIORITY:
            raise ValueError(f"triage_observed inválido para patient_id={row['patient_id']}: {observed!r}")
        row["triage_observed"] = observed
        assigned = row.get("triage_assigned")
        if assigned is not None:
            assigned = str(assigned).upper()
            if assigned not in SEVERITY_TO_PRIORITY:
                raise ValueError(f"triage_assigned inválido para patient_id={row['patient_id']}: {assigned!r}")
            row["triage_assigned"] = assigned
        row["_input_order"] = index
    ids = [row["patient_id"] for row in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("patient_id debe ser único dentro de la cohorte")
    are_numeric = [isinstance(row["arrival_time"], Integral) and not isinstance(row["arrival_time"], bool) for row in materialized]
    if all(are_numeric):
        first = min(int(row["arrival_time"]) for row in materialized)
        for row in materialized:
            row["arrival_minute"] = int(row["arrival_time"]) - first
    elif any(are_numeric):
        raise ValueError("La cohorte no puede mezclar arrival_time entero y timestamp")
    else:
        for row in materialized:
            row["_arrival_datetime"] = _coerce_datetime(row["arrival_time"])
        timestamps = [row["_arrival_datetime"] for row in materialized]
        if any((ts.tzinfo is None) != (timestamps[0].tzinfo is None) for ts in timestamps):
            raise ValueError("La cohorte mezcla timestamps con y sin zona horaria")
        start = min(timestamps).replace(second=0, microsecond=0)
        for row in materialized:
            seconds = (row["_arrival_datetime"] - start).total_seconds()
            row["arrival_minute"] = int(round(seconds / 60.0))
            row.pop("_arrival_datetime", None)
    materialized.sort(key=lambda r: (int(r["arrival_minute"]), r["patient_id"], r["_input_order"]))
    return materialized


class PatientAgent(Agent):
    """Patient entity; service/resource decisions remain explicit in the model."""

    def __init__(self, model: EmergencyDepartmentModelMesa, cohort_row: Mapping[str, object]):
        super().__init__(model)
        self.patient_id = str(cohort_row["patient_id"])
        self.data = dict(cohort_row)
        self.arrival_minute = int(cohort_row["arrival_minute"])
        self.triage_observed = str(cohort_row["triage_observed"])
        self.triage_assigned: str | None = None
        self.state = "not_arrived"
        self.queue_entry_minute: int | None = None
        self.registration_start: int | None = None
        self.registration_end: int | None = None
        self.triage_start: int | None = None
        self.triage_end: int | None = None
        self.attention_start: int | None = None
        self.doctor_release: int | None = None
        self.box_release: int | None = None
        self.departure: int | None = None

    def step(self) -> None:
        # Patient transitions are coordinated in a fixed model stage order.
        return None


class EmergencyDepartmentModelMesa(Model):
    """Minute-step Mesa port for fixed cohorts, preserving H.1-R6 event order.

    Durations are passed as empirical samples (minutes). This module deliberately
    does not infer service distributions from arrival-to-triage or LOS fields.
    """

    def __init__(
        self,
        cohort: Iterable[Mapping[str, object]],
        service_samples: Mapping[str, Sequence[float]] | EmpiricalInputModel,
        *,
        config: ModelConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        super().__init__(rng=self.config.master_seed if seed is None else seed)
        self.cohort = normalize_cohort(cohort)
        self.calibration_metadata: Mapping[str, object] | None = None
        if isinstance(service_samples, EmpiricalInputModel):
            self.calibration_metadata = service_samples.metadata
            service_samples = service_samples.as_service_samples()
        self._validate_samples(service_samples)
        self.service_samples = {key: tuple(sorted(float(v) for v in values)) for key, values in service_samples.items()}
        self.clock_minute = 0
        self._next_patient_index = 0
        self.patients: dict[str, PatientAgent] = {}
        self.queues: dict[str, list[PatientAgent]] = {"registration": [], "triage": [], "attention": []}
        self._active: dict[str, list[tuple[int, PatientAgent]]] = {name: [] for name in PROCESS_NAMES}
        self._doctors_busy_until: list[int] = []
        self._boxes_busy_until: list[int] = []
        self.patient_results: list[dict[str, object]] = []
        self.minute_results: list[dict[str, int | float]] = []
        self._arrival_minutes = [int(row["arrival_minute"]) for row in self.cohort]
        self._drain_limit = self._arrival_minutes[-1] + self.config.drain_days * 24 * 60
        self.running = True

    @staticmethod
    def _validate_samples(samples: Mapping[str, Sequence[float]]) -> None:
        missing = set(PROCESS_NAMES) - set(samples)
        if missing:
            raise ValueError(f"Faltan distribuciones empíricas: {sorted(missing)}")
        for process, values in samples.items():
            if not values or any(not 1 <= float(v) <= 720 for v in values):
                raise ValueError(f"{process}: se requieren duraciones en el intervalo 1–720 min")

    def _sample_minutes(self, patient: PatientAgent, process: str, category: str | None = None) -> int:
        key = process
        if process == "attention" and category is not None:
            by_category = self.service_samples.get(f"attention_{category}")
            if by_category:
                key = f"attention_{category}"
        values = self.service_samples[key]
        u = stable_uniform(self.config.master_seed, self.config.replication_id, patient.patient_id, process)
        sampled = empirical_quantile_sample_sorted(values, u)
        factor = {
            "registration": self.config.registration_factor,
            "triage": self.config.triage_factor,
            "attention": self.config.attention_factor,
        }[process]
        if process == "attention" and category is not None:
            factor *= self.config.attention_factor_critical if category in ("C1", "C2") else self.config.attention_factor_noncritical
        return max(1, round(sampled * factor))

    def _complete_and_release(self) -> None:
        for stage in PROCESS_NAMES:
            remaining = []
            for end, patient in self._active[stage]:
                if end > self.clock_minute:
                    remaining.append((end, patient))
                    continue
                if stage == "registration":
                    patient.registration_end = self.clock_minute
                    patient.state = "waiting_triage"
                    patient.queue_entry_minute = self.clock_minute
                    self.queues["triage"].append(patient)
                elif stage == "triage":
                    patient.triage_end = self.clock_minute
                    patient.triage_assigned = str(patient.data.get("triage_assigned") or patient.triage_observed)
                    patient.state = "waiting_attention"
                    patient.queue_entry_minute = self.clock_minute
                    self.queues["attention"].append(patient)
                else:
                    patient.departure = self.clock_minute
                    patient.state = "completed"
                    self.patient_results.append(self._patient_record(patient))
            self._active[stage] = remaining
        self._doctors_busy_until = [t for t in self._doctors_busy_until if t > self.clock_minute]
        self._boxes_busy_until = [t for t in self._boxes_busy_until if t > self.clock_minute]

    def _inject_arrivals(self) -> None:
        while self._next_patient_index < len(self.cohort) and self._arrival_minutes[self._next_patient_index] <= self.clock_minute:
            row = self.cohort[self._next_patient_index]
            patient = PatientAgent(self, row)
            patient.state = "waiting_registration"
            patient.queue_entry_minute = self.clock_minute
            self.patients[patient.patient_id] = patient
            self.queues["registration"].append(patient)
            self._next_patient_index += 1

    def _assign_single_resource(self, stage: str, capacity: int) -> None:
        free = capacity - len(self._active[stage])
        while free > 0 and self.queues[stage]:
            patient = self.queues[stage].pop(0)
            if stage == "registration" and self.config.registration_service_minutes is not None:
                duration = self.config.registration_service_minutes
            else:
                duration = self._sample_minutes(patient, stage)
            if stage == "registration":
                patient.registration_start = self.clock_minute
                patient.state = "registration"
                if duration == 0:
                    patient.registration_end = self.clock_minute
                    patient.state = "waiting_triage"
                    patient.queue_entry_minute = self.clock_minute
                    self.queues["triage"].append(patient)
                    continue
            else:
                patient.triage_start = self.clock_minute
                patient.state = "triage"
            self._active[stage].append((self.clock_minute + duration, patient))
            free -= 1

    def _effective_priority(self, patient: PatientAgent) -> int:
        base = SEVERITY_TO_PRIORITY[patient.triage_assigned or patient.triage_observed]
        if base <= 2 or patient.queue_entry_minute is None:
            return base
        improvement = (self.clock_minute - patient.queue_entry_minute) // self.config.aging_interval_min
        return max(self.config.aging_floor_priority, base - int(improvement))

    def _assign_attention(self) -> None:
        available = min(self.config.doctor_capacity - len(self._doctors_busy_until), self.config.box_capacity - len(self._boxes_busy_until))
        ordered = sorted(self.queues["attention"], key=lambda p: (self._effective_priority(p), SEVERITY_TO_PRIORITY[p.triage_assigned or p.triage_observed], p.queue_entry_minute or 0, p.arrival_minute, p.patient_id))
        for patient in ordered[:max(0, available)]:
            self.queues["attention"].remove(patient)
            category = patient.triage_assigned or patient.triage_observed
            duration = self._sample_minutes(patient, "attention", category)
            doctor_minutes = min(duration, max(1, ceil(duration * self.config.doctor_occupancy_fraction)))
            box_minutes = min(duration, max(1, ceil(duration * self.config.box_occupancy_fraction)) + self.config.box_release_delay_min)
            patient.attention_start = self.clock_minute
            patient.doctor_release = self.clock_minute + doctor_minutes
            patient.box_release = self.clock_minute + box_minutes
            patient.state = "attention"
            self._doctors_busy_until.append(patient.doctor_release)
            self._boxes_busy_until.append(patient.box_release)
            self._active["attention"].append((self.clock_minute + duration, patient))

    def _record_minute(self) -> None:
        registration_busy = len(self._active["registration"])
        triage_busy = len(self._active["triage"])
        doctors_busy = len(self._doctors_busy_until)
        boxes_busy = len(self._boxes_busy_until)
        self.minute_results.append({
            "minute": self.clock_minute,
            "queue_registration": len(self.queues["registration"]),
            "queue_triage": len(self.queues["triage"]),
            "queue_attention": len(self.queues["attention"]),
            "active_registration": registration_busy,
            "active_triage": triage_busy,
            "active_attention": len(self._active["attention"]),
            "active_patients": sum(p.state != "completed" for p in self.patients.values()),
            "completed_patients": len(self.patient_results),
            "doctors_busy": doctors_busy,
            "boxes_busy": boxes_busy,
            "registration_utilization": registration_busy / self.config.registration_capacity,
            "triage_utilization": triage_busy / self.config.triage_capacity,
            "doctor_utilization": doctors_busy / self.config.doctor_capacity,
            "box_utilization": boxes_busy / self.config.box_capacity,
        })

    def _check_invariants(self) -> None:
        occupied = {
            "admisión": (len(self._active["registration"]), self.config.registration_capacity),
            "triage": (len(self._active["triage"]), self.config.triage_capacity),
            "médicos": (len(self._doctors_busy_until), self.config.doctor_capacity),
            "boxes": (len(self._boxes_busy_until), self.config.box_capacity),
        }
        exceeded = {name: (used, capacity) for name, (used, capacity) in occupied.items() if used > capacity}
        if exceeded:
            raise RuntimeError(f"Se excedió la capacidad de recursos: {exceeded}")
        completed = len(self.patient_results)
        active = sum(patient.state != "completed" for patient in self.patients.values())
        if len(self.patients) != completed + active:
            raise RuntimeError("La conservación de pacientes no se cumple")

    @staticmethod
    def _patient_record(patient: PatientAgent) -> dict[str, object]:
        assert patient.departure is not None and patient.attention_start is not None
        assert patient.arrival_minute is not None and patient.triage_start is not None and patient.triage_end is not None
        assert patient.doctor_release is not None and patient.box_release is not None
        return {
            "patient_id": patient.patient_id,
            "triage_observed": patient.triage_observed,
            "triage_assigned": patient.triage_assigned,
            "arrival_minute": patient.arrival_minute,
            "registration_start": patient.registration_start,
            "registration_end": patient.registration_end,
            "triage_start": patient.triage_start,
            "triage_end": patient.triage_end,
            "attention_start": patient.attention_start,
            "doctor_release": patient.doctor_release,
            "box_release": patient.box_release,
            "departure": patient.departure,
            "registration_queue_wait_sim": patient.registration_start - patient.arrival_minute,
            "registration_time_sim": patient.registration_end - patient.registration_start,
            "triage_queue_wait_sim": patient.triage_start - patient.registration_end,
            "triage_time_sim": patient.triage_end - patient.triage_start,
            "wait_triage_sim": patient.triage_start - patient.arrival_minute,
            "wait_attention_sim": patient.attention_start - patient.triage_end,
            "door2doctor_sim": patient.attention_start - patient.arrival_minute,
            "attention_time_sim": patient.departure - patient.attention_start,
            "doctor_contact_sim": patient.doctor_release - patient.attention_start,
            "box_occupation_sim": patient.box_release - patient.attention_start,
            "los_sim": patient.departure - patient.arrival_minute,
        }

    def step(self) -> None:
        """Advance one minute in the H.1-R6 event order."""
        if not self.running:
            return
        self._complete_and_release()
        self._inject_arrivals()
        self._assign_single_resource("registration", self.config.registration_capacity)
        self._assign_single_resource("triage", self.config.triage_capacity)
        self._assign_attention()
        self._check_invariants()
        self._record_minute()
        if len(self.patient_results) == len(self.cohort):
            self.running = False
        elif self.clock_minute >= self._drain_limit:
            unfinished = [p.patient_id for p in self.patients.values() if p.state != "completed"]
            raise RuntimeError(f"Cohorte no completada dentro de drain_days={self.config.drain_days}; pendientes={len(unfinished)}")
        self.clock_minute += 1

    def run_to_completion(self) -> tuple[list[dict[str, object]], list[dict[str, int | float]]]:
        while self.running:
            self.step()
        return self.patient_results, self.minute_results

