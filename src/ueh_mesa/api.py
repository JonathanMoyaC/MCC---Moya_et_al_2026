"""FastAPI endpoints for the Mesa prototype.

Runtime inputs are supplied through UEH_ARRIVAL_PROFILE_PATH and
UEH_CALIBRATION_CACHE_PATH. Source workbooks and the restricted calibration
cache are intentionally excluded from the distributable package.
"""
from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .calibration import EmpiricalInputModel, with_adult_triage_samples
from .demand import load_arrival_profile
from .service import SimulationRequest, SimulationService


class SimulationRunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: datetime
    horizon_hours: int = Field(ge=1, le=168)
    master_seed: int = Field(default=42, ge=0)
    replication_id: int = Field(default=0, ge=0)
    severity_probabilities: Literal["raw", "smoothed"] = "smoothed"
    registration_capacity: int = Field(default=2, ge=1, le=50)
    registration_service_minutes: int = Field(default=1, ge=0, le=2)
    triage_capacity: int = Field(default=1, ge=1, le=50)
    triage_duration_factor: float = Field(default=1.0, ge=0.75, le=1.25)
    doctor_capacity: int = Field(default=12, ge=1, le=200)
    box_capacity: int = Field(default=18, ge=1, le=300)


def _runtime_service() -> SimulationService:
    profile_path = os.getenv("UEH_ARRIVAL_PROFILE_PATH")
    cache_path = os.getenv("UEH_CALIBRATION_CACHE_PATH")
    service_samples_path = os.getenv("UEH_SERVICE_SAMPLES_PATH")
    if not profile_path or not cache_path or not service_samples_path:
        raise RuntimeError("Runtime inputs are not configured")
    profile = load_arrival_profile(profile_path)
    calibration = EmpiricalInputModel.from_f3_cache(cache_path)
    calibration = with_adult_triage_samples(calibration, service_samples_path)
    return SimulationService(profile, calibration)


def create_app(service: SimulationService | None = None):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - exercised in deployed environment
        raise RuntimeError("Instala las dependencias API con: pip install -e '.[api]'") from exc

    app = FastAPI(
        title="UEH Mesa Simulator API",
        version="0.1.0",
        description="Simulación sintética de urgencias y sus indicadores agregados.",
    )
    injected_service = service

    @lru_cache(maxsize=1)
    def get_service() -> SimulationService:
        return injected_service if injected_service is not None else _runtime_service()

    @app.get("/health")
    def health():
        profile_path = os.getenv("UEH_ARRIVAL_PROFILE_PATH")
        cache_path = os.getenv("UEH_CALIBRATION_CACHE_PATH")
        service_samples_path = os.getenv("UEH_SERVICE_SAMPLES_PATH")
        configured = injected_service is not None or bool(
            profile_path and cache_path and service_samples_path
            and os.path.isfile(profile_path) and os.path.isfile(cache_path)
            and os.path.isfile(service_samples_path)
        )
        return {"status": "ok" if configured else "needs_runtime_inputs", "configured": configured}

    @app.get("/v1/version")
    def version():
        return {"service": "ueh-mesa-prototype", "version": "0.1.0", "engine": "Mesa"}

    @app.post("/v1/simulations")
    def run_simulation(body: SimulationRunBody):
        if body.start.minute or body.start.second or body.start.microsecond:
            raise HTTPException(status_code=422, detail="start debe estar alineado al inicio de una hora")
        try:
            simulation_service = get_service()
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="El servicio requiere insumos de ejecución válidos") from exc
        try:
            return simulation_service.run(SimulationRequest(**body.model_dump()))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
