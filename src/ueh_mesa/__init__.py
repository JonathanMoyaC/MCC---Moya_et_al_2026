"""Prototipo de port operacional de urgencias a Mesa."""

from .model import EmergencyDepartmentModelMesa, ModelConfig
from .calibration import EmpiricalInputModel, stable_uniform
from .equivalence import compare_h1_patient_outputs
from .demand import generate_synthetic_cohort, load_arrival_profile, validate_arrival_profile
from .model_contract import audit_model_input_contract
from .service import SimulationRequest, SimulationService
from .h6 import (
    H6_TOTAL_PATIENTS,
    H6_WINDOW_SIZES,
    build_observed_h6_assignments,
    prepare_h6_windows,
    read_h6_assignments,
    read_h6_reference_cohort,
    run_h6_mesa_scenario,
)

__all__ = [
    "EmergencyDepartmentModelMesa", "ModelConfig", "EmpiricalInputModel",
    "stable_uniform", "H6_TOTAL_PATIENTS", "H6_WINDOW_SIZES",
    "compare_h1_patient_outputs",
    "generate_synthetic_cohort", "load_arrival_profile", "validate_arrival_profile",
    "audit_model_input_contract",
    "SimulationRequest", "SimulationService",
    "build_observed_h6_assignments",
    "prepare_h6_windows", "read_h6_assignments", "read_h6_reference_cohort",
    "run_h6_mesa_scenario",
]
