import pytest

from ueh_mesa import EmpiricalInputModel, EmergencyDepartmentModelMesa, ModelConfig
from ueh_mesa.calibration import empirical_quantile_sample, stable_uniform
from ueh_mesa.model import normalize_cohort


SAMPLES = {"registration": [1], "triage": [1], "attention": [4]}


def cohort(rows):
    return [
        {"patient_id": pid, "arrival_time": minute, "triage_observed": sev}
        for pid, minute, sev in rows
    ]


def test_patient_pipeline_finishes_and_reports_core_indicators():
    model = EmergencyDepartmentModelMesa(cohort([("p1", 0, "C3")]), SAMPLES)
    patients, minute_log = model.run_to_completion()
    row = patients[0]
    assert row["registration_start"] == 0
    assert row["triage_start"] == row["registration_end"]
    assert row["attention_start"] == row["triage_end"]
    assert row["attention_time_sim"] == 4
    assert row["los_sim"] == row["departure"] - row["arrival_minute"]
    assert minute_log[0]["active_patients"] == 1


def test_duration_sampling_is_keyed_and_independent_of_cohort_order():
    first = EmergencyDepartmentModelMesa(cohort([("a", 0, "C3"), ("b", 0, "C4")]), {"registration": [1, 4], "triage": [1], "attention": [2]})
    second = EmergencyDepartmentModelMesa(cohort([("b", 0, "C4"), ("a", 0, "C3")]), {"registration": [1, 4], "triage": [1], "attention": [2]})
    first.run_to_completion()
    second.run_to_completion()
    first_rows = {r["patient_id"]: r for r in first.patient_results}
    second_rows = {r["patient_id"]: r for r in second.patient_results}
    assert first_rows["a"]["registration_end"] - first_rows["a"]["registration_start"] == second_rows["a"]["registration_end"] - second_rows["a"]["registration_start"]


def test_required_empirical_inputs_and_known_severity_are_enforced():
    with pytest.raises(ValueError, match="Faltan distribuciones"):
        EmergencyDepartmentModelMesa(cohort([("p1", 0, "C3")]), {"registration": [1]})
    with pytest.raises(ValueError, match="triage_observed inválido"):
        EmergencyDepartmentModelMesa([{"patient_id": "p1", "arrival_time": 0, "triage_observed": "C6"}], SAMPLES)


def test_attention_reserves_doctor_and_box_until_distinct_release_times():
    cfg = ModelConfig(doctor_occupancy_fraction=0.25, box_occupancy_fraction=0.60, doctor_capacity=1, box_capacity=1)
    model = EmergencyDepartmentModelMesa(cohort([("p1", 0, "C2"), ("p2", 0, "C3")]), SAMPLES, config=cfg)
    model.run_to_completion()
    rows = {r["patient_id"]: r for r in model.patient_results}
    assert rows["p1"]["doctor_contact_sim"] == 1
    assert rows["p1"]["box_occupation_sim"] == 3
    assert rows["p2"]["attention_start"] >= rows["p1"]["attention_start"] + 3


def test_minute_order_and_resource_utilization_are_recorded():
    model = EmergencyDepartmentModelMesa(cohort([("p1", 0, "C3")]), SAMPLES)
    patients, minutes = model.run_to_completion()
    by_minute = {row["minute"]: row for row in minutes}
    assert patients[0]["registration_start"] == 0
    assert patients[0]["triage_start"] == 1
    assert patients[0]["attention_start"] == 2
    assert patients[0]["departure"] == 6
    for row in minutes:
        for key in ("registration_utilization", "triage_utilization", "doctor_utilization", "box_utilization"):
            assert 0 <= row[key] <= 1
    assert by_minute[0]["registration_utilization"] == 0.5
    assert by_minute[1]["triage_utilization"] == 1.0


def test_aging_can_raise_a_long_waiting_c5_ahead_of_a_new_c3():
    long_service = {"registration": [1], "triage": [1], "attention": [400]}
    rows = cohort([("blocker", 0, "C1"), ("long_wait_C5", 0, "C5"), ("new_C3", 390, "C3")])
    model = EmergencyDepartmentModelMesa(rows, long_service, config=ModelConfig(doctor_capacity=1, box_capacity=1))
    patients, _ = model.run_to_completion()
    by_id = {row["patient_id"]: row for row in patients}
    assert by_id["long_wait_C5"]["attention_start"] < by_id["new_C3"]["attention_start"]


def test_f3_keyed_randomness_and_discrete_empirical_quantile_match_reference():
    assert stable_uniform(42, 7, "p-123", "attention") == pytest.approx(0.8700423163387891)
    assert empirical_quantile_sample([4.6, 1.2, 3.4], 0.67) == 5


def test_registration_zero_is_instant_and_wait_triage_components_are_recorded():
    model = EmergencyDepartmentModelMesa(
        cohort([("p1", 0, "C3"), ("p2", 0, "C4")]), SAMPLES,
        config=ModelConfig(registration_capacity=1, triage_capacity=1, registration_service_minutes=0),
    )
    patients, _ = model.run_to_completion()
    by_id = {row["patient_id"]: row for row in patients}
    assert len(by_id) == 2
    assert all(row["registration_start"] == row["registration_end"] == 0 for row in by_id.values())
    assert all(row["registration_time_sim"] == 0 for row in by_id.values())
    assert all(row["registration_queue_wait_sim"] == 0 for row in by_id.values())
    assert by_id["p1"]["triage_start"] == 0
    assert by_id["p2"]["triage_start"] == 1
    assert by_id["p2"]["triage_queue_wait_sim"] == 1
    assert all(row["wait_triage_sim"] == row["triage_start"] - row["arrival_minute"] for row in by_id.values())


def test_triage_factor_scales_empirical_service_duration():
    services = {"registration": [1], "triage": [8], "attention": [4]}
    faster = EmergencyDepartmentModelMesa(
        cohort([("p1", 0, "C3")]), services,
        config=ModelConfig(triage_factor=0.75),
    )
    slower = EmergencyDepartmentModelMesa(
        cohort([("p1", 0, "C3")]), services,
        config=ModelConfig(triage_factor=1.25),
    )
    faster.run_to_completion()
    slower.run_to_completion()
    assert faster.patient_results[0]["triage_time_sim"] == 6
    assert slower.patient_results[0]["triage_time_sim"] == 10


def test_model_accepts_calibration_bundle_and_retains_provenance():
    bundle = EmpiricalInputModel(
        registration=(3, 5), triage=(1, 2), attention_global=(10, 20),
        attention_by_class={"C2": (25, 30)}, metadata={"cache_sha256": "abc"},
    )
    model = EmergencyDepartmentModelMesa(cohort([("p1", 0, "C2")]), bundle)
    model.run_to_completion()
    assert model.calibration_metadata == {"cache_sha256": "abc"}
    assert model.patient_results[0]["attention_time_sim"] in (25, 30)


def test_timestamp_arrivals_follow_f3_floor_then_minute_rounding_and_id_tie_break():
    rows = [
        {"patient_id": "z", "arrival_time": "2017-09-18T00:00:45", "triage_observed": "C3"},
        {"patient_id": "a", "arrival_time": "2017-09-18T00:01:15", "triage_observed": "C3"},
    ]
    normalized = normalize_cohort(rows)
    assert [(r["patient_id"], r["arrival_minute"]) for r in normalized] == [("a", 1), ("z", 1)]

