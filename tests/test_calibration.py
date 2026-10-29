import pandas as pd

from ueh_mesa.calibration import (
    OBS_ADMISSION_MIN,
    OBS_TRIAGE_MIN,
    _build_attention_by_class,
    select_f3_calibration,
)


def test_f3_frozen_registration_and_triage_vectors_are_present():
    assert len(OBS_ADMISSION_MIN) == 75
    assert len(OBS_TRIAGE_MIN) == 79
    assert min(OBS_ADMISSION_MIN) == 1
    assert min(OBS_TRIAGE_MIN) == 1


def test_f3_temporal_quality_filter_selects_calibration_window_and_valid_attention():
    frame = pd.DataFrame({
        "patient_id": ["valid", "too_long", "bad_order", "missing", "outside"],
        "arrival_time": pd.to_datetime([
            "2017-09-18 00:10", "2017-09-19 01:00", "2017-09-19 02:00",
            "2017-09-20 01:00", "2017-09-21 00:00",
        ]),
        "triage_time": pd.to_datetime([
            "2017-09-18 00:15", "2017-09-19 01:05", "2017-09-19 01:55",
            "2017-09-20 01:05", "2017-09-21 00:05",
        ]),
        "medical_start_time": pd.to_datetime([
            "2017-09-18 00:25", "2017-09-19 01:15", "2017-09-19 02:10",
            None, "2017-09-21 00:15",
        ]),
        "departure_time": pd.to_datetime([
            "2017-09-18 01:00", "2017-09-19 14:00", "2017-09-19 03:00",
            "2017-09-20 02:00", "2017-09-21 01:00",
        ]),
        "triage_observed": ["C2", "C3", "C4", "C1", "C5"],
        "wait_triage_real": [5, 5, 5, 5, 5],
        "door2doctor_real": [15, 15, 15, 15, 15],
        "los_real": [50, 780, 60, 60, 60],
        "attention_time_real": [35, 765, 40, 40, 40],
    })
    selected, audit = select_f3_calibration(frame)
    assert selected["patient_id"].tolist() == ["valid"]
    assert audit == {
        "calibration_window_rows": 4,
        "quality_valid": 2,
        "quality_review_extreme": 0,
        "quality_exclude_missing": 1,
        "quality_exclude_order": 1,
        "eligible_attention_global": 1,
        "attention_class_nonmissing_counts": {
            "C1": 0, "C2": 1, "C3": 1, "C4": 0, "C5": 0,
        },
    }


def test_h1_threshold_is_applied_before_duration_range_cleaning():
    selected = pd.DataFrame({
        "triage_observed": ["C4"] * 29,
        "attention_time_real": [20.0] * 29,
    })
    by_class = _build_attention_by_class(selected, {"C4": 30})
    assert by_class == {"C4": (20.0,) * 29}
    assert _build_attention_by_class(selected, {"C4": 29}) == {}


def test_adult_service_inputs_override_attention_with_frozen_adult_vectors(tmp_path):
    import json
    from ueh_mesa.calibration import EmpiricalInputModel, with_adult_triage_samples

    samples_path = tmp_path / "adult_service_samples.json"
    samples_path.write_text(json.dumps({
        "schema_version": "adult-service-samples-v1",
        "population": "confirmed age >=18 years only",
        "triage_measurement_year": 2025,
        "registration_minutes": [1, 1, 1, 4, 5, 6, 6, 7, 7, 8, 10, 10, 12, 13, 20, 21, 21],
        "triage_minutes": list(range(1, 39)),
        "attention_global_minutes": list(range(1, 458)),
        "attention_by_class_minutes": {
            "C2": list(range(1, 201)),
            "C3": list(range(1, 221)),
            "C4": list(range(1, 30)),
        },
    }), encoding="utf-8")
    base = EmpiricalInputModel(
        registration=(9.0,), triage=(5.0,), attention_global=(12.0,),
        attention_by_class={"C2": (14.0,)}, metadata={"source": "F3"},
    )
    updated = with_adult_triage_samples(base, samples_path, expected_sha256=None)
    assert updated.registration == (1.0,)
    assert updated.triage == tuple(float(value) for value in range(1, 39))
    assert updated.attention_global == tuple(float(value) for value in range(1, 458))
    assert updated.attention_by_class == {
        "C2": tuple(float(value) for value in range(1, 201)),
        "C3": tuple(float(value) for value in range(1, 221)),
        "C4": tuple(float(value) for value in range(1, 30)),
    }
    assert updated.metadata["triage_service_sample_count"] == 38
    assert updated.metadata["attention_service_sample_count"] == 457
    assert updated.metadata["attention_service_class_counts"] == {"C2": 200, "C3": 220, "C4": 29}
    assert updated.metadata["registration_service_source"].startswith("fixed one-minute")
    assert updated.metadata["registration_interval_observation_count_not_used"] == 17
