import pandas as pd
import pytest

from ueh_mesa.equivalence import compare_h1_patient_outputs


def _patient_frame():
    return pd.DataFrame({
        "patient_id": ["p1", "p2"],
        "window_id": ["w1", "w1"],
        "triage_observed": ["C2", "C3"],
        "triage_assigned": [2, 3],
        "arrival_minute": [0, 1],
        "registration_start": [0, 1],
        "registration_end": [2, 3],
        "triage_start": [2, 3],
        "triage_end": [4, 5],
        "attention_start": [5, 6],
        "doctor_release": [10, 11],
        "box_release": [12, 13],
        "departure_minute": [12, 13],
        "wait_triage_sim": [0, 0],
        "wait_attention_sim": [1, 1],
        "door2doctor_sim": [5, 5],
        "attention_time_sim": [5, 5],
        "doctor_contact_sim": [5, 5],
        "box_occupation_sim": [7, 7],
        "los_sim": [12, 12],
    })


def test_patient_comparator_reports_exact_aggregate_without_ids():
    reference = _patient_frame()
    candidate = reference.rename(columns={"departure_minute": "departure"})
    report = compare_h1_patient_outputs(reference, candidate)
    assert report["exact_equivalence"] is True
    assert report["patient_window_rows"] == 2
    assert report["metrics"]["arrival_minute"]["equal_rows"] == 2
    assert "patient_id" not in str(report)
    assert "p1" not in str(report)


def test_patient_comparator_counts_differences_and_normalizes_categories():
    reference = _patient_frame()
    candidate = reference.copy()
    candidate.loc[0, "los_sim"] = 13
    candidate["triage_assigned"] = ["C2", "C3"]
    report = compare_h1_patient_outputs(reference, candidate)
    assert report["exact_equivalence"] is False
    assert report["categories"]["triage_assigned"]["equal_rows"] == 2
    assert report["metrics"]["los_sim"]["equal_rows"] == 1
    assert report["metrics"]["los_sim"]["max_absolute_error"] == 1


def test_patient_comparator_rejects_duplicate_patient_window_keys():
    reference = _patient_frame()
    candidate = pd.concat([reference, reference.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="claves paciente–ventana duplicadas"):
        compare_h1_patient_outputs(candidate, candidate)
