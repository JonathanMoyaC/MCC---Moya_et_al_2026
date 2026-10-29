import pandas as pd

from validation.triage_time_staffing_sensitivity.run_triage_time_sensitivity import make_rows


def cohort(patient_ids, *, evaluation=(True, True), observed=("C2", "C3")):
    return pd.DataFrame({
        "patient_id": patient_ids,
        "window_id": ["W1"] * len(patient_ids),
        "arrival_time": pd.to_datetime(["2018-01-01 00:00"] * len(patient_ids)),
        "triage_observed": list(observed),
        "evaluation_case": list(evaluation),
    })


def assignments(patient_ids):
    return pd.DataFrame({
        "patient_id": patient_ids,
        "window_id": ["W1"] * len(patient_ids),
        "triage_observed": [2, 3],
        "triage_assigned": [3, 2],
    })


def test_make_rows_joins_using_patient_id_when_cohort_has_no_internal_folio():
    rows = make_rows(cohort(["2018-001", "2018-002"]), assignments(["2018-001", "2018-002"]))
    assert rows.triage_assigned.tolist() == ["C3", "C2"]
    assert rows.patient_id.tolist() == ["2018-001", "2018-002"]


def test_make_rows_supports_2018_prefixed_patient_id_mapping():
    rows = make_rows(cohort(["001", "002"]), assignments(["2018-001", "2018-002"]))
    assert rows.triage_assigned.tolist() == ["C3", "C2"]


def test_make_rows_keeps_warmup_cases_out_of_classifier_assignment_join():
    frame = cohort(
        ["2018-001", "2018-002", "warmup"],
        evaluation=(True, True, False),
        observed=("C2", "C3", "C4"),
    )
    assignment = assignments(["2018-001", "2018-002"])
    rows = make_rows(frame, assignment)
    assert rows.triage_assigned.tolist() == ["C3", "C2", "C4"]
