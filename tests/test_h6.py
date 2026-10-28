import hashlib

import pandas as pd
import pytest

from ueh_mesa import (
    ModelConfig,
    build_observed_h6_assignments,
    prepare_h6_windows,
    read_h6_assignments,
    run_h6_mesa_scenario,
)


def small_h6_inputs():
    assignments = pd.DataFrame([
        {"replica": 3, "scenario": "empirica", "patient_id": "p1", "window_id": "W1",
         "triage_observed": 2, "triage_assigned": 1},
        {"replica": 3, "scenario": "empirica", "patient_id": "p2", "window_id": "W1",
         "triage_observed": 3, "triage_assigned": 3},
    ])
    reference = pd.DataFrame([
        {"patient_id": "p1", "arrival_time": "2018-01-01T00:00:00", "triage_observed": "C2"},
        {"patient_id": "p2", "arrival_time": "2018-01-01T00:01:00", "triage_observed": "C3"},
    ])
    return assignments, reference


def test_h6_assignments_join_to_locked_arrivals_and_keep_fixed_scenario():
    assignments, reference = small_h6_inputs()
    windows = prepare_h6_windows(
        assignments, reference, replication_id=3, scenario="empirical",
        expected_window_sizes={"W1": 2},
    )
    assert list(windows) == ["W1"]
    assert [row["patient_id"] for row in windows["W1"]] == ["p1", "p2"]
    assert [row["triage_assigned"] for row in windows["W1"]] == ["C1", "C3"]
    assert all(row["severity_source"] == "empirical" for row in windows["W1"])
    assert windows["W1"][0]["arrival_time"] == pd.Timestamp("2018-01-01 00:00:00")


def test_h6_rejects_patient_or_observed_severity_mismatch():
    assignments, reference = small_h6_inputs()
    missing = reference.iloc[:1].copy()
    with pytest.raises(ValueError, match="no aparecen"):
        prepare_h6_windows(assignments, missing, expected_window_sizes={"W1": 2})
    reference.loc[0, "triage_observed"] = "C5"
    with pytest.raises(ValueError, match="categorías observadas"):
        prepare_h6_windows(assignments, reference, expected_window_sizes={"W1": 2})


def test_h6_rejects_wrong_replica_scenario_and_duplicate_keys():
    assignments, reference = small_h6_inputs()
    with pytest.raises(ValueError, match="no coincide"):
        prepare_h6_windows(assignments, reference, replication_id=4, expected_window_sizes={"W1": 2})
    duplicate = pd.concat([assignments, assignments.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicadas"):
        prepare_h6_windows(duplicate, reference, expected_window_sizes={"W1": 3})


def test_observed_control_is_derived_from_locked_cohort():
    assignments, reference = small_h6_inputs()
    observed = build_observed_h6_assignments(assignments, reference)
    assert set(observed["scenario"]) == {"observed"}
    assert observed["triage_assigned"].tolist() == ["C2", "C3"]
    assert observed["triage_observed"].tolist() == ["C2", "C3"]


def test_h6_input_loader_checks_manifest_hash_and_row_count(tmp_path):
    path = tmp_path / "H6_input_empirica_rep_003.csv.gz"
    table = pd.DataFrame([{"replica": 3, "scenario": "empirica"}])
    table.to_csv(path, index=False, compression="gzip")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{
        "replica": 3, "scenario": "empirica", "rows": 1,
        "file": path.name, "sha256": digest,
    }]).to_csv(manifest, index=False)
    loaded = read_h6_assignments(path, manifest_path=manifest, replication_id=3, scenario="empirical")
    assert loaded.equals(table)
    pd.DataFrame([{
        "replica": 3, "scenario": "empirica", "rows": 2,
        "file": path.name, "sha256": digest,
    }]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="declara 2 filas"):
        read_h6_assignments(path, manifest_path=manifest, replication_id=3, scenario="empirical")


def test_h6_mesa_runner_keeps_scenario_assignment_and_replica():
    assignments, reference = small_h6_inputs()
    samples = {"registration": [1], "triage": [1], "attention": [2]}
    patients, minutes = run_h6_mesa_scenario(
        assignments, reference, samples,
        config=ModelConfig(master_seed=8),
        expected_window_sizes={"W1": 2},
    )
    assert len(patients) == 2
    assert set(patients["scenario"]) == {"empirical"}
    assert set(patients["replication_id"]) == {3}
    assert set(patients["window_id"]) == {"W1"}
    expected = dict(zip(assignments.patient_id, ["C1", "C3"]))
    assert patients.set_index("patient_id")["triage_assigned"].to_dict() == expected
    assert len(minutes) > 0
