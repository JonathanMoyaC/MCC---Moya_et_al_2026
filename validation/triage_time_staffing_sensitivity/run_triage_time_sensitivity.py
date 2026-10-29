"""Reproduce paired H6 sensitivity results for triage capacity (1 or 2)."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from ueh_mesa import EmpiricalInputModel, EmergencyDepartmentModelMesa, ModelConfig

EXPECTED_SERVICE_SAMPLES_SHA256 = "340f27d51b7638ab105b65708a98757141eacf4c1094fea23f46cbd796ae94c5"
POLICIES = {
    "observed_priority_reference": "observed",
    "flat_multiclass_classifier": "empirica",
    "hierarchical_classifier": "jerarquica",
}
SUBGROUPS = (
    "all_patients", "misclassified_patients", "undertriaged_patients",
    "overtriaged_patients", "high_acuity_undertriaged_patients", "C2_to_C3_patients",
)
STAGE_METRICS = (
    "registration_queue_wait_min", "registration_duration_min",
    "triage_queue_wait_min", "triage_duration_min",
    "arrival_to_triage_start_min", "post_triage_wait_for_medical_care_min",
    "door_to_doctor_min", "length_of_stay_min",
)
BASE_CONFIG = {
    "registration_capacity": 2,
    "doctor_capacity": 12,
    "box_capacity": 11,
    "doctor_occupancy_fraction": 0.25,
    "box_occupancy_fraction": 0.60,
}


def load_service_inputs(path: str) -> EmpiricalInputModel:
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_SERVICE_SAMPLES_SHA256:
        raise RuntimeError(f"Adult service sample SHA-256 mismatch: {digest}")
    x = json.loads(source.read_text(encoding="utf-8"))
    # These are 17 arrival-to-triage intervals, not verified active admission
    # service durations. They are intentionally excluded from the simulation.
    registration_intervals = tuple(map(float, x["registration_minutes"]))
    if len(registration_intervals) != 17 or any(not 1 <= value <= 720 for value in registration_intervals):
        raise RuntimeError("Expected the 17 audited admission-to-triage intervals (not used as service times)")
    return EmpiricalInputModel(
        registration=(1.0,),
        triage=tuple(map(float, x["triage_minutes"])),
        attention_global=tuple(map(float, x["attention_global_minutes"])),
        attention_by_class={k: tuple(map(float, v)) for k, v in x["attention_by_class_minutes"].items()},
        metadata=x["provenance"],
    )


def load_assignment(root: str, manifest: pd.DataFrame, assignment_rep: int, raw_policy: str) -> pd.DataFrame:
    # These legacy filenames are inputs only; all generated outputs use descriptive scientific names.
    name = f"H6_input_{raw_policy}_rep_{assignment_rep:03d}.csv.gz"
    path = Path(root) / name
    row = manifest[(manifest.replica == assignment_rep) & (manifest.scenario == raw_policy)]
    if len(row) != 1 or row.iloc[0]["file"] != name:
        raise RuntimeError(f"Assignment manifest mismatch: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != row.iloc[0].sha256:
        raise RuntimeError(f"Assignment hash mismatch: {name}")
    return pd.read_csv(path, dtype={"patient_id": str, "window_id": str})


def _normalize_categories(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    return text.str.replace(r"^C?([1-5])$", r"C\1", regex=True)


def _with_2018_prefix(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    return text.where(text.str.startswith("2018-"), "2018-" + text)


def _assignment_join_id(frame: pd.DataFrame, assignment: pd.DataFrame) -> pd.Series:
    """Choose a cohort key only when it joins all evaluation rows and acuities."""
    required = {"patient_id", "window_id", "triage_observed", "triage_assigned"}
    if missing := required - set(assignment.columns):
        raise ValueError(f"Assignment file is missing columns: {sorted(missing)}")

    eval_mask = frame["evaluation_case"].astype(bool)
    eval_rows = frame.loc[eval_mask]
    if eval_rows.empty:
        raise ValueError("Cohort has no evaluation cases")

    lookup = assignment[["patient_id", "window_id", "triage_observed"]].copy()
    lookup["patient_id"] = lookup["patient_id"].astype(str).str.strip()
    lookup["window_id"] = lookup["window_id"].astype(str).str.strip()
    lookup["_assignment_observed"] = _normalize_categories(lookup["triage_observed"])
    lookup = lookup.rename(columns={"patient_id": "_assignment_id"})
    if lookup.duplicated(["_assignment_id", "window_id"]).any():
        raise ValueError("Assignment file has duplicate patient/window keys")

    candidates: list[tuple[str, pd.Series]] = []
    if "internal_folio" in frame.columns:
        candidates.append(("2018-prefixed internal_folio", _with_2018_prefix(frame["internal_folio"])))
    if "patient_id" in frame.columns:
        patient_id = frame["patient_id"].astype(str).str.strip()
        candidates.append(("patient_id direct", patient_id))
        if not patient_id.str.startswith("2018-").all():
            candidates.append(("2018-prefixed patient_id", _with_2018_prefix(patient_id)))

    diagnostics: list[str] = []
    valid: list[tuple[str, pd.Series]] = []
    seen: set[tuple[str, ...]] = set()
    for label, values in candidates:
        candidate = values.astype(str).str.strip()
        signature = tuple(candidate.loc[eval_mask].tolist())
        if signature in seen:
            continue
        seen.add(signature)

        probe = pd.DataFrame({
            "_candidate_id": candidate.loc[eval_mask].to_numpy(),
            "window_id": eval_rows["window_id"].astype(str).str.strip().to_numpy(),
            "_cohort_observed": _normalize_categories(eval_rows["triage_observed"]).to_numpy(),
        })
        joined = probe.merge(
            lookup[["_assignment_id", "window_id", "_assignment_observed"]],
            left_on=["_candidate_id", "window_id"],
            right_on=["_assignment_id", "window_id"],
            how="left",
            validate="one_to_one",
        )
        matched = int(joined["_assignment_id"].notna().sum())
        acuity_matches = int(
            (joined["_cohort_observed"] == joined["_assignment_observed"]).sum()
        )
        diagnostics.append(f"{label}: {matched}/{len(eval_rows)} IDs, {acuity_matches} acuity matches")
        if matched == len(eval_rows) and acuity_matches == len(eval_rows):
            valid.append((label, candidate))

    if not valid:
        raise RuntimeError(
            "Could not match assignment IDs to the complete evaluation cohort. "
            + "; ".join(diagnostics)
        )
    if len(valid) > 1:
        first = valid[0][1].astype(str).to_numpy()
        if any(not first.tolist() == values.astype(str).to_numpy().tolist() for _, values in valid[1:]):
            raise RuntimeError("Multiple distinct cohort ID mappings match the assignments; refusing an ambiguous join")
    return valid[0][1]


def make_rows(cohort: pd.DataFrame, assignment: pd.DataFrame | None) -> pd.DataFrame:
    frame = cohort.copy()
    frame["window_id"] = frame["window_id"].astype(str).str.strip()
    frame["triage_observed"] = _normalize_categories(frame["triage_observed"])
    frame["evaluation_case"] = frame["evaluation_case"].astype(bool)
    if assignment is None:
        frame["triage_assigned"] = frame["triage_observed"]
    else:
        frame["join_id"] = _assignment_join_id(frame, assignment)
        a = assignment[["patient_id", "window_id", "triage_observed", "triage_assigned"]].copy()
        a["patient_id"] = a["patient_id"].astype(str).str.strip()
        a["window_id"] = a["window_id"].astype(str).str.strip()
        a["triage_observed"] = _normalize_categories(a["triage_observed"])
        a["triage_assigned"] = _normalize_categories(a["triage_assigned"])
        a = a.rename(columns={
            "patient_id": "join_id",
            "triage_observed": "triage_observed_assignment",
        })
        frame = frame.merge(
            a[["join_id", "window_id", "triage_observed_assignment", "triage_assigned"]],
            on=["join_id", "window_id"],
            how="left",
            validate="one_to_one",
        )
        eval_mask = frame["evaluation_case"]
        if frame.loc[eval_mask, "triage_assigned"].isna().any():
            raise RuntimeError("Unmatched classifier assignments in evaluation cohort")
        if (frame.loc[eval_mask, "triage_observed"] != frame.loc[eval_mask, "triage_observed_assignment"]).any():
            raise RuntimeError("Observed triage mismatch after assignment join")
        frame.loc[~eval_mask, "triage_assigned"] = frame.loc[~eval_mask, "triage_observed"]
    return frame[["patient_id", "window_id", "arrival_time", "triage_observed", "triage_assigned", "evaluation_case"]]


def subgroup_mask(frame: pd.DataFrame, name: str) -> np.ndarray:
    observed = frame.triage_observed.str[1:].astype(int)
    assigned = frame.triage_assigned.str[1:].astype(int)
    return {
        "all_patients": np.ones(len(frame), dtype=bool),
        "misclassified_patients": observed != assigned,
        "undertriaged_patients": assigned > observed,
        "overtriaged_patients": assigned < observed,
        "high_acuity_undertriaged_patients": (observed <= 2) & (assigned > observed),
        "C2_to_C3_patients": (observed == 2) & (assigned == 3),
    }[name]


def add_stage_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f["registration_queue_wait_min"] = f.registration_start - f.arrival_minute
    f["registration_duration_min"] = f.registration_end - f.registration_start
    f["triage_queue_wait_min"] = f.triage_start - f.registration_end
    f["triage_duration_min"] = f.triage_end - f.triage_start
    f["arrival_to_triage_start_min"] = f.triage_start - f.arrival_minute
    f["post_triage_wait_for_medical_care_min"] = f.attention_start - f.triage_end
    f["door_to_doctor_min"] = f.attention_start - f.arrival_minute
    f["length_of_stay_min"] = f.departure - f.arrival_minute
    return f


def run_one(task: tuple) -> dict:
    (triage_capacity, triage_duration_factor, sim_rep, cohort_path, service_path,
     assignment_root, manifest_path, checkpoint_root, master_seed) = task
    factor_label = f"{triage_duration_factor:.2f}".replace(".", "p")
    cap_dir = Path(checkpoint_root) / f"triage_capacity_{triage_capacity}__duration_factor_{factor_label}"
    cap_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cap_dir / f"replicate_{sim_rep:03d}_summary.csv"
    delta_path = cap_dir / f"replicate_{sim_rep:03d}_patient_deltas.csv.gz"
    if summary_path.exists() and delta_path.exists():
        return {"triage_capacity": triage_capacity, "simulation_replication": sim_rep, "status": "reused"}

    cohort = pd.read_csv(cohort_path, dtype={"patient_id": str, "internal_folio": str, "window_id": str})
    cohort.arrival_time = pd.to_datetime(cohort.arrival_time)
    cohort.evaluation_case = cohort.evaluation_case.astype(bool)
    manifest = pd.read_csv(manifest_path)
    services = load_service_inputs(service_path)
    assignment_rep = sim_rep % 64
    assignments = {
        "observed": None,
        "empirica": load_assignment(assignment_root, manifest, assignment_rep, "empirica"),
        "jerarquica": load_assignment(assignment_root, manifest, assignment_rep, "jerarquica"),
    }
    config = dict(BASE_CONFIG, triage_capacity=int(triage_capacity), triage_factor=float(triage_duration_factor))
    patients: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for policy_name, raw_policy in POLICIES.items():
        rows = make_rows(cohort, assignments[raw_policy])
        patient_parts = []
        minute_parts = []
        for window_id, window in rows.groupby("window_id", sort=False):
            model = EmergencyDepartmentModelMesa(
                window[["patient_id", "arrival_time", "triage_observed", "triage_assigned"]].to_dict("records"),
                services,
                config=ModelConfig(master_seed=master_seed, replication_id=sim_rep, **config),
            )
            patient_records, minute_records = model.run_to_completion()
            pf = pd.DataFrame(patient_records).merge(
                window[["patient_id", "evaluation_case"]], on="patient_id", validate="one_to_one"
            )
            pf = add_stage_metrics(pf.loc[pf.evaluation_case].copy())
            pf["window_id"] = window_id
            patient_parts.append(pf)
            mf = pd.DataFrame(minute_records)
            mf["window_id"] = window_id
            minute_parts.append(mf)
        patient_frame = pd.concat(patient_parts, ignore_index=True)
        minute_frame = pd.concat(minute_parts, ignore_index=True)
        patients[policy_name] = patient_frame
        row = {
            "triage_capacity": triage_capacity,
            "triage_duration_factor": triage_duration_factor,
            "simulation_replication": sim_rep,
            "assignment_replication": assignment_rep,
            "policy": policy_name,
            "evaluation_patients": len(patient_frame),
        }
        for metric in STAGE_METRICS:
            row[f"mean_{metric}"] = float(patient_frame[metric].mean())
            row[f"median_{metric}"] = float(patient_frame[metric].median())
            row[f"p90_{metric}"] = float(patient_frame[metric].quantile(0.90))
        for queue in ("queue_registration", "queue_triage", "queue_attention"):
            row[f"mean_{queue}"] = float(minute_frame[queue].mean())
            row[f"p90_{queue}"] = float(minute_frame[queue].quantile(0.90))
        for util in ("registration_utilization", "triage_utilization", "doctor_utilization", "box_utilization"):
            row[f"mean_{util}"] = float(minute_frame[util].mean())
        summary_rows.append(row)

    ref = patients["observed_priority_reference"][
        ["patient_id", "window_id", "post_triage_wait_for_medical_care_min", "length_of_stay_min"]
    ].rename(columns={
        "post_triage_wait_for_medical_care_min": "reference_post_triage_wait_min",
        "length_of_stay_min": "reference_length_of_stay_min",
    })
    delta_parts = []
    for policy_name in ("flat_multiclass_classifier", "hierarchical_classifier"):
        d = patients[policy_name].merge(ref, on=["patient_id", "window_id"], validate="one_to_one")
        d["added_post_triage_wait_min"] = d.post_triage_wait_for_medical_care_min - d.reference_post_triage_wait_min
        d["added_length_of_stay_min"] = d.length_of_stay_min - d.reference_length_of_stay_min
        d["triage_capacity"] = triage_capacity
        d["triage_duration_factor"] = triage_duration_factor
        d["simulation_replication"] = sim_rep
        d["assignment_replication"] = assignment_rep
        d["policy"] = policy_name
        keep = ["triage_capacity", "triage_duration_factor", "simulation_replication", "assignment_replication", "policy", "patient_id",
                "window_id", "triage_observed", "triage_assigned", "added_post_triage_wait_min",
                "added_length_of_stay_min"]
        delta_parts.append(d[keep])
        for subgroup in SUBGROUPS:
            z = d.loc[subgroup_mask(d, subgroup)]
            n = len(z)
            summary_rows.append({
                "triage_capacity": triage_capacity,
                "triage_duration_factor": triage_duration_factor,
                "simulation_replication": sim_rep,
                "assignment_replication": assignment_rep,
                "policy": policy_name,
                "subgroup": subgroup,
                "subgroup_n": n,
                "mean_added_post_triage_wait_min": float(z.added_post_triage_wait_min.mean()) if n else np.nan,
                "median_added_post_triage_wait_min": float(z.added_post_triage_wait_min.median()) if n else np.nan,
                "p90_added_post_triage_wait_min": float(z.added_post_triage_wait_min.quantile(0.90)) if n else np.nan,
                "proportion_added_wait_over_30_min": float((z.added_post_triage_wait_min > 30).mean()) if n else np.nan,
                "proportion_added_wait_over_60_min": float((z.added_post_triage_wait_min > 60).mean()) if n else np.nan,
                "mean_added_length_of_stay_min": float(z.added_length_of_stay_min.mean()) if n else np.nan,
            })

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.concat(delta_parts, ignore_index=True).to_csv(delta_path, index=False, compression="gzip")
    return {"triage_capacity": triage_capacity, "simulation_replication": sim_rep, "status": "completed"}


def percentile_interval(values: pd.Series) -> tuple[float, float, float]:
    v = values.dropna().astype(float)
    return float(v.mean()), float(v.quantile(0.025)), float(v.quantile(0.975))


def aggregate(checkpoint_root: Path, output_dir: Path, scenarios: list[tuple[int, float, int]], master_seed: int) -> None:
    summary_files = []
    delta_files = []
    for capacity, factor, replications in scenarios:
        factor_label = f"{factor:.2f}".replace(".", "p")
        scenario_dir = checkpoint_root / f"triage_capacity_{capacity}__duration_factor_{factor_label}"
        summary_files.extend(scenario_dir / f"replicate_{r:03d}_summary.csv" for r in range(replications))
        delta_files.extend(scenario_dir / f"replicate_{r:03d}_patient_deltas.csv.gz" for r in range(replications))
    missing = [str(p) for p in summary_files + delta_files if not p.exists()]
    if missing:
        raise RuntimeError(f"Cannot aggregate; {len(missing)} checkpoint files are missing. First: {missing[0]}")
    summaries = pd.concat((pd.read_csv(p) for p in summary_files), ignore_index=True)
    deltas = pd.concat((pd.read_csv(p) for p in delta_files), ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(output_dir / "replication_level_results.csv", index=False)

    absolute = summaries[summaries.subgroup.isna()].copy()
    absolute_rows = []
    numeric_cols = [c for c in absolute.columns if c.startswith(("mean_", "median_", "p90_"))]
    for (cap, factor, policy), z in absolute.groupby(["triage_capacity", "triage_duration_factor", "policy"]):
        for metric in numeric_cols:
            estimate, low, high = percentile_interval(z[metric])
            absolute_rows.append({
                "triage_capacity": cap, "triage_duration_factor": factor,
                "policy": policy, "metric": metric, "estimate": estimate,
                "ci95_low": low, "ci95_high": high,
                "replications": z.simulation_replication.nunique(),
            })
    pd.DataFrame(absolute_rows).to_csv(output_dir / "absolute_process_estimands.csv", index=False)

    relative = summaries[summaries.subgroup.notna()].copy()
    relative_metrics = [
        "mean_added_post_triage_wait_min", "median_added_post_triage_wait_min",
        "p90_added_post_triage_wait_min", "proportion_added_wait_over_30_min",
        "proportion_added_wait_over_60_min", "mean_added_length_of_stay_min",
    ]
    relative_rows = []
    for (cap, factor, policy, subgroup), z in relative.groupby(
        ["triage_capacity", "triage_duration_factor", "policy", "subgroup"]
    ):
        for metric in relative_metrics:
            estimate, low, high = percentile_interval(z[metric])
            relative_rows.append({
                "triage_capacity": cap, "triage_duration_factor": factor,
                "policy": policy, "subgroup": subgroup, "metric": metric,
                "estimate": estimate, "ci95_low": low, "ci95_high": high,
                "replications": z.simulation_replication.nunique(),
                "median_subgroup_n": float(z.subgroup_n.median()),
            })
    pd.DataFrame(relative_rows).to_csv(output_dir / "paired_priority_effect_estimands.csv", index=False)

    transition_rows = []
    for keys, z in deltas.groupby(
        ["triage_capacity", "triage_duration_factor", "policy", "triage_observed", "triage_assigned"]
    ):
        cap, factor, policy, observed, assigned = keys
        by_rep = z.groupby("simulation_replication").added_post_triage_wait_min.mean()
        if by_rep.size < 30:
            continue
        estimate, low, high = percentile_interval(by_rep)
        transition_rows.append({
            "triage_capacity": cap, "triage_duration_factor": factor, "policy": policy,
            "observed_category": observed, "assigned_category": assigned,
            "estimate_added_wait_min": estimate, "ci95_low": low, "ci95_high": high,
            "replications_present": by_rep.size, "patient_replication_rows": len(z),
        })
    pd.DataFrame(transition_rows).to_csv(output_dir / "classification_transition_estimands.csv", index=False)

    manifest = {
        "experiment": "triage duration and staffing sensitivity",
        "run_stage": "final",
        "master_seed": master_seed,
        "common_random_numbers": True,
        "assignment_realizations": 64,
        "policies": list(POLICIES),
        "registration_duration_assumption_minutes": 1,
        "registration_assumption_status": "unvalidated deterministic placeholder",
        "service_samples_sha256": EXPECTED_SERVICE_SAMPLES_SHA256,
        "ci_method": "2.5th–97.5th percentiles across replication-level estimands",
        "interpretation": "Exploratory capacity and service-time sensitivity; no staffing level is declared empirically correct.",
        "scenarios": [
            {"triage_capacity": capacity, "triage_duration_factor": factor,
             "simulation_replications_per_policy": reps}
            for capacity, factor, reps in scenarios
        ],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired Mesa sensitivity scenarios for triage staffing and service duration."
    )
    parser.add_argument("--cohort", required=True, help="Private adult-only H6 cohort CSV.")
    parser.add_argument("--service-inputs", required=True, help="Approved adult_service_samples.json.")
    parser.add_argument("--assignment-root", required=True, help="Private directory with H6 assignment files.")
    parser.add_argument("--assignment-manifest", required=True, help="Private manifest for H6 assignments.")
    parser.add_argument("--checkpoint-root", required=True, help="Private local checkpoint directory; contains patient-level deltas.")
    parser.add_argument("--output-dir", required=True, help="Directory for aggregated outputs only.")
    parser.add_argument(
        "--scenario", action="append", default=None, metavar="CAPACITY,FACTOR,REPLICATIONS",
        help="Repeat per scenario, e.g. --scenario 1,0.75,32. Default reproduces the six supplied configurations.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--master-seed", type=int, default=20261027)
    args = parser.parse_args()

    scenario_specs = args.scenario or [
        "1,0.75,32", "1,1.00,128", "1,1.25,32",
        "2,0.75,32", "2,1.00,32", "2,1.25,32",
    ]
    scenarios = []
    for item in scenario_specs:
        parts = item.split(",")
        if len(parts) != 3:
            raise ValueError(f"Invalid --scenario {item!r}; use CAPACITY,FACTOR,REPLICATIONS")
        capacity, factor, reps = int(parts[0]), float(parts[1]), int(parts[2])
        if capacity not in (1, 2) or factor not in (0.75, 1.0, 1.25) or reps < 1:
            raise ValueError(f"Unsupported scenario values: {item!r}")
        scenarios.append((capacity, factor, reps))
    if len({(c, f) for c, f, _ in scenarios}) != len(scenarios):
        raise ValueError("Each capacity/factor configuration must be specified once.")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    tasks = [
        (capacity, factor, rep, args.cohort, args.service_inputs,
         args.assignment_root, args.assignment_manifest, args.checkpoint_root, args.master_seed)
        for capacity, factor, replications in scenarios for rep in range(replications)
    ]
    counts = {"completed": 0, "reused": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            counts[result["status"]] += 1
            if index % 8 == 0 or index == len(futures):
                print(
                    f"Progress {index}/{len(futures)} | completed={counts['completed']} reused={counts['reused']}",
                    flush=True,
                )
    aggregate(Path(args.checkpoint_root), Path(args.output_dir), scenarios, args.master_seed)
    print(f"Finished. Aggregated results: {args.output_dir}")


if __name__ == "__main__":
    main()

