from datetime import date

from ueh_mesa.web_client import aggregate_replication_results, build_simulation_payload


def test_ui_payload_matches_api_contract_and_uses_hour_aligned_start():
    payload = build_simulation_payload(
        start_date=date(2018, 1, 1), start_hour=7, horizon_hours=24,
        master_seed=42, replication_id=3, severity_probabilities="smoothed",
        registration_capacity=2, registration_service_minutes=1,
        triage_capacity=1, triage_duration_factor=1.0, doctor_capacity=12, box_capacity=18,
    )
    assert payload == {
        "start": "2018-01-01T07:00:00",
        "horizon_hours": 24,
        "master_seed": 42,
        "replication_id": 3,
        "severity_probabilities": "smoothed",
        "registration_capacity": 2,
        "registration_service_minutes": 1,
        "triage_capacity": 1,
        "triage_duration_factor": 1.0,
        "doctor_capacity": 12,
        "box_capacity": 18,
    }


def test_replication_summary_reports_mean_and_between_replication_spread():
    def result(replication_id, arrivals, mean_wait, p95_wait):
        return {
            "status": "completed",
            "run": {"replication_id": replication_id, "start": "2018-01-01T00:00:00"},
            "indicators": {
                "arrivals": arrivals,
                "completed": arrivals,
                "severity_counts": {"C1": 1, "C2": 0, "C3": arrivals - 1, "C4": 0, "C5": 0},
                "simulation_minutes": 20,
                "registration_queue_wait": {"mean": 0, "median": 0, "p90": 0, "p95": 0},
                "registration_duration": {"mean": 1, "median": 1, "p90": 1, "p95": 1},
                "triage_queue_wait": {"mean": mean_wait, "median": mean_wait, "p90": p95_wait - 1, "p95": p95_wait},
                "triage_duration": {"mean": 2, "median": 2, "p90": 3, "p95": 4},
                "wait_triage": {"mean": mean_wait + 1, "median": mean_wait + 1, "p90": p95_wait, "p95": p95_wait + 1},
                "wait_attention": {"mean": 4, "median": 4, "p90": 5, "p95": 6},
                "door_to_doctor": {"mean": 5, "median": 5, "p90": 6, "p95": 7},
                "attention_time": {"mean": 8, "median": 8, "p90": 9, "p95": 10},
                "length_of_stay": {"mean": 13, "median": 13, "p90": 14, "p95": 15},
                "queues": {"queue_registration": {"mean": 1, "max": arrivals}},
                "resource_utilization": {"doctor": 0.5},
            },
            "method": {"engine": "Mesa"},
        }

    summary = aggregate_replication_results([result(4, 3, 2, 4), result(5, 5, 6, 9)])
    assert summary["run"]["replication_ids"] == [4, 5]
    assert summary["indicators"]["arrivals"] == 4
    assert summary["indicators"]["arrivals_sd"] > 0
    assert summary["indicators"]["wait_triage"]["mean"] == 5
    assert summary["indicators"]["triage_queue_wait"]["mean"] == 4
    assert summary["indicators"]["wait_triage"]["p95"] == 7.5
    assert "not pooled" in summary["method"]["replication_aggregation"]
