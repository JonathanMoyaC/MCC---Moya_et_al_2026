"""Small, dependency-light client helpers shared by the Streamlit UI."""
from __future__ import annotations

import json
from datetime import date
from statistics import mean, stdev
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def build_simulation_payload(
    *, start_date: date, start_hour: int, horizon_hours: int,
    master_seed: int, replication_id: int, severity_probabilities: str,
    registration_capacity: int, triage_capacity: int,
    doctor_capacity: int, box_capacity: int, demand_multiplier: float = 1.0,
) -> dict[str, object]:
    """Build the API request from explicit UI values."""
    return {
        "start": f"{start_date.isoformat()}T{int(start_hour):02d}:00:00",
        "horizon_hours": int(horizon_hours),
        "master_seed": int(master_seed),
        "replication_id": int(replication_id),
        "severity_probabilities": severity_probabilities,
        "demand_multiplier": float(demand_multiplier),
        "registration_capacity": int(registration_capacity),
        "triage_capacity": int(triage_capacity),
        "doctor_capacity": int(doctor_capacity),
        "box_capacity": int(box_capacity),
    }


def post_simulation(api_base_url: str, payload: dict[str, object], timeout: int = 180) -> dict[str, object]:
    """Call the simulation endpoint and preserve useful API error messages."""
    url = api_base_url.rstrip("/") + "/v1/simulations"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail", str(exc))
        except (ValueError, AttributeError):
            detail = str(exc)
        raise RuntimeError(f"API respondió HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"No fue posible conectar con la API: {exc.reason}") from exc


def aggregate_replication_results(results: list[dict[str, object]]) -> dict[str, object]:
    """Summarize per-replication indicators without pooling patient rows."""
    if not results:
        raise ValueError("Se requiere al menos una réplica.")
    if any(item.get("status") != "completed" for item in results):
        raise ValueError("Todas las réplicas deben finalizar antes de resumirlas.")
    first = results[0]

    def series(path: tuple[str, ...]) -> list[float]:
        found: list[float] = []
        for item in results:
            value = item
            for key in path:
                value = value[key]
            if value is not None:
                found.append(float(value))
        return found

    def stats(values: list[float]) -> dict[str, float | None]:
        return {
            "mean": float(mean(values)) if values else None,
            "sd": float(stdev(values)) if len(values) > 1 else (0.0 if values else None),
        }

    metrics: dict[str, dict[str, float | None]] = {}
    for metric in ("wait_triage", "wait_attention", "door_to_doctor", "attention_time", "length_of_stay"):
        row = {
            field: stats(series(("indicators", metric, field)))["mean"]
            for field in ("mean", "median", "p90", "p95")
        }
        row["between_replication_sd_mean"] = stats(series(("indicators", metric, "mean")))["sd"]
        metrics[metric] = row

    categories = first["indicators"]["severity_counts"].keys()
    severity = {c: stats(series(("indicators", "severity_counts", c)))["mean"] for c in categories}
    severity_sd = {c: stats(series(("indicators", "severity_counts", c)))["sd"] for c in categories}
    queues = {
        key: {
            "mean": stats(series(("indicators", "queues", key, "mean")))["mean"],
            "max": stats(series(("indicators", "queues", key, "max")))["mean"],
        }
        for key in first["indicators"]["queues"]
    }
    utilization = {
        key: stats(series(("indicators", "resource_utilization", key)))["mean"]
        for key in first["indicators"]["resource_utilization"]
    }
    run = dict(first["run"])
    run.update({
        "replication_count": len(results),
        "replication_ids": [int(item["run"]["replication_id"]) for item in results],
    })
    arrival_stats = stats(series(("indicators", "arrivals")))
    completed_stats = stats(series(("indicators", "completed")))
    minute_stats = stats(series(("indicators", "simulation_minutes")))
    return {
        "status": "completed",
        "run": run,
        "indicators": {
            "arrivals": arrival_stats["mean"], "arrivals_sd": arrival_stats["sd"],
            "completed": completed_stats["mean"], "completed_sd": completed_stats["sd"],
            "severity_counts": severity, "severity_counts_sd": severity_sd,
            "simulation_minutes": minute_stats["mean"], "simulation_minutes_sd": minute_stats["sd"],
            **metrics, "queues": queues, "resource_utilization": utilization,
        },
        "method": {
            **first["method"],
            "replication_aggregation": "mean of per-replication summaries; quantiles are not pooled across patient rows",
        },
        "replication_results": results,
    }
