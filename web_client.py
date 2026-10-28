"""Small, dependency-light client helpers shared by the Streamlit UI."""
from __future__ import annotations

import json
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def build_simulation_payload(
    *, start_date: date, start_hour: int, horizon_hours: int,
    master_seed: int, replication_id: int, severity_probabilities: str,
    registration_capacity: int, triage_capacity: int,
    doctor_capacity: int, box_capacity: int,
) -> dict[str, object]:
    """Build the API request from explicit UI values."""
    return {
        "start": f"{start_date.isoformat()}T{int(start_hour):02d}:00:00",
        "horizon_hours": int(horizon_hours),
        "master_seed": int(master_seed),
        "replication_id": int(replication_id),
        "severity_probabilities": severity_probabilities,
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
