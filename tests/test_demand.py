from datetime import datetime

import pandas as pd
import pytest

from ueh_mesa.demand import CATEGORIES, DAY_TYPES, generate_synthetic_cohort, validate_arrival_profile


def _profile():
    rows = []
    for month in range(1, 13):
        for day_type in DAY_TYPES:
            for hour in range(24):
                row = {
                    "month": month,
                    "day_type": day_type,
                    "hour": hour,
                    "mean_arrivals_per_comparable_day_hour": 0.0,
                }
                for category in CATEGORIES:
                    row[f"p_{category}_raw"] = 0.2
                    row[f"p_{category}_smoothed"] = 0.2
                rows.append(row)
    frame = pd.DataFrame(rows)
    target = frame.index[(frame.month == 1) & (frame.day_type == "dia_habil") & (frame.hour == 0)][0]
    frame.loc[target, "mean_arrivals_per_comparable_day_hour"] = 40
    return frame


def test_generation_is_reproducible_and_returns_only_synthetic_cohort_fields():
    profile = _profile()
    start = datetime(2018, 1, 1, 0, 0)  # Monday
    first, meta1 = generate_synthetic_cohort(profile, start, 1, master_seed=9, replication_id=2)
    second, meta2 = generate_synthetic_cohort(profile, start, 1, master_seed=9, replication_id=2)
    pd.testing.assert_frame_equal(first, second)
    assert meta1 == meta2
    assert len(first) > 0
    assert first.columns.tolist() == ["patient_id", "arrival_time", "triage_observed"]
    assert first.patient_id.is_unique
    assert set(first.triage_observed).issubset(CATEGORIES)
    assert first.arrival_time.map(lambda ts: ts.minute in range(60)).all()
    assert not any("folio" in column.lower() for column in first.columns)
    assert meta1["arrival_count_model"].startswith("Poisson")


def test_generation_rejects_unaligned_start_and_invalid_profile_probabilities():
    with pytest.raises(ValueError, match="alineado"):
        generate_synthetic_cohort(_profile(), datetime(2018, 1, 1, 0, 30), 1)
    broken = _profile()
    broken.loc[0, "p_C1_smoothed"] = 0.9
    with pytest.raises(ValueError, match="sumar 1"):
        validate_arrival_profile(broken)


def test_demand_multiplier_scales_hourly_intensity_and_is_recorded():
    profile = _profile()
    one, one_meta = generate_synthetic_cohort(
        profile, datetime(2018, 1, 1), 1, master_seed=9, replication_id=2,
        demand_multiplier=1.0,
    )
    increased, increased_meta = generate_synthetic_cohort(
        profile, datetime(2018, 1, 1), 1, master_seed=9, replication_id=2,
        demand_multiplier=1.5,
    )
    assert one_meta["hourly_arrivals"][0]["expected_arrivals"] == 40
    assert increased_meta["hourly_arrivals"][0]["expected_arrivals"] == 60
    assert increased_meta["demand_multiplier"] == 1.5
    assert len(one) >= 0 and len(increased) >= 0


def test_demand_multiplier_must_be_in_supported_range():
    with pytest.raises(ValueError, match="entre 0.1 y 3.0"):
        generate_synthetic_cohort(_profile(), datetime(2018, 1, 1), 1, demand_multiplier=0)
