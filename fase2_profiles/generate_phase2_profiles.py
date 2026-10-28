#!/usr/bin/env python3
"""Generate aggregate-only profiles for the emergency unit simulator.
Reads only event timestamps and triage category from the source workbooks.
No encounter-level output is written.
"""
from pathlib import Path
import argparse, json, calendar
import pandas as pd
import numpy as np

FILES = [
    "01-Unificado2016_01-05-2025.xlsx",
    "03-Unificado2017_19-04-2025.xlsx",
    "02-Unificado2018_19-04-2025.xlsx",
]
CATEGORIES = ["C1", "C2", "C3", "C4", "C5"]
FIELDS = ["urg_fecha_hora", "tri_fecha_hora", "urg_fecha_egreso", "niv_cod"]
ALPHA = 20.0
YEARS = [2016, 2017, 2018]


def day_type(ts):
    return "fin_de_semana" if ts.weekday() >= 5 else "dia_habil"


def read_sources(input_dir):
    frames = []
    source_summary = []
    for filename in FILES:
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Falta el archivo esperado: {path}")
        df = pd.read_excel(path, usecols=FIELDS)
        year = 2016 if filename.startswith("01-") else (2017 if filename.startswith("03-") else 2018)
        # Parse only fields needed for aggregation; never load identifiers or free text.
        for col in FIELDS[:3]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        df["year"] = year
        df["category"] = df["niv_cod"].astype("string").str.strip().str.upper()
        n_raw = len(df)
        valid_category = df["category"].isin(CATEGORIES)
        source_summary.append({
            "year": year,
            "source_file": filename,
            "rows_read": n_raw,
            "rows_with_c1_c5": int(valid_category.sum()),
            "rows_other_or_missing_category": int((~valid_category).sum()),
            "arrival_parse_missing": int(df["urg_fecha_hora"].isna().sum()),
            "triage_parse_missing": int(df["tri_fecha_hora"].isna().sum()),
            "discharge_parse_missing": int(df["urg_fecha_egreso"].isna().sum()),
        })
        df = df.loc[valid_category].copy()
        frames.append(df[["urg_fecha_hora", "tri_fecha_hora", "urg_fecha_egreso", "year", "category"]])
    return pd.concat(frames, ignore_index=True), source_summary


def make_profiles(df, output_dir):
    # Arrival-rate exposure: all calendar days in the complete source years.
    cal = pd.date_range("2016-01-01", "2018-12-31", freq="D")
    exposure = pd.DataFrame({"date": cal})
    exposure["month"] = exposure.date.dt.month
    exposure["day_type"] = exposure.date.map(day_type)
    exposure_days = exposure.groupby(["month", "day_type"]).size().rename("exposure_days").reset_index()

    # Use valid C1-C5 encounters with parseable arrival timestamp for the profile.
    arrivals = df.loc[df["urg_fecha_hora"].notna()].copy()
    arrivals["month"] = arrivals["urg_fecha_hora"].dt.month
    arrivals["day_type"] = arrivals["urg_fecha_hora"].map(day_type)
    arrivals["hour"] = arrivals["urg_fecha_hora"].dt.hour
    child = arrivals.groupby(["month", "day_type", "hour", "category"]).size().rename("count").reset_index()
    child_wide = child.pivot_table(index=["month", "day_type", "hour"], columns="category", values="count", fill_value=0)
    child_wide = child_wide.reindex(columns=CATEGORIES, fill_value=0)
    parent = arrivals.groupby(["month", "day_type", "category"]).size().unstack(fill_value=0).reindex(columns=CATEGORIES, fill_value=0)
    parent_n = parent.sum(axis=1)
    parent_p = parent.div(parent_n.replace(0, np.nan), axis=0).fillna(0)

    grid = pd.MultiIndex.from_product([range(1, 13), ["dia_habil", "fin_de_semana"], range(24)], names=["month", "day_type", "hour"]).to_frame(index=False)
    profile = grid.merge(exposure_days, on=["month", "day_type"], how="left")
    profile = profile.merge(child_wide.reset_index(), on=["month", "day_type", "hour"], how="left")
    profile[CATEGORIES] = profile[CATEGORIES].fillna(0).astype(int)
    profile["n_arrivals"] = profile[CATEGORIES].sum(axis=1)
    profile["mean_arrivals_per_comparable_day_hour"] = profile["n_arrivals"] / profile["exposure_days"]
    profile["fallback_parent_n"] = profile.apply(lambda r: int(parent_n.get((int(r.month), r.day_type), 0)), axis=1)
    for cat in CATEGORIES:
        profile[f"p_{cat}_raw"] = np.where(profile.n_arrivals > 0, profile[cat] / profile.n_arrivals, np.nan)
        def smoothed(row):
            key=(int(row.month), row.day_type)
            p=float(parent_p.loc[key, cat]) if key in parent_p.index else 0.0
            return (float(row[cat]) + ALPHA*p)/(float(row.n_arrivals)+ALPHA)
        profile[f"p_{cat}_smoothed"] = profile.apply(smoothed, axis=1)
    profile["severity_smoothing_alpha"] = ALPHA
    profile.to_csv(output_dir / "hourly_arrival_severity_profile.csv", index=False, encoding="utf-8-sig")

    # Empirical quantile functions for observed event intervals; negative/missing intervals are excluded.
    rows=[]
    qvals=sorted(set([0.0,*[x/100 for x in range(1,100)],0.995,0.999,1.0]))
    for event, end_col in [("ingreso_a_triage", "tri_fecha_hora"), ("ingreso_a_egreso_LOS", "urg_fecha_egreso")]:
        d=df[["urg_fecha_hora",end_col,"year","category"]].copy()
        d["minutes"]=(d[end_col]-d["urg_fecha_hora"]).dt.total_seconds()/60
        d=d.loc[d.minutes.notna() & (d.minutes>=0)]
        for scope_year, g in [("2016-2018",d), *[(str(y),d[d.year==y]) for y in YEARS]]:
            for cat, h in g.groupby("category"):
                if len(h)==0: continue
                quant=np.quantile(h.minutes.to_numpy(),qvals,method="linear")
                for q,v in zip(qvals,quant):
                    rows.append({"event_interval":event,"year_scope":scope_year,"category":cat,"n_valid":int(len(h)),"quantile":float(q),"minutes":float(v)})
    qdf=pd.DataFrame(rows)
    qdf.to_csv(output_dir / "observed_interval_quantiles.csv", index=False, encoding="utf-8-sig")

    # Aggregate temporal quality checks, never output event-level records.
    qsummary=[]
    for y,g in df.groupby("year"):
        arrival=g.urg_fecha_hora
        tri=g.tri_fecha_hora
        exit_=g.urg_fecha_egreso
        wt=(tri-arrival).dt.total_seconds()/60
        los=(exit_-arrival).dt.total_seconds()/60
        tri_exit=(exit_-tri).dt.total_seconds()/60
        qsummary.append({
            "year":int(y),"n_c1_c5":int(len(g)),
            "missing_arrival_after_parse":int(arrival.isna().sum()),
            "missing_triage_after_parse":int(tri.isna().sum()),
            "missing_exit_after_parse":int(exit_.isna().sum()),
            "negative_arrival_to_triage":int((wt<0).sum()),
            "negative_arrival_to_exit":int((los<0).sum()),
            "negative_triage_to_exit":int((tri_exit<0).sum()),
            "median_arrival_to_triage_min":float(wt[wt>=0].median()),
            "p95_arrival_to_triage_min":float(wt[wt>=0].quantile(.95)),
            "median_arrival_to_exit_min":float(los[los>=0].median()),
            "p95_arrival_to_exit_min":float(los[los>=0].quantile(.95)),
        })
    pd.DataFrame(qsummary).to_csv(output_dir / "temporal_quality_summary.csv", index=False, encoding="utf-8-sig")

    yearly=arrivals.groupby("year").size().rename("valid_c1_c5_arrivals").reset_index()
    yearly.to_csv(output_dir / "annual_volume_summary.csv", index=False, encoding="utf-8-sig")

    meta={
      "purpose":"Aggregate profiles for emergency-unit simulation",
      "source_years":YEARS,
      "profile_strata":{"month":"1-12","day_type":["dia_habil","fin_de_semana"],"hour":"0-23"},
      "arrival_denominator":"Number of calendar days of the same month and day_type across 2016-2018",
      "severity_policy":"Only niv_cod values C1-C5 included; other/missing categories excluded and counted in source summary",
      "severity_smoothing":{"method":"Dirichlet-style shrinkage toward month x day_type category proportions","pseudo_count_alpha":ALPHA,"raw_and_smoothed_probabilities_both_exported":True},
      "interval_policy":"Store aggregate empirical quantiles by category and year; include only parseable, nonnegative intervals",
      "intervals":{"ingreso_a_triage":"tri_fecha_hora - urg_fecha_hora; observed interval, not pure queue wait or service duration","ingreso_a_egreso_LOS":"urg_fecha_egreso - urg_fecha_hora; observed LOS interval"},
      "privacy":"No row-level records, identifiers, exact dates, birth dates, staff codes, or free text are exported",
      "known_limitations":["Door-to-Doctor cannot be calculated from these source columns","Historical staffing by shift is unavailable","Year-to-year volume shift should be reviewed before treating the pooled profile as current demand","Clinical feature missingness/range issues require a separate model-input audit"],
      "sampling_note":"For application sampling, interpolate the quantile function using a deterministic seed; report source year scope and seed."
    }
    (output_dir/"profile_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",type=Path,default=Path(__file__).resolve().parent.parent/"project_sources")
    ap.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent)
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    df, source_summary=read_sources(args.input_dir)
    make_profiles(df,args.output_dir)
    (args.output_dir/"source_aggregate_counts.json").write_text(json.dumps(source_summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"rows_profiled":len(df),"outputs":["hourly_arrival_severity_profile.csv","observed_interval_quantiles.csv","temporal_quality_summary.csv","annual_volume_summary.csv","profile_metadata.json","source_aggregate_counts.json"]},ensure_ascii=False))

if __name__=="__main__": main()
