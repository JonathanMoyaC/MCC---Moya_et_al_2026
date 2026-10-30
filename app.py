from __future__ import annotations

import json
import os
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ueh_mesa.web_client import aggregate_replication_results, build_simulation_payload, post_simulation


st.set_page_config(
    page_title="Simulador de Unidad de Emergencia Hospitalaria HRT",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#0B2C66"
BLUE = "#0866D9"
CYAN = "#20BFC8"
PALE = "#F3F7FC"
ESI_COLORS = {"C1": "#EF4444", "C2": "#F28C28", "C3": "#FACC15", "C4": "#79C96B", "C5": "#078A58"}
ESI = ["C1", "C2", "C3", "C4", "C5"]

st.markdown(
    f"""<style>
    :root {{ --ueh-navy:{NAVY}; --ueh-blue:{BLUE}; --ueh-cyan:{CYAN}; }}
    .stApp {{ background: {PALE}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{ background:#fff; border-right:1px solid #D8E2F0; }}
    .brandbar {{ display:flex; align-items:center; gap:16px; background:#073B83;
      color:white; padding:12px 20px; border-radius:12px; margin:0 0 8px 0; }}
    .brand-institution {{ font-weight:800; font-size:24px; color:#14C4F4; letter-spacing:.4px; }}
    .brand-divider {{ height:30px; border-left:1px solid #B6CEE9; }}
    .brand-title {{ font-size:20px; font-weight:700; line-height:1.2; flex:1; }}
    .brand-school {{ font-size:11px; text-transform:uppercase; letter-spacing:.6px;
      font-weight:700; line-height:1.25; text-align:right; }}
    .brand-sub {{ color:#1E3A68; font-size:13px; font-weight:650; padding:4px 2px 12px; }}
    div[role="radiogroup"] {{ gap:12px; }}
    div[role="radiogroup"] label {{ background:#fff; border:1px solid #DAE4F0;
      border-radius:10px; padding:8px 18px; color:{NAVY}; font-weight:600; }}
    div[role="radiogroup"] label:has(input:checked) {{ border-color:{BLUE}; color:{BLUE}; }}
    div[data-testid="stMetric"] {{ background:white; border:1px solid #DFE7F1;
      padding:14px 16px; border-radius:12px; box-shadow:0 1px 2px #0b2c660d; }}
    div[data-testid="stMetricLabel"] {{ color:{NAVY}; }}
    div[data-testid="stMetricValue"] {{ color:{BLUE}; }}
    .section-card {{ background:white; border:1px solid #DFE7F1; border-radius:12px;
      padding:14px 16px; }}
    .note {{ color:#536780; font-size:13px; }}
    </style>""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="brandbar">
      <div class="brand-institution">UCM</div>
      <div class="brand-divider"></div>
      <div class="brand-title">Simulador de Unidad de Emergencia Hospitalaria HRT</div>
      <div class="brand-divider"></div>
      <div class="brand-school">Magíster en Ciencias<br>de la Computación</div>
    </div>
    <div class="brand-sub">Unidad de Emergencia Hospitalaria · análisis operacional con simulación Mesa</div>""",
    unsafe_allow_html=True,
)

view = st.radio(
    "Vista del análisis",
    ["Retrospectivo", "Contrafactual", "Predictivo"],
    horizontal=True,
    label_visibility="collapsed",
    key="analysis_view",
)

api_base_url = os.getenv("UEH_API_BASE_URL", "http://127.0.0.1:8000")

scenario_choices = {}
selected_scenarios: list[str] = []
if view == "Contrafactual":
    st.markdown("### Escenarios contrafactuales")
    st.caption("Compara una línea base con ajustes simulados. Son escenarios bajo supuestos, no efectos causales observados.")
    demand_change = st.slider("Variación de demanda en escenarios (%)", 5, 50, 10, 5, key="scenario_demand_change")
    resource_cols = st.columns(2)
    with resource_cols[0]:
        doctor_change = st.number_input("Cambio de médicos (+/−)", min_value=1, max_value=5, value=1, step=1, key="scenario_doctor_change")
    with resource_cols[1]:
        box_change = st.number_input("Cambio de box (+/−)", min_value=1, max_value=5, value=1, step=1, key="scenario_box_change")
    scenario_choices = {
        f"Demanda +{demand_change}%": (1 + demand_change / 100, 0, 0),
        f"Demanda −{demand_change}%": (1 - demand_change / 100, 0, 0),
        f"+{doctor_change} médico(s)": (1.0, int(doctor_change), 0),
        f"−{box_change} box(es)": (1.0, 0, -int(box_change)),
    }
    selected_scenarios = st.multiselect(
        "Selecciona los escenarios a comparar",
        list(scenario_choices),
        default=list(scenario_choices),
        help="Cada escenario reutiliza las mismas semillas y réplicas que la línea base.",
    )

with st.sidebar:
    st.header("Parámetros de simulación")
    st.caption("Fuente disponible: perfil histórico agregado 2016–2018.")
    with st.form("simulation_config", clear_on_submit=False):
        if view == "Predictivo":
            today = date.today()
            default_period = (today, today + timedelta(days=2))
        else:
            default_period = (date(2018, 1, 1), date(2018, 1, 3))
        period = st.date_input(
            "Período (fechas inclusivas)",
            value=default_period,
            min_value=date(2016, 1, 1),
            max_value=date(2035, 12, 31),
            key=f"period_{view}",
        )
        st.selectbox(
            "Demanda",
            ["Perfil histórico agrupado (2016–2018)"],
            help="El motor genera llegadas sintéticas desde tasas agregadas por mes, tipo de día y hora.",
        )
        demand_scale = st.slider(
            "Multiplicador de demanda",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.05,
            help="Factor aplicado a la tasa horaria histórica agrupada. 1,00 conserva la tasa base.",
        )
        st.subheader("Capacidad")
        registration_capacity = st.number_input("Admisión", min_value=1, max_value=50, value=2, step=1)
        triage_capacity = st.number_input("Enfermería / puestos de triage", min_value=1, max_value=50, value=1, step=1)
        doctor_capacity = st.number_input("Médicos", min_value=1, max_value=200, value=12, step=1)
        box_capacity = st.number_input("Box", min_value=1, max_value=300, value=18, step=1)
        st.subheader("Reproducibilidad")
        master_seed = st.number_input("Semilla maestra", min_value=0, max_value=2_147_483_647, value=20260730, step=1)
        replication_id = st.number_input("Primera réplica", min_value=0, max_value=2_147_483_637, value=0, step=1)
        replication_count = st.number_input("Número de réplicas", min_value=1, max_value=10, value=3, step=1)
        severity_probabilities = st.selectbox(
            "Perfil de severidad",
            options=["smoothed", "raw"],
            format_func=lambda value: "Suavizado" if value == "smoothed" else "Bruto",
        )
        submitted = st.form_submit_button("Ejecutar simulación", type="primary", use_container_width=True)
    st.caption(f"API: `{api_base_url}` · Motor: Mesa")

def period_to_run(value) -> tuple[date, int] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    start_date, end_date = value
    if start_date > end_date:
        return None
    hours = (end_date - start_date).days * 24 + 24
    if hours > 168:
        return None
    return start_date, hours


def run_replicates(payload: dict[str, object], count: int, first_replication: int, progress, progress_offset: int, progress_total: int):
    rows = []
    for index in range(count):
        request = {**payload, "replication_id": int(first_replication) + index}
        rows.append(post_simulation(api_base_url, request))
        progress_offset += 1
        progress.progress(min(progress_offset / max(progress_total, 1), 1.0), text=f"Réplica {index + 1}/{count}")
    return aggregate_replication_results(rows), progress_offset


def fmt(value, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "No disponible"
    return f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def category_chart(result: dict, reference: pd.DataFrame | None = None, title: str = "Pacientes por categoría ESI"):
    sim_counts = result["indicators"].get("severity_counts", {})
    rows = [{"Categoría": c, "Pacientes": float(sim_counts.get(c, 0) or 0), "Serie": "Simulado"} for c in ESI]
    if reference is not None:
        observed = reference.loc[reference.metric.eq("severity_count")].set_index("category")["value"].to_dict()
        rows.extend({"Categoría": c, "Pacientes": float(observed[c]), "Serie": "Observado"} for c in ESI if c in observed)
    frame = pd.DataFrame(rows)
    bars = alt.Chart(frame).mark_bar().encode(
        x=alt.X("Categoría:N", title=None, sort=ESI, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Serie:N", sort=["Observado", "Simulado"]),
        y=alt.Y("Pacientes:Q", title="Cantidad de pacientes", scale=alt.Scale(zero=True)),
        color=alt.Color(
            "Categoría:N",
            scale=alt.Scale(domain=ESI, range=[ESI_COLORS[c] for c in ESI]),
            legend=alt.Legend(title="Categoría ESI"),
        ),
        opacity=alt.Opacity("Serie:N", scale=alt.Scale(domain=["Observado", "Simulado"], range=[1.0, 0.55]),
                            legend=alt.Legend(title="Serie")),
        tooltip=["Categoría", "Serie", alt.Tooltip("Pacientes:Q", format=".1f")],
    ).properties(title=title, height=310)
    labels = alt.Chart(frame).mark_text(dy=-5, fontSize=11, color=NAVY).encode(
        x=alt.X("Categoría:N", sort=ESI),
        xOffset=alt.XOffset("Serie:N", sort=["Observado", "Simulado"]),
        y=alt.Y("Pacientes:Q"),
        text=alt.Text("Pacientes:Q", format=".1f"),
    )
    st.altair_chart(bars + labels, use_container_width=True)
    st.caption("Colores por gravedad: C1 rojo · C2 naranjo · C3 amarillo · C4 verde claro · C5 verde oscuro.")


def utilization_panel(result: dict):
    values = result["indicators"].get("resource_utilization", {})
    labels = {"doctor": "Médicos", "triage": "Enfermería / triage", "box": "Box", "registration": "Admisión"}
    rows = [{"Recurso": labels.get(key, key), "Utilización (%)": float(value) * 100}
            for key, value in values.items() if key in labels and value is not None]
    if not rows:
        st.info("No hay indicadores de utilización para este resultado.")
        return
    chart = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=5, color=CYAN).encode(
        x=alt.X("Utilización (%):Q", title="Porcentaje del tiempo simulado", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Recurso:N", title=None, sort=None),
        tooltip=["Recurso", alt.Tooltip("Utilización (%):Q", format=".1f")],
    ).properties(height=155)
    st.altair_chart(chart, use_container_width=True)


def aggregate_daily_arrivals(result: dict) -> pd.DataFrame:
    raw = []
    for rep in result.get("replication_results", []):
        rep_id = int(rep["run"]["replication_id"])
        for point in rep.get("demand", {}).get("hourly_arrivals", []):
            raw.append({"Réplica": rep_id, "Fecha": point["time"][:10], "Llegadas": int(point["arrivals"])})
    if not raw:
        return pd.DataFrame()
    hourly = pd.DataFrame(raw)
    daily = hourly.groupby(["Réplica", "Fecha"], as_index=False)["Llegadas"].sum()
    return daily


def predictive_demand_chart(result: dict):
    daily = aggregate_daily_arrivals(result)
    if daily.empty:
        st.info("El backend no devolvió la serie de demanda para este resultado.")
        return
    summary = daily.groupby("Fecha")["Llegadas"].agg(
        mediana="median",
        p10=lambda s: s.quantile(0.10),
        p90=lambda s: s.quantile(0.90),
        p025=lambda s: s.quantile(0.025),
        p975=lambda s: s.quantile(0.975),
    ).reset_index()
    st.markdown("#### Proyección de demanda (llegadas diarias)")
    if daily["Réplica"].nunique() < 2:
        st.caption("Una réplica no permite estimar intervalos; ejecuta varias réplicas para ver bandas de incertidumbre.")
        line = alt.Chart(summary).mark_line(point=True, color=BLUE).encode(
            x=alt.X("Fecha:T", title=None), y=alt.Y("mediana:Q", title="Llegadas simuladas"),
            tooltip=["Fecha", alt.Tooltip("mediana:Q", title="Llegadas")],
        ).properties(height=300)
        st.altair_chart(line, use_container_width=True)
        return
    base = alt.Chart(summary).encode(x=alt.X("Fecha:T", title=None))
    outer = base.mark_area(color="#B7D2FA", opacity=0.45).encode(
        y=alt.Y("p025:Q", title="Llegadas simuladas"), y2="p975:Q",
        tooltip=["Fecha", alt.Tooltip("p025:Q", title="P2,5"), alt.Tooltip("p975:Q", title="P97,5")],
    )
    inner = base.mark_area(color="#5C9CF0", opacity=0.42).encode(y="p10:Q", y2="p90:Q")
    median = base.mark_line(point=True, color=BLUE, strokeWidth=2.5).encode(
        y=alt.Y("mediana:Q", title="Llegadas simuladas"),
        tooltip=["Fecha", alt.Tooltip("mediana:Q", title="Mediana")],
    )
    st.altair_chart((outer + inner + median).properties(height=310), use_container_width=True)
    st.caption("Bandas P2,5–P97,5 y P10–P90 entre réplicas de simulación; no son intervalos de un modelo estadístico validado.")


def scenario_chart(results: dict[str, dict], metric: str, label: str, divisor: float = 1.0):
    rows = []
    for scenario, result in results.items():
        reps = result.get("replication_results", [])
        vals = [float(rep["indicators"][metric]["mean"]) / divisor for rep in reps
                if rep["indicators"].get(metric, {}).get("mean") is not None]
        rows.extend({"Escenario": scenario, "Valor": v} for v in vals)
    if not rows:
        st.info(f"{label}: sin resultados disponibles.")
        return
    frame = pd.DataFrame(rows)
    points = alt.Chart(frame).mark_boxplot(extent="min-max", size=38).encode(
        x=alt.X("Escenario:N", title=None, sort=list(results), axis=alt.Axis(labelAngle=-20)),
        y=alt.Y("Valor:Q", title=label),
        color=alt.Color("Escenario:N", legend=None, scale=alt.Scale(scheme="blues")),
        tooltip=["Escenario", alt.Tooltip("Valor:Q", format=".2f")],
    ).properties(height=270)
    st.altair_chart(points, use_container_width=True)


def _template_bytes() -> bytes:
    frame = pd.DataFrame(columns=["period_start", "period_end", "metric", "category", "value", "unit", "definition", "source"])
    return frame.to_csv(index=False).encode("utf-8")


def parse_reference(uploaded, start_date: date, end_date: date) -> pd.DataFrame | None:
    if uploaded is None:
        return None
    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"No se pudo leer el CSV de referencia: {exc}")
        return None
    required = {"period_start", "period_end", "metric", "category", "value", "unit", "definition", "source"}
    missing = required - set(frame.columns)
    if missing:
        st.error(f"Faltan columnas del formato agregado: {', '.join(sorted(missing))}.")
        return None
    if frame.empty:
        st.error("El CSV no contiene indicadores agregados.")
        return None
    frame["period_start"] = pd.to_datetime(frame["period_start"], errors="coerce").dt.date
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce").dt.date
    if frame["period_start"].isna().any() or frame["period_end"].isna().any():
        st.error("Las fechas del CSV deben usar formato YYYY-MM-DD.")
        return None
    if not (frame["period_start"].eq(start_date).all() and frame["period_end"].eq(end_date).all()):
        st.error("El período del archivo no coincide con el período seleccionado en la barra lateral.")
        return None
    valid_metrics = {"length_of_stay", "wait_triage", "door_to_doctor", "abandonment", "utilization", "severity_count"}
    frame["metric"] = frame["metric"].astype(str).str.strip()
    if not frame["metric"].isin(valid_metrics).all():
        st.error(f"Métricas permitidas: {', '.join(sorted(valid_metrics))}.")
        return None
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if frame["value"].isna().any() or not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        st.error("La columna value debe contener valores numéricos finitos.")
        return None
    if frame[["definition", "source"]].isna().any().any():
        st.error("Cada indicador debe documentar su definición y fuente.")
        return None
    frame["category"] = frame["category"].fillna("").astype(str).str.strip()
    for metric, allowed in (("severity_count", set(ESI)), ("utilization", {"doctor", "triage", "box", "registration"})):
        values = frame.loc[frame.metric.eq(metric), "category"]
        if any(value not in allowed for value in values):
            st.error(f"Categoría inválida para {metric}: use {', '.join(sorted(allowed))}.")
            return None
    if frame.duplicated(["metric", "category"]).any():
        st.error("No debe haber filas duplicadas para la misma métrica y categoría.")
        return None
    return frame


def sim_value(result: dict, metric: str, category: str = ""):
    indicators = result["indicators"]
    if metric in {"length_of_stay", "wait_triage", "door_to_doctor"}:
        key = {"length_of_stay": "length_of_stay", "wait_triage": "wait_triage", "door_to_doctor": "door_to_doctor"}[metric]
        return indicators[key].get("mean")
    if metric == "utilization":
        return indicators.get("resource_utilization", {}).get(category)
    if metric == "severity_count":
        return indicators.get("severity_counts", {}).get(category)
    return None


def render_kpis(result: dict, reference: pd.DataFrame | None = None):
    indicators = result["indicators"]
    reps = result.get("replication_results", [])
    definitions = {
        "LOS": ("length_of_stay", "min", "Llegada a salida del simulador. El valor mostrado es la media entre réplicas."),
        "Door-to-doctor": ("door_to_doctor", "min", "Llegada al inicio de atención médica simulado. No está observado en la base HRT disponible."),
        "Espera triage": ("wait_triage", "min", "Llegada al inicio del triage; no es la duración del triage."),
    }
    columns = st.columns(5)
    for column, (label, (metric, unit, help_text)) in zip(columns[:3], definitions.items()):
        value = indicators[metric].get("mean")
        ref = None
        if reference is not None and metric in set(reference.metric):
            row = reference.loc[reference.metric.eq(metric)].iloc[0]
            ref = float(row.value)
            if metric == "length_of_stay" and row.unit == "hours":
                ref *= 60
        delta = None if ref is None or value is None else f"Referencia: {fmt(ref)} {unit}"
        column.metric(label, f"{fmt(value)} {unit}" if value is not None else "No disponible", delta=delta, help=help_text)
    columns[3].metric("% abandono", "No disponible", help="El motor Mesa actual no modela abandono; no se calcula ni se imputa.")
    columns[4].metric("Réplicas", str(result["run"]["replication_count"]), help="Se muestran promedios entre réplicas; el detalle queda en la descarga JSON.")
    if reference is not None:
        st.caption("Los valores de referencia provienen del CSV agregado cargado para el período; revisa las definiciones declaradas.")


def render_comparison_table(result: dict, reference: pd.DataFrame):
    rows = []
    chart_rows = []
    for _, row in reference.iterrows():
        metric, category, unit = row.metric, row.category, row.unit
        simulated = sim_value(result, metric, category)
        observed = float(row.value)
        if metric == "utilization":
            if unit == "%":
                observed /= 100
            elif unit != "proportion":
                st.error(f"Utilización requiere unidad % o proportion (fila {category}).")
                continue
            simulated_display, observed_display = simulated * 100 if simulated is not None else None, observed * 100
            display_unit = "%"
        elif metric == "length_of_stay":
            if unit not in {"minutes", "hours"}:
                st.error("LOS requiere unit=minutes u hours.")
                continue
            observed = observed * 60 if unit == "hours" else observed
            simulated_display, observed_display, display_unit = simulated, observed, "min"
        elif metric in {"wait_triage", "door_to_doctor"}:
            if unit != "minutes":
                st.error(f"{metric} requiere unit=minutes.")
                continue
            simulated_display, observed_display, display_unit = simulated, observed, "min"
        elif metric == "severity_count":
            if unit != "patients":
                st.error("severity_count requiere unit=patients.")
                continue
            simulated_display, observed_display, display_unit = simulated, observed, "pacientes"
        else:
            st.info("El abandono observado se recibió, pero el simulador actual no calcula ese resultado; no se compara.")
            continue
        if simulated_display is None:
            delta, similarity = None, None
        else:
            delta = float(simulated_display) - float(observed_display)
            similarity = None if abs(float(observed_display)) < 1e-9 else max(0.0, 100.0 * (1.0 - abs(delta) / abs(float(observed_display))))
        label = f"{metric} {category}".strip()
        if simulated_display is not None:
            chart_rows.extend([
                {"Indicador": label, "Serie": "Observado", "Valor": float(observed_display)},
                {"Indicador": label, "Serie": "Simulado", "Valor": float(simulated_display)},
            ])
        rows.append({
            "Indicador": label,
            "Observado": f"{fmt(observed_display)} {display_unit}",
            "Simulado": "No disponible" if simulated_display is None else f"{fmt(simulated_display)} {display_unit}",
            "Simulado − observado": "No calculable" if delta is None else f"{fmt(delta)} {display_unit}",
            "Concordancia relativa": "No calculable" if similarity is None else f"{fmt(similarity)}%",
            "Definición / fuente": f"{row.definition} · {row.source}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Concordancia relativa = max(0, 100 × [1 − |simulado − observado| / |observado|]). No es probabilidad ni similitud de distribuciones; no se calcula si el observado es cero.")
    if chart_rows:
        frame = pd.DataFrame(chart_rows)
        bars = alt.Chart(frame).mark_bar().encode(
            x=alt.X("Valor:Q", title="Valor agregado (escala independiente por indicador)"),
            y=alt.Y("Serie:N", title=None, sort=["Observado", "Simulado"]),
            color=alt.Color("Serie:N", scale=alt.Scale(domain=["Observado", "Simulado"], range=[NAVY, CYAN]), legend=None),
            tooltip=["Indicador", "Serie", alt.Tooltip("Valor:Q", format=".2f")],
        ).properties(height=85)
        st.altair_chart(
            bars.facet(column=alt.Column("Indicador:N", title=None, header=alt.Header(labelAngle=-25))).resolve_scale(x="independent"),
            use_container_width=True,
        )


def show_run_metadata(result: dict):
    run = result["run"]
    st.caption(
        f"Período simulado: {run['start']} · {run['horizon_hours']} h · "
        f"Semilla: {run['master_seed']} · Réplicas: {run['replication_count']} "
        f"({run['replication_ids'][0]}–{run['replication_ids'][-1]}) · "
        f"Capacidades: admisión {run['capacities']['registration']}, triage {run['capacities']['triage']}, "
        f"médicos {run['capacities']['doctors']}, box {run['capacities']['boxes']}"
    )


def display_result(result: dict, reference: pd.DataFrame | None = None, predictive: bool = False):
    run = result["run"]
    indicators = result["indicators"]
    st.markdown("#### Indicadores del escenario")
    render_kpis(result, reference)
    show_run_metadata(result)

    left, right = st.columns([1.5, 1])
    with left:
        if predictive:
            predictive_demand_chart(result)
        else:
            st.markdown("#### Pacientes por categoría de triage")
            category_chart(result, reference)
    with right:
        st.markdown("#### Utilización de recursos")
        utilization_panel(result)
    if predictive:
        st.markdown("#### Pacientes por categoría ESI (proyección)")
        category_chart(result, title="Pacientes por categoría ESI · proyección de simulación")

    st.markdown("#### Indicadores de tiempo (minutos)")
    metric_names = {
        "length_of_stay": "LOS",
        "door_to_doctor": "Door-to-doctor simulado",
        "wait_triage": "Espera hasta inicio del triage",
        "wait_attention": "Espera después del triage hasta atención",
        "attention_time": "Duración de atención simulada",
    }
    timing_rows = []
    for key, label in metric_names.items():
        vals = [rep["indicators"][key]["mean"] for rep in result.get("replication_results", [])
                if rep["indicators"][key].get("mean") is not None]
        if not vals:
            timing_rows.append({"Indicador": label, "Media": "No disponible", "DE entre réplicas": "—", "P10–P90 entre réplicas": "—", "P2,5–P97,5 entre réplicas": "—"})
        else:
            timing_rows.append({
                "Indicador": label,
                "Media": f"{fmt(np.mean(vals))} min",
                "DE entre réplicas": f"{fmt(np.std(vals, ddof=1))} min" if len(vals) > 1 else "No estimable (1 réplica)",
                "P10–P90 entre réplicas": f"{fmt(np.quantile(vals, .1))}–{fmt(np.quantile(vals, .9))} min" if len(vals) > 1 else "No estimable (1 réplica)",
                "P2,5–P97,5 entre réplicas": f"{fmt(np.quantile(vals, .025))}–{fmt(np.quantile(vals, .975))} min" if len(vals) > 1 else "No estimable (1 réplica)",
            })
    st.dataframe(pd.DataFrame(timing_rows), hide_index=True, use_container_width=True)

    st.markdown("#### Colas")
    queue_labels = {"queue_registration": "Admisión", "queue_triage": "Triage", "queue_attention": "Atención médica"}
    queue_rows = [{"Etapa": queue_labels.get(key, key), "Media durante el horizonte": value["mean"], "Máximo": value["max"]}
                  for key, value in indicators.get("queues", {}).items()]
    st.dataframe(pd.DataFrame(queue_rows), hide_index=True, use_container_width=True)

    method = result.get("method", {})
    st.info(
        "La demanda es sintética y se genera desde perfiles agregados históricos. "
        "La severidad se muestrea desde proporciones; el modelo ESI preentrenado no se aplica. "
        "El motor no modela abandono. Door-to-doctor es un resultado simulado, no una medida observada de HRT."
    )
    st.caption(
        f"Demanda: multiplicador {fmt(method.get('demand_multiplier', 1), 2)} sobre tasas históricas agregadas; "
        f"llegadas Poisson por hora comparable y distribución uniforme dentro de cada hora. "
        "Los intervalos mostrados son entre réplicas; no representan incertidumbre clínica individual."
    )
    st.download_button(
        "Descargar resultados agregados (JSON)",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name=f"ueh_hrt_seed{run['master_seed']}_reps{run['replication_count']}.json",
        mime="application/json",
        key=f"download_{run['master_seed']}_{run['replication_ids'][0]}_{run['replication_count']}_{predictive}",
    )


with st.sidebar:
    st.divider()
    st.caption("Las simulaciones no contienen identificadores de pacientes. Los datos ilustrativos de la maqueta no se usan como resultados.")

# A compact aggregate reference upload is optional. It never accepts patient-level rows.
reference_upload = None
if view == "Retrospectivo":
    with st.expander("Referencia observada agregada (opcional)", expanded=True):
        reference_upload = st.file_uploader(
            "Carga un CSV agregado para el período seleccionado",
            type=["csv"],
            help="Debe contener indicadores ya agregados, sin folios ni filas de pacientes.",
        )
        st.download_button("Descargar plantilla CSV", _template_bytes(), "plantilla_referencia_observada_hrt.csv", "text/csv")
        st.caption("Esquema y unidades: `length_of_stay` (minutes u hours), `wait_triage` y `door_to_doctor` (minutes), `utilization` (% o proportion), `severity_count` (patients), `abandonment` (%). Documenta definición y fuente por fila.")

reference = None
parsed_period = period_to_run(period)
if parsed_period is not None:
    selected_start, selected_hours = parsed_period
    reference = parse_reference(reference_upload, selected_start, selected_start + timedelta(days=selected_hours // 24 - 1)) if view == "Retrospectivo" else None

if submitted:
    if parsed_period is None:
        st.error("Selecciona un período válido de hasta 7 días (fechas inclusivas). El motor admite horizontes de hasta 168 horas.")
    else:
        start_date, horizon_hours = parsed_period
        base_payload = build_simulation_payload(
            start_date=start_date,
            start_hour=0,
            horizon_hours=horizon_hours,
            master_seed=master_seed,
            replication_id=replication_id,
            severity_probabilities=severity_probabilities,
            registration_capacity=registration_capacity,
            triage_capacity=triage_capacity,
            doctor_capacity=doctor_capacity,
            box_capacity=box_capacity,
            demand_multiplier=float(demand_scale),
        )
        try:
            if view == "Contrafactual":
                scenarios = {"Base (configuración vigente)": (float(demand_scale), int(doctor_capacity), int(box_capacity))}
                for name in selected_scenarios:
                    multiplier, doctors_delta, boxes_delta = scenario_choices[name]
                    if name.startswith("Demanda"):
                        factor = float(demand_scale) * multiplier
                    else:
                        factor = float(demand_scale) * multiplier
                    scenarios[name] = (factor, max(1, int(doctor_capacity) + doctors_delta), max(1, int(box_capacity) + boxes_delta))
                total_calls = int(replication_count) * len(scenarios)
                progress = st.progress(0, text="Preparando escenarios pareados")
                results = {}
                completed_calls = 0
                for name, (factor, doctors, boxes) in scenarios.items():
                    payload = {
                        **base_payload,
                        "demand_multiplier": factor,
                        "doctor_capacity": doctors,
                        "box_capacity": boxes,
                    }
                    results[name], completed_calls = run_replicates(
                        payload, int(replication_count), int(replication_id), progress, completed_calls, total_calls
                    )
                st.session_state["contrast_results"] = results
            else:
                progress = st.progress(0, text="Preparando réplicas")
                result, _ = run_replicates(
                    base_payload, int(replication_count), int(replication_id), progress, 0, int(replication_count)
                )
                st.session_state[f"result_{view}"] = result
                st.session_state[f"payload_{view}"] = {**base_payload, "replication_count": int(replication_count)}
            st.session_state.pop("simulation_error", None)
        except (RuntimeError, ValueError, KeyError) as exc:
            st.session_state["simulation_error"] = str(exc)

if st.session_state.get("simulation_error"):
    st.error(st.session_state["simulation_error"])

if view == "Retrospectivo":
    st.markdown("### Validación retrospectiva")
    st.caption("Comparación de simulación con una referencia observada agregada del mismo período.")
    result = st.session_state.get("result_Retrospectivo")
    if not result:
        st.info("Ejecuta la simulación. Sin un CSV de referencia observada, el panel mostrará resultados simulados y señalará qué indicadores no están disponibles como observaciones.")
    else:
        display_result(result, reference)
        st.markdown("#### Comparación observado vs. simulado")
        if reference is None:
            st.warning("No se cargó una referencia observada para este período. No se inventan valores reales ni porcentajes de similitud.")
        else:
            render_comparison_table(result, reference)

elif view == "Contrafactual":
    st.markdown("### Comparación de escenarios")
    results = st.session_state.get("contrast_results")
    if not results:
        st.info("Selecciona escenarios en el panel superior y ejecuta la comparación.")
    else:
        baseline = results["Base (configuración vigente)"]
        show_run_metadata(baseline)
        st.caption("Se reutilizan semilla maestra e identificadores de réplica entre escenarios. La demanda cambia por su tasa Poisson; las llegadas pueden diferir entre políticas.")
        c1, c2 = st.columns(2)
        with c1:
            scenario_chart(results, "length_of_stay", "LOS medio por paciente (horas)", divisor=60)
        with c2:
            scenario_chart(results, "door_to_doctor", "Door-to-doctor simulado (minutos)")
        selected = st.selectbox("Distribución ESI del escenario", list(results), index=0)
        category_chart(results[selected], title=f"Pacientes por categoría ESI · {selected}")
        st.markdown("#### Resumen por escenario")
        rows = []
        for name, item in results.items():
            ind = item["indicators"]
            rows.append({
                "Escenario": name,
                "Llegadas medias": fmt(ind.get("arrivals")),
                "LOS medio (h)": fmt((ind["length_of_stay"].get("mean") or 0) / 60) if ind["length_of_stay"].get("mean") is not None else "No disponible",
                "Door-to-doctor medio (min)": fmt(ind["door_to_doctor"].get("mean")),
                "Utilización médicos (%)": fmt((ind["resource_utilization"].get("doctor") or 0) * 100),
                "Utilización enfermería (%)": fmt((ind["resource_utilization"].get("triage") or 0) * 100),
                "Utilización box (%)": fmt((ind["resource_utilization"].get("box") or 0) * 100),
                "Configuración": f"Demanda ×{fmt(item['run'].get('demand_multiplier', 1), 2)}; médicos {item['run']['capacities']['doctors']}; box {item['run']['capacities']['boxes']}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.info("El motor no modela abandono; este indicador queda fuera de la comparación contrafactual.")
        st.download_button(
            "Descargar comparación de escenarios (JSON)",
            json.dumps(results, ensure_ascii=False, indent=2),
            "ueh_hrt_escenarios_contrafactuales.json",
            "application/json",
        )

else:
    st.markdown("### Proyección de demanda y desempeño")
    st.caption("Proyección de simulación basada en el perfil histórico agrupado. No se presenta como pronóstico estadístico entrenado.")
    result = st.session_state.get("result_Predictivo")
    if not result:
        st.info("Elige un horizonte de hasta 7 días y ejecuta varias réplicas para estimar bandas de variación entre réplicas.")
    else:
        display_result(result, predictive=True)
