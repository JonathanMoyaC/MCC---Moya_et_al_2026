from __future__ import annotations

import json
import os
from datetime import date

import pandas as pd
import streamlit as st

from ueh_mesa.web_client import aggregate_replication_results, build_simulation_payload, post_simulation


st.set_page_config(page_title="Simulador UEH | Mesa", page_icon="🏥", layout="wide")
st.title("Simulador operacional de urgencias")
st.caption("Prototipo de investigación basado en Mesa · escenarios sintéticos · indicadores agregados")

api_base_url = os.getenv("UEH_API_BASE_URL", "http://127.0.0.1:8000")

with st.sidebar:
    st.header("Configuración de la corrida")
    with st.form("simulation_config"):
        start_date = st.date_input("Fecha de inicio", value=date(2018, 1, 1))
        start_hour = st.selectbox("Hora de inicio", options=list(range(24)), index=0, format_func=lambda h: f"{h:02d}:00")
        horizon_hours = st.number_input("Horizonte (horas)", min_value=1, max_value=168, value=24, step=1)
        st.subheader("Recursos disponibles")
        registration_capacity = st.number_input("Admisión", min_value=1, max_value=50, value=2, step=1)
        triage_capacity = st.number_input("Puestos de triage", min_value=1, max_value=50, value=1, step=1)
        doctor_capacity = st.number_input("Médicos", min_value=1, max_value=200, value=12, step=1)
        box_capacity = st.number_input("Boxes", min_value=1, max_value=300, value=18, step=1)
        st.subheader("Reproducibilidad")
        master_seed = st.number_input("Semilla maestra", min_value=0, max_value=2_147_483_647, value=42, step=1)
        replication_id = st.number_input("Primera réplica", min_value=0, max_value=2_147_483_637, value=0, step=1)
        replication_count = st.number_input("Número de réplicas", min_value=1, max_value=10, value=1, step=1)
        severity_probabilities = st.selectbox(
            "Perfil de severidad", options=["smoothed", "raw"],
            format_func=lambda value: "Suavizado (recomendado)" if value == "smoothed" else "Bruto",
        )
        submitted = st.form_submit_button("Ejecutar simulación", type="primary", use_container_width=True)

if submitted:
    payload = build_simulation_payload(
        start_date=start_date,
        start_hour=start_hour,
        horizon_hours=horizon_hours,
        master_seed=master_seed,
        replication_id=replication_id,
        severity_probabilities=severity_probabilities,
        registration_capacity=registration_capacity,
        triage_capacity=triage_capacity,
        doctor_capacity=doctor_capacity,
        box_capacity=box_capacity,
    )
    with st.spinner("Ejecutando el escenario en Mesa…"):
        try:
            replication_results = []
            progress = st.progress(0, text="Preparando réplicas")
            for index in range(int(replication_count)):
                payload["replication_id"] = int(replication_id) + index
                replication_results.append(post_simulation(api_base_url, payload))
                progress.progress((index + 1) / int(replication_count), text=f"Réplica {index + 1}/{int(replication_count)}")
            st.session_state["simulation_result"] = aggregate_replication_results(replication_results)
            st.session_state["simulation_payload"] = {
                **payload,
                "replication_count": int(replication_count),
                "replication_id_start": int(replication_id),
            }
            st.session_state.pop("simulation_error", None)
        except RuntimeError as exc:
            st.session_state["simulation_error"] = str(exc)

if st.session_state.get("simulation_error"):
    st.error(st.session_state["simulation_error"])

result = st.session_state.get("simulation_result")
if not result:
    st.info("Configura el escenario en el panel izquierdo y ejecuta la simulación para ver sus indicadores.")
    st.caption(f"API configurada: `{api_base_url}`")
else:
    run = result["run"]
    indicators = result["indicators"]
    method = result["method"]
    count = int(run["replication_count"])
    def show_replication_value(value, sd):
        if count == 1:
            return f"{value:,.0f}"
        return f"{value:,.1f} ± {sd:,.1f}"
    st.subheader("Resumen del escenario")
    st.write(
        f"Inicio: **{run['start']}** · Horizonte: **{run['horizon_hours']} h** · "
        f"Semilla: **{run['master_seed']}** · Réplicas: **{count}** ({run['replication_ids'][0]}–{run['replication_ids'][-1]})"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Llegadas sintéticas (media/réplica)", show_replication_value(indicators["arrivals"], indicators["arrivals_sd"]))
    c2.metric("Completados (media/réplica)", show_replication_value(indicators["completed"], indicators["completed_sd"]))
    c3.metric("Minutos simulados (media/réplica)", show_replication_value(indicators["simulation_minutes"], indicators["simulation_minutes_sd"]))

    st.subheader("Distribución de severidad asignada")
    severity = pd.DataFrame(
        [{"Categoría": key, "Pacientes (media/réplica)": value} for key, value in indicators["severity_counts"].items()]
    ).set_index("Categoría")
    st.bar_chart(severity, color="#2563EB")

    st.subheader("Indicadores de tiempo (minutos simulados)")
    metric_labels = {
        "wait_triage": "Espera hasta triage",
        "wait_attention": "Espera hasta atención",
        "door_to_doctor": "Door-to-Doctor",
        "attention_time": "Tiempo de atención",
        "length_of_stay": "Estadía en urgencias (LOS)",
    }
    timing = pd.DataFrame(
        [
            {"Indicador": label, **indicators[key]}
            for key, label in metric_labels.items()
        ]
    ).set_index("Indicador").rename(columns={
        "mean": "Media por réplica", "median": "Mediana por réplica",
        "p90": "P90 por réplica", "p95": "P95 por réplica",
        "between_replication_sd_mean": "DE de medias entre réplicas",
    })
    st.dataframe(timing.round(1), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Colas")
        queue_labels = {
            "queue_registration": "Admisión",
            "queue_triage": "Triage",
            "queue_attention": "Atención médica",
        }
        queues = pd.DataFrame(
            [{"Etapa": queue_labels[key], "Media": value["mean"], "Máximo": value["max"]}
             for key, value in indicators["queues"].items()]
        ).set_index("Etapa")
        st.dataframe(queues.round(1), width="stretch")
    with right:
        st.subheader("Utilización media de recursos")
        resource_labels = {"registration": "Admisión", "triage": "Triage", "doctor": "Médicos", "box": "Boxes"}
        utilization = pd.DataFrame(
            [{"Recurso": resource_labels[key], "Utilización": value * 100}
             for key, value in indicators["resource_utilization"].items()]
        ).set_index("Recurso")
        st.bar_chart(utilization, color="#0F766E", y="Utilización")
        st.dataframe(utilization.round(1).astype(str) + "%", width="stretch")

    st.warning(
        "La severidad se muestrea desde proporciones históricas agregadas. "
        "El modelo ESI preentrenado no se aplicó; los tiempos y colas son resultados simulados."
    )
    st.caption(
        "Los percentiles presentados son el promedio de los percentiles de cada réplica; "
        "no son percentiles agrupados de todos los pacientes."
    )
    st.caption(
        f"Llegadas: {method['arrival_count_assumption']}; "
        f"ubicación intrahoraria: {method['within_hour_arrival_assumption']}. "
        "No son estimaciones clínicas individuales."
    )
    st.download_button(
        "Descargar resultados agregados (JSON)",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name=f"ueh_mesa_seed{run['master_seed']}_reps{count}.json",
        mime="application/json",
    )
