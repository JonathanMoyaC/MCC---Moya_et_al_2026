# Port estructural del simulador de urgencias a Mesa

Prototipo de los pasos 2–5: una entidad `PatientAgent` por paciente, flujo operacional de un minuto, distribuciones alineadas a F3.1/H.1-R6 y adaptador para cohortes bloqueadas H.6.

## Alcance de esta iteración

- Cohorte fija/exógena con `patient_id`, `arrival_time` y `triage_observed`.
- Secuencia: completar y liberar → llegadas → admisión → triage → atención → registro de indicadores.
- Capacidades base 2/1/12/18 y priorización C1–C5 con envejecimiento para C3–C5.
- Muestreo empírico determinista, keyed por semilla/réplica/paciente/proceso.
- Sampler discreto F3.1: `BLAKE2b(digest_size=8, person=b"UEH_VE")`, conversión al punto medio del intervalo y selección por `floor(u × n)` sobre la distribución empírica ordenada.
- Indicadores por paciente y series de colas, recursos y pacientes activos/completados.
- Utilización minuto a minuto de admisión, triage, médicos y boxes, más invariantes de capacidad y conservación de pacientes.
- Integración de las asignaciones H.6 `observada`, `empirica` y `jerarquica` con las horas de llegada de la cohorte bloqueada H.1-R6.
- Validación de réplica, escenario, hashes del manifiesto, conteos por ventana, IDs únicos y consistencia de severidad observada.
- Ejecución por ventana independiente: las cuatro ventanas no se concatenan como si los meses entre ellas fueran tiempo simulado.

## Insumos obligatorios

La clase `EmpiricalInputModel.from_f3_cache(path)` reproduce los insumos congelados:

- Registro: los 75 valores de `OBS_ADMISSION_MIN` definidos en F3.1.
- Triage: los 79 valores de `OBS_TRIAGE_MIN` definidos en F3.1.
- Atención: `attention_time_real` de la ventana de calibración 2017-09-18 a 2017-09-20, con calidad temporal elegible y límite inclusivo de 1–720 min.
- Atención específica por categoría cuando H.1-R6 encuentra al menos 30 valores no faltantes antes de limpiar el rango 1–720 min. En la caché auditada, eso habilita C2 (204 valores válidos), C3 (240) y C4 (29); C1 y C5 usan la distribución global (482 valores). C4 tiene 38 valores no faltantes y 29 dentro del rango, por lo que conserva la distribución de clase del motor de referencia.

El lector comprueba SHA-256 y restringe los globals del pickle a las clases de reconstrucción Pandas/NumPy observadas. El pickle clínico no se distribuye dentro de este paquete. SHA-256 auditado: `5edef6b238a0723954b3c678d6b8192a3af672afb227245332daa58ad2a74cea`.

El resumen de auditoría sin filas de pacientes está en `calibration_audit_F1_R1_v2.json`.

También se puede pasar `service_samples` directamente, con muestras en minutos para `registration`, `triage` y `attention`, y, opcionalmente, claves `attention_C1`…`attention_C5`. Se filtran duraciones fuera del intervalo 1–720 min.

Cada fila puede aportar `triage_assigned` para un escenario predictivo ya calculado. Si no existe, se usa `triage_observed`, lo que representa el escenario observado/control. El motor no ejecuta todavía modelos ML ni el bootstrap H.6.

```python
from ueh_mesa import EmpiricalInputModel, EmergencyDepartmentModelMesa, ModelConfig

inputs = EmpiricalInputModel.from_f3_cache("version_d_base_preparada_F1_R1_v2.pkl")
model = EmergencyDepartmentModelMesa(
    cohort=rows,
    service_samples=inputs,
    config=ModelConfig(master_seed=42, replication_id=0),
)
patient_rows, minute_rows = model.run_to_completion()
print(model.calibration_metadata)
```

## Paso 5: cohortes y escenarios H.6

Los CSV H.6 fijan `triage_assigned`, pero no incluyen `arrival_time`. El adaptador `ueh_mesa.h6` cruza esas asignaciones con las columnas `patient_id`, `arrival_time` y `triage_observed` de la caché H.1-R6. La tabla fuente permanece local; no se incluye en este paquete.

```python
import pandas as pd
from ueh_mesa import (
    EmpiricalInputModel,
    ModelConfig,
    build_observed_h6_assignments,
    read_h6_assignments,
    read_h6_reference_cohort,
    run_h6_mesa_scenario,
)
from ueh_mesa.calibration import EXPECTED_F3_CACHE_SHA256

manifest = "H6_input_manifest_confirmatorio.csv"
predictive = read_h6_assignments(
    "H6_input_empirica_rep_063.csv.gz",
    manifest_path=manifest,
    replication_id=63,
    scenario="empirica",
)
reference = read_h6_reference_cohort(
    "version_d_base_preparada_F1_R1_v2.pkl",
    expected_sha256=EXPECTED_F3_CACHE_SHA256,
)
services = EmpiricalInputModel.from_f3_cache(
    "version_d_base_preparada_F1_R1_v2.pkl"
)
observed_assignments = build_observed_h6_assignments(predictive, reference)
patients, minutes = run_h6_mesa_scenario(
    observed_assignments,
    reference,
    services,
    config=ModelConfig(master_seed=42),
    replication_id=63,
    scenario="observed",
)
patients_emp, minutes_emp = run_h6_mesa_scenario(
    predictive, reference, services,
    config=ModelConfig(master_seed=42),
    replication_id=63, scenario="empirical",
)
```

El protocolo H.6 tiene 64 réplicas (0–63), cuatro ventanas con 3.948 pacientes por réplica y tres ramas: control observado, empírica y jerárquica. El manifiesto confirmatorio registra las dos entradas predictivas; el control observado se obtiene de `triage_observed` en la cohorte bloqueada. Se procesa cada archivo réplica–escenario por separado. Las asignaciones empírica y jerárquica ya vienen fijadas por paciente en los artefactos H.6; este paso no vuelve a entrenar ni a muestrear los clasificadores. No incorpora la política oráculo/salvaguarda como escenario desplegable.

## Instalación y prueba

```bash
python -m pip install -e '.[test]'
pytest
```

Dependencia fijada en Mesa 3.5.1. Usa `Model` y `Agent` actuales; no depende del antiguo `mesa.time` ni de SimPy.

## Paso 6: equivalencia paciente a paciente

Se comparó la réplica 63 de las ramas empírica y jerárquica, usando las mismas asignaciones H.6, la cohorte bloqueada y la calibración F3.1/H.1-R6. La función `compare_h1_patient_outputs()` alinea por `patient_id` y `window_id` y devuelve solo estadísticas agregadas, sin identificadores ni filas individuales.

En ambas ramas coincidieron exactamente los 3.948 pacientes, las categorías observada y asignada, las nueve marcas temporales del proceso y los siete KPI por paciente. El error absoluto máximo fue 0 en cada variable comparada. La referencia H.1-R6 disponible contiene marcas de eventos por paciente, pero no series minuto a minuto de colas y utilización; por eso esta verificación no valida la equivalencia de esas series. El resumen auditable sin datos de pacientes está en `step6_equivalence_replica_063.json`.

Esta es una verificación de una réplica de equivalencia para cada rama predictiva, no la ejecución confirmatoria completa de 64 réplicas ni una validación de resultados estadísticos agregados. Antes de afirmar reproducibilidad confirmatoria queda ejecutar las réplicas restantes con un manifiesto de salida y revisar los agregados H.6. El control observado tampoco se incluye en esta comparación porque no hay un checkpoint H.1-R6 correspondiente entre los archivos verificados.

El envejecimiento sí puede cambiar el orden entre pacientes con esperas distintas; los empates conservan la precedencia de la categoría base. Para la réplica 63, el motor Mesa reproduce exactamente las salidas de paciente de ambas ramas predictivas en los artefactos H.1-R6 contrastados. La equivalencia de trazas minuto a minuto y la validación confirmatoria agregada siguen pendientes.

## Paso 7: generación sintética de demanda agregada

El módulo independiente `ueh_mesa.demand` convierte el perfil agregado de Fase 2 en una cohorte sintética para ejecutar en Mesa. No altera el adaptador H.6 ni la prueba de equivalencia: H.6 sigue recibiendo la misma cohorte bloqueada; este generador se usa en corridas de escenarios con llegadas nuevas.

Antes de implementarlo, se reconstruyeron los perfiles desde los tres libros de 2016–2018 leyendo solo `urg_fecha_hora`, `tri_fecha_hora`, `urg_fecha_egreso` y `niv_cod`. Los seis artefactos agregados resultaron idénticos a los perfiles entregados. Los libros fuente no forman parte de este paquete ni son necesarios al ejecutar el simulador.

El muestreador:

- selecciona la celda según mes, día hábil/fin de semana y hora;
- genera el conteo horario con una Poisson parametrizada por llegadas medias por día comparable y hora;
- distribuye llegadas uniformemente entre los 60 minutos discretos de cada hora;
- asigna C1–C5 con probabilidades históricas suavizadas por defecto, o probabilidades brutas si se solicitan;
- usa flujos pseudoaleatorios separados y deterministas por hora, réplica y proceso;
- devuelve claves `syn-r…` sintéticas y metadatos agregados, sin filas fuente ni identificadores originales.

Poisson e intrahora uniforme son **supuestos de simulación aún por validar**; los perfiles solo describen frecuencias horarias agregadas. El perfil combina 2016–2018, período en que el volumen incluido creció 42,9 %, por lo que no debe presentarse como estimación de demanda actual sin validación temporal. Los intervalos ingreso–triage e ingreso–egreso no se usan como tiempos de servicio.

```python
from datetime import datetime
from ueh_mesa import (
    EmpiricalInputModel,
    EmergencyDepartmentModelMesa,
    ModelConfig,
    generate_synthetic_cohort,
    load_arrival_profile,
)

profile = load_arrival_profile("fase2_profiles/hourly_arrival_severity_profile.csv")
cohort, generation = generate_synthetic_cohort(
    profile,
    start=datetime(2018, 1, 1, 0, 0),
    horizon_hours=24,
    master_seed=42,
    replication_id=0,
)
services = EmpiricalInputModel.from_f3_cache("version_d_base_preparada_F1_R1_v2.pkl")
model = EmergencyDepartmentModelMesa(
    cohort.to_dict("records"), services,
    config=ModelConfig(master_seed=42, replication_id=0),
)
patient_rows, minute_rows = model.run_to_completion()
```

La prueba integrada del 1 de enero de 2018 produjo 339 llegadas sintéticas y Mesa completó los 339 pacientes. En el paso 7 pasaron 25 pruebas unitarias. El resultado verifica la conexión perfil → cohorte sintética → motor; no valida empíricamente los supuestos de llegada ni integra el clasificador LightGBM, que requiere una auditoría separada de sus variables de entrada sintéticas.

## Paso 8: inventario y contrato del modelo ESI

Se fija como candidato la versión ESI recientemente validada `prioria_esi_triage_v1.0` (LightGBM), con el artefacto `prioria_esi_triage_v1.0_lightgbm_tuned.joblib` y su sidecar. El sidecar reporta accuracy de prueba 0,6923, balanced accuracy 0,5199, macro-F1 0,5110 y recalls C1–C5 de 0,4598, 0,7935, 0,6453, 0,6235 y 0,0772. Son métricas registradas por ese artefacto; en este paso no se reentrenó ni se recalcularon.

La entrada declarada contiene 769 variables: `AÑO` más `beto_clinico_000`…`beto_clinico_767`. La cohorte de demanda sintética solo contiene `patient_id`, `arrival_time` y `triage_observed`; faltan las 769 entradas requeridas. El modelo depende de representaciones textuales BETO que el generador no produce. Por eso se mantiene el modelo en el inventario, pero se bloquea la inferencia y no se sustituyen embeddings por valores inventados ni se usa la categoría histórica como si fuera una predicción ML.

El módulo `ueh_mesa.model_contract.audit_model_input_contract()` hace explícito este gate antes de cualquier integración. Las clases declaradas son C1–C5 con codificación 0–4. El sidecar no incluye hash del artefacto serializado ni declara fuente de entrenamiento; tampoco se deserializó ni ejecutó el `joblib`, porque la interfaz de entrada ya es incompatible. El reporte completo está en `step8_esi_model_contract_audit.json`. Los artefactos XGBoost anteriores quedan fuera de la selección vigente, conforme a la decisión de usar la versión ESI validada recientemente.

Para habilitar inferencia habrá que especificar y validar la generación de `AÑO` y del texto clínico sintético, reproducir el pipeline BETO-CLINICAL-ES y su tokenizador, y después comprobar carga, orden de features, decodificación de clases y calibración dentro del runtime fijado. Hasta entonces, la severidad del simulador sigue proviniendo del muestreador de proporciones históricas, no del modelo ESI.


## API de simulación (Fase 5 del plan)

`ueh_mesa.service.SimulationService` conecta los perfiles agregados, la calibración F3.1/H.1-R6 y el motor Mesa. La API FastAPI en `ueh_mesa.api` ofrece:

- `GET /health`: informa si los insumos runtime están configurados.
- `GET /v1/version`: nombre, versión y motor.
- `POST /v1/simulations`: ejecuta una simulación y devuelve conteos y KPI agregados. Nunca devuelve filas ni identificadores de pacientes.

La respuesta marca explícitamente `esi_model_applied: false` y describe la severidad como muestreada desde proporciones históricas agregadas. También declara que conteos Poisson y minutos intrahorarios uniformes son supuestos del generador. Los estadísticos de atención y espera son resultados simulados, no estimaciones clínicas individuales.

Para instalar y levantar el servicio desde la raíz `mesa_prototype` (Python 3.12 o superior):

```bash
uv sync --locked --extra api --extra web
export UEH_ARRIVAL_PROFILE_PATH="$PWD/fase2_profiles/hourly_arrival_severity_profile.csv"
export UEH_CALIBRATION_CACHE_PATH="/ruta/runtime/restringida/calibration_cache_F1_R1_v2.pkl"
uv run --no-sync uvicorn ueh_mesa.api:app --host 127.0.0.1 --port 8000
```

En una segunda terminal, levanta la interfaz: `UEH_API_BASE_URL=http://127.0.0.1:8000 uv run --no-sync streamlit run app.py --server.address 127.0.0.1 --server.port 8501`.

La caché de calibración debe estar disponible en el entorno de ejecución y coincidir con el SHA-256 F3.1 esperado. No se incluye en el ZIP junto con la API, ni se incluyen libros fuente. Ejemplo de cuerpo para una corrida:

```json
{
  "start": "2018-01-01T00:00:00",
  "horizon_hours": 24,
  "master_seed": 42,
  "replication_id": 0,
  "severity_probabilities": "smoothed",
  "registration_capacity": 2,
  "triage_capacity": 1,
  "doctor_capacity": 12,
  "box_capacity": 18
}
```

La ventana se limita a 1–168 horas; `start` debe estar alineado al inicio de una hora. La interfaz admite de 1 a 10 réplicas consecutivas; recursos, horizonte y campos desconocidos se validan en la API. Las dependencias de API y Streamlit están declaradas en los extras `api` y `web`, fijadas en `uv.lock`. Pasaron 30 pruebas; además, la interfaz se probó conectada a FastAPI/Uvicorn y Mesa. La corrida de 24 horas del 1 de enero de 2018 produjo 339 llegadas y completó 339, sin devolver IDs de paciente.


## Empaquetado y despliegue privado (paso 9 del orden de implementación)

La API y la interfaz Streamlit quedan preparadas para empaquetarse en una imagen Docker para Python 3.12, con el grafo de dependencias fijado en `uv.lock`. La imagen incorpora el código y los seis perfiles agregados de Fase 2. No incorpora los libros Excel, la caché de calibración ni modelos serializados. La caché aprobada se monta como secreto de solo lectura al arrancar.

Requisitos: Docker Engine y Docker Compose v2. Desde esta carpeta, indica la ruta absoluta de la caché F3.1/H.1-R6 validada y ejecuta:

```bash
UEH_CALIBRATION_CACHE_HOST_PATH=/ruta/aprobada/calibration_cache_F1_R1_v2.pkl \
  docker compose up --build -d
docker compose ps
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/version
```

La API queda en `http://127.0.0.1:8000` y la interfaz en `http://127.0.0.1:8501`. Ambas quedan ligadas al loopback del host. Para acceso desde una red privada, deben quedar detrás de un proxy con TLS y autenticación; no se publican directamente a Internet. Para detenerlas: `docker compose down`. Los contenedores corren como usuario no privilegiado, con sistema de archivos de solo lectura y capacidades Linux descartadas; la caché se monta fuera de la imagen.

La pantalla Streamlit ya está implementada. Con varias réplicas, los percentiles son promedios de percentiles por réplica, no percentiles agrupados de pacientes; las corridas se ejecutan secuencialmente. Se añadieron `deploy/README.md`, un preflight con verificación de hash/permisos de la caché y tags de imagen para rollback. Las comprobaciones estáticas pasan, pero no se desplegó en un servidor ni se construyó la imagen: este entorno no dispone de Docker/Compose ni de destino privado. Además, la caché disponible tiene permisos de lectura demasiado amplios y el preflight la bloquea; sus permisos no se modificaron. El recorrido UI → API → Mesa sí se probó localmente. La construcción, lectura segura del secreto y prueba del contenedor deben ejecutarse en el host antes de marcar el despliegue como verificado. Estado detallado: `deploy/deployment_readiness.json`.


## Fase 6: validación exploratoria con datos 2018

Se ejecutó una comprobación condicional en cuatro ventanas independientes de 7 días, usando horas de llegada y categorías C1–C5 observadas de 2018, con la calibración 2017 F3.1/H.1-R6. Se leyeron solo `urg_fecha_hora`, `tri_fecha_hora`, `urg_fecha_egreso` y `niv_cod`; no se guardaron filas de la fuente. La comprobación evaluó 10.436 encuentros.

Con la capacidad base 2/1/12/18, la tasa observada de llegadas supera la capacidad estimada de admisión para los periodos, y la cola crece; los tiempos simulados de triage y estadía quedan muy por encima de los observados. En una sensibilidad exploratoria de la ventana iniciada el 8 de enero, pasar admisión de 2 a 3 puestos redujo el promedio simulado de estadía de aproximadamente 2.273 a 164 minutos; el promedio hasta completar triage bajó de aproximadamente 2.183 a 66 minutos, aún mayor que los 9,5 minutos observados entre llegada y `tri_fecha_hora`. Esto no determina el número real de puestos: identifica una configuración que requiere contrastarse con registros operacionales.

La semántica de `tri_fecha_hora` (inicio o término de triage) todavía no está confirmada, por lo que esa comparación es provisional. La validación es condicional a llegadas observadas; no valida el generador sintético, la inferencia LightGBM ni el muestreador de severidad. La fase 6 sigue abierta hasta confirmar capacidades, ejecutar las 64 réplicas confirmatorias y completar rendimiento y revisión de logs. Los resultados agregados están en `validation/phase6_validation_summary.json`; los libros fuente no se distribuyen en el paquete.
