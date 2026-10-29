# Sensibilidad de espera y duración hasta triage

## Propósito

Documentar el análisis final de sensibilidad entregado para ajustar la representación de admisión y triage en el simulador Mesa. Se comparan una o dos enfermeras/puestos de triage y tres factores de duración del servicio. La corrida mantiene el mismo conjunto de llegadas y las mismas realizaciones aleatorias entre configuraciones comparables.

## Insumos y límites de interpretación

- Población de evaluación: adultos de las cuatro ventanas H.6 de 2018; 3.514 casos de evaluación por réplica.
- Llegadas y categoría observada: cohorte histórica bloqueada de 2018. Las ventanas se simulan por separado con su calentamiento.
- Tiempo de admisión: un minuto fijo por paciente, usado como placeholder determinístico no validado.
- Duración de triage: distribución empírica de 38 mediciones aproximadas tomadas in situ en adultos durante 2025. Se escala por 0,75×, 1,00× y 1,25×; son parámetros exploratorios, no datos de 2018.
- Tiempo de atención: insumo F3.1/H.1-R6 del período 18–20 de septiembre de 2017.
- Políticas de prioridad comparadas en el análisis: referencia con prioridad observada, clasificación multiclase plana y clasificación jerárquica.
- Réplicas: la celda de referencia capacidad 1/factor 1,00× tiene 128; las otras cinco celdas, 32 cada una. Las asignaciones predictivas usan realizaciones compartidas de 64 réplicas.
- Intervalos: percentiles 2,5 y 97,5 de los estimandos por réplica. No corresponden a intervalos calculados agrupando pacientes.

Los resultados son una sensibilidad condicional al conjunto histórico, al placeholder de admisión y a la muestra externa pequeña de triage. No identifican la dotación real ni constituyen validación clínica. Las tablas del directorio contienen solo estimandos agregados; se excluyeron filas e identificadores individuales.

## Resultado descriptivo: referencia con prioridad observada

| Puestos de triage | Factor de duración | Espera en cola de triage (media, min) | Duración de triage (media, min) | Llegada a inicio de triage (media, min) | Llegada a inicio de triage (P90, min) | Espera posterior a triage hasta atención médica (media, min) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,75× | 1,70 | 2,42 | 2,70 | 6,91 | 277,02 |
| 1 | 1,00× | 3,19 | 2,92 | 4,19 | 11,23 | 279,49 |
| 1 | 1,25× | 5,82 | 3,43 | 6,82 | 19,37 | 277,31 |
| 2 | 0,75× | 0,14 | 2,42 | 1,14 | 1,00 | 276,47 |
| 2 | 1,00× | 0,27 | 2,93 | 1,27 | 1,32 | 276,65 |
| 2 | 1,25× | 0,45 | 3,43 | 1,45 | 2,03 | 276,87 |

En este escenario y bajo estas suposiciones, dos puestos reducen la espera hasta iniciar triage frente a uno. Sin embargo, la espera media posterior a triage permanece alrededor de 276–279 minutos, lo que señala que este experimento no resuelve el cuello de botella posterior de atención médica. No se debe presentar la reducción simulada como una recomendación de dotación hasta contrastarla con turnos y registros operacionales.

## Archivos

- `experiment_manifest.json`: población, supuestos, tamaños de réplica e intervalos.
- `run_triage_time_sensitivity.py`: ejecutor reproducible de la cuadrícula, con identificadores de entrada y salida mantenidos fuera del repositorio.
- `replication_level_results.csv`: estadísticas agregadas por réplica y política; no contiene identificadores de pacientes.
- `absolute_process_estimands.csv`: indicadores absolutos por política, capacidad y factor.
- `paired_priority_effect_estimands.csv`: diferencias pareadas entre cada política predictiva y la referencia.
- `classification_transition_estimands.csv`: efectos agregados por transición de categoría.

Para reproducir exactamente este análisis con las ventanas H.6 y las asignaciones de prioridad, se usa el ejecutor (no el endpoint de demanda sintética). Desde la raíz del proyecto, con los insumos privados disponibles localmente:

```bash
uv run python validation/triage_time_staffing_sensitivity/run_triage_time_sensitivity.py \
  --cohort /ruta/privada/h6_2018_mesa_cohort_adults.csv \
  --service-inputs /ruta/privada/adult_service_samples.json \
  --assignment-root /ruta/privada/H6_assignments \
  --assignment-manifest /ruta/privada/H6_input_manifest_confirmatorio.csv \
  --checkpoint-root /ruta/privada/triage_sensitivity_checkpoints \
  --output-dir /ruta/privada/triage_sensitivity_outputs \
  --workers 2 --master-seed 20261027
```

Sin repetir `--scenario`, el ejecutor usa la cuadrícula exacta de seis configuraciones y los tamaños de réplica indicados arriba. Las salidas de checkpoint contienen diferencias por paciente y deben permanecer fuera del repositorio; `--output-dir` contiene solo agregados. Cohorte, asignaciones H.6 y archivo de muestras no se distribuyen en GitHub. El JSON de muestras está fijado por SHA-256 en el ejecutor.

La interfaz online también permite variar puestos de admisión/triage, admisión fija de 0–2 min y el factor de duración 0,75–1,25×. Esa interfaz genera llegadas sintéticas, así que sus corridas no reproducen las cifras H.6 de esta tabla. Se mantiene el insumo de atención F3.1/H.1-R6.


