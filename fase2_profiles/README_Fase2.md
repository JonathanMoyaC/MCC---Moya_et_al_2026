# Fase 2 — Perfiles agregados para el simulador de urgencias

## Contenido del paquete

- **hourly_arrival_severity_profile.csv**: perfil por mes, tipo de día y hora; contiene número de llegadas C1–C5, días de exposición, tasa media de llegadas por día comparable/hora y proporciones de gravedad observadas y suavizadas.
- **observed_interval_quantiles.csv**: cuantiles empíricos por categoría para ingreso–triage e ingreso–egreso (LOS), tanto por año como para el conjunto 2016–2018. Son intervalos observados, no tiempos de servicio.
- **temporal_quality_summary.csv**: faltantes de fechas tras conversión, secuencias temporales negativas y resúmenes por año.
- **annual_volume_summary.csv**: volumen C1–C5 válido por año.
- **source_aggregate_counts.json**: conteo de registros leídos y registros incluidos/excluidos por categoría, sin registros individuales.
- **profile_metadata.json**: definición de estratos, suavizado, política de inclusión, privacidad y limitaciones.
- **generate_phase2_profiles.py**: script reproducible que lee solo las cuatro columnas necesarias de los libros fuente.
- **README_Fase2.md**: descripción de método y uso.

Los archivos Excel originales no están incluidos. El script nunca exporta filas individuales, folios, fechas exactas, fecha de nacimiento, texto clínico ni códigos de personal.

## Perfil de llegadas

Se generaron 576 estratos: 12 meses × 2 tipos de día (lunes–viernes y fin de semana) × 24 horas. El denominador de cada tasa es la cantidad de días calendario comparables del mismo mes y tipo de día entre 2016 y 2018. Por tanto, cada celda informa llegadas promedio por día comparable y hora; no es una proyección para una fecha real específica.

Las proporciones C1–C5 incluyen tanto la frecuencia empírica por hora como una estimación suavizada. El suavizado usa una pseudocuenta de 20 hacia la distribución del mes y tipo de día, sumando estabilidad a categorías escasas sin ocultar la proporción bruta. El valor puede ajustarse posteriormente.

No hubo celdas sin llegadas ni estratos con menos de 30 registros C1–C5 en el perfil agregado. Esto no elimina la incertidumbre en categorías raras como C1; conviene conservar réplicas y semillas para comparar escenarios.

Al simular llegadas a partir de estas tasas, una posible primera implementación es generar el conteo de cada hora desde una Poisson con la tasa de la celda y ubicar los tiempos uniformemente dentro de esa hora. Esa forma intrahoraria es un supuesto del simulador, no algo observado en el perfil agregado.

## Intervalos de tiempo observados

El archivo de cuantiles almacena la distribución empírica de:

1. **Ingreso a triage:** triage menos ingreso.
2. **Ingreso a egreso (LOS):** egreso menos ingreso.

Se excluyen intervalos con fecha inválida o duración negativa. Los cuantiles se entregan por categoría y año, además del conjunto agrupado 2016–2018. Ingreso–triage no equivale a tiempo puro en cola ni a duración del procedimiento de triage; el significado operacional preciso de las marcas debe confirmarse. No existe una marca explícita de primera atención médica, por lo que el paquete no permite estimar Door-to-Doctor observado ni tiempo de atención médica.

## Resultados de volumen

| Año | Llegadas C1–C5 incluidas |
|---|---:|
| 2016 | 98.917 |
| 2017 | 118.979 |
| 2018 | 141.357 |

El volumen incluido aumenta 20,3 % de 2016 a 2017 y 18,8 % de 2017 a 2018 (42,9 % entre 2016 y 2018). Puede representar cambios de demanda, cobertura o captura de datos. El perfil pooled 2016–2018 debe interpretarse como un promedio histórico, no como una estimación de demanda actual sin validación adicional.

## Categorías de gravedad

Se incluyeron únicamente valores reconocidos C1, C2, C3, C4 y C5. En total se usaron 359.253 registros. Se excluyeron del perfil por gravedad 342 registros (SC, C o sin categoría). El desglose por fuente se encuentra en source_aggregate_counts.json.

## Ejecución reproducible

Requiere Python, pandas, NumPy y un motor Excel compatible con pandas (por ejemplo, openpyxl). Con los tres archivos fuente en una carpeta local:

    python generate_phase2_profiles.py --input-dir /ruta/local/project_sources --output-dir /ruta/de/salida

El script lee únicamente urg_fecha_hora, tri_fecha_hora, urg_fecha_egreso y niv_cod. Los libros originales deben mantenerse fuera del servidor. Despliega únicamente perfiles agregados revisados.

## Próximas decisiones antes de integrar al modelo Mesa

1. Confirmar el significado de las marcas de ingreso y triage con el diccionario de datos.
2. Decidir si el perfil pooled 2016–2018 es suficiente o si los escenarios deben permitir elegir período/año de referencia.
3. Acordar cómo modelar tiempo de servicio de triage y atención médica: no se observa de forma separada en estas planillas.
4. Validar si se usarán tasas Poisson y uniformidad intrahoraria para generar llegadas sintéticas.
5. Mantener Door-to-Doctor como salida modelada y rotulada, no observada.
