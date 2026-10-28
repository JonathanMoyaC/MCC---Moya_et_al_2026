# Estado de las fases del prototipo UEH Mesa

**Corte:** 23 de octubre de 2026  
**Plan de referencia:** `Plan_Implementacion_Despliegue_Simulador_Mesa.md`

| Fase | Estado | Evidencia y pendientes principales |
|---|---|---|
| 0. Requisitos y protocolo | Cerrada como especificación inicial | La especificación funcional y el diccionario están en el paquete. Supuestos editables y alcance deben versionarse si cambian. |
| 1. Auditoría y privacidad | Cerrada para el conjunto 2016–2018 disponible | Se documentaron campos permitidos/excluidos y se construyeron perfiles agregados. Los libros originales se excluyen del runtime web. |
| 2. Perfiles de demanda | Implementada | Perfil de 576 estratos y generador reproducible. Conteos Poisson y distribución uniforme dentro de cada hora siguen siendo supuestos que necesitan validación empírica. |
| 3. Adaptación a Mesa | Implementada con validación estructural parcial | Motor minuto a minuto; 30 pruebas de software pasan. La equivalencia paciente a paciente se verificó para la réplica 63 de las ramas predictivas; falta la corrida confirmatoria completa y análisis agregado de 64 réplicas. |
| 4. Modelos preentrenados | Integración evaluada y descartada para la interfaz actual | Se conserva como referencia el LightGBM ESI v1.0 validado recientemente. No se invoca: requiere `AÑO` y 768 variables BETO que la cohorte sintética no genera. La categoría se muestrea desde perfiles agregados y la interfaz lo informa. |
| 5. API y pantalla | Implementada y probada localmente | FastAPI + Streamlit; configuración de escenario, 1–10 réplicas secuenciales, indicadores y descarga JSON. Prueba integrada UI → API → Mesa con dos réplicas. |
| 6. Validación del prototipo | Parcial; holdout exploratorio ejecutado | 30 pruebas pasan, incluido mismo resultado para la misma semilla y variación de llegadas al cambiarla. En 4 ventanas de 2018 (10.436 encuentros evaluados), capacidad 2/1/12/18 genera colas inestables frente a la demanda observada. Una sensibilidad inicial con 3 puestos de admisión reduce la estadía simulada en la ventana de enero, pero la espera a triage sigue alta. Pendiente confirmar capacidades y significado de `tri_fecha_hora`, ejecutar protocolo confirmatorio de 64 réplicas, medir rendimiento/memoria y revisar logs. Resumen: `validation/phase6_validation_summary.json`. |
| 7. Docker y despliegue | Preparación avanzada; validación en host bloqueada | Se añadieron runbook, preflight con hash y permisos de caché, tags de imagen para rollback y comprobaciones estáticas de Compose. Las 30 pruebas pasan. Este entorno carece de Docker/Compose y la caché disponible es legible por otros usuarios del host; no se modificaron permisos. Falta validar en el servidor privado y completar TLS/autenticación, health checks y rollback. Detalle: `deploy/deployment_readiness.json`. |
| 8. Piloto y mantenimiento | No iniciado | Requiere cerrar validación, desplegar en entorno privado, definir usuarios y ejecutar escenarios controlados. |

## Próximo trabajo

1. Resolver con evidencia las capacidades y la semántica de `tri_fecha_hora`; luego completar el protocolo de fase 6 (64 réplicas, sensibilidad y rendimiento).
2. Corregir permisos de la caché aprobada para acceso mínimo y ejecutar `scripts/deploy_preflight.sh`; después construir y probar Docker/Compose en el host privado.
3. Desplegar detrás de proxy autenticado con TLS en un servidor privado y probar reversión antes del piloto.
4. Iniciar fase 8 solo después de aprobar los controles anteriores.

## Alcance de seguridad y uso

El prototipo entrega resultados de simulación operacional, no recomendaciones clínicas. La API y la pantalla exponen estadísticas agregadas; no retornan filas ni identificadores de pacientes. Las planillas fuente, la caché restringida y el artefacto `joblib` no se copian a la imagen.
