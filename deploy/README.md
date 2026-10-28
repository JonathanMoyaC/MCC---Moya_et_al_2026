# Despliegue privado de UEH Mesa

Este runbook cubre la construcción y operación inicial en un host Linux privado con Docker Engine y Docker Compose v2. El servicio se considera **no desplegado** hasta completar las comprobaciones de este documento. La publicación para usuarios externos queda bloqueada mientras la fase 6 siga pendiente.

## 1. Preparar el host

- Instalar Docker Engine y el plugin Docker Compose v2.
- Mantener el repositorio en una ruta de servicio con acceso de escritura restringido.
- Guardar la caché aprobada F3.1/H.1-R6 fuera del repositorio y del contexto de construcción.
- Asegurar que el archivo de caché sea legible dentro del contenedor por UID 10001, que corre sin privilegios. Con Docker rootless, verificar el mapeo de UID del daemon. No ampliar los permisos de lectura a otros usuarios del host.
- No copiar los libros Excel, artefactos `joblib` ni la caché restringida al repositorio o a la imagen.

## 2. Validar entradas y configuración

Desde la carpeta `mesa_prototype`:

```bash
export UEH_CALIBRATION_CACHE_HOST_PATH=/ruta/privada/calibration_cache_F1_R1_v2.pkl
export UEH_IMAGE_TAG=0.1.0
bash scripts/deploy_preflight.sh
```

El preflight verifica el hash de la caché, que esta quede fuera del contexto del proyecto, y que Docker/Compose puedan resolver el YAML. El hash requerido está fijado en `src/ueh_mesa/calibration.py`. Si falla, detener el despliegue y resolver la causa; no sustituir la caché por otra versión sin auditarla.

Si se usa Docker rootful con el UID/GID del contenedor sin remapeo, una configuración posible para el archivo es propietario `root`, grupo numérico `10001` y modo `0440`; verificar primero que el grupo de servicio corresponda a UID/GID 10001 y que no haya ACLs que amplíen acceso. No ejecutar esos cambios sobre una caché compartida sin evaluar los demás consumidores.

## 3. Construir y probar en el host privado

```bash
docker compose build --pull
docker compose up -d
docker compose ps
curl --fail --silent http://127.0.0.1:8000/health
curl --fail --silent http://127.0.0.1:8501/_stcore/health
```

La API y la interfaz se enlazan a `127.0.0.1`; no quedan expuestas directamente en interfaces públicas. La API es accesible desde la red interna de Compose para Streamlit. El proceso corre como UID 10001, con filesystem raíz de solo lectura, capacidades Linux eliminadas y `/tmp` como volumen temporal.

Antes de permitir uso real, completar también un recorrido de interfaz que ejecute una simulación, confirmar que solo aparecen indicadores agregados, revisar que los logs no incluyan filas ni identificadores de pacientes y medir CPU/memoria en el host de destino. No almacenar respuestas de simulación en logs.

## 4. Publicar detrás de proxy

Para acceso desde otros equipos, el proxy inverso debe terminar TLS y exigir autenticación. Publicar solo Streamlit en el proxy. Mantener el puerto 8000 de la API en loopback, salvo que exista un requisito explícito de acceso API independiente y autenticado. Restringir firewall a la red autorizada.

## 5. Actualizar y volver a una versión anterior

Usar tags de imagen inmutables por versión. Construir una nueva versión sin sobrescribir la anterior:

```bash
export UEH_IMAGE_TAG=0.1.1
docker compose build --pull
docker compose up -d
docker compose ps
```

Si falla el smoke test, volver al tag anterior que permanezca disponible localmente:

```bash
export UEH_IMAGE_TAG=0.1.0
docker compose up -d
docker compose ps
```

Confirmar de nuevo los dos health checks. Conservar el tag anterior hasta aprobar la nueva versión.

## 6. Apagado

```bash
docker compose down
```

La caché se monta como secreto de Compose en modo de solo lectura. Compose local monta archivos del host; antes del arranque se debe comprobar que UID 10001 pueda leer la caché y que ningún otro usuario del host obtenga acceso adicional.
