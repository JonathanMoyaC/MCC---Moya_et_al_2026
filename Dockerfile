FROM ghcr.io/astral-sh/uv:0.12.23 AS uv-bin
FROM python:3.12-slim

COPY --from=uv-bin /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UEH_ARRIVAL_PROFILE_PATH=/app/fase2_profiles/hourly_arrival_severity_profile.csv \
    UEH_CALIBRATION_CACHE_PATH=/run/secrets/calibration_cache \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY fase2_profiles/ ./fase2_profiles/
COPY .streamlit/ ./.streamlit/
COPY app.py ./

RUN uv sync --locked --no-dev --extra api --extra web --no-editable \
    && groupadd --gid 10001 simulator \
    && useradd --uid 10001 --gid simulator --no-create-home simulator \
    && chown -R simulator:simulator /app

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["/app/.venv/bin/uvicorn", "ueh_mesa.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
