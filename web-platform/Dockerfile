FROM node:22-alpine AS frontend

WORKDIR /build

COPY web-platform/frontend/package.json web-platform/frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY web-platform/frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./requirements-engine.txt
COPY web-platform/backend/requirements.txt ./requirements-api.txt

RUN sed -i '/^streamlit==/d; /^matplotlib==/d; /^plotly==/d; /^pytest==/d' \
        requirements-engine.txt \
    && python -m pip install --upgrade pip \
    && python -m pip install -r requirements-engine.txt \
    && python -m pip install -r requirements-api.txt \
    && python -m pip install "psycopg[binary]==3.2.3"

COPY orbital_core/ ./orbital_core/
COPY pilot/ ./pilot/
COPY LICENSE README.md ./

COPY web-platform/backend/app/ ./app/
COPY web-platform/backend/tests/ ./tests/
COPY web-platform/backend/pytest.ini ./pytest.ini

COPY --from=frontend /build/dist ./app/static

RUN useradd --create-home --uid 10001 orbital \
    && mkdir -p /app/pilot_data \
    && chown -R orbital:orbital /app
USER orbital

RUN PILOT_ENV=test \
    PILOT_PARTICIPANT_SALT=build-check-not-a-real-secret \
    python -m pytest tests/test_engine_parity.py -q

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1} --proxy-headers --forwarded-allow-ips='*'"]
