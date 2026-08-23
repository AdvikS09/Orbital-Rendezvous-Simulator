# Production image for the Orbital Rendezvous Simulator web platform.
#
# Multi-stage:
#   1. build the React application with Node
#   2. copy the built assets into a Python image that also holds the engine
#
# FastAPI then serves the API and the frontend from the SAME origin, so there
# is one deployment, one custom domain and no CORS configuration to get wrong.
#
# Build context is the REPOSITORY ROOT, not this directory, because the image
# needs orbital_core/ and pilot/ which live above it:
#
#   docker build -f web-platform/Dockerfile -t orbital-web .
#   docker run -p 8000:8000 --env-file .env orbital-web
#
# The image contains no secrets. Everything is supplied at run time; see
# web-platform/.env.example.

# =========================================================== stage 1: frontend
FROM node:22-alpine AS frontend

WORKDIR /build

# Dependencies first so an application change does not invalidate the layer.
COPY web-platform/frontend/package.json web-platform/frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY web-platform/frontend/ ./
RUN npm run build


# ============================================================ stage 2: runtime
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- dependencies -----------------------------------------------------------
COPY requirements.txt ./requirements-engine.txt
COPY web-platform/backend/requirements.txt ./requirements-api.txt

# Streamlit and matplotlib are not needed to serve the API. They are dropped
# here so the image does not carry a second web framework it never runs; the
# legacy Streamlit application has its own image at the repository root.
RUN sed -i '/^streamlit==/d; /^matplotlib==/d; /^plotly==/d; /^pytest==/d' \
        requirements-engine.txt \
    && python -m pip install --upgrade pip \
    && python -m pip install -r requirements-engine.txt \
    && python -m pip install -r requirements-api.txt \
    && python -m pip install "psycopg[binary]==3.2.3"

# --- the engine and the platform layer -------------------------------------
# orbital_core is the physics. pilot holds the scenario catalogue, validation,
# run path, analytics and exports, shared with the legacy application.
COPY orbital_core/ ./orbital_core/
COPY pilot/ ./pilot/
COPY LICENSE README.md ./

# --- the API ---------------------------------------------------------------
COPY web-platform/backend/app/ ./app/
COPY web-platform/backend/tests/ ./tests/
COPY web-platform/backend/pytest.ini ./pytest.ini

# --- the built frontend ----------------------------------------------------
COPY --from=frontend /build/dist ./app/static

# --- non-root user ---------------------------------------------------------
RUN useradd --create-home --uid 10001 orbital \
    && mkdir -p /app/pilot_data \
    && chown -R orbital:orbital /app
USER orbital

# --- build-time regression gate --------------------------------------------
# An image whose numerical results have moved cannot be produced. This runs the
# engine-parity suite: 338k values compared against a direct engine call.
RUN PILOT_ENV=test \
    PILOT_PARTICIPANT_SALT=build-check-not-a-real-secret \
    python -m pytest tests/test_engine_parity.py -q

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=4).status==200 else 1)"

# One worker by design: the engine is pure-Python and CPU-bound, so extra
# workers multiply memory without improving throughput on a small host. Raise
# WEB_CONCURRENCY only alongside more CPU. Note that the rate limiter is
# in-process, so N workers means N times the configured limit.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1} --proxy-headers --forwarded-allow-ips='*'"]
