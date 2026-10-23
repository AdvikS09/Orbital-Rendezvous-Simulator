FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt requirements-lock.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-lock.txt \
    && python -m pip install "psycopg[binary]==3.2.3"

COPY orbital_core/ ./orbital_core/
COPY src/ ./src/
COPY pilot/ ./pilot/
COPY tests/ ./tests/
COPY pilot_app.py app.py pytest.ini LICENSE README.md ./
COPY scripts/ ./scripts/
COPY .streamlit/ ./.streamlit/

RUN useradd --create-home --uid 10001 pilot \
    && mkdir -p /app/pilot_data \
    && chown -R pilot:pilot /app
USER pilot

RUN PILOT_ENV=test PILOT_PARTICIPANT_SALT=build-check \
    python -m pytest tests/test_regression_baseline.py -q

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "pilot_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
