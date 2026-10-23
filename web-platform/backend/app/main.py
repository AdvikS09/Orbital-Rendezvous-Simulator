from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import bootstrap  # noqa: F401  (side effect: repo root on sys.path)

import orbital_core  # noqa: E402
from pilot.config import get_settings  # noqa: E402
from pilot.db import get_engine, run_migrations  # noqa: E402

from app import __version__  # noqa: E402
from app.api.v1 import pilot as pilot_routes  # noqa: E402
from app.api.v1 import scenarios as scenario_routes  # noqa: E402
from app.api.v1 import system as system_routes  # noqa: E402
from app.schemas.common import ErrorResponse, FieldError  # noqa: E402

logging.basicConfig(
    level=os.environ.get("PILOT_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("api")

API_PREFIX = "/api/v1"

_ERROR_CLASSES: dict[int, str] = {
    400: "BadRequest",
    401: "Unauthorized",
    403: "Forbidden",
    404: "NotFound",
    409: "Conflict",
    422: "ValidationError",
    429: "RateLimited",
    500: "InternalError",
    503: "ServiceUnavailable",
}

STATIC_DIR = Path(
    os.environ.get("FRONTEND_DIST", Path(__file__).resolve().parent / "static")
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    try:
        applied = run_migrations(get_engine(settings))
        if applied:
            logger.info("applied database migrations %s", applied)
    except Exception as exc:  # noqa: BLE001
        logger.error("database setup failed: %s", type(exc).__name__)

    for warning in settings.deployment_warnings():
        logger.warning("configuration: %s", warning)

    logger.info(
        "api %s ready (engine %s, environment %s)",
        __version__, orbital_core.__version__, settings.environment,
    )
    yield


app = FastAPI(
    title="Orbital Rendezvous Simulator API",
    version=__version__,
    summary=(
        "Research and educational orbital-mechanics simulation. Not NASA-approved, "
        "not flight-certified, and not for operational mission planning."
    ),
    description=(
        "Every endpoint returns numerical data; no plot is rendered server-side. "
        "The physics is the validated Python engine in `orbital_core`, shared "
        "unchanged with the legacy Streamlit application."
    ),
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json",
    lifespan=lifespan,
)

_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Session-Token", "Authorization"],
    )

app.include_router(system_routes.router, prefix=API_PREFIX)
app.include_router(scenario_routes.router, prefix=API_PREFIX)
app.include_router(pilot_routes.router, prefix=API_PREFIX)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_request: Request, exc: StarletteHTTPException):
    detail = exc.detail
    if isinstance(detail, list):
        fields = [
            FieldError(
                field=str(item.get("field")) if isinstance(item, dict) and item.get("field") else None,
                message=str(item.get("message", item)) if isinstance(item, dict) else str(item),
            )
            for item in detail
        ]
        message = fields[0].message if fields else "The request was refused."
    else:
        message = str(detail)
        fields = [FieldError(field=None, message=message)]

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_class=_ERROR_CLASSES.get(exc.status_code, "HTTPError"),
            message=message,
            detail=fields,
            rejected=400 <= exc.status_code < 500,
        ).model_dump(),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError):
    details = [
        FieldError(
            field=".".join(str(part) for part in error["loc"][1:]) or None,
            message=error.get("msg", "Invalid value."),
        )
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error_class="RequestValidationError",
            message="One or more values were not accepted.",
            detail=details,
            rejected=True,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_request: Request, exc: Exception):
    logger.exception("unhandled error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_class=type(exc).__name__,
            message=(
                "The request could not be completed. If this persists, please "
                "report it with the scenario and parameters you used."
            ),
        ).model_dump(),
    )


if STATIC_DIR.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=STATIC_DIR / "assets"),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str):
        """Serve the React application, letting client-side routing work.

        A request for a real file returns that file; anything else returns
        index.html so a deep link such as /scenarios/pd_rendezvous resolves.
        API paths are excluded, so a mistyped endpoint still 404s as JSON
        rather than silently returning HTML.
        """
        if full_path.startswith("api/"):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=ErrorResponse(
                    error_class="NotFound",
                    message=f"No API route at /{full_path}.",
                ).model_dump(),
            )
        candidate = (STATIC_DIR / full_path).resolve()
        if (
            full_path
            and candidate.is_file()
            and STATIC_DIR.resolve() in candidate.parents
        ):
            return FileResponse(candidate)
        return FileResponse(STATIC_DIR / "index.html")

else:  # pragma: no cover - development convenience only
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "Orbital Rendezvous Simulator API",
            "api_version": __version__,
            "engine_version": orbital_core.__version__,
            "docs": f"{API_PREFIX}/docs",
            "note": (
                "No built frontend found. In development run the Vite dev server "
                "separately; in production the Docker build copies it here."
            ),
        }
