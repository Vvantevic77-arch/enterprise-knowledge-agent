"""Exception handlers producing one consistent error envelope."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core_exceptions import AppError

logger = logging.getLogger(__name__)


def _envelope(code: str, message: str, details: dict[str, Any], trace_id: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details, "trace_id": trace_id}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
        # Denials are expected traffic; server errors are not. Log accordingly.
        if exc.status_code >= 500:
            logger.exception("unhandled app error on %s %s", request.method, request.url.path)
        else:
            logger.info(
                "request rejected",
                extra={"code": exc.code, "path": request.url.path, "trace_id": trace_id},
            )
        headers = {}
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            headers["WWW-Authenticate"] = "Bearer"
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, exc.details, trace_id),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_envelope(
                "VALIDATION_ERROR",
                "Request payload is invalid",
                {"errors": exc.errors()},
                trace_id,
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope(
                "INTERNAL_ERROR",
                "Internal server error",
                {},
                trace_id,
            ),
        )
