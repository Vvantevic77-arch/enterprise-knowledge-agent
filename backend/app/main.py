import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db import init_db
from app.errors import install_error_handlers
from app.routes.router import router


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    try:
        yield
    finally:
        from app.services import events, rag

        await events.close()
        await asyncio.to_thread(rag.close_embedding_client)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Payment Operations Knowledge & Ticket Agent",
        version="0.2.0",
        description=(
            "Multi-tenant merchant operations knowledge base with a "
            "retrieval-augmented agent, risk-tiered approval gates, and a "
            "ticket execution lifecycle."
        ),
        lifespan=lifespan,
    )

    install_error_handlers(app)

    @app.middleware("http")
    async def _attach_trace_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Give every request a trace ID so audit rows and logs can be joined."""
        import uuid

        request.state.trace_id = request.headers.get("X-Trace-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Trace-ID"] = request.state.trace_id
        return response

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    if settings.app_env.lower() in {"development", "dev", "local"}:
        # Only in development: the browser UI is served from the same origin in
        # production, so cross-origin access is not needed there.
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(router)

    # The operator UI is served from the same origin so the demo needs no
    # second dev server. Mounted last so it cannot shadow /v1 or /docs.
    frontend_dir = Path(__file__).parent / "static"
    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


app = create_app()
