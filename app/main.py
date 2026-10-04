from contextlib import asynccontextmanager

import httpx
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from app.api.v1.routers.assistant import router as assistant_router
from app.core.config import Settings
from app.core.exceptions import AuthError, LLMError, UpstreamError
from app.core.logging import setup_logging
from app.infrastructure.container import build_container


def create_app() -> FastAPI:
    settings = Settings()
    setup_logging(settings.debug)
    container = build_container()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await container.close()

    app = FastAPI(title=settings.app_name, lifespan=lifespan, docs_url="/ai/docs",
                  openapi_url="/ai/openapi.json")
    setup_dishka(container, app=app)
    app.include_router(assistant_router)

    @app.exception_handler(AuthError)
    async def auth_error(_: Request, exc: AuthError):
        return JSONResponse(status_code=401, content={"error": str(exc)})

    @app.exception_handler(UpstreamError)
    async def upstream_error(_: Request, exc: UpstreamError):
        status = exc.status_code if exc.status_code in (401, 403) else 502
        return JSONResponse(status_code=status, content={"error": exc.detail})

    @app.exception_handler(LLMError)
    async def llm_error(_: Request, exc: LLMError):
        return JSONResponse(status_code=502, content={"error": str(exc)})

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        async with container() as request_container:
            redis = await request_container.get(Redis)
            http = await request_container.get(httpx.AsyncClient)
            await redis.ping()
            # Absolute URL bypasses the client's /api/v1 base_url: Go /healthz is at root.
            response = await http.get(settings.bookly_api_url.rstrip("/") + "/healthz")
            response.raise_for_status()
        return {"status": "ok"}

    return app


app = create_app()
