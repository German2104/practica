"""Local composition root, process ownership and offline dashboard assets."""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
import typing

import httpx2
from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from ec_analysis.agents.providers import create_interpreter
from ec_analysis.api.jobs import JobManager
from ec_analysis.api.routes import router
from ec_analysis.config import AppSettings


def create_app(settings: AppSettings | None = None) -> FastAPI:
    configuration = settings or AppSettings()
    assets_path = pathlib.Path(__file__).with_name("static")

    @contextlib.asynccontextmanager
    async def lifespan(application: FastAPI) -> typing.AsyncIterator[None]:
        manager = JobManager(
            root_directory=pathlib.Path(configuration.jobs_directory).resolve(),
            project_root=pathlib.Path.cwd(),
        )
        application.state.job_manager = manager
        llm_client = httpx2.AsyncClient(timeout=configuration.llm_timeout_seconds)
        application.state.interpreter = create_interpreter(configuration, llm_client)

        async def monitor_jobs() -> None:
            while True:
                await run_in_threadpool(manager.refresh_jobs)
                await asyncio.sleep(1)

        monitor_task = asyncio.create_task(monitor_jobs())
        try:
            yield
        finally:
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task
            await run_in_threadpool(manager.shutdown)
            await llm_client.aclose()

    application = FastAPI(title="Факторный анализ NPV", version="0.3.0", lifespan=lifespan)
    application.include_router(router)
    application.mount("/static", StaticFiles(directory=assets_path), name="static")

    @application.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(assets_path / "index.html")

    return application


app = create_app()
