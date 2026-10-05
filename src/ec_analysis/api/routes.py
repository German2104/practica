"""Local dashboard API: upload, owned jobs and checked result exports."""

from __future__ import annotations

import pathlib
import typing

from ag_ui.core import RunAgentInput
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, StreamingResponse

from ec_analysis.agents.interpreter import InterpretationService
from ec_analysis.api.interpretation import CheckedAGUIStream
from ec_analysis.api.jobs import AnalysisJob, JobManager
from ec_analysis.config import AppSettings
from ec_analysis.engine.benchmark import BenchmarkRunner
from ec_analysis.schemas.dataset import DatasetLoader
from ec_analysis.schemas.interpretation import CheckedExplanation
from ec_analysis.schemas.state import AnalysisState

router = APIRouter()


def _manager(request: Request) -> JobManager:
    return typing.cast(JobManager, request.app.state.job_manager)


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/benchmark", response_model=AnalysisState)
async def get_benchmark() -> AnalysisState:
    def calculate() -> AnalysisState:
        settings = AppSettings()
        dataset = DatasetLoader(target_metric=settings.target_metric).load_csv(pathlib.Path(settings.dataset_path))
        return BenchmarkRunner().run(dataset)

    try:
        return await run_in_threadpool(calculate)
    except (OSError, ValueError) as failure:
        raise HTTPException(status_code=422, detail=str(failure)) from failure


class ExplanationQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


def _interpreter(request: Request) -> InterpretationService:
    interpreter = typing.cast(InterpretationService | None, request.app.state.interpreter)
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Настройте EC_LLM_MODEL и EC_OPENROUTER_API_KEY локально в .env.")
    return interpreter


def _completed_state(identifier: str, request: Request) -> AnalysisState:
    try:
        return _manager(request).read_result(identifier)
    except KeyError as failure:
        raise HTTPException(status_code=404, detail="Анализ не найден.") from failure
    except RuntimeError as failure:
        raise HTTPException(status_code=409, detail="Дождитесь завершения расчёта.") from failure


@router.get("/api/llm/status")
def interpretation_status(request: Request) -> dict[str, str | bool]:
    interpreter = typing.cast(InterpretationService | None, request.app.state.interpreter)
    return {"configured": interpreter is not None, "model": interpreter.model_name if interpreter else ""}


@router.post("/api/analyses/{identifier}/explain", response_model=CheckedExplanation)
async def explain_analysis(identifier: str, question: ExplanationQuestion, request: Request) -> CheckedExplanation:
    interpreter = _interpreter(request)
    state = _completed_state(identifier, request)
    try:
        return await interpreter.explain_result(state, question.question)
    except (ValueError, TimeoutError, ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded) as failure:
        raise HTTPException(
            status_code=502, detail="Проверенное объяснение не получено. Проверьте модель и подключение."
        ) from failure


@router.post("/ag-ui")
async def checked_ag_ui(run_input: RunAgentInput, request: Request) -> StreamingResponse:
    interpreter = _interpreter(request)
    forwarded = run_input.forwarded_props
    if not isinstance(forwarded, dict) or not isinstance(forwarded.get("analysisId"), str):
        raise HTTPException(status_code=422, detail="forwardedProps.analysisId обязателен.")
    questions = [message.content for message in run_input.messages if message.role == "user"]
    if not questions or not isinstance(questions[-1], str) or not 1 <= len(questions[-1]) <= 4000:
        raise HTTPException(status_code=422, detail="Требуется текстовый вопрос пользователя до 4000 символов.")
    state = _completed_state(forwarded["analysisId"], request)
    stream = CheckedAGUIStream(interpreter=interpreter).stream_events(run_input, state, questions[-1])
    return StreamingResponse(stream, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@router.post("/api/analyses", response_model=AnalysisJob, status_code=202)
async def create_analysis(
    request: Request, target: str = Query(default="npv", min_length=1, max_length=100), thorough: bool = False
) -> AnalysisJob:
    manager = _manager(request)
    contents = bytearray()
    async for chunk in request.stream():
        contents.extend(chunk)
        if len(contents) > manager.maximum_bytes:
            raise HTTPException(status_code=413, detail="Максимальный размер CSV — 10 МиБ.")
    try:
        return await run_in_threadpool(manager.submit, bytes(contents), target, thorough)
    except (ValueError, UnicodeError) as failure:
        raise HTTPException(status_code=422, detail=str(failure)) from failure
    except RuntimeError as failure:
        raise HTTPException(status_code=409, detail=str(failure)) from failure


@router.get("/api/analyses/{identifier}", response_model=AnalysisJob)
def analysis_status(identifier: str, request: Request) -> AnalysisJob:
    try:
        return _manager(request).read_job(identifier)
    except KeyError as failure:
        raise HTTPException(status_code=404, detail="Задание не найдено.") from failure


@router.delete("/api/analyses/{identifier}", response_model=AnalysisJob)
def cancel_analysis(identifier: str, request: Request) -> AnalysisJob:
    try:
        return _manager(request).cancel_job(identifier)
    except KeyError as failure:
        raise HTTPException(status_code=404, detail="Задание не найдено.") from failure


@router.get("/api/analyses/{identifier}/result", response_model=AnalysisState)
def analysis_result(identifier: str, request: Request) -> AnalysisState:
    try:
        return _manager(request).read_result(identifier)
    except KeyError as failure:
        raise HTTPException(status_code=404, detail="Задание не найдено.") from failure
    except RuntimeError as failure:
        raise HTTPException(status_code=409, detail=str(failure)) from failure


@router.get("/api/analyses/{identifier}/files/{filename}")
def analysis_file(identifier: str, filename: str, request: Request) -> FileResponse:
    try:
        return FileResponse(_manager(request).export_path(identifier, filename), filename=filename)
    except KeyError as failure:
        raise HTTPException(status_code=404, detail="Файл не найден.") from failure
    except RuntimeError as failure:
        raise HTTPException(status_code=409, detail=str(failure)) from failure


@router.get("/api/example")
def example_csv() -> FileResponse:
    input_path = pathlib.Path(AppSettings().dataset_path)
    if not input_path.is_file():
        raise HTTPException(status_code=404, detail="Демонстрационный CSV отсутствует.")
    return FileResponse(input_path, filename="portfolio-example.csv", media_type="text/csv")
