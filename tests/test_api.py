"""HTTP contract for endpoints that do not need an LLM call."""

import pathlib

import pytest
from fastapi.testclient import TestClient

from ec_analysis.api.app import app, create_app
from ec_analysis.config import AppSettings


def test_health_endpoint() -> None:
    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ag_ui_requires_configured_provider() -> None:
    application = create_app(AppSettings(llm_model="", openrouter_api_key=None))
    with TestClient(application) as test_client:
        response = test_client.post(
            "/ag-ui",
            json={"threadId": "t", "runId": "r", "messages": [], "tools": [], "context": [], "forwardedProps": {}},
        )

    assert response.status_code == 503


def test_benchmark_endpoint_returns_balanced_provisional_state(
    monkeypatch: pytest.MonkeyPatch, portfolio_csv: pathlib.Path
) -> None:
    monkeypatch.setenv("EC_DATASET_PATH", str(portfolio_csv))
    with TestClient(app) as test_client:
        response = test_client.get("/benchmark")

    assert response.status_code == 200
    analysis_state = response.json()
    assert analysis_state["selected_model"] in {"mean", "ridge", "gam_spline"}
    assert analysis_state["final_test"]["scope"] == "test"
    assert analysis_state["confidence_level"] == "low"
    contribution_total = sum(item["contribution"] for item in analysis_state["factor_contributions"])
    contribution_total += sum(item["contribution"] for item in analysis_state["interactions"])
    assert analysis_state["total_change"] == pytest.approx(contribution_total + analysis_state["unexplained_remainder"])
    assert analysis_state["selection_status"] == "attribution_checked"


def test_benchmark_endpoint_reports_missing_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setenv("EC_DATASET_PATH", str(tmp_path / "absent.csv"))
    with TestClient(app) as test_client:
        response = test_client.get("/benchmark")

    assert response.status_code == 422


@pytest.mark.parametrize("thorough", [False, True])
def test_uploaded_job_completes_and_exports_only_checked_files(
    tmp_path: pathlib.Path, portfolio_csv: pathlib.Path, thorough: bool
) -> None:
    import time

    application = create_app(AppSettings(jobs_directory=str(tmp_path / "jobs")))
    with TestClient(application) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        invalid = client.post("/api/analyses", content=b"id,period,npv\n1,T0,5")
        assert invalid.status_code == 422
        response = client.post(f"/api/analyses?thorough={str(thorough).lower()}", content=portfolio_csv.read_bytes())
        assert response.status_code == 202
        identifier = response.json()["identifier"]
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            status = client.get(f"/api/analyses/{identifier}").json()
            if status["status"] != "running":
                break
            time.sleep(0.1)
        assert status["status"] == "completed", status
        result = client.get(f"/api/analyses/{identifier}/result").json()
        assert result["dataset_sha256"]
        assert result["selected_model"] == "stable_gam"
        assert len(result["benchmark_summary"]) == 1
        selected = result["benchmark_summary"][0]
        assert len(selected["tuning_trials"]) == 3
        assert selected["stability"]["requested_repeats"] == 20
        assert client.get(f"/api/analyses/{identifier}/files/sections.json").status_code == 200
        assert client.get(f"/api/analyses/{identifier}/files/analysis.json").json() == result
        assert client.get(f"/api/analyses/{identifier}/files/model-ridge.joblib").status_code == 404
        assert client.get("/api/analyses/unknown").status_code == 404
        assert client.get(f"/api/analyses/{identifier}/files/worker.log").status_code == 404


def test_job_cancellation_timeout_and_concurrency_limit(tmp_path: pathlib.Path, portfolio_csv: pathlib.Path) -> None:
    from ec_analysis.api.jobs import JobManager

    manager = JobManager(root_directory=tmp_path / "jobs", project_root=pathlib.Path.cwd())
    try:
        information = manager.submit(portfolio_csv.read_bytes(), "npv", True)
        with pytest.raises(RuntimeError, match="Уже"):
            manager.submit(portfolio_csv.read_bytes(), "npv", False)
        assert manager.cancel_job(information.identifier).status == "cancelled"
        assert manager.jobs[information.identifier].process.poll() is not None
        manager.timeout_seconds = 0
        information = manager.submit(portfolio_csv.read_bytes(), "npv", True)
        assert manager.read_job(information.identifier).status == "timed_out"
        assert manager.jobs[information.identifier].process.poll() is not None
        manager.maximum_bytes = 10
        with pytest.raises(ValueError, match="MiB"):
            manager.submit(portfolio_csv.read_bytes(), "npv", False)
    finally:
        manager.shutdown()
