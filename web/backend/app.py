"""PaperOrchestra Web API — FastAPI application.

Endpoints:
  POST   /api/keys              — store an API key
  GET    /api/keys              — list stored key names
  DELETE /api/keys/{name}       — remove a key

  POST   /api/uploads           — upload files (templates, scripts, data)
  GET    /api/uploads           — list uploaded files

  POST   /api/pipeline/start    — start a pipeline run
  GET    /api/pipeline/runs     — list all runs
  GET    /api/pipeline/runs/{id}— get a single run's status
  WS     /api/pipeline/ws/{id}  — stream live events for a run

  GET    /api/pipeline/runs/{id}/artifacts  — list output files
  GET    /api/pipeline/runs/{id}/artifacts/{path} — download an artifact
  GET    /api/pipeline/runs/{id}/overleaf.zip — manuscript, bibliography and figures for Overleaf
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import aiofiles
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from paper_orchestra.utils.overleaf import NoManuscript, build_overleaf_zip
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from paper_orchestra.domains import DOMAINS

from .config import settings, vault
from .runner import _reuse_dir, colab_problem, get_event_queue, get_run, list_runs, start_pipeline
from .schemas import ApiKeyIn, ApiKeyOut, PipelineRunStatus, PipelineRunSummary, PipelineStartRequest, RunArtifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("po.api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PaperOrchestra",
    version="0.1.0",
    description="Web interface for the PaperOrchestra multi-agent research pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    logger.info("PaperOrchestra API ready  data_dir=%s", settings.data_dir)


# ===================================================================
# API-key vault
# ===================================================================

@app.post("/api/keys", response_model=ApiKeyOut, status_code=201)
async def set_key(body: ApiKeyIn) -> ApiKeyOut:
    """Store (or overwrite) an API key."""
    vault.set(body.name, body.value)
    return ApiKeyOut(name=body.name, is_set=True)


@app.get("/api/keys", response_model=list[ApiKeyOut])
async def list_keys() -> list[ApiKeyOut]:
    return [ApiKeyOut(name=n, is_set=True) for n in vault.list_keys()]


@app.delete("/api/keys/{name}", status_code=204)
async def delete_key(name: str) -> None:
    if not vault.has(name):
        raise HTTPException(404, f"Key '{name}' not found")
    vault.delete(name)


# ===================================================================
# File uploads (templates, experiment scripts, datasets)
# ===================================================================

@app.post("/api/uploads", status_code=201)
async def upload_files(files: list[UploadFile]) -> list[dict]:
    """Accept one or more file uploads, store in the uploads dir."""
    results = []
    for f in files:
        dest = settings.uploads_dir / f.filename
        async with aiofiles.open(dest, "wb") as out:
            while chunk := await f.read(1024 * 256):
                await out.write(chunk)
        results.append({"filename": f.filename, "size": dest.stat().st_size})
        logger.info("Uploaded %s (%d bytes)", f.filename, dest.stat().st_size)
    return results


@app.get("/api/uploads")
async def list_uploads() -> list[dict]:
    return [
        {"filename": p.name, "size": p.stat().st_size}
        for p in sorted(settings.uploads_dir.iterdir())
        if p.is_file()
    ]


# ===================================================================
# Pipeline runs
# ===================================================================

@app.post("/api/pipeline/start", status_code=202)
async def pipeline_start(req: PipelineStartRequest) -> dict:
    """Start a new pipeline run. Returns immediately with the run_id."""
    er = req.experiment_requirements
    if er and (er.dataset.domain not in DOMAINS or er.dataset.source not in ("kaggle", "zenodo", "local")):
        raise HTTPException(400, f"unknown dataset domain {er.dataset.domain!r} or source {er.dataset.source!r}; "
                                 f"domains: {', '.join(DOMAINS)}; sources: kaggle, zenodo, local")
    if er and er.reuse_run_id:
        try:
            _reuse_dir(er.reuse_run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    if er and er.colab.enabled:  # fail now rather than after the outline step
        problem = await asyncio.to_thread(colab_problem)
        if problem:
            raise HTTPException(409, f"Colab training is enabled but {problem}. Fix that, or untick Google Colab.")
    run_id = start_pipeline(req, asyncio.get_running_loop())
    return {"run_id": run_id, "status": "queued"}


@app.get("/api/pipeline/runs", response_model=list[PipelineRunSummary])
async def pipeline_list() -> list[PipelineRunSummary]:
    runs = list_runs()
    return [
        PipelineRunSummary(
            run_id=r.run_id,
            status=r.status,
            idea_title=r.idea_title,
            started_at=r.started_at,
            finished_at=r.finished_at,
        )
        for r in runs
    ]


@app.get("/api/pipeline/runs/{run_id}", response_model=PipelineRunStatus)
async def pipeline_status(run_id: str) -> PipelineRunStatus:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return run


# ===================================================================
# WebSocket — live pipeline events
# ===================================================================

@app.websocket("/api/pipeline/ws/{run_id}")
async def pipeline_ws(ws: WebSocket, run_id: str) -> None:
    """Stream real-time events for a pipeline run.

    Events are JSON objects with a 'type' field:
      pipeline_started, agent_update, log, pipeline_completed,
      pipeline_failed, done.
    """
    run = get_run(run_id)
    if run is None:
        await ws.close(code=4004, reason="run not found")
        return

    await ws.accept()
    queue = get_event_queue(run_id)
    if queue is None:
        await ws.send_json({"type": "error", "message": "No event queue for run"})
        await ws.close()
        return

    try:
        while True:
            event = await queue.get()
            await ws.send_json(event)
            if event.get("type") == "done":
                break
    except WebSocketDisconnect:
        logger.info("WS client disconnected for run %s", run_id)
    except Exception:
        logger.exception("WS error for run %s", run_id)
    finally:
        await ws.close()


# ===================================================================
# Run artifacts (output files)
# ===================================================================

@app.get("/api/pipeline/runs/{run_id}/artifacts", response_model=list[RunArtifact])
async def list_artifacts(run_id: str) -> list[RunArtifact]:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run '{run_id}' not found")
    out_dir = Path(run.output_dir) if run.output_dir else None
    if out_dir is None or not out_dir.exists():
        return []
    artifacts = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            artifacts.append(RunArtifact(
                filename=str(p.relative_to(out_dir)),
                size=p.stat().st_size,
                url=f"/api/pipeline/runs/{run_id}/artifacts/{p.relative_to(out_dir)}",
            ))
    return artifacts


@app.get("/api/pipeline/runs/{run_id}/artifacts/{path:path}")
async def download_artifact(run_id: str, path: str) -> FileResponse:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run '{run_id}' not found")
    out_dir = Path(run.output_dir) if run.output_dir else None
    if out_dir is None:
        raise HTTPException(404, "No output directory")
    file_path = (out_dir / path).resolve()
    # Prevent path traversal
    if not str(file_path).startswith(str(out_dir.resolve())):
        raise HTTPException(400, "Invalid path")
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {path}")
    return FileResponse(file_path, filename=file_path.name)


@app.get("/api/pipeline/runs/{run_id}/overleaf.zip")
async def download_overleaf_zip(run_id: str) -> Response:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run '{run_id}' not found")
    out_dir = Path(run.output_dir) if run.output_dir else None
    if out_dir is None or not out_dir.exists():
        raise HTTPException(404, "No output directory")
    try:
        data = build_overleaf_zip(out_dir)
    except NoManuscript as exc:
        raise HTTPException(409, str(exc))
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="paper-{run_id}-overleaf.zip"'},
    )
