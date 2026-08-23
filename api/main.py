from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import settings
from graph.build import app_graph
from graph.state import AgentState

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_dirs()
    if settings.scheduler_enabled:
        if not settings.default_playlist_url:
            logger.warning(
                "SCHEDULER_ENABLED is true but DEFAULT_PLAYLIST_URL is unset; "
                "scheduled job will be skipped."
            )
        else:
            scheduler.add_job(
                _run_scheduled_job,
                CronTrigger(hour=settings.scheduler_cron_hour, minute=0),
                id="daily_playlist_digest",
                replace_existing=True,
            )
            scheduler.start()
            logger.info(
                "Scheduler started: daily run at %02d:00 for %s",
                settings.scheduler_cron_hour,
                settings.default_playlist_url,
            )
    yield
    if scheduler.running:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="YouTube Study Agent",
    description="Summarizes a YouTube study playlist into structured Markdown notes.",
    version="0.1.0",
    lifespan=lifespan,
)

# Schemas
class TriggerRequest(BaseModel):
    playlist_url: str = Field(
        description="Full URL of the YouTube playlist to summarize.",
        examples=["https://www.youtube.com/playlist?list=PLxxxxxxxxxxxxxxxx"],
    )


class VideoErrorOut(BaseModel):
    video_id: str | None = None
    url: str | None = None
    stage: str
    message: str


class TriggerResponse(BaseModel):
    output_path: str
    videos_discovered: int
    videos_summarized: int
    errors: list[VideoErrorOut]
    digest_markdown: str


# Core run logic
async def _run_pipeline(playlist_url: str) -> TriggerResponse:
    initial_state = AgentState(playlist_url=playlist_url)
    result = await app_graph.ainvoke(initial_state)

    output_path = result.get("output_path")
    if not output_path:
        raise RuntimeError("Graph completed without producing an output_path.")

    digest_text = Path(output_path).read_text(encoding="utf-8")

    return TriggerResponse(
        output_path=output_path,
        videos_discovered=len(result.get("videos", [])),
        videos_summarized=len(result.get("notes", [])),
        errors=[
            VideoErrorOut(**(e if isinstance(e, dict) else e.model_dump()))
            for e in result.get("errors", [])
        ],
        digest_markdown=digest_text,
    )


async def _run_scheduled_job() -> None:
    if not settings.default_playlist_url:
        return
    logger.info("Running scheduled digest for %s", settings.default_playlist_url)
    try:
        response = await _run_pipeline(settings.default_playlist_url)
        logger.info(
            "Scheduled run complete: %d/%d videos summarized, %d errors -> %s",
            response.videos_summarized,
            response.videos_discovered,
            len(response.errors),
            response.output_path,
        )
    except Exception:
        logger.exception("Scheduled digest run failed")


# Routes
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/trigger", response_model=TriggerResponse)
async def trigger(request: TriggerRequest) -> TriggerResponse:
    try:
        return await _run_pipeline(request.playlist_url)
    except Exception as e:  # noqa: BLE001 - surface as a clean 500 to the caller
        logger.exception("Pipeline run failed for %s", request.playlist_url)
        raise HTTPException(status_code=500, detail=str(e)) from e
