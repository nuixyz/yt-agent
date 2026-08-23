from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from browser.session import camoufox_session
from config import settings
from graph.state import AgentState, StudyNote, VideoError, VideoRef
from llm.summarizer import summarize_transcript

logger = logging.getLogger(__name__)

PLAYLIST_ITEM_SELECTOR = "ytd-playlist-video-renderer"
PLAYLIST_ITEM_TITLE_SELECTOR = "#video-title"
TRANSCRIPT_BUTTON_SELECTOR = 'button[aria-label*="transcript" i]'
TRANSCRIPT_MORE_ACTIONS_SELECTOR = "button#expand"  # "...more" under description
TRANSCRIPT_SEGMENT_SELECTOR = "ytd-transcript-segment-renderer"
TRANSCRIPT_PANEL_SELECTOR = "ytd-transcript-segment-list-renderer"

DEFAULT_WAIT_MS = 15000


def _video_id_from_url(url: str) -> str | None:
    match = re.search(r"[?&]v=([\w-]+)", url)
    return match.group(1) if match else None


# Node: navigate_playlist
async def navigate_playlist(state: AgentState) -> dict:
    logger.info("Navigating to playlist: %s", state.playlist_url)
    async with camoufox_session() as page:
        try:
            await page.goto(state.playlist_url, wait_until="domcontentloaded")
            await page.wait_for_selector(
                PLAYLIST_ITEM_SELECTOR, timeout=DEFAULT_WAIT_MS
            )
        except PlaywrightTimeoutError as e:
            return {
                "errors": [
                    VideoError(
                        stage="navigate_playlist",
                        message=f"Playlist page did not load video items in time: {e}",
                    )
                ]
            }
    return {}


# Node: extract_video_list
async def extract_video_list(state: AgentState) -> dict:
    videos: list[VideoRef] = []

    async with camoufox_session() as page:
        await page.goto(state.playlist_url, wait_until="domcontentloaded")
        try:
            await page.wait_for_selector(
                PLAYLIST_ITEM_SELECTOR, timeout=DEFAULT_WAIT_MS
            )
        except PlaywrightTimeoutError as e:
            return {
                "errors": [
                    VideoError(
                        stage="extract_video_list",
                        message=f"No playlist items found: {e}",
                    )
                ]
            }

        items = await page.query_selector_all(PLAYLIST_ITEM_SELECTOR)
        for position, item in enumerate(items, start=1):
            if len(videos) >= settings.max_videos_per_run:
                break
            try:
                title_el = await item.query_selector(PLAYLIST_ITEM_TITLE_SELECTOR)
                if title_el is None:
                    continue
                title = (await title_el.inner_text()).strip()
                href = await title_el.get_attribute("href")
                if not href:
                    continue
                url = href if href.startswith("http") else f"https://www.youtube.com{href}"
                video_id = _video_id_from_url(url)
                if not video_id:
                    continue
                videos.append(
                    VideoRef(video_id=video_id, url=url, title=title, position=position)
                )
            except Exception as e:  # skip upon finding a bad item in the playlist
                logger.warning("Skipping unparsable playlist item at position %d: %s", position, e)
                continue

    if not videos:
        return {
            "errors": [
                VideoError(
                    stage="extract_video_list",
                    message="Playlist parsed but no valid videos were extracted.",
                )
            ]
        }

    logger.info(
        "Discovered %d videos (capped at %d)", len(videos), settings.max_videos_per_run
    )
    return {
        "videos": videos,
        "pending_video_ids": [v.video_id for v in videos],
    }


# Node: open_video
async def open_video(state: AgentState) -> dict:
    if not state.pending_video_ids:
        return {}

    next_id = state.pending_video_ids[0]
    remaining = state.pending_video_ids[1:]
    video = next((v for v in state.videos if v.video_id == next_id), None)

    if video is None:
        return {
            "pending_video_ids": remaining,
            "errors": [
                VideoError(
                    video_id=next_id,
                    stage="open_video",
                    message="Video ID in queue but missing from discovered videos list.",
                )
            ],
        }

    return {"pending_video_ids": remaining, "current_video": video}


# Node: extract_transcript
async def extract_transcript(state: AgentState) -> dict:
    video = state.current_video
    if video is None:
        return {}

    async with camoufox_session() as page:
        try:
            await page.goto(video.url, wait_until="domcontentloaded")
            try:
                expand_btn = await page.wait_for_selector(
                    TRANSCRIPT_MORE_ACTIONS_SELECTOR, timeout=5000
                )
                if expand_btn:
                    await expand_btn.click()
            except PlaywrightTimeoutError:
                pass

            transcript_btn = await page.wait_for_selector(
                TRANSCRIPT_BUTTON_SELECTOR, timeout=DEFAULT_WAIT_MS
            )
            await transcript_btn.click()

            await page.wait_for_selector(
                TRANSCRIPT_PANEL_SELECTOR, timeout=DEFAULT_WAIT_MS
            )
            await page.wait_for_selector(
                TRANSCRIPT_SEGMENT_SELECTOR, timeout=DEFAULT_WAIT_MS
            )

            segments = await page.query_selector_all(TRANSCRIPT_SEGMENT_SELECTOR)
            lines: list[str] = []
            for seg in segments:
                text = (await seg.inner_text()).strip()
                if text:
                    lines.append(text.replace("\n", " "))

            transcript = "\n".join(lines)

        except PlaywrightTimeoutError as e:
            return {
                "current_video": None,
                "errors": [
                    VideoError(
                        video_id=video.video_id,
                        url=video.url,
                        stage="extract_transcript",
                        message=f"Transcript unavailable or failed to load: {e}",
                    )
                ],
            }

    if not transcript:
        return {
            "current_video": None,
            "errors": [
                VideoError(
                    video_id=video.video_id,
                    url=video.url,
                    stage="extract_transcript",
                    message="Transcript panel loaded but no text was extracted.",
                )
            ],
        }

    return {"current_transcript": transcript}


# Node: summarize_node
async def summarize_node(state: AgentState) -> dict:
    """Calls the local LLM to turn the current transcript into a StudyNote."""
    video = state.current_video
    transcript = state.current_transcript
    if video is None or transcript is None:
        return {}

    try:
        note: StudyNote = await summarize_transcript(video=video, transcript=transcript)
    except Exception as e:  # LLM parsing failures doesn't kill the run
        return {
            "current_video": None,
            "current_transcript": None,
            "errors": [
                VideoError(
                    video_id=video.video_id,
                    url=video.url,
                    stage="summarize_node",
                    message=f"Summarization failed: {e}",
                )
            ],
        }

    return {
        "notes": [note],
        "current_video": None,
        "current_transcript": None,
    }


# Node: handle_error
async def handle_error(state: AgentState) -> dict:
    return {"current_video": None, "current_transcript": None}


# Node: write_markdown
async def write_markdown(state: AgentState) -> dict:
    """Renders all accumulated notes (and any errors) into a single Markdown
    digest and writes it to OUTPUT_DIR."""
    settings.ensure_dirs()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out_path = Path(settings.output_dir) / f"digest_{timestamp}.md"

    lines: list[str] = [
        f"# Study Digest — {timestamp} UTC",
        "",
        f"Playlist: {state.playlist_url}",
        f"Videos summarized: {len(state.notes)} / {len(state.videos)} discovered",
        "",
    ]

    for note in state.notes:
        lines.append(f"## {note.title}")
        lines.append(f"[Watch]({note.url})")
        lines.append("")
        lines.append(note.summary)
        lines.append("")
        if note.key_concepts:
            lines.append("**Key concepts:**")
            lines.extend(f"- {c}" for c in note.key_concepts)
            lines.append("")
        if note.timestamps:
            lines.append("**Notable moments:**")
            lines.extend(f"- {t}" for t in note.timestamps)
            lines.append("")
        if note.action_items:
            lines.append("**Action items:**")
            lines.extend(f"- [ ] {a}" for a in note.action_items)
            lines.append("")
        lines.append("---")
        lines.append("")

    if state.errors:
        lines.append("## Errors")
        for err in state.errors:
            ref = err.url or err.video_id or "unknown video"
            lines.append(f"- `{err.stage}` — {ref}: {err.message}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote digest to %s", out_path)

    return {"output_path": str(out_path)}
