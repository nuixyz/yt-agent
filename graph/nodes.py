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

PLAYLIST_ITEM_STRATEGIES: list[dict[str, str]] = [
    {
        "item": "yt-lockup-view-model",
        "title": "a.ytLockupMetadataViewModelTitle",
    },
    {
        "item": "ytd-playlist-video-renderer",
        "title": "#video-title",
    },
]
SELECTOR_DETECT_TIMEOUT_MS = 6000
DEFAULT_WAIT_MS = 15000

TRANSCRIPT_BUTTON_SELECTOR = 'button[aria-label*="transcript" i]'


async def _find_expanded_panel(page: Page, selector: str, *, timeout_ms: int = DEFAULT_WAIT_MS):
    deadline = timeout_ms / 1000
    elapsed = 0.0
    interval = 0.5
    while elapsed < deadline:
        candidates = await page.query_selector_all(selector)
        for el in candidates:
            try:
                visibility_attr = await el.get_attribute("visibility")
                if visibility_attr == "ENGAGEMENT_PANEL_VISIBILITY_EXPANDED":
                    return el
            except Exception:
                continue
        await page.wait_for_timeout(int(interval * 1000))
        elapsed += interval

    return await _find_first_visible(page, selector, timeout_ms=2000)


async def _find_first_visible(page: Page, selector: str, *, timeout_ms: int = DEFAULT_WAIT_MS):
    deadline = timeout_ms / 1000
    elapsed = 0.0
    interval = 0.5
    while elapsed < deadline:
        candidates = await page.query_selector_all(selector)
        for el in candidates:
            try:
                if await el.is_visible():
                    return el
            except Exception:
                continue
        await page.wait_for_timeout(int(interval * 1000))
        elapsed += interval
    return None


async def _click_first_visible(
    page: Page, selector: str, *, timeout_ms: int = DEFAULT_WAIT_MS
) -> bool:
    deadline = timeout_ms / 1000
    elapsed = 0.0
    interval = 0.5
    while elapsed < deadline:
        candidates = await page.query_selector_all(selector)
        for el in candidates:
            try:
                if await el.is_visible():
                    await el.scroll_into_view_if_needed()
                    await el.click()
                    return True
            except Exception:
                continue
        await page.wait_for_timeout(int(interval * 1000))
        elapsed += interval
    return False


EXPAND_DESCRIPTION_SELECTORS = [
    "#expand",
    "tp-yt-paper-button#expand",
    "button:has-text('...more')",
    "button:has-text('more')",
]


async def _expand_description_if_present(page: Page, *, timeout_ms: int = 4000) -> None:
    for selector in EXPAND_DESCRIPTION_SELECTORS:
        try:
            btn = await page.wait_for_selector(selector, timeout=timeout_ms)
            if btn and await btn.is_visible():
                await btn.click()
                logger.info("Expanded description via selector: %s", selector)
                await page.wait_for_timeout(500)
                return
        except PlaywrightTimeoutError:
            continue
        except Exception as e:  # noqa: BLE001 - never let this block the real flow
            logger.debug("Expand-description attempt failed for %s: %s", selector, e)
            continue


TRANSCRIPT_PANEL_SELECTOR = "ytd-engagement-panel-section-list-renderer[target-id*='transcript']"
TIMESTAMP_LINE_RE = re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?)\s*\n?\s*([^\n]+)")
MIN_PANEL_TEXT_LENGTH = 60


CONSENT_BUTTON_SELECTORS = [
    "button:has-text('Accept all')",
    "button:has-text('I agree')",
    "form[action*='consent'] button",
    "#introAgreeButton",
]

DEBUG_DIR = Path("./outputs/_debug")


async def _detect_playlist_strategy(page: Page) -> dict[str, str] | None:
    for strategy in PLAYLIST_ITEM_STRATEGIES:
        try:
            await page.wait_for_selector(
                strategy["item"], timeout=SELECTOR_DETECT_TIMEOUT_MS
            )
            logger.info("Playlist item selector matched: %s", strategy["item"])
            return strategy
        except PlaywrightTimeoutError:
            continue
    return None


async def _dismiss_consent_if_present(page: Page, *, timeout_ms: int = 3000) -> None:
    for selector in CONSENT_BUTTON_SELECTORS:
        try:
            btn = await page.wait_for_selector(selector, timeout=timeout_ms)
            if btn:
                await btn.click()
                logger.info("Dismissed a consent dialog via selector: %s", selector)
                await page.wait_for_timeout(1000)  # let the dialog close
                return
        except PlaywrightTimeoutError:
            continue
        except Exception as e:
            logger.debug("Consent dismissal attempt failed for %s: %s", selector, e)
            continue


async def _capture_debug_artifacts(page: Page, *, stage: str, video_id: str = "playlist") -> str | None:
    """Saves a screenshot + HTML snapshot on failure so timeouts are
    diagnosable after the fact instead of a bare 'timeout exceeded'.
    Returns the screenshot path, or None if capture itself failed."""
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base = DEBUG_DIR / f"{stage}_{video_id}_{timestamp}"
        screenshot_path = f"{base}.png"
        html_path = f"{base}.html"

        await page.screenshot(path=screenshot_path, full_page=True)
        html = await page.content()
        Path(html_path).write_text(html, encoding="utf-8")

        logger.info("Saved debug artifacts: %s , %s", screenshot_path, html_path)
        return screenshot_path
    except Exception as e:
        logger.warning("Failed to capture debug artifacts: %s", e)
        return None


def _video_id_from_url(url: str) -> str | None:
    match = re.search(r"[?&]v=([\w-]+)", url)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Node: navigate_playlist
# ---------------------------------------------------------------------------
async def navigate_playlist(state: AgentState) -> dict:
    """Opens the playlist page. Actual scraping happens in extract_video_list
    (kept separate so navigation failures are distinguishable from parse
    failures in logs/errors)."""
    logger.info("Navigating to playlist: %s", state.playlist_url)
    async with camoufox_session() as page:
        try:
            await page.goto(state.playlist_url, wait_until="domcontentloaded")
            await _dismiss_consent_if_present(page)
            strategy = await _detect_playlist_strategy(page)
            if strategy is None:
                raise PlaywrightTimeoutError(
                    "None of the known playlist-item selectors matched."
                )
        except PlaywrightTimeoutError as e:
            debug_path = await _capture_debug_artifacts(page, stage="navigate_playlist")
            hint = f" (screenshot saved to {debug_path})" if debug_path else ""
            return {
                "errors": [
                    VideoError(
                        stage="navigate_playlist",
                        message=(
                            f"Playlist page did not load video items in time: {e}{hint}"
                        ),
                    )
                ]
            }
    return {}


# ---------------------------------------------------------------------------
# Node: extract_video_list
# ---------------------------------------------------------------------------
async def extract_video_list(state: AgentState) -> dict:
    """Scrapes video refs from the playlist page, applies the configured cap,
    and seeds the processing queue."""
    videos: list[VideoRef] = []

    async with camoufox_session() as page:
        await page.goto(state.playlist_url, wait_until="domcontentloaded")
        await _dismiss_consent_if_present(page)

        strategy = await _detect_playlist_strategy(page)
        if strategy is None:
            debug_path = await _capture_debug_artifacts(page, stage="extract_video_list")
            hint = f" (screenshot saved to {debug_path})" if debug_path else ""
            return {
                "errors": [
                    VideoError(
                        stage="extract_video_list",
                        message=f"No playlist items found (all selector strategies failed){hint}",
                    )
                ]
            }

        items = await page.query_selector_all(strategy["item"])
        for position, item in enumerate(items, start=1):
            if len(videos) >= settings.max_videos_per_run:
                break
            try:
                title_el = await item.query_selector(strategy["title"])
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
            except Exception as e:
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


# ---------------------------------------------------------------------------
# Node: open_video
# ---------------------------------------------------------------------------
async def open_video(state: AgentState) -> dict:
    """Pops the next pending video ID off the queue and sets it as current."""
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


# ---------------------------------------------------------------------------
# Node: extract_transcript
# ---------------------------------------------------------------------------
async def extract_transcript(state: AgentState) -> dict:
    """Opens the current video and extracts its transcript via the DOM,
    using dynamic waits rather than fixed sleeps at every step."""
    video = state.current_video
    if video is None:
        return {}

    async with camoufox_session() as page:
        try:
            await page.goto(video.url, wait_until="domcontentloaded")
            try:
                await page.wait_for_selector("ytd-watch-metadata", timeout=DEFAULT_WAIT_MS)
            except PlaywrightTimeoutError:
                pass
            await _expand_description_if_present(page)

            clicked = await _click_first_visible(page, TRANSCRIPT_BUTTON_SELECTOR)
            if not clicked:
                raise PlaywrightTimeoutError(
                    "No visible 'Show transcript' button found "
                    "(matches existed in the DOM but none were visible)."
                )

            panel = await _find_expanded_panel(page, TRANSCRIPT_PANEL_SELECTOR)
            if panel is None:
                raise PlaywrightTimeoutError(
                    "Transcript panel did not become visible after clicking "
                    "'Show transcript'."
                )

            try:
              await page.wait_for_function(
                """(el) => {
                const text = el.innerText || '';
                const stillLoading = el.querySelector('tp-yt-paper-spinner[active]');
                return !stillLoading && text.trim().length > %d;
                }"""
                % MIN_PANEL_TEXT_LENGTH,
                arg=panel,
                timeout=30000,
              )
            except PlaywrightTimeoutError as e:
              raise PlaywrightTimeoutError(
                f"Transcript panel opened but content never finished loading "
                f"(stuck on loading spinner past 30000ms): {e}"
              )

            panel_text = await panel.inner_text()
            matches = TIMESTAMP_LINE_RE.findall(panel_text)
            lines = [
                f"{ts.strip()} - {text.strip()}"
                for ts, text in matches
                if text.strip()
            ]
            transcript = "\n".join(lines)

        except PlaywrightTimeoutError as e:
            debug_path = await _capture_debug_artifacts(
                page, stage="extract_transcript", video_id=video.video_id
            )
            hint = f" (screenshot saved to {debug_path})" if debug_path else ""
            return {
                "current_video": None,
                "errors": [
                    VideoError(
                        video_id=video.video_id,
                        url=video.url,
                        stage="extract_transcript",
                        message=f"Transcript unavailable or failed to load: {e}{hint}",
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


# ---------------------------------------------------------------------------
# Node: summarize_node
# ---------------------------------------------------------------------------
async def summarize_node(state: AgentState) -> dict:
    """Calls the local LLM to turn the current transcript into a StudyNote."""
    video = state.current_video
    transcript = state.current_transcript
    if video is None or transcript is None:
        return {}

    try:
        note: StudyNote = await summarize_transcript(video=video, transcript=transcript)
    except Exception as e:
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


# ---------------------------------------------------------------------------
# Node: handle_error
# ---------------------------------------------------------------------------
async def handle_error(state: AgentState) -> dict:
    """Clears per-video scratch state so the loop can continue to the next
    pending video after a failure. Errors themselves are already appended
    by whichever node raised them."""
    return {"current_video": None, "current_transcript": None}


# ---------------------------------------------------------------------------
# Node: write_markdown
# ---------------------------------------------------------------------------
async def write_markdown(state: AgentState) -> dict:
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
