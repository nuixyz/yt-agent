# YouTube Study Agent

An autonomous browser agent that logs into YouTube, opens a study playlist, extracts each video's transcript, and uses a local LLM (via Ollama) to generate structured Markdown study notes — orchestrated end-to-end with LangGraph and exposed through a FastAPI trigger endpoint.

Built for the "Autonomous Browser Agent with Camoufox, LangGraph & Local/Fine-Tuned LLMs" take-home challenge (Option 2: Study / Content Reader).

---

## Table of contents

- [What this does](#what-this-does)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Clone and install dependencies](#1-clone-and-install-dependencies)
  - [2. Install the Camoufox browser](#2-install-the-camoufox-browser)
  - [3. Install and pull a local LLM via Ollama](#3-install-and-pull-a-local-llm-via-ollama)
  - [4. Configure environment variables](#4-configure-environment-variables)
  - [5. One-time interactive login](#5-one-time-interactive-login)
- [Running the agent](#running-the-agent)
  - [Via the API (recommended)](#via-the-api-recommended)
  - [Optional: background daily scheduler](#optional-background-daily-scheduler)
- [Configuration reference](#configuration-reference)
- [How the pipeline works](#how-the-pipeline-works)
- [Output format](#output-format)
- [Error handling & resilience](#error-handling--resilience)
- [Design decisions](#design-decisions)
- [Troubleshooting](#troubleshooting)
- [Limitations & future work](#limitations--future-work)

---

## What this does

Point it at a YouTube playlist URL, and it will:

1. Launch **Camoufox** (an anti-detect browser) reusing a persistent, already-authenticated session.
2. Open the playlist and discover videos (capped to a configurable number per run - default: 5).
3. For each video: open it, expand and extract its transcript from the DOM, and pass the transcript to a **local LLM** running in Ollama.
4. The LLM returns a structured note — summary, key concepts, notable timestamps, and action items — validated against a strict schema.
5. All notes (plus any per-video errors) are compiled into a single **Markdown digest** and saved to `outputs/`.

The entire flow is orchestrated as a **LangGraph** state machine, and triggered via a **FastAPI** `POST /trigger` endpoint that accepts any playlist URL at request time — no restart or redeploy needed to point it at a different playlist.

---

## Architecture

```
                POST /trigger {"playlist_url": "..."}
                              │
                              ▼
                    ┌───────────────────┐
                    │  navigate_playlist│
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │ extract_video_list│  (applies MAX_VIDEOS_PER_RUN cap)
                    └─────────┬─────────┘
                              │
                   no videos ◄┴► videos found
                       │              │
                       ▼              ▼
                write_markdown   ┌──────────┐
                       ▲         │open_video│◄────────────────┐
                       │         └────┬─────┘                 │
                       │              ▼                        │
                       │    ┌───────────────────┐               │
                       │    │ extract_transcript│               │
                       │    └─────────┬─────────┘               │
                       │        fail  │  success                │
                       │         ┌────┴────┐                    │
                       │         ▼         ▼                    │
                       │  handle_error  summarize_node           │
                       │         │         │  (success or fail)  │
                       │         └────┬────┘                    │
                       │              ▼                         │
                       │        route_next_step                  │
                       │         │           │                   │
                       │    queue empty   queue non-empty ───────┘
                       │         │
                       └─────────┘
                              │
                              ▼
                             END
```

**a failed video never blocks the rest of the playlist.** If a transcript can't be extracted or the LLM output can't be parsed, that failure is recorded as a structured error and the graph immediately moves to the next queued video — no retries, no aborts.

---

## Project structure

```
youtube-study-agent/
├── pyproject.toml           # dependencies & project metadata
├── requirements.txt          # plain pip equivalent
├── .env.example               # all configurable settings, documented
├── config.py                   # pydantic-settings config, loaded from .env
├── browser/
│   └── session.py                # Camoufox launch, persistent profile, interactive login
├── graph/
│   ├── state.py                    # AgentState / VideoRef / StudyNote / VideoError schemas
│   ├── nodes.py                     # the 7 node functions
│   └── build.py                      # StateGraph wiring, conditional routing, compiled app
├── llm/
│   └── summarizer.py                  # Ollama call, prompt, JSON extraction + retry
├── api/
│   └── main.py                          # FastAPI app: POST /trigger, GET /health, scheduler
├── outputs/                                # generated Markdown digests land here
```

---

## Prerequisites

- **Python 3.11+**
- **Ollama** installed and running locally ([ollama.com](https://ollama.com))
- A Google account you're comfortable using for the browser session (YouTube login)
- ~4–8 GB free RAM for a 7B-class local model (less if you pick a smaller model)

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd youtube-study-agent

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# or, if you prefer the pyproject-based install:
pip install -e .
```

### 2. Install the Camoufox browser

Camoufox ships its own patched Firefox build, fetched separately from the Python package:

```bash
camoufox fetch
```

This downloads the anti-detect browser binary Camoufox needs. You only need to do this once per machine.

### 3. Install and pull a local LLM via Ollama

If you haven't already, install Ollama, then pull the default model this project expects:

```bash
ollama pull qwen2.5:7b
```

Make sure the Ollama server is running (it typically runs as a background service after install; otherwise start it with `ollama serve`).

You can use a different model — see [Configuration reference](#configuration-reference).

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and adjust as needed. Every setting has a sane default (see the [Configuration reference](#configuration-reference) below); the only one you're likely to want to change immediately is `OLLAMA_MODEL` if you pulled a different model.

### 5. One-time interactive login

Before the agent can access your playlists, it needs an authenticated session. This is a **one-time, manual step** — a visible browser window opens, you sign in (and complete 2FA if prompted), and the resulting session is saved to a persistent profile directory for all future runs.

```bash
python -m browser.session --login
```

- A Firefox window (Camoufox) will open to the Google sign-in page.
- Sign in normally, complete any 2FA challenge YouTube/Google presents.
- The script polls in the background and will print a confirmation and close automatically once it detects you're signed in.
- If it times out (10 minutes), just re-run the command.

You can verify the saved session is still valid at any time (headless, no window) with:

```bash
python -m browser.session --check
```

If this ever reports "Not signed in," just re-run `--login`.

---

## Running the agent

### Via the API (recommended)

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

Then trigger a run against any playlist:

```bash
curl -X POST http://localhost:8000/trigger \
  -H "Content-Type: application/json" \
  -d '{"playlist_url": "https://www.youtube.com/playlist?list=PLxxxxxxxxxxxxxxxx"}'
```

The response includes the discovery/summarization counts, any per-video errors, the path to the saved digest, and the full digest content inline:

```json
{
  "output_path": "outputs/digest_2026-08-22_101530.md",
  "videos_discovered": 5,
  "videos_summarized": 4,
  "errors": [
    {
      "video_id": "abc123",
      "url": "https://www.youtube.com/watch?v=abc123",
      "stage": "extract_transcript",
      "message": "Transcript unavailable or failed to load: ..."
    }
  ],
  "digest_markdown": "# Study Digest — 2026-08-22_101530 UTC\n\n..."
}
```

You can also check the server is up with:

```bash
curl http://localhost:8000/health
```

> **Note:** `/trigger` runs synchronously — the request blocks until the whole playlist (up to `MAX_VIDEOS_PER_RUN` videos) finishes processing. For the default cap of 5 videos this is typically well under a couple of minutes, depending on your LLM's speed.

### Optional: background daily scheduler

If you'd rather this run automatically once a day against a fixed playlist (instead of, or in addition to, calling `/trigger` manually), set in `.env`:

```bash
SCHEDULER_ENABLED=true
DEFAULT_PLAYLIST_URL=https://www.youtube.com/playlist?list=PLxxxxxxxxxxxxxxxx
SCHEDULER_CRON_HOUR=7
```

With the server running, a background job will fire daily at the configured hour and write its digest to `outputs/` the same way `/trigger` does. This is independent of `/trigger` — you can still call the endpoint manually for other playlists at any time.

---

## Configuration reference

All settings live in `config.py` (backed by `.env`). Every value has a default, so `.env` is optional except where noted.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model tag used for summarization. |
| `OLLAMA_HOST` | `http://localhost:11434` | Base URL of the local Ollama server. |
| `CAMOUFOX_PROFILE_DIR` | `./.camoufox-profile` | Persistent browser profile directory (cookies, session). |
| `CAMOUFOX_HEADLESS` | `true` | Runs headless for normal use; the login script overrides this to `false` regardless. |
| `MAX_VIDEOS_PER_RUN` | `5` | Cap on how many playlist videos are processed per run. |
| `DEFAULT_PLAYLIST_URL` | *(unset)* | Only used by the background scheduler, not `/trigger`. |
| `OUTPUT_DIR` | `./outputs` | Where generated Markdown digests are written. |
| `SCHEDULER_ENABLED` | `false` | Enables the daily background job. |
| `SCHEDULER_CRON_HOUR` | `7` | Hour (24h, local time) the scheduled job runs at. |

---

## How the pipeline works

The graph (`graph/build.py`) wires together 7 nodes (`graph/nodes.py`):

1. **`navigate_playlist`** — opens the playlist URL and confirms video items are present (dynamic wait, no fixed sleep).
2. **`extract_video_list`** — scrapes video refs (id, title, URL, position) from the DOM, applies the `MAX_VIDEOS_PER_RUN` cap, and seeds the processing queue.
3. **`open_video`** — pops the next video ID off the queue and sets it as "current."
4. **`extract_transcript`** — opens the video, expands the description, clicks "Show transcript," waits for the transcript panel *and* at least one segment to render (dynamic waits at each stage), then concatenates all segment text.
5. **`summarize_node`** — sends the transcript to the local LLM (`llm/summarizer.py`) with a strict JSON-only prompt; validates the response against the `StudyNote` schema; retries once with a stricter prompt if the first response isn't valid JSON.
6. **`handle_error`** — clears per-video scratch state so the loop can continue; the actual `VideoError` is recorded by whichever node failed.
7. **`write_markdown`** — once the queue is empty (or discovery itself failed), renders all accumulated notes and errors into one timestamped Markdown file in `outputs/`.

State (`graph/state.py`) is a Pydantic model (`AgentState`) with `videos`, `notes`, and `errors` using additive reducers, so results accumulate cleanly across the per-video loop without manual merging logic.

---

## Output format

Each run produces a file like `outputs/digest_2026-08-22_101530.md`:

```markdown
# Study Digest — 2026-08-22_101530 UTC

Playlist: https://www.youtube.com/playlist?list=PLxxxx
Videos summarized: 4 / 5 discovered

## Introduction to Neural Networks
[Watch](https://www.youtube.com/watch?v=abc123)

This video introduces the basic building blocks of neural networks...

**Key concepts:**
- perceptron
- activation function
- backpropagation

**Notable moments:**
- 02:15 - explanation of the sigmoid function
- 08:40 - worked backpropagation example

**Action items:**
- [ ] Review the chain rule before the next video
- [ ] Implement a single-layer perceptron from scratch

---

## Error handling & resilience

- **Per-video isolation:** a bad transcript or a malformed LLM response on one video is recorded as a `VideoError` and the pipeline moves straight to the next video — deliberately **no retries** at the graph level, so one problematic video never stalls or aborts the whole run.
- **LLM output validation:** the summarization prompt forces JSON-only output matched against the `StudyNote` schema; if the first response fails to parse, exactly one retry is attempted with a stricter follow-up prompt before giving up on that video.
- **Discovery failure:** if the playlist page never renders any video items, or none can be parsed, the graph still proceeds to `write_markdown` so you get a digest documenting what went wrong rather than a silent crash.
- **Pipeline-level failures** (e.g. the graph itself throws) surface as a clean `500` from `/trigger` rather than an unhandled server crash.

---

## Design decisions

- **DOM-scraped transcripts, not `youtube-transcript-api`.** This was chosen deliberately over the simpler library-based approach because it exercises exactly what the rubric weights heavily: resilient element discovery and dynamic DOM waits, using real `wait_for_selector` calls at each stage rather than a single library call.
- **Per-video `StudyNote`, not one aggregated summary.** Keeps `summarize_node` isolated and independently testable (one transcript in, one validated note out), and produces a more genuinely useful digest.
- **Capped run size (`MAX_VIDEOS_PER_RUN`, default 5).** Keeps demo runs fast and avoids long-playlist timeouts; configurable via `.env` if you want to process more.
- **Fail immediately, no retry, on a bad video.** Keeps the control flow simple and predictable, and matches the actual failure modes here (a missing transcript or a bad LLM response usually won't fix itself on retry within the same run).
- **Synchronous `/trigger`.** For a capped run this completes in well under a couple of minutes, and a single blocking call is far easier to demo and reason about than adding a job store/polling endpoint the rubric doesn't ask for.

---

## Troubleshooting

**`camoufox.exceptions.CamoufoxNotInstalled`**
Run `camoufox fetch` — the Python package and the browser binary are installed separately.

**`python -m browser.session --check` reports "Not signed in"**
Your session expired or was never established. Re-run `python -m browser.session --login`.

**Ollama connection errors**
Confirm the server is running (`ollama serve`, or check it's running as a background service) and that `OLLAMA_HOST` in `.env` matches where it's listening (default `http://localhost:11434`).

**Transcript extraction fails on every video**
YouTube's DOM structure changes periodically. Check the selectors in `graph/nodes.py` (`TRANSCRIPT_BUTTON_SELECTOR`, `TRANSCRIPT_PANEL_SELECTOR`, etc.) against the current page — some videos also simply don't have captions available, which will correctly show up as a per-video error rather than crash the run.

**Playlist page times out / "did not load video items in time"**
This is most commonly a **cookie-consent interstitial** ("Before you continue to YouTube") blocking the page on a fresh browser profile — it can appear even when the session is otherwise signed in. The agent tries to dismiss this automatically before waiting for playlist content, but if it still fails, check `outputs/_debug/` — every timeout in `navigate_playlist` or `extract_video_list` saves a screenshot (`.png`) and full HTML snapshot (`.html`) of what the page actually looked like at the moment of failure, so you can see exactly what's blocking it instead of guessing from a bare timeout message.

**LLM responses keep failing to parse as JSON**
Try a different model — smaller/less-instructable models sometimes ignore formatting instructions more often. `qwen2.5:7b` and `llama3.1:8b` have both proven reliable for structured JSON output.

---

## Limitations & future work

- Transcript extraction relies on YouTube's current DOM structure and will need selector updates if YouTube changes its UI.
- No automatic retry at the graph level for transient failures (e.g. a slow network causing a one-off timeout)
- The background scheduler runs against a single fixed `DEFAULT_PLAYLIST_URL`; supporting multiple scheduled playlists would need a small extension (e.g. a list of playlist configs instead of one URL).
- No persistence/database layer — each run's digest is a standalone file; there's no cross-run deduplication if the same video appears in multiple runs.
