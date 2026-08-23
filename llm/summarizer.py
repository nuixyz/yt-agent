from __future__ import annotations

import ast
import json
import logging
import re

import ollama

from config import settings
from graph.state import StudyNote, VideoRef

logger = logging.getLogger(__name__)

MAX_TRANSCRIPT_CHARS = 12000

SYSTEM_PROMPT = """You are a study-notes assistant. You are given the transcript \
of an educational YouTube video and must produce a structured study note.

Respond with ONLY a single JSON object -- no markdown fences, no commentary, \
no preamble or explanation. The JSON object must have exactly these keys:

{
  "summary": "2-4 sentence high-level summary of the video",
  "key_concepts": ["short phrase", "..."],
  "action_items": ["concrete follow-up action", "..."],
  "timestamps": ["MM:SS - short description", "..."]
}

Rules:
- Base everything strictly on the transcript. Do not invent facts, tools, or \
claims that are not present in the text.
- If the transcript gives no clear timestamps or action items, return an empty \
list for that field rather than guessing.
- key_concepts should be short noun phrases (e.g. "gradient descent"), not sentences.
- Keep the JSON compact and valid: double-quoted strings, no trailing commas.
"""

RETRY_SYSTEM_SUFFIX = """

IMPORTANT: Your previous response could not be parsed as valid JSON. \
Return ONLY the JSON object this time, with no surrounding text, no markdown \
code fences, and no explanation before or after it."""

_BARE_TIMESTAMP_ITEM_RE = re.compile(
    r'(?<![\w"\'])(\d{1,2}:\d{2}(?::\d{2})?\s*-\s*[^,\]"]+?)(?=\s*[,\]])'
)

def _quote_bare_unquoted_timestamps(text: str) -> str:
    def _wrap(m: re.Match) -> str:
        inner = m.group(1).strip().replace('"', '\\"')
        return f'"{inner}"'
    return _BARE_TIMESTAMP_ITEM_RE.sub(_wrap, text)

def _extract_json(raw: str) -> dict:
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)

    text = _quote_bare_unquoted_timestamps(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    normalized = re.sub(r"\btrue\b", "True", text)
    normalized = re.sub(r"\bfalse\b", "False", normalized)
    normalized = re.sub(r"\bnull\b", "None", normalized)
    result = ast.literal_eval(normalized)
    if not isinstance(result, dict):
        raise json.JSONDecodeError("Parsed value was not a JSON object", text, 0)
    return result


def _build_user_prompt(video: VideoRef, transcript: str) -> str:
    truncated = transcript[:MAX_TRANSCRIPT_CHARS]
    note = ""
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        note = "\n\n[Transcript truncated for length.]"
    return (
        f"Video title: {video.title}\n\n"
        f"Transcript:\n{truncated}{note}"
    )


async def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    response = ollama.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.2},
    )
    return response["message"]["content"]


async def summarize_transcript(video: VideoRef, transcript: str) -> StudyNote:
    user_prompt = _build_user_prompt(video, transcript)

    raw = await _call_ollama(SYSTEM_PROMPT, user_prompt)
    try:
        payload = _extract_json(raw)
        return StudyNote(
            video_id=video.video_id,
            title=video.title,
            url=video.url,
            summary=payload.get("summary", ""),
            key_concepts=payload.get("key_concepts", []) or [],
            action_items=payload.get("action_items", []) or [],
            timestamps=payload.get("timestamps", []) or [],
        )
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning(
            "First summarization attempt for %s produced invalid JSON (%s); retrying once.",
            video.video_id,
            e,
        )

    raw_retry = await _call_ollama(SYSTEM_PROMPT + RETRY_SYSTEM_SUFFIX, user_prompt)
    payload = _extract_json(raw_retry)
    return StudyNote(
        video_id=video.video_id,
        title=video.title,
        url=video.url,
        summary=payload.get("summary", ""),
        key_concepts=payload.get("key_concepts", []) or [],
        action_items=payload.get("action_items", []) or [],
        timestamps=payload.get("timestamps", []) or [],
    )
