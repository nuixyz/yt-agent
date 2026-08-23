from __future__ import annotations

from typing import Annotated, Optional

import operator
from pydantic import BaseModel, Field


class VideoRef(BaseModel):
    video_id: str = Field(description="YouTube video ID, e.g. 'dQw4w9WgXcQ'.")
    url: str = Field(description="Full watch URL for the video.")
    title: str = Field(description="Video title as shown in the playlist listing.")
    position: int = Field(description="1-indexed position within the playlist.")


class StudyNote(BaseModel):
    video_id: str
    title: str
    url: str
    summary: str = Field(description="2-4 sentence high-level summary of the video.")
    key_concepts: list[str] = Field(
        default_factory=list, description="Core concepts/terms covered."
    )
    action_items: list[str] = Field(
        default_factory=list,
        description="Concrete follow-ups: things to practice, re-watch, or look up.",
    )
    timestamps: list[str] = Field(
        default_factory=list,
        description="Notable moments as 'MM:SS - description' strings, if available.",
    )


class VideoError(BaseModel):
    video_id: str | None = None
    url: str | None = None
    stage: str = Field(description="Which node the failure occurred in.")
    message: str


def _keep_last(_current: str | None, new: str | None) -> str | None:
    return new


class AgentState(BaseModel):
    # --- input ---
    playlist_url: str = Field(description="Playlist URL supplied via /trigger.")

    # --- discovery ---
    videos: Annotated[list[VideoRef], operator.add] = Field(default_factory=list)

    # --- per-video processing ---
    pending_video_ids: list[str] = Field(
        default_factory=list,
        description="Video IDs still queued for transcript extraction + summarization.",
    )
    current_video: Optional[VideoRef] = None
    current_transcript: Optional[str] = None

    # --- results ---
    notes: Annotated[list[StudyNote], operator.add] = Field(default_factory=list)
    errors: Annotated[list[VideoError], operator.add] = Field(default_factory=list)

    output_path: Optional[str] = Field(
        default=None, description="Path to the written Markdown digest, once complete."
    )

    class Config:
        arbitrary_types_allowed = True
