from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from graph.nodes import (
    extract_transcript,
    extract_video_list,
    handle_error,
    navigate_playlist,
    open_video,
    summarize_node,
    write_markdown,
)
from graph.state import AgentState

logger = logging.getLogger(__name__)


# Conditional routing functions
def route_after_discovery(state: AgentState) -> str:
    if not state.videos:
        return "write_markdown"
    return "open_video"


def route_next_step(state: AgentState) -> str:
    if state.pending_video_ids:
        return "open_video"
    return "write_markdown"


def route_after_extract(state: AgentState) -> str:
    if state.current_transcript is not None:
        return "summarize_node"
    return "handle_error"


def route_after_summarize(state: AgentState) -> str:
    return "route_next_step"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("navigate_playlist", navigate_playlist)
    graph.add_node("extract_video_list", extract_video_list)
    graph.add_node("open_video", open_video)
    graph.add_node("extract_transcript", extract_transcript)
    graph.add_node("summarize_node", summarize_node)
    graph.add_node("handle_error", handle_error)
    graph.add_node("write_markdown", write_markdown)

    graph.set_entry_point("navigate_playlist")

    graph.add_edge("navigate_playlist", "extract_video_list")

    graph.add_conditional_edges(
        "extract_video_list",
        route_after_discovery,
        {"open_video": "open_video", "write_markdown": "write_markdown"},
    )

    graph.add_edge("open_video", "extract_transcript")

    graph.add_conditional_edges(
        "extract_transcript",
        route_after_extract,
        {"summarize_node": "summarize_node", "handle_error": "handle_error"},
    )

    graph.add_conditional_edges(
        "summarize_node",
        lambda _state: "handle_error",
        {"handle_error": "handle_error"},
    )

    graph.add_conditional_edges(
        "handle_error",
        route_next_step,
        {"open_video": "open_video", "write_markdown": "write_markdown"},
    )

    graph.add_edge("write_markdown", END)

    return graph.compile()


# Module-level compiled app, reused across API requests.
app_graph = build_graph()
