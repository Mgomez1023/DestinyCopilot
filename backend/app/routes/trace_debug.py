from typing import Any

from fastapi import APIRouter, HTTPException

from app.dependencies import ServicesDep

router = APIRouter(prefix="/api/debug/chat-traces", tags=["debug"])


@router.get("")
async def get_recent_chat_traces(services: ServicesDep, limit: int = 20) -> dict[str, Any]:
    if not services.settings.enable_debug_tools:
        raise HTTPException(status_code=404, detail="Debug chat traces are disabled.")
    bounded_limit = min(max(limit, 1), 100)
    traces = services.recommendations.recent_traces(bounded_limit)
    return {"traces": traces, "count": len(traces)}


@router.get("/latest")
async def get_latest_chat_trace(services: ServicesDep) -> dict[str, Any]:
    if not services.settings.enable_debug_tools:
        raise HTTPException(status_code=404, detail="Debug chat traces are disabled.")
    trace = services.recommendations.latest_trace()
    if trace is None:
        raise HTTPException(status_code=404, detail="No chat trace has been captured.")
    return trace


@router.get("/{trace_id}")
async def get_chat_trace(trace_id: str, services: ServicesDep) -> dict[str, Any]:
    if not services.settings.enable_debug_tools:
        raise HTTPException(status_code=404, detail="Debug chat traces are disabled.")
    trace = services.recommendations.trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Chat trace was not found.")
    return trace
