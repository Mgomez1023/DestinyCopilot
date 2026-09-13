import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.dependencies import ServicesDep
from app.destiny_knowledge import (
    INDEXED_DEFINITION_TYPES,
    UnknownKnowledgeToolError,
)
from app.guardian_tools import ToolInvocationResponse

router = APIRouter(prefix="/api/debug/destiny-knowledge", tags=["debug"])
logger = logging.getLogger(__name__)


def _require_debug_enabled(services: ServicesDep) -> None:
    if not services.settings.enable_debug_tools:
        raise HTTPException(status_code=404, detail="Debug Destiny knowledge tools are disabled.")


@router.get("")
async def list_destiny_knowledge_tools(services: ServicesDep) -> dict[str, Any]:
    _require_debug_enabled(services)
    return {
        "tools": services.knowledge.definitions(),
        "configured_definition_types": INDEXED_DEFINITION_TYPES,
    }


@router.get("/status")
async def destiny_knowledge_status(services: ServicesDep) -> dict[str, Any]:
    _require_debug_enabled(services)
    return await services.knowledge.index_status()


@router.post("/{tool_name}", response_model=ToolInvocationResponse)
async def invoke_destiny_knowledge_tool(
    tool_name: str,
    services: ServicesDep,
    arguments: dict[str, Any] | None = None,
) -> ToolInvocationResponse:
    _require_debug_enabled(services)
    try:
        result = await services.knowledge.execute(tool_name, arguments)
    except UnknownKnowledgeToolError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:
        logger.exception("Destiny knowledge debug tool failed tool=%s", tool_name)
        raise HTTPException(
            status_code=502,
            detail="Destiny knowledge could not be loaded.",
        ) from exc
    return ToolInvocationResponse(tool_name=tool_name, result=result)
