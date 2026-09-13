from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from app.dependencies import ServicesDep
from app.guardian_tools import (
    CharacterNotFoundError,
    GuardianToolService,
    ToolInvocationResponse,
    UnknownGuardianToolError,
)
from app.routes.guardian import _refresh_service, guardian_credentials

router = APIRouter(prefix="/api/debug/guardian-tools", tags=["debug"])


def _require_debug_enabled(services: ServicesDep) -> None:
    if not services.settings.enable_debug_tools:
        raise HTTPException(status_code=404, detail="Debug Guardian tools are disabled.")


@router.get("")
async def list_guardian_tools(services: ServicesDep) -> dict[str, Any]:
    _require_debug_enabled(services)
    return {"tools": GuardianToolService.definitions()}


@router.post("/{tool_name}", response_model=ToolInvocationResponse)
async def invoke_guardian_tool(
    tool_name: str,
    request: Request,
    services: ServicesDep,
    arguments: dict[str, Any] | None = None,
) -> ToolInvocationResponse:
    _require_debug_enabled(services)
    account_key, access_token = await guardian_credentials(request, services)
    context = (
        await _refresh_service(services).for_tool(account_key, access_token, tool_name)
    ).context
    try:
        result = GuardianToolService(context).execute(tool_name, arguments)
    except CharacterNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UnknownGuardianToolError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return ToolInvocationResponse(tool_name=tool_name, result=result)
