from fastapi import APIRouter, HTTPException, Request

from app.dependencies import ServicesDep
from app.models import ChatRequest, ChatResponse, GuardianContext
from app.routes.guardian import _refresh_service, guardian_credentials, load_guardian

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    services: ServicesDep,
) -> ChatResponse:
    guardian = await load_guardian(request, services, stale_while_revalidate=False)
    try:
        refresh_callback = None
        if getattr(services, "guardian_refresh", None) is not None:
            session_id, access_token = await guardian_credentials(request, services)
            refresh = _refresh_service(services)
            guardian = (await refresh.for_prompt(session_id, access_token, body.message)).context

            async def refresh_callback(tool_name: str) -> GuardianContext:
                return (await refresh.for_tool(session_id, access_token, tool_name)).context

        return await services.recommendations.chat(
            body, guardian, refresh_guardian=refresh_callback
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
