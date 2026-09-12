from fastapi import APIRouter, HTTPException, Request

from app.dependencies import ServicesDep
from app.models import ChatRequest, ChatResponse
from app.routes.guardian import load_guardian

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    services: ServicesDep,
) -> ChatResponse:
    guardian = await load_guardian(request, services)
    try:
        return await services.recommendations.chat(body, guardian)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
