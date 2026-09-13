from collections.abc import Awaitable

from fastapi import APIRouter, HTTPException, Request

from app.bungie.client import BungieAPIError
from app.bungie.guardian import GuardianNormalizationError, GuardianNotFoundError
from app.bungie.oauth import OAuthError
from app.dependencies import Services, ServicesDep
from app.guardian_refresh import GuardianRefreshService, RefreshOutcome
from app.models import GuardianContext
from app.routes.auth import SESSION_COOKIE

router = APIRouter(prefix="/api/guardian", tags=["guardian"])


def _refresh_service(services: Services) -> GuardianRefreshService:
    if services.guardian_refresh is None:
        services.guardian_refresh = GuardianRefreshService(services.guardian, services.settings)
    return services.guardian_refresh


async def guardian_credentials(request: Request, services: Services) -> tuple[str, str]:
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id:
        raise HTTPException(status_code=401, detail="Connect your Bungie account first.")
    token = await services.oauth_store.get_session(session_id)
    account_key = (
        f"bungie:{token.bungie_membership_id}"
        if token and token.bungie_membership_id
        else f"session:{session_id}"
    )
    try:
        access_token = await services.oauth.valid_access_token(session_id)
        return account_key, access_token
    except OAuthError as exc:
        if services.guardian_refresh is not None:
            await services.guardian_refresh.clear(account_key)
        raise HTTPException(status_code=401, detail=str(exc)) from exc


async def _execute_refresh(operation: Awaitable[RefreshOutcome]) -> RefreshOutcome:
    try:
        return await operation
    except GuardianNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GuardianNormalizationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except BungieAPIError as exc:
        status = 401 if exc.status_code == 401 else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc


async def load_guardian(
    request: Request,
    services: Services,
    *,
    stale_while_revalidate: bool = True,
) -> GuardianContext:
    session_id, access_token = await guardian_credentials(request, services)
    outcome = await _execute_refresh(
        _refresh_service(services).get_context(
            session_id,
            access_token,
            stale_while_revalidate=stale_while_revalidate,
        )
    )
    return outcome.context


@router.get("/me", response_model=GuardianContext)
async def guardian_me(request: Request, services: ServicesDep) -> GuardianContext:
    return await load_guardian(request, services)


@router.post("/resume")
async def guardian_resume(request: Request, services: ServicesDep) -> dict[str, object]:
    session_id, access_token = await guardian_credentials(request, services)
    outcome = await _execute_refresh(_refresh_service(services).app_open(session_id, access_token))
    return {
        "guardian": outcome.context.model_dump(mode="json"),
        "refresh": await _refresh_service(services).status(session_id),
        "reason": outcome.reason,
    }


@router.post("/refresh")
async def guardian_refresh(
    request: Request, services: ServicesDep, level: str = "normal"
) -> dict[str, object]:
    if level not in {"normal", "full"}:
        raise HTTPException(status_code=422, detail="Refresh level must be normal or full.")
    session_id, access_token = await guardian_credentials(request, services)
    refresh = _refresh_service(services)
    outcome = await _execute_refresh(
        refresh.full_refresh(session_id, access_token)
        if level == "full"
        else refresh.normal_refresh(session_id, access_token)
    )
    return {
        "guardian": outcome.context.model_dump(mode="json"),
        "refresh": await refresh.status(session_id),
        "reason": outcome.reason,
    }


@router.get("/refresh/status")
async def guardian_refresh_status(request: Request, services: ServicesDep) -> dict[str, object]:
    session_id, _ = await guardian_credentials(request, services)
    return await _refresh_service(services).status(session_id)
