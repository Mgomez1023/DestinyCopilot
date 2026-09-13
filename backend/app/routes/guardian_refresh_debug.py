from fastapi import APIRouter, HTTPException, Request

from app.dependencies import ServicesDep
from app.guardian_refresh import GuardianSlice
from app.routes.debug import _require_debug_enabled
from app.routes.guardian import _refresh_service, guardian_credentials

router = APIRouter(prefix="/api/debug/guardian-refresh", tags=["debug"])


@router.get("")
async def refresh_status(request: Request, services: ServicesDep) -> dict[str, object]:
    _require_debug_enabled(services)
    session_id, _ = await guardian_credentials(request, services)
    return await _refresh_service(services).status(session_id)


@router.post("/slice/{slice_name}")
async def refresh_slice(
    slice_name: str, request: Request, services: ServicesDep
) -> dict[str, object]:
    _require_debug_enabled(services)
    try:
        selected = GuardianSlice(slice_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Unknown Guardian cache slice.") from exc
    session_id, access_token = await guardian_credentials(request, services)
    refresh = _refresh_service(services)
    outcome = await refresh.refresh(
        session_id,
        access_token,
        slices={selected},
        force=True,
        reason=f"debug_slice:{selected.value}",
    )
    return {
        "guardian": outcome.context.model_dump(mode="json"),
        "refresh": await refresh.status(session_id),
    }


@router.post("/clear")
async def clear_refresh_cache(request: Request, services: ServicesDep) -> dict[str, bool]:
    _require_debug_enabled(services)
    session_id, _ = await guardian_credentials(request, services)
    await _refresh_service(services).clear(session_id)
    return {"cleared": True}


@router.post("/{level}")
async def refresh_level(level: str, request: Request, services: ServicesDep) -> dict[str, object]:
    _require_debug_enabled(services)
    if level not in {"normal", "full"}:
        raise HTTPException(status_code=404, detail="Unknown refresh level.")
    session_id, access_token = await guardian_credentials(request, services)
    refresh = _refresh_service(services)
    outcome = (
        await refresh.full_refresh(session_id, access_token)
        if level == "full"
        else await refresh.normal_refresh(session_id, access_token)
    )
    return {
        "guardian": outcome.context.model_dump(mode="json"),
        "refresh": await refresh.status(session_id),
    }
