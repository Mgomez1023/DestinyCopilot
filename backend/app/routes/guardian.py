from fastapi import APIRouter, HTTPException, Request

from app.bungie.client import BungieAPIError
from app.bungie.guardian import GuardianNormalizationError, GuardianNotFoundError
from app.bungie.oauth import OAuthError
from app.dependencies import Services, ServicesDep
from app.models import GuardianContext
from app.routes.auth import SESSION_COOKIE

router = APIRouter(prefix="/api/guardian", tags=["guardian"])


async def load_guardian(request: Request, services: Services) -> GuardianContext:
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id:
        raise HTTPException(status_code=401, detail="Connect your Bungie account first.")
    try:
        access_token = await services.oauth.valid_access_token(session_id)
        return await services.guardian.load(access_token)
    except OAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except GuardianNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GuardianNormalizationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except BungieAPIError as exc:
        status = 401 if exc.status_code == 401 else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.get("/me", response_model=GuardianContext)
async def guardian_me(
    request: Request, services: ServicesDep
) -> GuardianContext:
    return await load_guardian(request, services)
