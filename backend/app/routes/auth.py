import logging
import secrets
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse

from app.bungie.oauth import OAuthError
from app.config import Settings
from app.dependencies import ServicesDep
from app.models import AuthStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])
SESSION_COOKIE = "guardian_session"
OAUTH_STATE_COOKIE = "guardian_oauth_state"


def _frontend_redirect(settings: Settings, **query_values: str) -> str:
    query = urlencode(query_values)
    suffix = f"?{query}" if query else ""
    return f"{settings.frontend_url.rstrip('/')}/{suffix}"


def _delete_cookie(response: Response, name: str, settings: Settings, *, path: str) -> None:
    response.delete_cookie(
        name,
        path=path,
        secure=settings.cookie_secure,
        httponly=True,
        samesite="lax" if name == OAUTH_STATE_COOKIE else settings.cookie_samesite,
    )


def _protect_oauth_redirect(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"


@router.get("/status", response_model=AuthStatus)
async def auth_status(request: Request, services: ServicesDep) -> AuthStatus:
    session_id = request.cookies.get(SESSION_COOKIE)
    authenticated = bool(session_id and await services.oauth_store.get_session(session_id))
    message = (
        None if services.settings.bungie_configured else "Bungie credentials are not configured."
    )
    return AuthStatus(
        configured=services.settings.bungie_configured,
        authenticated=authenticated,
        debug_enabled=services.settings.debug_tools_enabled,
        message=message,
    )


@router.get("/login")
async def login(services: ServicesDep) -> RedirectResponse:
    if not services.settings.bungie_configured:
        raise HTTPException(status_code=503, detail="Bungie credentials are not configured.")
    state = await services.oauth_store.create_state()
    response = RedirectResponse(services.oauth.authorization_url(state), status_code=302)
    _protect_oauth_redirect(response)
    response.set_cookie(
        OAUTH_STATE_COOKIE,
        state,
        httponly=True,
        secure=services.settings.cookie_secure,
        samesite="lax",
        max_age=services.settings.oauth_state_ttl_seconds,
        path="/api/auth/callback",
    )
    return response


@router.get("/callback")
async def callback(
    request: Request,
    services: ServicesDep,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    error_key: str | None = Query(default=None, alias="errorKey"),
) -> RedirectResponse:
    cookie_state = request.cookies.get(OAUTH_STATE_COOKIE)
    state_is_valid = bool(
        state
        and cookie_state
        and secrets.compare_digest(state, cookie_state)
        and await services.oauth_store.consume_state(state)
    )
    if not state_is_valid:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth callback state.")
    oauth_error = error or error_key
    if oauth_error:
        response = RedirectResponse(_frontend_redirect(services.settings, auth_error=oauth_error))
        _protect_oauth_redirect(response)
        _delete_cookie(response, OAUTH_STATE_COOKIE, services.settings, path="/api/auth/callback")
        return response
    if not code:
        raise HTTPException(status_code=400, detail="Bungie did not return an authorization code.")
    try:
        token = await services.oauth.exchange_code(code)
    except OAuthError as exc:
        logger.warning("OAuth code exchange failed: %s", exc)
        response = RedirectResponse(_frontend_redirect(services.settings, auth_error=str(exc)))
        _protect_oauth_redirect(response)
        _delete_cookie(response, OAUTH_STATE_COOKIE, services.settings, path="/api/auth/callback")
        return response

    if services.guardian_refresh is not None and token.bungie_membership_id:
        await services.guardian_refresh.clear(f"bungie:{token.bungie_membership_id}")
    session_id = await services.oauth_store.create_session(token)
    response = RedirectResponse(_frontend_redirect(services.settings, connected="1"))
    _protect_oauth_redirect(response)
    _delete_cookie(response, OAUTH_STATE_COOKIE, services.settings, path="/api/auth/callback")
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        httponly=True,
        secure=services.settings.cookie_secure,
        samesite=services.settings.cookie_samesite,
        max_age=services.settings.session_cookie_max_age_seconds,
        path="/",
    )
    return response


@router.post("/logout", status_code=204)
async def logout(request: Request, response: Response, services: ServicesDep) -> None:
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id:
        token = await services.oauth_store.get_session(session_id)
        if services.guardian_refresh is not None:
            account_key = (
                f"bungie:{token.bungie_membership_id}"
                if token and token.bungie_membership_id
                else f"session:{session_id}"
            )
            await services.guardian_refresh.clear(account_key)
        await services.oauth_store.delete_session(session_id)
    _delete_cookie(response, SESSION_COOKIE, services.settings, path="/")
