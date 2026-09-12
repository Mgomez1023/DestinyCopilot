import logging
import secrets
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse

from app.bungie.oauth import OAuthError, oauth_store
from app.dependencies import ServicesDep
from app.models import AuthStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])
SESSION_COOKIE = "guardian_session"
OAUTH_STATE_COOKIE = "guardian_oauth_state"


@router.get("/status", response_model=AuthStatus)
async def auth_status(
    request: Request, services: ServicesDep
) -> AuthStatus:
    session_id = request.cookies.get(SESSION_COOKIE)
    authenticated = bool(session_id and await oauth_store.get_session(session_id))
    message = (
        None
        if services.settings.bungie_configured
        else "Bungie credentials are not configured."
    )
    return AuthStatus(
        configured=services.settings.bungie_configured,
        authenticated=authenticated,
        message=message,
    )


@router.get("/login")
async def login(services: ServicesDep) -> RedirectResponse:
    if not services.settings.bungie_configured:
        raise HTTPException(status_code=503, detail="Bungie credentials are not configured.")
    state = await oauth_store.create_state()
    response = RedirectResponse(services.oauth.authorization_url(state), status_code=302)
    response.set_cookie(
        OAUTH_STATE_COOKIE,
        state,
        httponly=True,
        secure=services.settings.cookie_secure,
        samesite="lax",
        max_age=600,
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
        and await oauth_store.consume_state(state)
    )
    if not state_is_valid:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth callback state.")
    oauth_error = error or error_key
    if oauth_error:
        query = urlencode({"auth_error": oauth_error})
        response = RedirectResponse(f"{services.settings.frontend_url}/?{query}")
        response.delete_cookie(OAUTH_STATE_COOKIE, path="/api/auth/callback")
        return response
    if not code:
        raise HTTPException(status_code=400, detail="Bungie did not return an authorization code.")
    try:
        token = await services.oauth.exchange_code(code)
    except OAuthError as exc:
        logger.warning("OAuth code exchange failed: %s", exc)
        query = urlencode({"auth_error": str(exc)})
        return RedirectResponse(f"{services.settings.frontend_url}/?{query}")

    session_id = await oauth_store.create_session(token)
    response = RedirectResponse(f"{services.settings.frontend_url}/?connected=1")
    response.delete_cookie(OAUTH_STATE_COOKIE, path="/api/auth/callback")
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        httponly=True,
        secure=services.settings.cookie_secure,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
        path="/",
    )
    return response


@router.post("/logout", status_code=204)
async def logout(
    request: Request, response: Response, services: ServicesDep
) -> None:
    del services
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id:
        await oauth_store.delete_session(session_id)
    response.delete_cookie(SESSION_COOKIE, path="/")
