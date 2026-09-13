import asyncio
import secrets
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode

import httpx

from app.config import Settings

BUNGIE_AUTHORIZE_URL = "https://www.bungie.net/en/OAuth/Authorize"
BUNGIE_TOKEN_URL = "https://www.bungie.net/Platform/App/OAuth/Token/"


class OAuthError(RuntimeError):
    pass


@dataclass
class OAuthToken:
    access_token: str
    refresh_token: str | None
    expires_at: float
    refresh_expires_at: float | None
    bungie_membership_id: str | None


class OAuthSessionStore(Protocol):
    """Storage boundary for OAuth CSRF state and server-side token sessions."""

    async def create_state(self) -> str: ...

    async def consume_state(self, state: str) -> bool: ...

    async def create_session(self, token: OAuthToken) -> str: ...

    async def get_session(self, session_id: str) -> OAuthToken | None: ...

    async def update_session(self, session_id: str, token: OAuthToken) -> None: ...

    async def delete_session(self, session_id: str) -> None: ...


class InMemoryOAuthSessionStore:
    """Single-process beta store. Restarting or scaling the backend signs users out."""

    def __init__(self, state_ttl_seconds: int = 600) -> None:
        self._states: dict[str, float] = {}
        self._sessions: dict[str, OAuthToken] = {}
        self._lock = asyncio.Lock()
        self._state_ttl_seconds = state_ttl_seconds

    async def create_state(self) -> str:
        state = secrets.token_urlsafe(32)
        async with self._lock:
            now = time.time()
            self._states = {key: expiry for key, expiry in self._states.items() if expiry > now}
            self._states[state] = now + self._state_ttl_seconds
        return state

    async def consume_state(self, state: str) -> bool:
        async with self._lock:
            expiry = self._states.pop(state, 0)
        return expiry > time.time()

    async def create_session(self, token: OAuthToken) -> str:
        session_id = secrets.token_urlsafe(32)
        async with self._lock:
            self._sessions[session_id] = token
        return session_id

    async def get_session(self, session_id: str) -> OAuthToken | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def update_session(self, session_id: str, token: OAuthToken) -> None:
        async with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id] = token

    async def delete_session(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)


def create_oauth_session_store(settings: Settings) -> OAuthSessionStore:
    """Build the configured store without coupling callers to its implementation."""

    if settings.session_backend == "memory":
        return InMemoryOAuthSessionStore(settings.oauth_state_ttl_seconds)
    raise ValueError(f"Unsupported session backend: {settings.session_backend}")


class BungieOAuth:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient,
        store: OAuthSessionStore | None = None,
    ) -> None:
        self.settings = settings
        self.http = http_client
        self.store = store or create_oauth_session_store(settings)

    def authorization_url(self, state: str) -> str:
        # Bungie requires scope to be configured in the application portal, not in this URL.
        query = urlencode(
            {
                "client_id": self.settings.bungie_client_id,
                "response_type": "code",
                "state": state,
                "redirect_uri": self.settings.bungie_redirect_uri,
            }
        )
        return f"{BUNGIE_AUTHORIZE_URL}?{query}"

    async def exchange_code(self, code: str) -> OAuthToken:
        return await self._request_token(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.settings.bungie_redirect_uri,
            }
        )

    async def refresh(self, refresh_token: str) -> OAuthToken:
        return await self._request_token(
            {"grant_type": "refresh_token", "refresh_token": refresh_token}
        )

    async def _request_token(self, form: dict[str, str]) -> OAuthToken:
        try:
            response = await self.http.post(
                BUNGIE_TOKEN_URL,
                data=form,
                auth=(self.settings.bungie_client_id, self.settings.bungie_client_secret),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            detail = "Bungie rejected the OAuth token request."
            if isinstance(exc, httpx.HTTPStatusError):
                with suppress(ValueError):
                    detail = exc.response.json().get("error_description", detail)
            raise OAuthError(detail) from exc

        now = time.time()
        return OAuthToken(
            access_token=payload["access_token"],
            refresh_token=payload.get("refresh_token"),
            expires_at=now + int(payload.get("expires_in", 3600)),
            refresh_expires_at=(
                now + int(payload["refresh_expires_in"])
                if payload.get("refresh_expires_in") is not None
                else None
            ),
            bungie_membership_id=payload.get("membership_id"),
        )

    async def valid_access_token(self, session_id: str) -> str:
        token = await self.store.get_session(session_id)
        if token is None:
            raise OAuthError("Session not found.")
        if token.expires_at > time.time() + 30:
            return token.access_token
        if not token.refresh_token or (
            token.refresh_expires_at is not None and token.refresh_expires_at <= time.time()
        ):
            await self.store.delete_session(session_id)
            raise OAuthError("Session expired. Connect with Bungie again.")

        refreshed = await self.refresh(token.refresh_token)
        if refreshed.refresh_token is None:
            refreshed.refresh_token = token.refresh_token
            refreshed.refresh_expires_at = token.refresh_expires_at
        if refreshed.bungie_membership_id is None:
            refreshed.bungie_membership_id = token.bungie_membership_id
        await self.store.update_session(session_id, refreshed)
        return refreshed.access_token
