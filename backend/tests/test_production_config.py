import asyncio
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.bungie.oauth import OAuthToken, create_oauth_session_store
from app.config import Settings
from app.main import create_app


def production_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "app_env": "production",
        "bungie_api_key": "bungie-api-key",
        "bungie_client_id": "bungie-client-id",
        "bungie_client_secret": "bungie-client-secret",
        "bungie_redirect_uri": "https://api.example.com/api/auth/callback",
        "openai_api_key": "openai-api-key",
        "frontend_origin": "https://app.example.com",
        "frontend_url": "https://app.example.com",
        "cookie_secure": True,
        "cookie_samesite": "none",
        "enable_debug_tools": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_production_settings_enforce_https_cross_site_cookie_safety() -> None:
    settings = production_settings()
    assert settings.is_production is True
    assert settings.cookie_secure is True
    assert settings.cookie_samesite == "none"
    assert production_settings(enable_debug_tools=True).debug_tools_enabled is False

    with pytest.raises(ValidationError, match="COOKIE_SECURE must be true"):
        production_settings(cookie_secure=False)
    with pytest.raises(ValidationError, match="COOKIE_SAMESITE must be none"):
        production_settings(cookie_samesite="lax")
    with pytest.raises(ValidationError, match="FRONTEND_ORIGIN must be an HTTPS origin"):
        production_settings(frontend_origin="http://app.example.com")
    with pytest.raises(ValidationError, match="Missing production settings: OPENAI_API_KEY"):
        production_settings(openai_api_key="")
    with pytest.raises(
        ValidationError,
        match="BUNGIE_REDIRECT_URI must be an HTTPS /api/auth/callback URL",
    ):
        production_settings(
            bungie_redirect_uri="http://api.example.com/api/auth/callback"
        )


def test_memory_session_store_is_replaceable_and_one_time_state_is_consumed() -> None:
    async def exercise() -> None:
        store = create_oauth_session_store(
            Settings(_env_file=None, session_backend="memory", oauth_state_ttl_seconds=600)
        )
        state = await store.create_state()
        assert await store.consume_state(state) is True
        assert await store.consume_state(state) is False

        token = OAuthToken("access", "refresh", 10_000, 20_000, "membership")
        session_id = await store.create_session(token)
        assert await store.get_session(session_id) == token
        await store.delete_session(session_id)
        assert await store.get_session(session_id) is None

    asyncio.run(exercise())


def test_production_app_has_exact_cors_and_no_debug_or_docs() -> None:
    application = create_app(production_settings())
    with TestClient(application, base_url="https://api.example.com") as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/docs").status_code == 404
        assert client.get("/api/debug/guardian-tools").status_code == 404

        allowed = client.options(
            "/api/health",
            headers={
                "Origin": "https://app.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert allowed.headers["access-control-allow-origin"] == "https://app.example.com"
        assert allowed.headers["access-control-allow-credentials"] == "true"

        blocked = client.options(
            "/api/health",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in blocked.headers


def test_production_oauth_session_cookie_is_secure_httponly_and_cross_site() -> None:
    application = create_app(production_settings())
    with TestClient(
        application,
        base_url="https://api.example.com",
        follow_redirects=False,
    ) as client:
        login = client.get("/api/auth/login")
        state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]

        async def exchange_code(code: str) -> OAuthToken:
            assert code == "authorization-code"
            return OAuthToken("access", "refresh", 10_000, 20_000, "membership")

        client.app.state.services.oauth.exchange_code = exchange_code
        callback = client.get(f"/api/auth/callback?code=authorization-code&state={state}")

    session_cookie = next(
        value
        for value in callback.headers.get_list("set-cookie")
        if value.startswith("guardian_session=")
    )
    assert "HttpOnly" in session_cookie
    assert "Secure" in session_cookie
    assert "SameSite=none" in session_cookie
    assert callback.headers["location"] == "https://app.example.com/?connected=1"
