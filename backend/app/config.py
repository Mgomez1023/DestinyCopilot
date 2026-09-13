from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    port: int = Field(default=8000, ge=1, le=65535)
    bungie_api_key: str = ""
    bungie_client_id: str = ""
    bungie_client_secret: str = ""
    bungie_redirect_uri: str = "https://localhost:8000/api/auth/callback"
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    openai_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "low"
    openai_max_output_tokens: int = Field(default=2000, ge=512, le=32768)
    frontend_origin: str = "https://localhost:5173"
    frontend_url: str = "https://localhost:5173"
    cookie_secure: bool = True
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"
    session_backend: Literal["memory"] = "memory"
    session_cookie_max_age_seconds: int = Field(default=60 * 60 * 24 * 30, ge=300)
    oauth_state_ttl_seconds: int = Field(default=600, ge=60, le=3600)
    tls_cert_file: Path = Path(".certs/localhost.pem")
    tls_key_file: Path = Path(".certs/localhost-key.pem")
    manifest_cache_dir: Path = Path(".cache/manifest")
    manifest_definition_cache_size: int = Field(default=1024, ge=0, le=10000)
    manifest_metadata_ttl_seconds: int = Field(default=900, ge=0, le=86400)
    guide_corpus_file: Path = Path("backend/app/data/destiny_guides.json")
    guide_cache_dir: Path = Path(".cache/guides")
    guide_cache_stable_ttl_seconds: int = Field(default=60 * 60 * 24 * 30, ge=0)
    guide_cache_semi_stable_ttl_seconds: int = Field(default=60 * 60 * 24, ge=0)
    live_cache_dir: Path = Path(".cache/live")
    live_cache_ttl_seconds: int = Field(default=300, ge=0, le=3600)
    guardian_cache_profile_ttl_seconds: int = Field(default=900, ge=0)
    guardian_cache_equipment_ttl_seconds: int = Field(default=180, ge=0)
    guardian_cache_inventory_ttl_seconds: int = Field(default=600, ge=0)
    guardian_cache_quests_progress_ttl_seconds: int = Field(default=300, ge=0)
    guardian_cache_activity_history_ttl_seconds: int = Field(default=600, ge=0)
    guardian_cache_collections_ttl_seconds: int = Field(default=1800, ge=0)
    guardian_soft_profile_min_seconds: int = Field(default=900, ge=0)
    guardian_soft_equipment_min_seconds: int = Field(default=45, ge=0)
    guardian_soft_inventory_min_seconds: int = Field(default=60, ge=0)
    guardian_soft_quests_progress_min_seconds: int = Field(default=60, ge=0)
    guardian_soft_activity_history_min_seconds: int = Field(default=120, ge=0)
    guardian_soft_collections_min_seconds: int = Field(default=600, ge=0)
    guardian_app_open_recent_seconds: int = Field(default=300, ge=0)
    guardian_app_open_full_seconds: int = Field(default=7200, ge=0)
    guardian_refresh_failure_backoff_seconds: int = Field(default=30, ge=0)
    enable_debug_tools: bool = True
    allow_production_debug: bool = False
    log_level: str = "INFO"
    request_timeout_seconds: float = Field(default=20.0, gt=0)

    @property
    def bungie_configured(self) -> bool:
        return all(
            (
                self.bungie_api_key,
                self.bungie_client_id,
                self.bungie_client_secret,
                self.bungie_redirect_uri,
            )
        )

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def debug_tools_enabled(self) -> bool:
        return self.enable_debug_tools and (not self.is_production or self.allow_production_debug)

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        if not self.is_production:
            return self

        required = {
            "BUNGIE_API_KEY": self.bungie_api_key,
            "BUNGIE_CLIENT_ID": self.bungie_client_id,
            "BUNGIE_CLIENT_SECRET": self.bungie_client_secret,
            "BUNGIE_REDIRECT_URI": self.bungie_redirect_uri,
            "OPENAI_API_KEY": self.openai_api_key,
            "FRONTEND_ORIGIN": self.frontend_origin,
            "FRONTEND_URL": self.frontend_url,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"Missing production settings: {', '.join(missing)}")
        if not self.cookie_secure:
            raise ValueError("COOKIE_SECURE must be true in production.")
        if self.cookie_samesite != "none":
            raise ValueError(
                "COOKIE_SAMESITE must be none when the production frontend and API use "
                "separate sites."
            )

        redirect = urlparse(self.bungie_redirect_uri)
        if (
            redirect.scheme != "https"
            or not redirect.netloc
            or redirect.path.rstrip("/") != "/api/auth/callback"
            or redirect.params
            or redirect.query
            or redirect.fragment
        ):
            raise ValueError(
                "BUNGIE_REDIRECT_URI must be an HTTPS /api/auth/callback URL in production."
            )
        for name, value in (
            ("FRONTEND_ORIGIN", self.frontend_origin),
            ("FRONTEND_URL", self.frontend_url),
        ):
            parsed = urlparse(value)
            if (
                parsed.scheme != "https"
                or not parsed.netloc
                or parsed.path not in {"", "/"}
                or parsed.params
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(f"{name} must be an HTTPS origin without a path or query.")
        if self.frontend_origin.rstrip("/") != self.frontend_url.rstrip("/"):
            raise ValueError("FRONTEND_ORIGIN and FRONTEND_URL must identify the same origin.")
        return self

    def resolve_local_path(self, path: Path) -> Path:
        return path if path.is_absolute() else ROOT_DIR / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
