from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

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
    tls_cert_file: Path = Path(".certs/localhost.pem")
    tls_key_file: Path = Path(".certs/localhost-key.pem")
    manifest_cache_dir: Path = Path(".cache/manifest")
    enable_debug_tools: bool = True
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

    def resolve_local_path(self, path: Path) -> Path:
        return path if path.is_absolute() else ROOT_DIR / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
