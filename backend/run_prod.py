"""Run Guardian Copilot behind a production host's HTTPS reverse proxy."""

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    if not settings.is_production:
        raise SystemExit("Set APP_ENV=production before using the production server command.")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
        access_log=False,
    )


if __name__ == "__main__":
    main()
