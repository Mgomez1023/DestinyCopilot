"""Run Guardian Copilot's local FastAPI server over HTTPS."""

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    cert_file = settings.resolve_local_path(settings.tls_cert_file)
    key_file = settings.resolve_local_path(settings.tls_key_file)
    missing = [str(path) for path in (cert_file, key_file) if not path.is_file()]
    if missing:
        files = "\n  - ".join(missing)
        raise SystemExit(
            "Local HTTPS certificate files are missing:\n"
            f"  - {files}\n"
            "Follow the mkcert setup in the root README, then try again."
        )

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        ssl_certfile=str(cert_file),
        ssl_keyfile=str(key_file),
    )


if __name__ == "__main__":
    main()
