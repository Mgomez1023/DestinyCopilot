import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ai import RecommendationService
from app.bungie.client import BungieClient
from app.bungie.guardian import GuardianService
from app.bungie.manifest import DefinitionResolver
from app.bungie.oauth import BungieOAuth, create_oauth_session_store
from app.config import Settings, get_settings
from app.dependencies import Services
from app.destiny_knowledge import BungieManifestProvider, DestinyKnowledgeService
from app.guardian_refresh import GuardianRefreshService
from app.guide_knowledge import GuideKnowledgeProvider
from app.live_destiny import LiveDestinyProvider
from app.routes import (
    auth,
    chat,
    debug,
    guardian,
    guardian_refresh_debug,
    knowledge_debug,
    trace_debug,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        http_client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)
        bungie_client = BungieClient(settings, http_client)
        resolver = DefinitionResolver(bungie_client)
        manifest_knowledge = BungieManifestProvider(resolver, settings)
        guide_knowledge = GuideKnowledgeProvider(manifest_knowledge, settings)
        live_knowledge = LiveDestinyProvider(
            bungie_client, resolver, settings, canonical=manifest_knowledge
        )
        knowledge = DestinyKnowledgeService([manifest_knowledge, guide_knowledge, live_knowledge])
        guardian_service = GuardianService(bungie_client, resolver)
        oauth_store = create_oauth_session_store(settings)
        application.state.services = Services(
            settings=settings,
            http=http_client,
            bungie=bungie_client,
            oauth=BungieOAuth(settings, http_client, oauth_store),
            oauth_store=oauth_store,
            guardian=guardian_service,
            knowledge=knowledge,
            recommendations=RecommendationService(settings, knowledge),
            guardian_refresh=GuardianRefreshService(guardian_service, settings),
        )
        yield
        if application.state.services.guardian_refresh is not None:
            await application.state.services.guardian_refresh.close()
        await application.state.services.recommendations.close()
        await http_client.aclose()

    application = FastAPI(
        title="Guardian Copilot API",
        description="Read-only Destiny 2 account context and AI recommendations.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin.rstrip("/")],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=600,
    )
    application.include_router(auth.router)
    application.include_router(guardian.router)
    application.include_router(chat.router)
    if settings.debug_tools_enabled:
        application.include_router(debug.router)
        application.include_router(guardian_refresh_debug.router)
        application.include_router(knowledge_debug.router)
        application.include_router(trace_debug.router)

    @application.get("/api/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
