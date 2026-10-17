import asyncio
import logging
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ai import RecommendationService
from app.bungie.client import BungieClient
from app.bungie.guardian import GuardianService
from app.bungie.manifest import DefinitionResolver
from app.bungie.oauth import BungieOAuth, create_oauth_session_store
from app.config import Settings, get_settings
from app.core_knowledge import CoreKnowledgeError, DestinyKnowledgeEngine
from app.core_knowledge_provider import CoreKnowledgeProvider
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

logger = logging.getLogger(__name__)


def core_knowledge_provider(settings: Settings) -> CoreKnowledgeProvider | None:
    """Load the curated corpus once at startup; a bad corpus disables only this provider."""

    if not settings.core_knowledge_enabled:
        return None
    try:
        engine = DestinyKnowledgeEngine.from_file(
            settings.resolve_local_path(settings.core_knowledge_file)
        )
    except CoreKnowledgeError as exc:
        logger.error("Core knowledge disabled: %s", exc)
        return None
    logger.info(
        "Core knowledge ready corpus=%s entities=%d", engine.corpus_version, engine.entity_count
    )
    return CoreKnowledgeProvider(engine)


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
        core_knowledge = core_knowledge_provider(settings)
        knowledge = DestinyKnowledgeService(
            [
                manifest_knowledge,
                guide_knowledge,
                live_knowledge,
                *([core_knowledge] if core_knowledge is not None else []),
            ]
        )
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
        # Background, not awaited: the server becomes healthy immediately, and a request
        # that arrives mid-warmup waits on the index lock instead of building it twice.
        warmup = (
            asyncio.create_task(manifest_knowledge.warm_index(), name="manifest-index-warmup")
            if settings.manifest_index_warmup and settings.bungie_api_key
            else None
        )
        if warmup is None:
            manifest_knowledge.warmup_state = {"state": "disabled"}
        yield
        if warmup is not None and not warmup.done():
            warmup.cancel()
            with suppress(asyncio.CancelledError):
                await warmup
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
