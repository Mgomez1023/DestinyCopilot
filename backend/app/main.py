import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ai import RecommendationService
from app.bungie.client import BungieClient
from app.bungie.guardian import GuardianService
from app.bungie.manifest import DefinitionResolver
from app.bungie.oauth import BungieOAuth
from app.config import get_settings
from app.dependencies import Services
from app.destiny_knowledge import DestinyKnowledgeService, ManifestKnowledgeProvider
from app.routes import auth, chat, debug, guardian, knowledge_debug

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    http_client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)
    bungie_client = BungieClient(settings, http_client)
    resolver = DefinitionResolver(bungie_client)
    knowledge = DestinyKnowledgeService([ManifestKnowledgeProvider(resolver, settings)])
    app.state.services = Services(
        settings=settings,
        http=http_client,
        bungie=bungie_client,
        oauth=BungieOAuth(settings, http_client),
        guardian=GuardianService(bungie_client, resolver),
        knowledge=knowledge,
        recommendations=RecommendationService(settings, knowledge),
    )
    yield
    await app.state.services.recommendations.close()
    await http_client.aclose()


app = FastAPI(
    title="Guardian Copilot API",
    description="Read-only Destiny 2 account context and AI recommendations.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(auth.router)
app.include_router(guardian.router)
app.include_router(chat.router)
app.include_router(debug.router)
app.include_router(knowledge_debug.router)


@app.get("/api/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
