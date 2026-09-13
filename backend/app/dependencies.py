from dataclasses import dataclass
from typing import Annotated

import httpx
from fastapi import Depends, Request

from app.ai import RecommendationService
from app.bungie.client import BungieClient
from app.bungie.guardian import GuardianService
from app.bungie.oauth import BungieOAuth, OAuthSessionStore
from app.config import Settings
from app.destiny_knowledge import DestinyKnowledgeService
from app.guardian_refresh import GuardianRefreshService


@dataclass
class Services:
    settings: Settings
    http: httpx.AsyncClient
    bungie: BungieClient
    oauth: BungieOAuth
    oauth_store: OAuthSessionStore
    guardian: GuardianService
    knowledge: DestinyKnowledgeService
    recommendations: RecommendationService
    guardian_refresh: GuardianRefreshService | None = None


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]
