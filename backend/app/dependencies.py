from dataclasses import dataclass
from typing import Annotated

import httpx
from fastapi import Depends, Request

from app.ai import RecommendationService
from app.bungie.client import BungieClient
from app.bungie.guardian import GuardianService
from app.bungie.oauth import BungieOAuth
from app.config import Settings
from app.destiny_knowledge import DestinyKnowledgeService


@dataclass
class Services:
    settings: Settings
    http: httpx.AsyncClient
    bungie: BungieClient
    oauth: BungieOAuth
    guardian: GuardianService
    knowledge: DestinyKnowledgeService
    recommendations: RecommendationService


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]
