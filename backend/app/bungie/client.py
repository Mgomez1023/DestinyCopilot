import logging
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

BUNGIE_PLATFORM_URL = "https://www.bungie.net/Platform"
BUNGIE_ROOT_URL = "https://www.bungie.net"


class BungieAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BungieClient:
    """Small read-only client for the official Bungie.net Platform API."""

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.http = http_client

    async def _get(
        self,
        path: str,
        *,
        access_token: str | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        headers = {"X-API-Key": self.settings.bungie_api_key}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        try:
            response = await self.http.get(
                f"{BUNGIE_PLATFORM_URL}{path}", headers=headers, params=params
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            logger.warning("Bungie request failed path=%s status=%s", path, status)
            raise BungieAPIError("Bungie.net could not be reached.", status_code=status) from exc

        if payload.get("ErrorCode") != 1:
            error_status = payload.get("ErrorStatus", "UnknownError")
            message = payload.get("Message", "Bungie.net returned an error.")
            logger.warning("Bungie API error path=%s error=%s", path, error_status)
            raise BungieAPIError(f"{error_status}: {message}")
        return payload.get("Response", {})

    async def get_current_memberships(self, access_token: str) -> dict[str, Any]:
        return await self._get("/User/GetMembershipsForCurrentUser/", access_token=access_token)

    async def get_profile(
        self,
        membership_type: int,
        membership_id: str,
        access_token: str,
        *,
        components: tuple[str, ...] | set[str] | None = None,
    ) -> dict[str, Any]:
        requested_components = components or (
            "Profiles",
            "ProfileInventories",
            "ProfileCurrencies",
            "ProfileProgression",
            "Characters",
            "CharacterInventories",
            "CharacterProgressions",
            "CharacterActivities",
            "CharacterEquipment",
            "ItemInstances",
            "ItemObjectives",
            "ItemSockets",
            "ItemStats",
            "Collectibles",
            "Records",
            "Craftables",
        )
        return await self._get(
            f"/Destiny2/{membership_type}/Profile/{membership_id}/",
            access_token=access_token,
            params={"components": ",".join(sorted(requested_components))},
        )

    async def get_activity_history(
        self,
        membership_type: int,
        membership_id: str,
        character_id: str,
        access_token: str,
        *,
        count: int = 5,
    ) -> dict[str, Any]:
        """Return recent activities, newest first, using Destiny2.GetActivityHistory."""
        return await self._get(
            (
                f"/Destiny2/{membership_type}/Account/{membership_id}/"
                f"Character/{character_id}/Stats/Activities/"
            ),
            access_token=access_token,
            params={"count": str(count), "page": "0"},
        )

    async def get_manifest(self) -> dict[str, Any]:
        return await self._get("/Destiny2/Manifest/")

    async def get_public_milestones(self) -> dict[str, Any]:
        """Return Bungie's public, current milestone view without player data."""
        return await self._get("/Destiny2/Milestones/")

    async def get_public_vendors(self) -> dict[str, Any]:
        """Return the small public vendor subset; this endpoint requires no OAuth token."""
        return await self._get("/Destiny2/Vendors/", params={"components": "400,401,402"})

    async def get_public_json(self, path: str) -> dict[str, Any]:
        """Fetch a Bungie-hosted JSON manifest component without Platform wrapping."""
        url = path if path.startswith("https://") else f"{BUNGIE_ROOT_URL}{path}"
        try:
            response = await self.http.get(url, timeout=120.0)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            logger.warning("Bungie manifest download failed url=%s status=%s", url, status)
            raise BungieAPIError(
                "The Destiny manifest could not be downloaded.", status_code=status
            ) from exc
        if not isinstance(payload, dict):
            raise BungieAPIError("The Destiny manifest component was not a JSON object.")
        return payload

    async def get_definition(self, entity_type: str, entity_hash: int) -> dict[str, Any]:
        return await self._get(f"/Destiny2/Manifest/{entity_type}/{entity_hash}/")
