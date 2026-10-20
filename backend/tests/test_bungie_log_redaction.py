"""Bungie client warnings must not carry membership or character IDs."""

import logging

import httpx
import pytest

from app.bungie.client import BungieAPIError, BungieClient, loggable_path
from app.config import Settings

MEMBERSHIP_ID = "4611686018467890123"
CHARACTER_ID = "2305843009301234567"


def _client(handler: httpx.MockTransport) -> BungieClient:
    return BungieClient(Settings(_env_file=None), httpx.AsyncClient(transport=handler))


def _record_text(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    "path",
    [
        f"/Destiny2/3/Profile/{MEMBERSHIP_ID}/",
        f"/Destiny2/3/Account/{MEMBERSHIP_ID}/Character/{CHARACTER_ID}/Stats/",
    ],
)
@pytest.mark.asyncio
async def test_http_failure_log_masks_identifiers_but_keeps_status(
    caplog: pytest.LogCaptureFixture, path: str
) -> None:
    client = _client(httpx.MockTransport(lambda _request: httpx.Response(503)))

    with (
        caplog.at_level(logging.WARNING, logger="app.bungie.client"),
        pytest.raises(BungieAPIError),
    ):
        await client._get(path, access_token="secret")

    text = _record_text(caplog)
    assert MEMBERSHIP_ID not in text and CHARACTER_ID not in text and "secret" not in text
    assert "status=503" in text
    assert "/Destiny2/3/" in text


@pytest.mark.asyncio
async def test_bungie_api_error_log_masks_identifiers_but_keeps_error_category(
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = {"ErrorCode": 1601, "ErrorStatus": "DestinyAccountNotFound", "Message": "x"}
    client = _client(httpx.MockTransport(lambda _request: httpx.Response(200, json=payload)))

    with (
        caplog.at_level(logging.WARNING, logger="app.bungie.client"),
        pytest.raises(BungieAPIError),
    ):
        await client._get(f"/Destiny2/3/Profile/{MEMBERSHIP_ID}/")

    text = _record_text(caplog)
    assert MEMBERSHIP_ID not in text
    assert "error=DestinyAccountNotFound" in text
    assert "/Profile/{id}/" in text


def test_loggable_path_keeps_endpoint_shape() -> None:
    assert loggable_path("/Destiny2/Milestones/") == "/Destiny2/Milestones/"
    assert loggable_path(f"/Destiny2/2/Profile/{MEMBERSHIP_ID}/") == "/Destiny2/2/Profile/{id}/"
