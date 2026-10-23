"""Live-model evaluation runtime: the exact production RecommendationService, wrapped only
to record model requests, usage, and tool calls.

The Manifest is read from the local mobile Manifest SQLite (read-only), so Manifest, Guide,
and Core tools need no Bungie network access. The live provider is optional; when enabled it
reaches Bungie's public endpoints with the configured API key.
"""

import asyncio
import json
import sqlite3
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.ai import CORE_FACTS_HEADER, SYSTEM_INSTRUCTIONS, RecommendationService
from app.config import Settings
from app.destiny_knowledge import DestinyKnowledgeService, ManifestKnowledgeProvider
from app.guardian_tools import GuardianToolService
from app.guide_knowledge import GuideKnowledgeProvider
from app.main import core_knowledge_provider
from app.models import ChatRequest, ChatTurn, GuardianContext
from evals.cassette import CassetteMismatch, CassetteWriter, ReplayClient
from evals.model import Scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = next(
    iter(sorted((REPO_ROOT / ".cache" / "manifest").glob("world-content-*.sqlite3"))), None
)
EVAL_CACHE = REPO_ROOT / ".cache" / "eval"


class LocalManifestResolver:
    """DefinitionResolver subset over a local Manifest SQLite file; never uses the network."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False
        )
        self._lock = asyncio.Lock()

    async def manifest_version(self) -> str:
        return f"local:{self.path.stem}"

    async def iter_definitions(self, entity_type: str) -> AsyncIterator[dict[str, Any]]:
        rows = await asyncio.to_thread(
            lambda: self._connection.execute(f'SELECT json FROM "{entity_type}"').fetchall()
        )
        for (raw,) in rows:
            yield json.loads(raw)

    async def resolve_many(
        self, entity_type: str, entity_hashes: set[int]
    ) -> dict[int, dict[str, Any]]:
        def query() -> dict[int, dict[str, Any]]:
            found: dict[int, dict[str, Any]] = {}
            for entity_hash in entity_hashes:
                signed = entity_hash - 2**32 if entity_hash >= 2**31 else entity_hash
                try:
                    row = self._connection.execute(
                        f'SELECT json FROM "{entity_type}" WHERE id = ?', (signed,)
                    ).fetchone()
                except sqlite3.Error:
                    return found
                if row:
                    found[entity_hash] = json.loads(row[0])
            return found

        async with self._lock:
            return await asyncio.to_thread(query)

    def release_table(self, entity_type: str) -> None:
        del entity_type


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class TurnRecord:
    message: str
    answer: str = ""
    source: str = ""
    trace: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    model_requests: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    web_search_calls: int = 0
    reasoning_tokens: int = 0
    instruction_chars: int = 0
    # Every assistant draft the model produced this turn, including ones the validator rejected.
    drafts: list[str] = field(default_factory=list)
    # ChatResponse.sources as the player sees them: title, url, and domain per source.
    sources: list[dict[str, Any]] = field(default_factory=list)
    core_block: str | None = None
    # Turn routing and planning context the model saw (instructions after the prefix).
    routing_context: str = ""
    latency_s: float = 0.0
    error: str | None = None


@dataclass
class Observation:
    scenario: Scenario
    context: GuardianContext
    turns: list[TurnRecord]

    @property
    def final(self) -> TurnRecord:
        return self.turns[-1]


class RecordingRecommendationService(RecommendationService):
    """Production service; overrides only record requests, usage, and tool results."""

    current: TurnRecord | None = None
    # Where the next call belongs in the cassette; set by run_scenario.
    cassette: CassetteWriter | None = None
    scenario_id: str = ""
    turn_index: int = 0

    async def _create_response(self, *, stream_status: Any, **request_options: Any) -> Any:
        response = await super()._create_response(stream_status=stream_status, **request_options)
        if self.cassette is not None and self.current is not None:
            self.cassette.write(
                self.scenario_id,
                self.turn_index,
                self.current.model_requests,
                request_options,
                response,
            )
        if self.current is not None:
            self.current.model_requests += 1
            self.current.web_search_calls += sum(
                1
                for item in self._field(response, "output", []) or []
                if self._field(item, "type") == "web_search_call"
            )
            draft = self._extract_assistant_text(response)
            if draft:
                self.current.drafts.append(draft)
            instructions = request_options.get("instructions") or ""
            self.current.instruction_chars = len(instructions)
            routing = instructions.removeprefix(SYSTEM_INSTRUCTIONS).strip()
            self.current.routing_context = routing.split(CORE_FACTS_HEADER, 1)[0].strip()
            if CORE_FACTS_HEADER in instructions:
                self.current.core_block = instructions.split(CORE_FACTS_HEADER + "\n", 1)[-1]
            usage = self._field(response, "usage")
            if usage is not None:
                self.current.input_tokens += int(self._field(usage, "input_tokens", 0) or 0)
                input_details = self._field(usage, "input_tokens_details")
                self.current.cached_input_tokens += int(
                    self._field(input_details, "cached_tokens", 0) or 0
                )
                self.current.output_tokens += int(self._field(usage, "output_tokens", 0) or 0)
                details = self._field(usage, "output_tokens_details")
                self.current.reasoning_tokens += int(
                    self._field(details, "reasoning_tokens", 0) or 0
                )
        return response

    async def _execute_tool_call(
        self, tools: GuardianToolService, name: str, arguments_json: str
    ) -> dict[str, Any]:
        result = await super()._execute_tool_call(tools, name, arguments_json)
        if self.current is not None:
            try:
                arguments = json.loads(arguments_json)
            except (TypeError, json.JSONDecodeError):
                arguments = {"raw": str(arguments_json)[:200]}
            self.current.tool_calls.append(ToolCallRecord(name, arguments, result))
        return result


def eval_settings(**overrides: Any) -> Settings:
    EVAL_CACHE.mkdir(parents=True, exist_ok=True)
    values = {
        "manifest_cache_dir": EVAL_CACHE / "manifest-index",
        "live_cache_dir": EVAL_CACHE / "live",
        "guide_cache_dir": EVAL_CACHE / "guides",
        "enable_debug_tools": True,
        **overrides,
    }
    return Settings(**values)


async def build_knowledge(
    settings: Settings, manifest_path: Path, *, live_provider: bool
) -> tuple[DestinyKnowledgeService, Any]:
    resolver = LocalManifestResolver(manifest_path)
    manifest = ManifestKnowledgeProvider(resolver, settings)  # type: ignore[arg-type]
    await manifest.warm_index()
    providers: list[Any] = [manifest, GuideKnowledgeProvider(manifest, settings)]
    http_client = None
    if live_provider:
        import httpx

        from app.bungie.client import BungieClient
        from app.live_destiny import LiveDestinyProvider

        http_client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)
        providers.append(
            LiveDestinyProvider(
                BungieClient(settings, http_client),
                resolver,  # type: ignore[arg-type]
                settings,
                canonical=manifest,
            )
        )
    core = core_knowledge_provider(settings)
    if core is not None:
        providers.append(core)
    return DestinyKnowledgeService(providers), http_client


async def run_scenario(
    settings: Settings,
    knowledge: DestinyKnowledgeService,
    scenario: Scenario,
    context: GuardianContext,
    *,
    cassette: CassetteWriter | None = None,
    client: ReplayClient | None = None,
    now: datetime | None = None,
) -> Observation:
    """Replay every user turn live; the model's own answers become the history.

    `cassette` records each model call. `client` swaps in a recorded-response stub (see
    evals.replay); a request that differs from the recording ends the scenario with an error
    that starts with "needs_live". `now` pins the time the prompts carry.
    """

    service = RecordingRecommendationService(settings, knowledge)
    if client is not None:
        service.client = client
    service.cassette = cassette
    service.scenario_id = scenario.id
    if now is not None:
        # The prompt carries the date; a run and its replay must carry the same one.
        service.clock = lambda: now
    history: list[ChatTurn] = []
    turns: list[TurnRecord] = []
    try:
        for index, message in enumerate(scenario.turns):
            record = TurnRecord(message=message)
            service.current = record
            service.turn_index = index
            if client is not None:
                client.begin(scenario.id, index)
            started = time.perf_counter()
            try:
                response = await service.chat(
                    ChatRequest(message=message, history=history[-12:]), context
                )
                record.answer = response.message
                record.source = response.source
                record.sources = [value.model_dump() for value in response.sources]
            except CassetteMismatch as exc:
                record.error = f"needs_live: {exc}"[:300]
            except Exception as exc:  # noqa: BLE001 - recorded as a tool-loop failure
                record.error = f"{type(exc).__name__}: {exc}"[:300]
            record.latency_s = round(time.perf_counter() - started, 2)
            record.trace = service.latest_trace() or {}
            turns.append(record)
            if record.error:
                break
            history.append(ChatTurn(role="user", content=message))
            history.append(ChatTurn(role="assistant", content=record.answer[:4000] or "(none)"))
    finally:
        await service.close()
    return Observation(scenario, context, turns)
