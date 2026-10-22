"""Cassette recording and offline replay: deterministic, no network, and honest about mismatches."""

import asyncio
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.ai import SYSTEM_INSTRUCTIONS
from app.config import Settings
from evals import replay as replay_module
from evals.cassette import CASSETTE_NAME, CassetteWriter, fingerprint, load_cassette
from evals.model import Expectations, Scenario
from evals.pricing import estimate_cost, load_prices
from evals.profiles import PROFILES
from evals.report import cost_line, usage_stats
from evals.runtime import run_scenario
from tests.test_guardian_tools import assistant_message
from tests.test_playtest_scenarios import Script

SCENARIO = Scenario(
    id="replay.demo",
    profile="titan_veteran",
    turns=("What should I do next?", "Anything else?"),
    expect=Expectations(),
)
ANSWERS = [
    "Run Kepler Uplink for the Edge of Fate quest; it fits your window.",
    "Bring a long range weapon for the boss.",
]


class RecordingScript(Script):
    """A scripted client whose responses carry usage with cached input tokens."""

    async def create(self, **kwargs: Any) -> Any:
        response = await super().create(**kwargs)
        response.usage = SimpleNamespace(
            input_tokens=1000,
            input_tokens_details=SimpleNamespace(cached_tokens=400),
            output_tokens=50,
            output_tokens_details=SimpleNamespace(reasoning_tokens=10),
        )
        return response


def settings() -> Settings:
    return Settings(openai_api_key="", enable_debug_tools=True)


def record_run(tmp_path: Path, answers: list[str] | None = None) -> Path:
    """A live-style run against a scripted client, written the way the runner writes it."""

    answers = answers or ANSWERS
    script = RecordingScript(*[[assistant_message(value)] for value in answers])
    writer = CassetteWriter(tmp_path / CASSETTE_NAME)

    async def go() -> Any:
        from evals import runtime

        original = runtime.RecordingRecommendationService.__init__

        def init(self: Any, *args: Any, **kwargs: Any) -> None:
            original(self, *args, **kwargs)
            self.client = SimpleNamespace(responses=script, close=_close)

        runtime.RecordingRecommendationService.__init__ = init  # type: ignore[method-assign]
        try:
            return await run_scenario(
                settings(), None, SCENARIO, PROFILES[SCENARIO.profile](), cassette=writer
            )
        finally:
            runtime.RecordingRecommendationService.__init__ = original  # type: ignore[method-assign]

    observation = asyncio.run(go())
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "meta": {"model": "m", "started_at": "t0", "manifest": "x"},
                "scenarios": [
                    {
                        "id": SCENARIO.id,
                        "turns": [{"answer": turn.answer} for turn in observation.turns],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


async def _close() -> None:
    return None


def replay(run_dir: Path, **kwargs: Any) -> Any:
    return asyncio.run(
        replay_module.replay_run(run_dir, settings(), None, pool={SCENARIO.id: SCENARIO}, **kwargs)
    )


# --- Recording --------------------------------------------------------------------------------


def test_the_cassette_records_each_call_per_scenario_turn_and_round(tmp_path: Path) -> None:
    run_dir = record_run(tmp_path)
    calls = load_cassette(run_dir)

    assert sorted(calls[SCENARIO.id]) == [0, 1]
    first = calls[SCENARIO.id][0][0]
    assert first["round"] == 0
    assert set(first["fingerprint"]) == {"model", "instructions", "input", "tools", "options"}
    assert first["response"]["output"][0]["content"][0]["text"] == ANSWERS[0]


def test_the_cassette_never_contains_keys_tokens_or_request_bodies(tmp_path: Path) -> None:
    text = (record_run(tmp_path) / CASSETTE_NAME).read_text(encoding="utf-8")

    for forbidden in ("sk-", "Bearer", "authorization", "access_token", "api_key"):
        assert forbidden.lower() not in text.lower()
    # Requests are stored as hashes only, never as the instructions or input themselves.
    assert SYSTEM_INSTRUCTIONS[:60] not in text
    assert "What should I do next?" not in text


def test_a_fingerprint_changes_with_each_part_of_the_request() -> None:
    base = {"model": "m", "instructions": "a", "input": [1], "tools": [], "store": False}
    prints = {
        key: fingerprint({**base, key: value})
        for key, value in {
            "instructions": "b",
            "input": [2],
            "tools": [{"x": 1}],
            "store": True,
        }.items()
    }
    original = fingerprint(base)
    assert prints["instructions"]["instructions"] != original["instructions"]
    assert prints["input"]["input"] != original["input"]
    assert prints["tools"]["tools"] != original["tools"]
    assert prints["store"]["options"] != original["options"]
    assert fingerprint(dict(base)) == original


# --- Replay -----------------------------------------------------------------------------------


def test_replay_reproduces_the_run_deterministically(tmp_path: Path) -> None:
    run_dir = record_run(tmp_path)

    first, second = replay(run_dir), replay(run_dir)

    assert [value.status for value in first.scenarios] == ["unchanged"]
    assert [t["status"] for t in first.scenarios[0].turns] == ["unchanged", "unchanged"]
    one, two = first.scenarios[0], second.scenarios[0]
    assert [t.answer for t in one.observation.turns] == ANSWERS
    assert [t.answer for t in one.observation.turns] == [t.answer for t in two.observation.turns]
    assert one.record["score"] == two.record["score"]
    assert [t["answer"] for t in one.record["turns"]] == [t["answer"] for t in two.record["turns"]]
    assert first.scenarios[0].observation.final.cached_input_tokens == 400


def test_replay_makes_no_network_calls_and_uses_only_the_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = record_run(tmp_path)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access during replay")

    real_connect = socket.socket.connect

    def local_only(self: socket.socket, address: Any) -> None:
        # The event loop's own wake-up socket pair connects to loopback; nothing else may.
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError(f"network access during replay: {address}")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", local_only)
    monkeypatch.setattr(socket, "create_connection", refuse)
    import app.ai

    monkeypatch.setattr(app.ai, "AsyncOpenAI", refuse)

    from evals.cassette import ReplayClient

    served: list[ReplayClient] = []
    original = ReplayClient.__init__

    def tracking(self: ReplayClient, calls: Any) -> None:
        original(self, calls)
        served.append(self)

    monkeypatch.setattr(ReplayClient, "__init__", tracking)
    result = replay(run_dir)

    assert result.scenarios[0].status == "unchanged"
    assert len(served) == 1 and served[0].requests_served == 2
    observation = result.scenarios[0].observation
    assert observation is not None and observation.final.model_requests == 1


def test_a_changed_request_stops_the_scenario_as_needs_live(tmp_path: Path) -> None:
    run_dir = record_run(tmp_path)
    changed = Scenario(
        id=SCENARIO.id,
        profile=SCENARIO.profile,
        turns=("What should I do next?", "Something different entirely?"),
        expect=SCENARIO.expect,
    )

    result = asyncio.run(
        replay_module.replay_run(run_dir, settings(), None, pool={SCENARIO.id: changed})
    )

    scenario = result.scenarios[0]
    assert scenario.status == "needs_live"
    assert [t["status"] for t in scenario.turns] == ["unchanged", "needs_live"]
    assert "input" in scenario.turns[1]["detail"]
    assert scenario.record is None  # never scored from a guess


def test_a_request_with_no_recording_left_is_needs_live_not_a_guess(tmp_path: Path) -> None:
    run_dir = record_run(tmp_path)
    lines = (run_dir / CASSETTE_NAME).read_text(encoding="utf-8").splitlines()
    (run_dir / CASSETTE_NAME).write_text(lines[0] + "\n", encoding="utf-8")

    scenario = replay(run_dir).scenarios[0]

    assert scenario.status == "needs_live"
    assert [t["status"] for t in scenario.turns] == ["unchanged", "needs_live"]
    assert "never recorded" in scenario.turns[1]["detail"]


def test_a_different_model_or_option_is_a_mismatch(tmp_path: Path) -> None:
    run_dir = record_run(tmp_path)
    other = Settings(openai_api_key="", enable_debug_tools=True, openai_model="another-model")

    result = asyncio.run(
        replay_module.replay_run(run_dir, other, None, pool={SCENARIO.id: SCENARIO})
    )

    assert result.scenarios[0].status == "needs_live"
    assert "model" in result.scenarios[0].turns[0]["detail"]


def test_ids_select_scenarios_and_unknown_scenarios_are_skipped(tmp_path: Path) -> None:
    run_dir = record_run(tmp_path)

    assert replay(run_dir, ids=["something.else"]).scenarios == []
    gone = asyncio.run(replay_module.replay_run(run_dir, settings(), None, pool={}))
    assert gone.scenarios == [] and "no longer in the scenario bank" in gone.skipped[0]


def test_a_changed_answer_is_reported_with_before_and_after(tmp_path: Path) -> None:
    run_dir = record_run(tmp_path)
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    results["scenarios"][0]["turns"][1]["answer"] = "An older second answer."
    (run_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")

    scenario = replay(run_dir).scenarios[0]

    assert scenario.status == "changed"
    assert scenario.turns[1]["before"] == "An older second answer."
    assert scenario.turns[1]["after"] == ANSWERS[1]


def test_replay_outputs_results_and_summary_with_the_turn_diff(tmp_path: Path) -> None:
    run_dir = record_run(tmp_path / "run")
    result = replay(run_dir)

    out = tmp_path / "out"
    replay_module.write_outputs(
        result, run_dir, {"model": "m", "started_at": "t0"}, settings(), out
    )

    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert data["meta"]["replay_of"] == "t0" and data["replay"][0]["status"] == "unchanged"
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Replay" in summary and "1:unchanged, 2:unchanged" in summary
    assert "Run totals" in summary


# --- Cost visibility --------------------------------------------------------------------------


def _usage(**totals: int) -> dict[str, Any]:
    turn = {
        "latency_s": 1.0,
        "model_requests": 1,
        "tool_calls": [],
        "input_tokens": totals.get("input", 0),
        "cached_input_tokens": totals.get("cached", 0),
        "output_tokens": totals.get("output", 0),
        "web_search_calls": totals.get("search", 0),
    }
    return usage_stats([{"turns": [turn]}])


def test_cost_is_estimated_from_configured_prices_and_labeled() -> None:
    prices = {
        "input_per_1m": 2.0,
        "cached_input_per_1m": 0.5,
        "output_per_1m": 8.0,
        "web_search_per_1k": 10.0,
    }
    usage = _usage(input=1_000_000, cached=400_000, output=100_000, search=20)

    # 600k uncached * 2 + 400k cached * 0.5 + 100k output * 8 + 20 searches * 0.01
    assert estimate_cost(
        {
            "input_tokens": 1_000_000,
            "cached_input_tokens": 400_000,
            "output_tokens": 100_000,
            "web_search_calls": 20,
        },
        prices,
    ) == pytest.approx(1.2 + 0.2 + 0.8 + 0.2)
    line = cost_line(usage, prices)
    assert "1000000 input (400000 cached)" in line and "20 web-search calls" in line
    assert "est. cost ~$2.40" in line and "estimate" in line


def test_without_prices_no_cost_is_invented() -> None:
    assert "not estimated" in cost_line(_usage(input=10, output=5), {})
    assert estimate_cost({"input_tokens": 10, "output_tokens": 5}, {}) is None


def test_prices_come_from_the_environment_not_from_code(tmp_path: Path) -> None:
    env = {"EVAL_PRICE_INPUT_PER_1M": "1.5", "EVAL_PRICE_OUTPUT_PER_1M": "6"}
    assert load_prices(env, dotenv=None) == {"input_per_1m": 1.5, "output_per_1m": 6.0}
    dotenv = tmp_path / ".env"
    dotenv.write_text("EVAL_PRICE_INPUT_PER_1M=3\nOPENAI_API_KEY=sk-secret\n", encoding="utf-8")
    assert load_prices({}, dotenv=dotenv) == {"input_per_1m": 3.0}
    assert load_prices({"EVAL_PRICE_INPUT_PER_1M": "oops"}, dotenv=None) == {}


def test_a_real_sdk_response_round_trips_through_the_cassette() -> None:
    from openai.types.responses import Response

    from evals.cassette import jsonable, rebuild_response

    raw = {
        "id": "resp_1",
        "created_at": 0,
        "model": "m",
        "object": "response",
        "output": [
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "q"},
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hi.", "annotations": []}],
            },
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 0},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 12,
        },
    }
    original = Response.model_validate(raw)
    rebuilt = rebuild_response(jsonable(original))

    assert isinstance(rebuilt, Response)
    assert rebuilt.output[1].content[0].text == "Hi."
    assert rebuilt.usage.input_tokens_details.cached_tokens == 4
    assert jsonable(rebuilt) == jsonable(original)
