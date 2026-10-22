"""Cassettes: every raw model request fingerprint and response of an eval run.

A live run records, per scenario/turn/round, a fingerprint of the OpenAI request (model,
instructions, input items including tool outputs, tools, and the remaining options, each
hashed) and the full raw response. `evals.replay` feeds those responses back to the current
production service through a stub client. Request bodies are never stored: only hashes. API
keys live in the client and are never part of the request options, so they cannot leak here.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CASSETTE_NAME = "cassette.jsonl"
HASHED_PARTS = ("instructions", "input", "tools")


class CassetteMismatch(BaseException):  # noqa: N818 - must not be swallowed by `except Exception`
    """The current request is not the recorded one (or nothing was recorded): needs a live run."""


def jsonable(value: Any) -> Any:
    """Plain JSON data for an SDK model, namespace, dict, or list."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(inner) for inner in value]
    if isinstance(value, SimpleNamespace):
        return {key: jsonable(inner) for key, inner in vars(value).items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _hash(value: Any) -> str:
    text = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def fingerprint(request_options: dict[str, Any]) -> dict[str, str]:
    """Hashes of the parts of a request that decide what the model can answer."""

    rest = {
        key: value for key, value in request_options.items() if key not in {"model", *HASHED_PARTS}
    }
    return {
        "model": str(request_options.get("model", "")),
        "instructions": _hash(request_options.get("instructions")),
        "input": _hash(request_options.get("input")),
        "tools": _hash(request_options.get("tools")),
        "options": _hash(rest),
    }


def differing_parts(recorded: dict[str, str], current: dict[str, str]) -> list[str]:
    return [key for key in recorded if recorded.get(key) != current.get(key)]


class CassetteWriter:
    """Appends one JSON line per model call. Safe to share between scenarios of one run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def write(
        self,
        scenario: str,
        turn: int,
        round_number: int,
        request_options: dict[str, Any],
        response: Any,
    ) -> None:
        line = {
            "scenario": scenario,
            "turn": turn,
            "round": round_number,
            "fingerprint": fingerprint(request_options),
            "response": jsonable(response),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def load_cassette(run_dir: Path) -> dict[str, dict[int, list[dict[str, Any]]]]:
    """scenario -> turn -> recorded calls in round order."""

    path = Path(run_dir) / CASSETTE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No {CASSETTE_NAME} in {run_dir}; record it with a live run.")
    calls: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        entry = json.loads(raw)
        calls.setdefault(entry["scenario"], {}).setdefault(entry["turn"], []).append(entry)
    for turns in calls.values():
        for rounds in turns.values():
            rounds.sort(key=lambda value: value["round"])
    return calls


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(inner) for key, inner in value.items()})
    if isinstance(value, list):
        return [_namespace(inner) for inner in value]
    return value


def rebuild_response(raw: dict[str, Any]) -> Any:
    """The recorded response as an SDK object when it validates, else as a namespace."""

    try:
        from openai.types.responses import Response

        return Response.model_validate(raw)
    except Exception:  # noqa: BLE001 - a partial recording still replays as attributes
        return _namespace(raw)


@dataclass
class _Position:
    scenario: str
    turn: int
    round: int = 0


class _Responses:
    def __init__(self, client: "ReplayClient") -> None:
        self.client = client

    async def create(self, **request_options: Any) -> Any:
        return self.client.next_response(request_options)

    def stream(self, **request_options: Any) -> Any:
        raise CassetteMismatch("streaming requests are not recorded by the eval runner")


class ReplayClient:
    """Stands in for AsyncOpenAI: returns recorded responses in order, never touches the network.

    Call `begin(scenario, turn)` before each turn. A request whose fingerprint differs from the
    recording, or that has no recording left, raises CassetteMismatch.
    """

    def __init__(self, calls: dict[str, dict[int, list[dict[str, Any]]]]) -> None:
        self.calls = calls
        self.responses = _Responses(self)
        self.position = _Position("", 0)
        self.requests_served = 0

    def begin(self, scenario: str, turn: int) -> None:
        self.position = _Position(scenario, turn)

    def next_response(self, request_options: dict[str, Any]) -> Any:
        where = self.position
        recorded = self.calls.get(where.scenario, {}).get(where.turn, [])
        if where.round >= len(recorded):
            raise CassetteMismatch(
                f"{where.scenario} turn {where.turn}: round {where.round} was never recorded"
            )
        entry = recorded[where.round]
        changed = differing_parts(entry["fingerprint"], fingerprint(request_options))
        if changed:
            raise CassetteMismatch(
                f"{where.scenario} turn {where.turn} round {where.round}: request differs in "
                + ", ".join(changed)
            )
        where.round += 1
        self.requests_served += 1
        return rebuild_response(entry["response"])

    async def close(self) -> None:
        return None
