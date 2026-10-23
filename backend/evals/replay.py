"""Replay a recorded eval run through the CURRENT production service, offline.

    python -m evals.replay <run_dir> [--ids a,b] [--manifest PATH] [--out DIR]

`<run_dir>` is a live run directory holding results.json and cassette.jsonl. Each scenario runs
against the same simulated profile with a stub client that returns the recorded model responses
in order. No OpenAI, Bungie, or web request is made; web-search results are whatever the
recorded responses contained. If the service now sends a request that differs from the
recording (any of model, instructions, input and tool outputs, tools, other options), the
scenario stops there and is marked `needs_live`: the replay never guesses what the model would
have said. Use it for changes to post-processing (validators, repairs, answer assembly); a
change to prompts, tools, or retrieval needs a live run.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.cassette import CASSETTE_NAME, ReplayClient, load_cassette
from evals.model import Scenario
from evals.pricing import load_prices
from evals.profiles import PROFILES, profile_facts
from evals.properties import evaluate_properties
from evals.report import record, summary_markdown, usage_stats
from evals.rubric import aggregate, score_card
from evals.runtime import (
    DEFAULT_MANIFEST,
    EVAL_CACHE,
    Observation,
    build_knowledge,
    eval_settings,
    run_scenario,
)
from evals.scenarios import SUBSETS

CHANGED, UNCHANGED, NEEDS_LIVE = "changed", "unchanged", "needs_live"


@dataclass
class ReplayedScenario:
    id: str
    status: str
    turns: list[dict[str, Any]]
    observation: Observation | None = None
    record: dict[str, Any] | None = None
    detail: str = ""


@dataclass
class ReplayResult:
    scenarios: list[ReplayedScenario] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def scenario_pool() -> dict[str, Scenario]:
    return {value.id: value for subset in SUBSETS.values() for value in subset}


def _base_id(scenario_id: str) -> str:
    return scenario_id.split("#", 1)[0]


def _diff_turns(recorded_turns: list[dict[str, Any]], observation: Observation) -> list[dict]:
    """Per-turn status of the final answer against the recorded run."""

    diffs: list[dict[str, Any]] = []
    for index, before in enumerate(recorded_turns):
        turn = observation.turns[index] if index < len(observation.turns) else None
        if turn is None or (turn.error or "").startswith(NEEDS_LIVE):
            reason = turn.error if turn is not None else "an earlier turn needs a live run"
            diffs.append({"turn": index, "status": NEEDS_LIVE, "detail": reason})
            continue
        status = UNCHANGED if turn.answer == before.get("answer") else CHANGED
        diffs.append(
            {
                "turn": index,
                "status": status,
                "before": before.get("answer"),
                "after": turn.answer,
            }
        )
    return diffs


async def replay_run(
    run_dir: Path,
    settings: Any,
    knowledge: Any,
    *,
    ids: list[str] | None = None,
    pool: dict[str, Scenario] | None = None,
) -> ReplayResult:
    """Replay the scenarios of a recorded run. Makes no network calls: the only model client
    is the stub built from the cassette."""

    recorded = json.loads((Path(run_dir) / "results.json").read_text(encoding="utf-8"))
    started = recorded.get("meta", {}).get("started_at")
    # The recorded run's clock; a run from before prompts carried the date has none to match.
    try:
        now = datetime.fromisoformat(started) if started else None
    except ValueError:
        now = None
    cassette = load_cassette(Path(run_dir))
    pool = pool if pool is not None else scenario_pool()
    result = ReplayResult()
    for entry in recorded["scenarios"]:
        scenario_id = entry["id"]
        if ids and scenario_id not in ids and _base_id(scenario_id) not in ids:
            continue
        base = pool.get(_base_id(scenario_id))
        if base is None:
            result.skipped.append(f"{scenario_id}: no longer in the scenario bank")
            continue
        scenario = replace(base, id=scenario_id)
        client = ReplayClient(cassette)
        context = PROFILES[scenario.profile]()
        observation = await run_scenario(
            settings, knowledge, scenario, context, client=client, now=now
        )
        diffs = _diff_turns(entry["turns"], observation)
        needs_live = any(value["status"] == NEEDS_LIVE for value in diffs)
        replayed = ReplayedScenario(
            id=scenario_id,
            status=NEEDS_LIVE
            if needs_live
            else CHANGED
            if any(value["status"] == CHANGED for value in diffs)
            else UNCHANGED,
            turns=diffs,
            observation=observation,
        )
        if not needs_live:
            facts = profile_facts(context)
            properties = evaluate_properties(observation, facts)
            card = score_card(properties, scenario.expect, observation.final.answer, facts, None)
            replayed.record = record(observation, properties, card)
            replayed.record["_card"] = card
        else:
            replayed.detail = next(v["detail"] for v in diffs if v["status"] == NEEDS_LIVE)
        result.scenarios.append(replayed)
    return result


def replay_section(result: ReplayResult, source: Path) -> str:
    counts = {
        status: sum(1 for value in result.scenarios if value.status == status)
        for status in (UNCHANGED, CHANGED, NEEDS_LIVE)
    }
    lines = [
        "## Replay",
        "",
        f"Replayed from `{source}` with recorded model responses; no network calls. Latency and "
        "token figures are the recorded ones, not new measurements.",
        "",
        f"**Scenarios** {counts[UNCHANGED]} unchanged · {counts[CHANGED]} changed · "
        f"{counts[NEEDS_LIVE]} needs_live",
        "",
        "| Scenario | Status | Turn diffs |",
        "|---|---|---|",
    ]
    for value in result.scenarios:
        turns = ", ".join(f"{t['turn'] + 1}:{t['status']}" for t in value.turns)
        lines.append(f"| {value.id} | {value.status} | {turns} |")
    for value in result.scenarios:
        for turn in value.turns:
            if turn["status"] == NEEDS_LIVE:
                lines.append(
                    f"\n- `{value.id}` turn {turn['turn'] + 1} needs_live: {turn['detail']}"
                )
    for value in result.scenarios:
        for turn in value.turns:
            if turn["status"] == CHANGED:
                lines += [
                    "",
                    f"### {value.id} turn {turn['turn'] + 1}: changed",
                    "",
                    "Before:",
                    "",
                    f"> {turn['before']}".replace("\n", "\n> "),
                    "",
                    "After:",
                    "",
                    f"> {turn['after']}".replace("\n", "\n> "),
                ]
    for note in result.skipped:
        lines.append(f"\n- skipped {note}")
    return "\n".join(lines) + "\n"


def write_outputs(
    result: ReplayResult, source: Path, recorded_meta: dict[str, Any], settings: Any, out: Path
) -> None:
    scored = [value for value in result.scenarios if value.record is not None]
    cards = [value.record.pop("_card") for value in scored]  # type: ignore[union-attr]
    records = [value.record for value in scored]
    meta = {
        "subset": "replay",
        "model": recorded_meta.get("model", settings.openai_model),
        "judge": False,
        "live_provider": False,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "manifest": recorded_meta.get("manifest"),
        "replay_of": recorded_meta.get("started_at"),
        "prices": load_prices(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "aggregate": aggregate(cards) if cards else None,
                "usage": usage_stats(records) if records else None,
                "replay": [
                    {"id": v.id, "status": v.status, "turns": v.turns, "detail": v.detail}
                    for v in result.scenarios
                ],
                "skipped": result.skipped,
                "scenarios": records,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    head = (
        summary_markdown(meta, records, cards)
        if cards
        else "# Destiny Copilot eval — replay (0 scored scenarios)\n\n"
    )
    (out / "summary.md").write_text(head + "\n" + replay_section(result, source), encoding="utf-8")


async def run(arguments: argparse.Namespace) -> int:
    source = Path(arguments.run_dir)
    if not (source / CASSETTE_NAME).is_file() or not (source / "results.json").is_file():
        print(f"{source} needs results.json and {CASSETTE_NAME} from a live run.", file=sys.stderr)
        return 2
    manifest = Path(arguments.manifest) if arguments.manifest else DEFAULT_MANIFEST
    if manifest is None or not manifest.is_file():
        print("No local Manifest SQLite found; pass --manifest.", file=sys.stderr)
        return 2
    # No API key: the production service builds no OpenAI client; the stub is the only one.
    settings = eval_settings(openai_api_key="")
    knowledge, http_client = await build_knowledge(settings, manifest, live_provider=False)
    ids = [value for value in (arguments.ids or "").split(",") if value] or None
    recorded_meta = json.loads((source / "results.json").read_text(encoding="utf-8"))["meta"]
    try:
        result = await replay_run(source, settings, knowledge, ids=ids)
    finally:
        if http_client is not None:
            await http_client.aclose()
    stamp = datetime.now(UTC).isoformat(timespec="seconds").replace(":", "")
    out = Path(arguments.out) if arguments.out else EVAL_CACHE / "runs" / f"replay-{stamp}"
    write_outputs(result, source, recorded_meta, settings, out)
    for value in result.scenarios:
        print(f"{value.id}: {value.status}", flush=True)
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline replay of a recorded eval run")
    parser.add_argument("run_dir")
    parser.add_argument("--ids", help="Comma-separated scenario ids to replay")
    parser.add_argument("--manifest")
    parser.add_argument("--out")
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
