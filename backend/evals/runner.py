"""Optional live-model evaluation runner. Never part of CI.

    python -m evals.runner --subset golden|representative|full [--ids a,b] [--no-judge]
                           [--live-provider] [--concurrency 3] [--out DIR] [--list]

Uses the configured OpenAI model and key (root .env) against synthetic Guardian profiles,
the local Manifest SQLite, the guide corpus, and the core knowledge corpus. Writes
results.json (machine-readable) and summary.md (human-readable).
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from evals.judge import judge
from evals.mechanics import mechanics_baseline
from evals.profiles import PROFILES, profile_facts
from evals.properties import evaluate_properties
from evals.report import record, summary_markdown, usage_stats
from evals.rubric import aggregate, score_card
from evals.runtime import DEFAULT_MANIFEST, EVAL_CACHE, build_knowledge, eval_settings, run_scenario
from evals.scenarios import SUBSETS


async def run(arguments: argparse.Namespace) -> int:
    scenarios = list(SUBSETS[arguments.subset])
    if arguments.ids:
        pool = {value.id: value for subset in SUBSETS.values() for value in subset}
        scenarios = [pool[value] for value in arguments.ids.split(",") if value in pool]
    if arguments.repeat > 1:
        from dataclasses import replace

        scenarios = [
            replace(value, id=f"{value.id}#{index}")
            for value in scenarios
            for index in range(1, arguments.repeat + 1)
        ]
    if arguments.list:
        for value in scenarios:
            print(value.id, value.profile, " | ".join(value.turns))
        return 0
    settings = eval_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not configured; live evaluation needs it.", file=sys.stderr)
        return 2
    manifest = Path(arguments.manifest) if arguments.manifest else DEFAULT_MANIFEST
    if manifest is None or not manifest.is_file():
        print("No local Manifest SQLite found; pass --manifest.", file=sys.stderr)
        return 2
    knowledge, http_client = await build_knowledge(
        settings, manifest, live_provider=arguments.live_provider
    )
    from openai import AsyncOpenAI

    judge_client = AsyncOpenAI(api_key=settings.openai_api_key) if arguments.judge else None
    semaphore = asyncio.Semaphore(arguments.concurrency)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    async def evaluate(scenario):  # noqa: ANN001, ANN202
        async with semaphore:
            context = PROFILES[scenario.profile]()
            facts = profile_facts(context)
            obs = await run_scenario(settings, knowledge, scenario, context)
            results = evaluate_properties(obs, facts)
            verdict = None
            if judge_client is not None and not obs.final.error:
                try:
                    verdict = await judge(judge_client, settings.openai_model, obs, facts)
                except Exception as exc:  # noqa: BLE001 - judge failure must not lose the run
                    print(f"judge failed for {scenario.id}: {exc}", file=sys.stderr)
            card = score_card(results, scenario.expect, obs.final.answer, facts, verdict)
            print(
                f"{scenario.id}: {card.total}/16 flags={card.flags} "
                f"latency={sum(t.latency_s for t in obs.turns):.1f}s",
                flush=True,
            )
            return record(obs, results, card), card

    try:
        outcomes = await asyncio.gather(*(evaluate(value) for value in scenarios))
    finally:
        if http_client is not None:
            await http_client.aclose()
        if judge_client is not None:
            await judge_client.close()
    records = [value[0] for value in outcomes]
    cards = [value[1] for value in outcomes]
    meta = {
        "subset": arguments.ids and "custom" or arguments.subset,
        "model": settings.openai_model,
        "judge": arguments.judge,
        "live_provider": arguments.live_provider,
        "started_at": started_at,
        "manifest": manifest.name,
    }
    out = (
        Path(arguments.out) if arguments.out else EVAL_CACHE / "runs" / started_at.replace(":", "")
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "aggregate": aggregate(cards),
                "usage": usage_stats(records),
                "mechanics_baseline": mechanics_baseline(records),
                "scenarios": records,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    (out / "summary.md").write_text(summary_markdown(meta, records, cards), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Destiny Copilot live-model evaluation")
    parser.add_argument("--subset", choices=sorted(SUBSETS), default="golden")
    parser.add_argument("--ids", help="Comma-separated scenario ids (overrides --subset)")
    parser.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-provider", action="store_true")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--manifest")
    parser.add_argument("--out")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="Run each scenario N times")
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
