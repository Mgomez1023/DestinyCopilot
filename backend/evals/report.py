"""Machine-readable records and the human-readable summary for an eval run."""

import json
from collections import Counter
from statistics import mean, median
from typing import Any

from evals.properties import PropertyResult
from evals.rubric import DIMENSIONS, MAX_SCORE, ScoreCard, aggregate
from evals.runtime import Observation

MAX_RESULT_CHARS = 1500


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def record(obs: Observation, results: list[PropertyResult], card: ScoreCard) -> dict[str, Any]:
    turns = []
    for turn in obs.turns:
        trace = turn.trace
        turns.append(
            {
                "message": turn.message,
                "answer": turn.answer,
                "source": turn.source,
                "error": turn.error,
                "latency_s": turn.latency_s,
                "model_requests": turn.model_requests,
                "input_tokens": turn.input_tokens,
                "output_tokens": turn.output_tokens,
                "reasoning_tokens": turn.reasoning_tokens,
                "instruction_chars": turn.instruction_chars,
                "tool_calls": [
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "result": json.dumps(call.result, ensure_ascii=False)[:MAX_RESULT_CHARS],
                    }
                    for call in turn.tool_calls
                ],
                "grounding": trace.get("grounding"),
                "knowledge_categories_used": trace.get("knowledge_categories_used"),
                "core_knowledge": trace.get("core_knowledge"),
                "core_block": turn.core_block,
                "drafts": turn.drafts,
                "current_character": trace.get("current_character"),
                "character_selections": trace.get("character_selections"),
                "ownership_validation": trace.get("ownership_validation"),
                "conversation_scope": trace.get("conversation_scope"),
                "validator": trace.get("planning_correction"),
                "initiative_retry": trace.get("initiative_retry"),
                "item_lookup_forced": trace.get("item_lookup_forced"),
                "trailing_handback_removed": bool(trace.get("trailing_handback_removed")),
                "tool_budget": trace.get("tool_budget"),
                "answer_kind": trace.get("answer"),
                "build_analysis": trace.get("build_analysis"),
                "required_live_topic": trace.get("required_live_topic"),
                "mechanics_claims": trace.get("mechanics_claims"),
            }
        )
    return {
        "id": obs.scenario.id,
        "profile": obs.scenario.profile,
        "golden": obs.scenario.golden,
        "tags": obs.scenario.tags,
        "note": obs.scenario.note,
        "turns": turns,
        "properties": [result.__dict__ for result in results if result.applies],
        "score": {
            "total": card.total,
            "max": MAX_SCORE,
            "scores": card.scores,
            "source": card.score_source,
            "deterministic_scores": card.deterministic_scores,
            "flags": card.flags,
            "deterministic_flags": card.deterministic_flags,
            "judge_only_flags": card.judge_only_flags,
            "passed": card.passed,
            "judge_notes": card.judge_notes,
            "grounded_claims": card.grounded_claims,
            "ungrounded_claims": card.ungrounded_claims,
        },
    }


def usage_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    finals = [value["turns"][-1] for value in records]
    all_turns = [turn for value in records for turn in value["turns"]]
    latency = [turn["latency_s"] for turn in all_turns]
    return {
        "turns": len(all_turns),
        "latency_s": {
            "mean": round(mean(latency), 2) if latency else 0,
            "p50": round(median(latency), 2) if latency else 0,
            "p90": round(_quantile(latency, 0.9), 2),
            "max": round(max(latency), 2) if latency else 0,
        },
        "model_requests_per_turn": round(mean(t["model_requests"] for t in all_turns), 2)
        if all_turns
        else 0,
        "tool_calls_per_turn": round(mean(len(t["tool_calls"]) for t in all_turns), 2)
        if all_turns
        else 0,
        "input_tokens_per_turn": round(mean(t["input_tokens"] for t in all_turns))
        if all_turns
        else 0,
        "output_tokens_per_turn": round(mean(t["output_tokens"] for t in all_turns))
        if all_turns
        else 0,
        "total_input_tokens": sum(t["input_tokens"] for t in all_turns),
        "total_output_tokens": sum(t["output_tokens"] for t in all_turns),
        "core_bundle_attached_final_turns": sum(
            bool((t.get("core_knowledge") or {}).get("bundle_attached")) for t in finals
        ),
        "core_tool_used_final_turns": sum(
            bool((t.get("core_knowledge") or {}).get("tool_used")) for t in finals
        ),
        "character_selectors": dict(
            sorted(
                Counter(
                    selection["selector"]
                    for t in all_turns
                    for selection in t.get("character_selections") or []
                ).items()
            )
        ),
        "model_chosen_class_without_user_scope": sum(
            bool(selection.get("class_without_user_scope"))
            for t in all_turns
            for selection in t.get("character_selections") or []
        ),
        "ownership_repairs": sum(
            bool(entry.get("repaired"))
            for t in all_turns
            for entry in (t.get("ownership_validation") or {}).get("passes", [])
        ),
        "ownership_fallbacks": sum(
            bool((t.get("ownership_validation") or {}).get("fallback_due_to_ownership"))
            for t in all_turns
        ),
        "validator_corrections": sum(
            bool((t.get("validator") or {}).get("attempted")) for t in all_turns
        ),
        "initiative_retries": sum(
            bool((t.get("initiative_retry") or {}).get("attempted")) for t in all_turns
        ),
        "handbacks_trimmed": sum(bool(t.get("trailing_handback_removed")) for t in all_turns),
        "validator_fallbacks": dict(
            sorted(
                {
                    fallback: sum(
                        1
                        for t in all_turns
                        if (t.get("validator") or {}).get("fallback") == fallback
                    )
                    for t in all_turns
                    if (fallback := (t.get("validator") or {}).get("fallback"))
                }.items()
            )
        ),
    }


def summary_markdown(
    meta: dict[str, Any], records: list[dict[str, Any]], cards: list[ScoreCard]
) -> str:
    totals = aggregate(cards)
    usage = usage_stats(records)
    lines = [
        f"# Destiny Copilot eval — {meta['subset']} ({len(records)} scenarios)",
        "",
        f"Model `{meta['model']}` · judge {'on' if meta['judge'] else 'off'} · live provider "
        f"{'on' if meta['live_provider'] else 'off'} · run {meta['started_at']}",
        "",
        f"**Passed without serious flags:** {totals['passed_without_flags']}/{totals['scenarios']}"
        f" · **mean score** {totals['mean_total']}/{MAX_SCORE}",
        "",
        "| Dimension | Mean (0-2) |",
        "|---|---|",
        *(f"| {key} | {totals['mean_by_dimension'][key]} |" for key in DIMENSIONS),
        "",
        "**Serious flags (deterministic):** "
        + (
            ", ".join(f"{k} {v}" for k, v in sorted(totals["flag_counts_deterministic"].items()))
            or "none"
        )
        + " · **judge-only (advisory):** "
        + (
            ", ".join(f"{k} {v}" for k, v in sorted(totals["flag_counts_judge_only"].items()))
            or "none"
        ),
        "",
        f"**Latency** mean {usage['latency_s']['mean']}s · p50 {usage['latency_s']['p50']}s · "
        f"p90 {usage['latency_s']['p90']}s · max {usage['latency_s']['max']}s",
        f"**Per turn** {usage['model_requests_per_turn']} model requests · "
        f"{usage['tool_calls_per_turn']} tool calls · {usage['input_tokens_per_turn']} input / "
        f"{usage['output_tokens_per_turn']} output tokens",
        f"**Core knowledge** bundle on {usage['core_bundle_attached_final_turns']} final turns · "
        f"tool used on {usage['core_tool_used_final_turns']} · validator corrections "
        f"{usage['validator_corrections']} · fallbacks {usage['validator_fallbacks'] or 'none'}",
        f"**Validator rounds** {usage['initiative_retries']} tool-enabled retries · "
        f"{usage['validator_corrections']} corrections · {usage['handbacks_trimmed']} closing "
        "hand-backs trimmed without a model round",
        *(
            []
            if meta["judge"]
            else [
                "",
                "_Judge off: scores come from deterministic properties only. Factual accuracy "
                "and usefulness beyond those properties are not assessed, and a dimension with no "
                "applicable property counts as met. Read the answers before trusting a high "
                "score._",
            ]
        ),
        "",
        "| Scenario | Score | Flags | Failed properties | Tools (final turn) | s |",
        "|---|---|---|---|---|---|",
    ]
    for value in records:
        failed = [p["name"] for p in value["properties"] if not p["passed"]]
        tools = [call["name"] for call in value["turns"][-1]["tool_calls"]]
        lines.append(
            f"| {value['id']} | {value['score']['total']} | {', '.join(value['score']['flags'])} | "
            f"{', '.join(failed)} | {', '.join(tools)} | "
            f"{sum(t['latency_s'] for t in value['turns']):.1f} |"
        )
    lines += ["", "## Failures", ""]
    for value in records:
        failed = [p for p in value["properties"] if not p["passed"]]
        if not failed and not value["score"]["flags"]:
            continue
        lines.append(f"### {value['id']}")
        lines.append(f"> {value['turns'][-1]['message']}")
        for result in failed:
            lines.append(f"- `{result['name']}` {result['flag'] or ''}: {result['detail']}")
        for claim in value["score"]["ungrounded_claims"]:
            lines.append(f"- ungrounded: {claim}")
        if value["score"]["judge_notes"]:
            lines.append(f"- judge: {value['score']['judge_notes']}")
        lines.append("")
    return "\n".join(lines) + "\n"
