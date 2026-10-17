"""Optional LLM judge: scores one observation against the rubric using the exact evidence the
product had (profile ground truth, tool results, grounded-facts block). Live runs only."""

import json
from typing import Any

from evals.model import Scenario
from evals.profiles import PROFILE_DESCRIPTIONS, ProfileFacts
from evals.rubric import DIMENSIONS, FLAGS
from evals.runtime import Observation

MAX_TOOL_CHARS = 7000
MAX_EVIDENCE_CHARS = 40000
MAX_ROUTING_CHARS = 9000

JUDGE_INSTRUCTIONS = f"""You grade answers from Destiny Copilot, a Destiny 2 assistant, for a
product evaluation. Be strict and evidence-based.

Score each dimension 0, 1, or 2 (2 = fully meets it):
{json.dumps(DIMENSIONS, indent=1)}

A dimension that does not apply to the question counts as fully met (2): a general factual
question needs no personalization, account data, or next step, so those score 2 when the
answer neither needs nor misuses them. Score a dimension down only for a real shortfall.

Raise a flag only for a clear, serious failure. Allowed flags: {", ".join(FLAGS)}.
- ownership_hallucination: names an item as owned or recommends an unowned item as available
  when the user limited advice to owned gear, contrary to the account ground truth.
- wrong_character: plans for or recommends gear usable only by a different class than the one
  requested.
- cross_character_leak: after the conversation switched characters, presents another
  character's quest, objective, or equipped gear as belonging to the current character.
- unsupported_active_quest: tells the user to finish a quest that the account does not have
  active.
- hallucinated_fact / invented_reward_or_mechanic: states a game fact, reward, perk effect, or
  mechanic as established when the evidence does not support it and it is wrong or unverifiable.
- stale_live_claim: presents rotating/current state as fact without live or cited web evidence.
- ambiguity_silently_resolved: answers about one entity when the term was ambiguous.

List up to 8 concrete claims from the answer as grounded_claims (supported by the evidence) and
up to 8 as ungrounded_claims (not supported by the evidence; prefix each with "memory:" when it
looks like general model knowledge, or "wrong:" when contradicted). General Destiny knowledge
that is widely true but not in the evidence is ungrounded, not necessarily wrong.
The routing_and_planning_context is application-generated evidence the model saw (for
example planning candidates with quest steps and objective progress); claims supported by
it are grounded. Tool results may be truncated with an ellipsis; do not call a claim
ungrounded only because it could sit in a truncated part. Keep notes under 80 words."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {
                key: {"type": "integer", "minimum": 0, "maximum": 2} for key in DIMENSIONS
            },
            "required": list(DIMENSIONS),
            "additionalProperties": False,
        },
        "flags": {"type": "array", "items": {"type": "string", "enum": list(FLAGS)}},
        "grounded_claims": {"type": "array", "items": {"type": "string"}},
        "ungrounded_claims": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["scores", "flags", "grounded_claims", "ungrounded_claims", "notes"],
    "additionalProperties": False,
}


def _expectations(scenario: Scenario) -> dict[str, Any]:
    expect = scenario.expect
    return {
        key: value
        for key, value in expect.__dict__.items()
        if value not in ((), None, False, "") and key != "style"
    } | {"style": expect.style}


def evidence(obs: Observation, facts: ProfileFacts) -> str:
    tools: list[dict[str, Any]] = []
    for call in obs.final.tool_calls:
        result = json.dumps(call.result, ensure_ascii=False, separators=(",", ":"))
        tools.append(
            {
                "tool": call.name,
                "arguments": call.arguments,
                "result": result[:MAX_TOOL_CHARS] + ("…" if len(result) > MAX_TOOL_CHARS else ""),
            }
        )
    conversation = [
        {"user": turn.message, "assistant": turn.answer[:1500]} for turn in obs.turns[:-1]
    ]
    payload = {
        "scenario": {
            "id": obs.scenario.id,
            "expectations": _expectations(obs.scenario),
            "tags": obs.scenario.tags,
        },
        "account_ground_truth": {
            "profile": PROFILE_DESCRIPTIONS[obs.scenario.profile],
            "characters": list(facts.classes),
            "owned_items": {
                name: {"class": owner or "any", "where": where}
                for name, (owner, where) in facts.owned_items.items()
            },
            "active_quests_by_class": facts.active_quests,
            "near_complete_records": list(facts.near_complete),
            "available_activities_by_class": facts.available_activities,
        },
        "earlier_turns": conversation,
        "final_user_message": obs.final.message,
        "routing_and_planning_context": obs.final.routing_context[:MAX_ROUTING_CHARS],
        "grounded_facts_block": obs.final.core_block,
        "tool_results_this_turn": tools,
        "live_provider_grounded": (obs.final.trace.get("grounding") or {}).get("live_provider"),
        "web_research": (obs.final.trace.get("grounding") or {}).get("web_research"),
        "answer": obs.final.answer,
    }
    return json.dumps(payload, ensure_ascii=False, indent=1)[:MAX_EVIDENCE_CHARS]


async def judge(client: Any, model: str, obs: Observation, facts: ProfileFacts) -> dict[str, Any]:
    response = await client.responses.create(
        model=model,
        instructions=JUDGE_INSTRUCTIONS,
        input=evidence(obs, facts),
        text={
            "format": {
                "type": "json_schema",
                "name": "eval_score",
                "schema": SCHEMA,
                "strict": True,
            }
        },
        reasoning={"effort": "low"},
        max_output_tokens=3000,
        store=False,
    )
    return json.loads(response.output_text)
