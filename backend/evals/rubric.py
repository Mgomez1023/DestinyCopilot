"""Scoring rubric: eight 0-2 dimensions (max 16) plus binary serious-failure flags.

Serious flags are never averaged away: a scenario with any flag is reported as failed even
when its score is high. Scores come from the LLM judge when it ran; otherwise from the
deterministic provisional scorer below. Deterministic flags are always kept.
"""

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Literal

Flag = Literal[
    "hallucinated_fact",
    "ownership_hallucination",
    "wrong_character",
    "stale_live_claim",
    "ambiguity_silently_resolved",
    "wrong_destination",
    "invented_reward_or_mechanic",
    "unsupported_active_quest",
    "false_refusal",
    "tool_loop_failed",
    "excessive_verbosity",
    "cross_character_leak",
    "unsupported_capability",
    "unfulfilled_request",
    "contradictory_claim",
    "ungrounded_advice",
]
FLAGS: tuple[str, ...] = Flag.__args__  # type: ignore[attr-defined]

DIMENSIONS: dict[str, str] = {
    "factual_correctness": "Every stated fact is correct for Destiny 2 and this account.",
    "account_grounding": "Account claims (ownership, progress, quests, characters) come from "
    "Guardian tool results.",
    "destiny_grounding": "Game facts come from Manifest, core knowledge, guide, live, or cited "
    "web evidence rather than unsupported memory; unknowns stay unknown.",
    "personalization": "Uses this player's actual characters, gear, quests, and progress.",
    "actionability": "Gives a concrete next step the player can take now.",
    "concision": "Answer first, no filler, length fits the question.",
    "confidence_calibration": "Certainty matches the evidence; uncertainty is stated briefly.",
    "follow_up_constraint_handling": "Honors explicit constraints and corrections; asks at most "
    "one necessary question.",
}
MAX_SCORE = 2 * len(DIMENSIONS)


@dataclass
class ScoreCard:
    scores: dict[str, int]
    flags: list[str]
    score_source: Literal["judge", "deterministic"]
    deterministic_scores: dict[str, int]
    judge_notes: str = ""
    # Provenance: deterministic flags are authoritative; judge-only flags are advisory.
    deterministic_flags: list[str] = field(default_factory=list)
    judge_only_flags: list[str] = field(default_factory=list)
    grounded_claims: list[str] = field(default_factory=list)
    ungrounded_claims: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.scores.values())

    @property
    def passed(self) -> bool:
        return not self.flags


def deterministic_scores(
    results: list[Any], expect: Any, answer: str, facts: Any
) -> dict[str, int]:
    """Provisional scores from property results; coarse by design."""

    failed = {value.name for value in results if value.applies and not value.passed}
    flags = {value.flag for value in results if value.flag}

    def score(serious: set[str], soft: set[str]) -> int:
        if flags & serious:
            return 0
        return 1 if failed & soft else 2

    personal_terms = [
        *facts.owned_items,
        *(quest for values in facts.active_quests.values() for quest in values),
        *facts.near_complete,
        *(name for values in facts.available_activities.values() for name in values),
    ]
    account_scenario = expect.style in {"planning", "build"}
    personalized = any(term.casefold() in answer.casefold() for term in personal_terms)
    return {
        "factual_correctness": score(
            {
                "hallucinated_fact",
                "wrong_destination",
                "invented_reward_or_mechanic",
                "contradictory_claim",
            },
            {"mentions_expected_fact", "no_forbidden_claims", "turn_expectations_met"},
        ),
        "account_grounding": score(
            {"ownership_hallucination", "wrong_character", "unsupported_active_quest"},
            {"calls_expected_evidence_tool"},
        ),
        "destiny_grounding": score(
            {"ambiguity_silently_resolved", "stale_live_claim", "hallucinated_fact"},
            {"fuzzy_match_not_presented_as_exact"},
        ),
        # Not-applicable dimensions count as met; without a judge, planning and build
        # actionability cannot be assessed deterministically and stays at 1.
        "personalization": (2 if personalized else 0) if account_scenario else 2,
        "actionability": 0
        if flags
        & {"false_refusal", "tool_loop_failed", "unsupported_capability", "unfulfilled_request"}
        else 1
        if account_scenario or "no_read_only_permission_request" in failed
        else 2,
        "concision": score(
            {"excessive_verbosity"},
            {"concise", "answer_first", "no_filler", "does_not_repeat_prompt"},
        ),
        "confidence_calibration": score(
            {"stale_live_claim", "hallucinated_fact", "contradictory_claim"},
            {"unknown_fields_not_filled_from_memory"},
        ),
        "follow_up_constraint_handling": score(
            {"wrong_character", "ownership_hallucination", "unfulfilled_request"},
            {"limited_follow_up_questions", "one_primary_and_brief_backup"},
        ),
    }


def score_card(
    results: list[Any],
    expect: Any,
    answer: str,
    facts: Any,
    judge: dict[str, Any] | None = None,
) -> ScoreCard:
    provisional = deterministic_scores(results, expect, answer, facts)
    flags = {value.flag for value in results if value.flag}
    deterministic = sorted(flags)
    if judge:
        judged = {value for value in judge.get("flags", []) if value in FLAGS}
        judge_only = sorted(judged - flags)
        flags |= judged
        scores = {key: int(judge["scores"].get(key, provisional[key])) for key in DIMENSIONS}
        return ScoreCard(
            scores=scores,
            flags=sorted(flags),
            score_source="judge",
            deterministic_scores=provisional,
            judge_notes=str(judge.get("notes", ""))[:1200],
            deterministic_flags=deterministic,
            judge_only_flags=judge_only,
            grounded_claims=list(judge.get("grounded_claims", []))[:12],
            ungrounded_claims=list(judge.get("ungrounded_claims", []))[:12],
        )
    return ScoreCard(
        provisional, sorted(flags), "deterministic", provisional, deterministic_flags=deterministic
    )


def aggregate(cards: list[ScoreCard]) -> dict[str, Any]:
    if not cards:
        return {"scenarios": 0}
    return {
        "scenarios": len(cards),
        "passed_without_flags": sum(card.passed for card in cards),
        "mean_total": round(mean(card.total for card in cards), 2),
        "max_total": MAX_SCORE,
        "mean_by_dimension": {
            key: round(mean(card.scores[key] for card in cards), 2) for key in DIMENSIONS
        },
        "flag_counts": dict(Counter(flag for card in cards for flag in card.flags)),
        "flag_counts_deterministic": dict(
            Counter(flag for card in cards for flag in card.deterministic_flags)
        ),
        "flag_counts_judge_only": dict(
            Counter(flag for card in cards for flag in card.judge_only_flags)
        ),
        "passed_without_deterministic_flags": sum(not card.deterministic_flags for card in cards),
    }
