"""Validators are non-destructive: each finding is repaired in place, and nothing is added.

Every behavior has a must-repair and a must-preserve case. A model rewrite that damages its draft
(drops its links or dates, the names it took from tool results, or most of its content) is
discarded in favor of the draft with only the flagged sentences removed.
"""

import asyncio
from types import SimpleNamespace

import pytest

from app.ai import RecommendationService
from app.config import Settings
from app.local_repairs import (
    dependents_of,
    drop_extra_questions,
    drop_repeated_sentences,
    flatten_headings,
    locally_handled,
    move_answer_first,
    remove_flagged_sentences,
    repair_style,
    rewrite_damage,
    trim_excess,
)
from app.models import ChatRequest
from app.response_quality import response_mode_context
from tests.test_guardian_tools import ScriptedWebResponses, assistant_message, guardian_context

# --- Damage guard -----------------------------------------------------------------------------

DRAFT = (
    "Reset was Oct 21, 2026, per https://example.com/reset. "
    "Run Kepler Uplink for the Edge of Fate quest. "
    "Haunted Altars of Sorrow is a good backup."
)


def test_a_rewrite_that_drops_the_link_or_date_is_discarded() -> None:
    rewrite = "Run Kepler Uplink for the Edge of Fate quest. Haunted Altars of Sorrow is a backup."
    assert rewrite_damage(DRAFT, rewrite, set()) == "lost_citations"


def test_a_rewrite_that_drops_a_tool_returned_name_is_discarded() -> None:
    rewrite = (
        "Reset was Oct 21, 2026, per https://example.com/reset. Run the quest for progress today."
    )
    assert rewrite_damage(DRAFT, rewrite, set(), {"kepler uplink"}) == "lost_names"


def test_a_rewrite_that_drops_most_of_the_sentences_is_discarded() -> None:
    draft = (
        "Run Kepler Uplink tonight. It advances your active quest. It fits thirty minutes. "
        "Rewards arrive at completion. Bring a long range weapon."
    )
    assert rewrite_damage(draft, "Try something else entirely.", set()) == "dropped_most_of_answer"


def test_a_rewrite_that_removes_exactly_the_flagged_claim_is_kept() -> None:
    flagged = {"Haunted Altars of Sorrow is a good backup."}
    rewrite = (
        "Reset was Oct 21, 2026, per https://example.com/reset. "
        "Run Kepler Uplink for the Edge of Fate quest."
    )
    assert rewrite_damage(DRAFT, rewrite, flagged, {"haunted altars of sorrow"}) is None


def test_a_rewrite_that_keeps_everything_reworded_is_kept() -> None:
    rewrite = (
        "Per https://example.com/reset, reset was Oct 21, 2026. Kepler Uplink is the Edge of "
        "Fate quest to run. Haunted Altars of Sorrow is a good backup."
    )
    assert rewrite_damage(DRAFT, rewrite, set(), {"kepler uplink"}) is None


def test_dropping_only_offers_and_questions_is_not_damage() -> None:
    draft = (
        "Run Kepler Uplink for the Edge of Fate quest. It fits your window and advances the "
        "active quest. Want me to check your build? I can also list backups. Which do you want?"
    )
    rewrite = "Run Kepler Uplink for the Edge of Fate quest; it fits your window and the quest."
    assert rewrite_damage(draft, rewrite, set()) is None


def test_lines_that_only_detail_a_removed_pick_count_as_flagged() -> None:
    answer = (
        "Primary: run Nightfall.\n- Why: best loot.\n- Steps: launch it.\n\nBackup: run Strikes."
    )
    assert dependents_of(answer, {"Primary: run Nightfall."}) >= {
        "Primary: run Nightfall.",
        "- Why: best loot.",
        "- Steps: launch it.",
    }


# --- Presentation-only findings: local fixes, never new content --------------------------------

MODE = response_mode_context("What next?", session_planning=True, continuation=False)


def _words(text: str) -> set[str]:
    return set(text.lower().replace("?", "").replace(".", "").split())


def test_the_answer_sentence_moves_first_and_the_filler_goes() -> None:
    answer = (
        "There are several things to consider. Run Kepler Uplink because it fits your window. "
        "It advances the quest."
    )
    fixed = move_answer_first(answer)
    assert fixed is not None and fixed.startswith("Run Kepler Uplink")
    assert "several things" not in fixed and "advances the quest" in fixed
    assert _words(fixed) <= _words(answer)


def test_a_filler_label_is_stripped_and_a_good_opening_is_untouched() -> None:
    assert move_answer_first("Primary recommendation: Run Kepler Uplink for the quest.") == (
        "Run Kepler Uplink for the quest."
    )
    assert move_answer_first("Run Kepler Uplink for the quest. It fits.") is None


def test_extra_questions_and_permission_asks_are_dropped_when_an_answer_remains() -> None:
    answer = "Run Kepler Uplink for the quest. Want me to check your build? Or your quests?"
    fixed = drop_extra_questions(answer, permission=True, clarification=False)
    assert fixed == "Run Kepler Uplink for the quest."
    two = "Run Kepler Uplink for the quest. Want tips? Or loot?"
    assert drop_extra_questions(two, permission=False, clarification=False) == (
        "Run Kepler Uplink for the quest. Want tips?"
    )


def test_a_draft_that_is_only_a_question_is_left_for_an_answer() -> None:
    only = "Which character? How much time? Story or loot?"
    assert drop_extra_questions(only, permission=False, clarification=True) is None
    assert locally_handled({"too_many_followups"}, only) == set()
    assert locally_handled(
        {"too_many_followups"}, "Run Kepler Uplink for the quest. Want tips? Or loot?"
    ) == {"too_many_followups"}


def test_a_single_question_after_an_answer_is_not_touched() -> None:
    answer = "Run Kepler Uplink for the quest. Want build tips?"
    assert drop_extra_questions(answer, permission=False, clarification=False) is None


def test_repeated_sentences_are_dropped_once() -> None:
    answer = (
        "Run Kepler Uplink because it fits your goal. Bring a long range weapon. "
        "Run Kepler Uplink because it fits your goal."
    )
    fixed = drop_repeated_sentences(answer)
    assert fixed is not None and fixed.count("Run Kepler Uplink") == 1
    assert drop_repeated_sentences("Run it now. Run it again later.") is None


def test_excess_lines_are_trimmed_from_the_end_and_links_and_limits_stay() -> None:
    lines = ["Run Kepler Uplink for the quest."]
    lines += [f"- Tip {i}: " + "word " * 20 for i in range(10)]
    lines += ["- Source: https://example.com/guide", "- I can't see your loot data."]
    answer = "\n".join(lines)
    fixed = trim_excess(answer, 60)
    assert fixed is not None and fixed.startswith("Run Kepler Uplink")
    assert "https://example.com/guide" in fixed and "can't see your loot" in fixed
    assert len(fixed.split()) < len(answer.split())
    assert trim_excess("Run it.", 60) is None


def test_heading_only_lines_are_flattened_and_text_under_them_stays() -> None:
    answer = "## Plan\nRun Kepler Uplink for the quest.\n**Why**\nIt fits your window today."
    fixed = flatten_headings(answer)
    assert fixed == "Run Kepler Uplink for the quest.\nIt fits your window today."
    assert flatten_headings("Run it today for the quest.") is None


def test_repair_style_records_each_decision_and_never_adds_words() -> None:
    answer = (
        "Short answer: Run Kepler Uplink because it fits your goal. Want tips? Or loot? "
        "Run Kepler Uplink because it fits your goal."
    )
    fixed, done = repair_style(
        answer,
        {"answer_not_first", "too_many_followups", "repeated_content"},
        mode=MODE,
        permission_allowed=True,
    )
    assert fixed.startswith("Run Kepler Uplink") and fixed.count("Run Kepler Uplink") == 1
    assert fixed.count("?") <= 1
    assert _words(fixed) <= _words(answer)
    assert {entry["validator"] for entry in done} == {
        "answer_not_first",
        "too_many_followups",
        "repeated_content",
    }
    assert all(entry["result"] in {"applied", "nothing_to_change"} for entry in done)


def test_a_permission_ask_before_any_lookup_is_left_for_the_tool_retry() -> None:
    answer = "Here is the plan for today. Want me to check your vault?"
    kept, done = repair_style(
        answer, {"unnecessary_read_only_permission"}, mode=MODE, permission_allowed=False
    )
    assert kept == answer
    assert done[0]["result"] == "nothing_to_change"


# --- Constraint findings: remove the breaking sentence, keep the rest --------------------------


def test_only_the_breaking_sentence_goes_and_the_rest_stays() -> None:
    answer = (
        "Run Kepler Uplink for the quest. Go with Nightfall for loot. Bring a long range weapon."
    )
    fixed = remove_flagged_sentences(answer, {"Go with Nightfall for loot."})
    assert fixed == "Run Kepler Uplink for the quest. Bring a long range weapon."


def test_a_removed_primary_takes_its_detail_and_the_backup_is_promoted() -> None:
    answer = (
        "Primary pick: run Nightfall for loot.\n"
        "- Why: Nightfalls drop the best gear.\n"
        "- Steps: launch it and clear the adds.\n\n"
        "Solo setup and tactics\n"
        "- Loadout: a heavy for the boss.\n\n"
        "Safer backup - run Strikes for the milestone.\n"
        "- Why: it is quick.\n\n"
        "Which do you want?"
    )
    fixed = remove_flagged_sentences(answer, {"Primary pick: run Nightfall for loot."})
    assert fixed is not None
    assert fixed.startswith("Run Strikes for the milestone.")
    assert "Nightfall" not in fixed and "Loadout" not in fixed and "launch it" not in fixed
    assert "- Why: it is quick." in fixed
    assert not fixed.startswith(("-", "Why"))


def test_a_promoted_backup_that_breaks_the_constraint_is_refused() -> None:
    answer = "Primary: run Nightfall.\n- Why: loot.\n\nBackup: run Raid Lair with a fireteam."
    refused = remove_flagged_sentences(
        answer, {"Primary: run Nightfall."}, acceptable_pick=lambda body: "fireteam" not in body
    )
    assert refused is None
    kept = remove_flagged_sentences(
        answer, {"Primary: run Nightfall."}, acceptable_pick=lambda body: True
    )
    assert kept is not None and kept.startswith("Run Raid Lair")


def test_nothing_left_to_recommend_means_no_local_repair() -> None:
    assert remove_flagged_sentences("Run Nightfall for loot.", {"Run Nightfall for loot."}) is None
    answer = "Run Nightfall for loot.\n- Why: best loot."
    assert remove_flagged_sentences(answer, {"Run Nightfall for loot."}) is None


def test_removing_a_bullets_first_sentence_never_leaves_a_fragment() -> None:
    answer = (
        "Run Kepler Uplink for the quest.\n"
        "- Go with Nightfall for loot. It is hard. Bring friends.\n"
        "- Bring a long range weapon."
    )
    fixed = remove_flagged_sentences(answer, {"- Go with Nightfall for loot."})
    assert fixed == "Run Kepler Uplink for the quest.\n- Bring a long range weapon."


def test_a_list_header_left_empty_is_removed_and_or_is_cleaned() -> None:
    answer = (
        "Run Kepler Uplink for the quest.\n"
        "Options:\n- Go with Nightfall for loot.\n- Go with a raid for loot.\n"
        "Bring a long range weapon."
    )
    fixed = remove_flagged_sentences(
        answer, {"- Go with Nightfall for loot.", "- Go with a raid for loot."}
    )
    assert fixed is not None and "Options" not in fixed
    assert fixed == "Run Kepler Uplink for the quest.\nBring a long range weapon."


# --- End to end: trace of each decision --------------------------------------------------------


def _chat(*drafts: str) -> tuple[RecommendationService, ScriptedWebResponses, str]:
    scripted = ScriptedWebResponses([[assistant_message(value)] for value in drafts], list(drafts))
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)
    response = asyncio.run(
        service.chat(ChatRequest(message="What should I do next?"), guardian_context())
    )
    return service, scripted, response.message


def test_a_damaging_rewrite_is_discarded_and_only_the_flagged_sentence_is_removed() -> None:
    draft = (
        "Reset was Oct 21, 2026, per https://example.com/reset. "
        "Run the Nightfall on your Titan for Vanguard rank progress. "
        "The Guardian tool returned that Nightfall fits."
    )
    rewrite = "Run the Nightfall on your Titan for Vanguard rank progress."
    service, scripted, message = _chat(draft, rewrite)

    assert "https://example.com/reset" in message and "Oct 21, 2026" in message
    assert "Guardian tool" not in message
    trace = service.latest_trace()
    assert trace["planning_correction"]["rewrite_discarded"] == "lost_citations"
    decisions = [(e["action"], e["result"]) for e in trace["local_repairs"]]
    assert ("rewrite", "lost_citations") in decisions
    assert ("remove_flagged_sentences", "applied") in decisions


def test_a_good_rewrite_is_kept_and_recorded() -> None:
    draft = "Run the Nightfall on your Titan. The Guardian tool returned that Nightfall fits."
    rewrite = "Run the Nightfall on your Titan; it fits your goal."
    service, _, message = _chat(draft, rewrite)

    assert message == rewrite
    assert ("rewrite", "kept") in [
        (e["action"], e["result"]) for e in service.latest_trace()["local_repairs"]
    ]


@pytest.mark.parametrize(
    "draft", ["Run the Nightfall on your Titan for rank. Want loot? Or story?"]
)
def test_a_style_only_draft_never_triggers_a_rewrite(draft: str) -> None:
    service, scripted, message = _chat(draft)

    assert len(scripted.requests) == 1
    assert message == "Run the Nightfall on your Titan for rank. Want loot?"
    entry = service.latest_trace()["local_repairs"][0]
    assert entry["validator"] == "too_many_followups" and entry["result"] == "applied"
