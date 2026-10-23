"""Word boundaries, orphan-free removals, the loot note, and offers no tool can back."""

import pytest

from app.item_fact_claims import (
    canonical_slot,
    collect_item_definitions,
    contradicted_item_facts,
    correct_item_facts,
)
from app.local_repairs import clean_layout, remove_flagged_sentences
from app.models import ChatTurn
from app.response_quality import (
    _CITED_FACT,
    ResponseQualityValidator,
    remove_unsupported_capability_offers,
)
from tests.test_playtest_scenarios import Script, assistant_message, run, service, titan_with

# --- 1. Whole words only ----------------------------------------------------------------------


def _crown() -> dict:
    definitions: dict = {}
    collect_item_definitions(
        [{"name": "Foxglove Crown", "class_restriction": "Warlock", "bucket": "Helmet"}],
        definitions,
    )
    return definitions


@pytest.mark.parametrize(
    "word", ["Legendary", "powerful", "remarks", "energetic", "kinetically", "helmets", "armsful"]
)
def test_slot_words_inside_other_words_are_never_slots(word: str) -> None:
    assert canonical_slot(word) == ""
    answer = f"Try Foxglove Crown ({word} Warlock helm) for the extra ability today."
    assert not contradicted_item_facts(answer, _crown())
    assert correct_item_facts(answer, _crown()) is None


def test_a_real_slot_word_is_still_caught_and_the_neighbours_are_untouched() -> None:
    answer = "Try Foxglove Crown (Legendary Hunter gauntlets) for the extra energetic ability."
    fixed = correct_item_facts(answer, _crown())
    assert fixed == "Try Foxglove Crown (Legendary Warlock helmet) for the extra energetic ability."


def test_one_class_claim_matched_twice_is_corrected_once() -> None:
    answer = "Try Foxglove Crown (Hunter helm) for the extra ability today."
    assert correct_item_facts(answer, _crown()) == (
        "Try Foxglove Crown (Warlock helm) for the extra ability today."
    )


def test_month_words_need_to_be_months() -> None:
    for text in ("Decided 5 things", "Marked 3 items", "Mayor 2 said", "Octave 4"):
        assert not _CITED_FACT.search(text), text
    for text in ("Oct 21, 2026", "March 3", "Sept 4", "2026-10-21"):
        assert _CITED_FACT.search(text), text


# --- 2. Removals leave no orphans --------------------------------------------------------------

REJECT_T2 = (
    "Got it — skip Nightfall. Do Vanguard Ops now — it’s quick and advances your Weekly Vanguard "
    "milestone.\n\nPlan\n"
    "- Primary (20–60 minutes): Run 1–2 Vanguard Ops. Each one nets milestone progress and "
    "regular rewards; they’re short and repeatable.\n"
    "- Backup (if you want something different): Run Duality: Standard (Dungeon) once to hunt "
    "the remaining chest.\n"
    "- Quick alternative: Queue Crucible: Control for short, repeatable play.\n\n"
    "Which of those do you want to run?"
)
CONSTRAINTS_T2 = (
    "Primary pick — run Haunted Altars of Sorrow (seasonal Altars variant).\n"
    "- Why: it’s available to your Titan and fits a 30-minute solo window.\n"
    "- How to run it:\n"
    "  - Clear waves fast with a high-mobility weapon.  \n"
    "  - Maximize drops: use ammo-finding mods.  \n"
    "  - Focus objectives that spawn chests or minibosses; these usually give the best "
    "immediate loot.  \n"
    "  - If you’re short on time, do two quick runs.\n\n"
    "Backup — finish the “Waves of Altars of Sorrow completed” progress (you’re at ~92%).\n"
    "- Why: it’s close to completion and will convert your remaining waves into a triumph.\n"
    "- How: play the same Haunted Altars waves until the record advances to 100%.\n\n"
    "Which do you want to start with?"
)


def _turn(message: str, answer: str) -> list[ChatTurn]:
    return [ChatTurn(role="user", content=message), ChatTurn(role="assistant", content=answer)]


def test_a_cut_duration_keeps_the_action_and_leaves_no_orphan_line() -> None:
    script = Script([assistant_message(REJECT_T2)], [assistant_message("unused")])
    value = service(script)
    history = _turn("What should I do next?", "Do Kepler Uplink now to advance your quest.")

    answer = run(
        value, "I don't want to do the Nightfall. Give me something else.", titan_with(), history
    )

    assert len(script.requests) == 1
    assert "20–60" not in answer.message
    assert "- Primary: Run 1–2 Vanguard Ops. Each one nets milestone progress" in answer.message
    assert "Plan\n- Primary" in answer.message
    assert "Which of those" not in answer.message
    assert "Loot quality" not in answer.message  # the player did not ask about loot


def test_removing_a_conflating_backup_removes_its_why_and_how_lines_too() -> None:
    script = Script([assistant_message(CONSTRAINTS_T2)], [assistant_message("unused")])
    value = service(script)
    history = _turn(
        "I have 30 minutes, solo, no Crucible and no dungeons. What should I do?",
        "Run the Kepler Uplink mission for The Edge of Fate.",
    )

    answer = run(value, "Something with better loot instead?", titan_with(), history).message

    assert answer.startswith("Primary pick — run Haunted Altars of Sorrow")
    assert "Waves of Altars of Sorrow completed" not in answer
    assert "close to completion" not in answer and "record advances" not in answer
    assert "best immediate loot" not in answer
    # No blank bullet, no run of blank lines, no trailing spaces.
    assert "\n\n\n" not in answer and "  \n" not in answer
    assert all(line.strip() not in {"-", "*"} for line in answer.splitlines())
    # The player asked about loot, so the cut loot ranking is acknowledged once.
    assert answer.count("Loot quality") == 1


def test_clean_layout_drops_blank_bullets_and_headings_with_nothing_under_them() -> None:
    text = "Run it now.\n\nOptions:\n\n- \n\nBring a long range weapon.\n   \n\n\nThen go.\nPlan"
    assert clean_layout(text) == "Run it now.\n\nBring a long range weapon.\n\nThen go."
    assert clean_layout("Run it now.\nSteps:\n- Launch.\n- Clear.") == (
        "Run it now.\nSteps:\n- Launch.\n- Clear."
    )


def test_removing_a_pick_removes_the_bullets_under_it_but_not_the_next_paragraph() -> None:
    answer = (
        "Run Kepler Uplink for the quest.\n"
        "Go with Nightfall for loot.\n"
        "- Why: best loot.\n"
        "- How: launch it.\n"
        "Bring a long range weapon."
    )
    fixed = remove_flagged_sentences(answer, {"Go with Nightfall for loot."})
    assert fixed == "Run Kepler Uplink for the quest.\nBring a long range weapon."


def test_an_answer_with_nothing_to_repair_is_not_reformatted() -> None:
    answer = "Run Kepler Uplink for the quest.  \n- Bring a long range weapon.  "
    assert ResponseQualityValidator.repair_planning_claims(answer) is None


# --- 3. The loot note answers a loot question only ---------------------------------------------


def test_the_loot_note_is_added_only_when_the_player_asked_about_loot() -> None:
    answer = "Run Kepler Uplink for the quest.\nNightfall has the best loot of all."
    asked = ResponseQualityValidator.repair_planning_claims(answer, note_loot=True)
    unasked = ResponseQualityValidator.repair_planning_claims(answer, note_loot=False)
    assert asked is not None and asked.endswith("Loot quality isn't established by your data.")
    assert unasked == "Run Kepler Uplink for the quest."


# --- 4. Offers no tool can back ----------------------------------------------------------------


@pytest.mark.parametrize(
    "offer",
    [
        "If you want, I can check your account to list exactly which Prismatic Facets you "
        "already own and which specific ones are still missing.",
        "I can run an account check to list owned Prismatic Facets if you want; shall I do that?",
        "I can run a check now to list which Facets you lack and then point to the quickest.",
        "I can compute your K/D from your Crucible matches if you want.",
        "If you want, I can fetch your recent Crucible match history and calculate a recent K/D "
        "estimate; should I do that for your Titan?",
    ],
)
def test_offers_of_abilities_no_tool_has_are_removed_and_the_rest_stays(offer: str) -> None:
    answer = f"Your Titan runs Sunbreaker today. {offer}"
    assert remove_unsupported_capability_offers(answer) == "Your Titan runs Sunbreaker today."


@pytest.mark.parametrize(
    "fine",
    [
        "I can explain how Facets unlock in the Pale Heart.",
        "I can look up what Ember of Ashes does.",
        "I can't tell which Prismatic Facets you own from equipped data.",
        "I can't find a season-wide Crucible K/D in the retrieved progression data.",
        "Your K/D is not something the retrieved data shows.",
    ],
)
def test_explanations_lookups_and_honest_limits_stay(fine: str) -> None:
    answer = f"Your Titan runs Sunbreaker today. {fine}"
    assert remove_unsupported_capability_offers(answer) == answer


def test_a_closing_offer_is_cut_without_a_model_round() -> None:
    draft = (
        "Your Titan runs Sunbreaker with its usual kit equipped right now, and "
        "the equipped-fragment data cannot show which Prismatic Facets you own. "
        "The fastest next step is the Pale Heart route in the Prismatic guide. "
        "If you want, I can run a check to list which Facets you lack and then point to the "
        "quickest."
    )
    script = Script([assistant_message(draft)], [assistant_message("unused")])
    value = service(script)

    answer = run(value, "What am I running right now?", titan_with(), []).message

    assert len(script.requests) == 1
    assert answer.endswith("in the Prismatic guide.")
    assert "I can run a check" not in answer


def test_an_offer_that_is_most_of_the_answer_still_goes_to_the_rewrite() -> None:
    draft = "I can run a check to list which Facets you lack. Want me to?"
    script = Script(
        [assistant_message(draft)],
        [
            assistant_message(
                "Prismatic Facets unlock in the Pale Heart; the guide lists the steps."
            )
        ],
    )
    value = service(script)

    answer = run(value, "What Facets am I missing?", titan_with(), []).message

    assert len(script.requests) == 2
    assert "Pale Heart" in answer
