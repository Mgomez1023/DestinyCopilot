"""The held-out scenarios stay unseen, runnable, and separate from the main bank."""

import json
import pathlib

from evals.heldout import HELDOUT, KNOWN_ISSUES
from evals.model import AXES
from evals.profiles import PROFILES
from evals.scenarios import BANK, SUBSETS
from tests.test_eval_bank import _tool_names

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _strings(value: object) -> set[str]:
    if isinstance(value, str):
        return {value.strip().lower()}
    if isinstance(value, dict):
        return set().union(*(_strings(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_strings(item) for item in value), set())
    return set()


def test_heldout_is_a_separate_subset_of_fifteen_runnable_scenarios() -> None:
    tools = _tool_names()
    assert SUBSETS["heldout"] == HELDOUT
    assert len(HELDOUT) == 15
    assert len({value.id for value in HELDOUT}) == len(HELDOUT)
    assert not {value.id for value in HELDOUT} & {value.id for value in BANK}
    assert all(value not in SUBSETS["full"] for value in HELDOUT)
    for value in HELDOUT:
        assert value.id.startswith("heldout."), value.id
        assert value.profile in PROFILES, value.id
        assert set(value.expect.tools_any) <= tools, value.id
        for axis, tag in value.tags.items():
            assert tag in AXES[axis], (value.id, axis, tag)


def test_heldout_covers_multi_turn_and_unavailable_data() -> None:
    assert sum(len(value.turns) >= 2 for value in HELDOUT) >= 5
    assert any(
        "can't" in value.expect.mentions_any for value in HELDOUT if value.id.endswith("trials")
    )
    assert {value.expect.style for value in HELDOUT} >= {"fact", "planning", "build"}
    assert len(KNOWN_ISSUES) == 5


def test_no_heldout_prompt_was_seen_in_the_bank_or_saved_drafts() -> None:
    seen = {turn.strip().lower() for value in BANK for turn in value.turns}
    for path in FIXTURES.glob("live_playtest_drafts*.json"):
        seen |= _strings(json.loads(path.read_text("utf-8")))
    for value in HELDOUT:
        for turn in value.turns:
            assert turn.strip().lower() not in seen, (value.id, turn)
