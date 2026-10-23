"""The turn instructions carry the current UTC date at the very end, and the ambiguity rule."""

from datetime import UTC, datetime

from app.ai import SYSTEM_INSTRUCTIONS, clock_instruction
from tests.test_playtest_scenarios import Script, assistant_message, run, service, titan_with

THURSDAY = datetime(2026, 10, 22, 21, 37, 5, tzinfo=UTC)


def _instructions(now: datetime, message: str = "What should I do next?") -> str:
    script = Script([assistant_message("Run Kepler Uplink for the quest today.")])
    value = service(script)
    value.clock = lambda: now
    run(value, message, titan_with())
    return script.requests[0]["instructions"]


def test_the_clock_names_the_utc_weekday_date_and_hour() -> None:
    text = clock_instruction(THURSDAY)
    assert "Thursday, 2026-10-22 21:00" in text and "UTC" in text


def test_the_clock_converts_other_time_zones_to_utc() -> None:
    from datetime import timedelta, timezone

    local = datetime(2026, 10, 23, 1, 5, tzinfo=timezone(timedelta(hours=5)))  # 20:05 UTC Oct 22
    assert "Thursday, 2026-10-22 20:00" in clock_instruction(local)


def test_the_clock_is_the_last_thing_in_the_instructions() -> None:
    text = _instructions(THURSDAY)
    assert text.endswith(clock_instruction(THURSDAY))
    assert text.startswith(SYSTEM_INSTRUCTIONS)


def test_everything_before_the_clock_is_the_same_at_any_time() -> None:
    first = _instructions(THURSDAY)
    later = _instructions(datetime(2026, 10, 27, 17, 2, tzinfo=UTC))
    stamp_first, stamp_later = (
        clock_instruction(THURSDAY),
        clock_instruction(datetime(2026, 10, 27, 17, 2, tzinfo=UTC)),
    )
    assert first.removesuffix(stamp_first) == later.removesuffix(stamp_later)
    assert first != later and "Tuesday, 2026-10-27 17:00" in later


def test_the_text_is_stable_within_an_hour() -> None:
    same_hour = datetime(2026, 10, 22, 21, 59, 59, tzinfo=UTC)
    assert clock_instruction(THURSDAY) == clock_instruction(same_hour)


def test_the_model_is_told_to_compare_dated_claims_with_the_clock() -> None:
    text = clock_instruction(THURSDAY)
    assert "dated source claim" in text and "active now, this week, or currently" in text
    assert "unconfirmed" in text


def test_an_ambiguous_question_gets_an_assumption_not_a_bare_question() -> None:
    assert "state that assumption in one" in SYSTEM_INSTRUCTIONS
    assert "do not reply with only a question" in SYSTEM_INSTRUCTIONS
    assert "no reasonable assumption exists" in SYSTEM_INSTRUCTIONS
