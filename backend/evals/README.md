# Destiny Copilot evaluation

Two layers:

- **Deterministic (CI).** `tests/test_eval_bank.py`, `tests/test_eval_properties.py`, and
  `tests/test_product_evals.py` run with the normal suite. No model calls.
- **Live model (manual, never CI).** `python -m evals.runner` drives the production
  `RecommendationService` against synthetic Guardian profiles with the configured OpenAI model.

## Running live evals

From `backend/`:

```
python -m evals.runner --subset golden            # 17 never-regress prompts
python -m evals.runner --subset representative    # 25-scenario routine subset
python -m evals.runner --subset full              # whole bank (127)
python -m evals.runner --subset playtest          # 10 production playtest failures (Operation Guardian)
python -m evals.runner --ids golden.weakest_build --repeat 3   # variance check
python -m evals.runner --subset golden --no-judge --live-provider --concurrency 4
python -m evals.runner --subset representative --list
```

Needs `OPENAI_API_KEY` (root `.env`) and the local Manifest SQLite under `.cache/manifest/`.
Manifest, guide, and core tools are offline; `--live-provider` reaches Bungie's public endpoints.
Each run writes `results.json` and `summary.md` to `.cache/eval/runs/<timestamp>/`.

## Cassettes, replay, and cost

Every live run also writes `cassette.jsonl` next to `results.json`: for each model call, per
scenario/turn/round, hashes of the request (model, instructions, input and tool-output items, tools,
other options) and the full raw response. Request bodies, API keys, tokens, and headers are never
recorded.

`python -m evals.replay <run_dir> [--ids a,b]` runs the **current** production service against the
same simulated profiles with a stub client that returns the recorded responses in order. No OpenAI,
Bungie, or web request is made (web-search results are whatever the recorded responses held). If a
request differs from the recording in any part, or was never recorded, that scenario stops there and
is marked `needs_live`; it is never scored from a guess. Output is `results.json` and `summary.md`
like the runner, plus a per-turn `unchanged` / `changed` / `needs_live` diff of final answers against
the recorded run (with before/after text for `changed`).

Workflow:

1. **Post-processing changes** (validators, repairs, answer assembly, sentence-level edits): replay
   the last full live run. It is free and deterministic. `changed` answers are the effect of the
   change; read them.
2. **Prompt, tool, retrieval, or profile changes** alter the requests, so replay reports
   `needs_live` from the first affected turn. Use a targeted live check:
   `python -m evals.runner --ids <id1>,<id2> --no-judge`.
3. **Full live runs only at milestones** (before a beta decision, after a batch of changes whose
   effect replay cannot show). Each one produces a new cassette for later replays.

Replay shows what the current code does with the *recorded* model outputs; it does not show what the
model would say to a changed request. Scores from a replay are deterministic-property scores only.

Cost: `summary.md` shows run totals (input, cached input, output tokens, web-search calls) and an
estimated USD cost. Prices are read from the environment or root `.env`, never from code:
`EVAL_PRICE_INPUT_PER_1M`, `EVAL_PRICE_CACHED_INPUT_PER_1M`, `EVAL_PRICE_OUTPUT_PER_1M`, and
optionally `EVAL_PRICE_WEB_SEARCH_PER_1K` (USD). Without input and output prices no estimate is
shown. It is an estimate from reported usage, not a bill; check current pricing before relying on it.

## Pieces

| File | Role |
|---|---|
| `profiles.py` | 11 deterministic Guardian profiles and their ground truth (`profile_facts`) |
| `golden.py` | The golden set; multi-turn entries replay earlier turns live |
| `scenarios.py` | Scenario bank across intent/character/time/social/ownership/knowledge/account |
| `properties.py` | Reusable behavioral properties and serious-failure flags |
| `rubric.py` | Eight 0-2 dimensions (max 16) plus flags that are never averaged away |
| `judge.py` | Optional LLM judge with the same evidence the product had |
| `runtime.py` | Recording subclass of the production service; local Manifest resolver |
| `report.py`, `runner.py` | JSON records, Markdown summary, CLI |
| `cassette.py`, `replay.py`, `pricing.py` | Recorded model calls, offline replay, cost estimate |

## Rules

A request for a *current* source (`Expectations.current_evidence`) is not satisfied by static
Manifest or guide hints alone: the answer needs live-provider or web-research evidence, or an
explicit statement that the current state is unverified (`current_source_not_overclaimed`).
`Expectations.forbidden_tools` marks tools that are unrelated to a request, so a follow-up that
should reuse earlier evidence is not credited for calling them (`no_irrelevant_tools`).

Golden prompts are regression tests, not special cases: never branch production code on them.
A failing property points at a behavior class; fix the class. Judge scores are advisory;
deterministic flags are authoritative for what they cover, and judge "ungrounded" notes
should be checked against `results.json` before acting on them.

## Held-out scenarios

`heldout.py` holds 15 unseen scenarios (6 multi-turn, casual wording, unavailable-data and
live-state cases) that were never used to repair the product and are not in `BANK`. A failure is a
behavior class to triage, not a prompt to special-case. Run them once:

```
python -m evals.runner --subset heldout --no-judge
```

Two real-model runs exposed held-out scenarios, which then count as seen. `heldout_retired.py` keeps
them under their original ids as regression scenarios (`--subset heldout_retired`); they are not
held-out and not counted toward the 15. Each was replaced by a new scenario for the same behavior
class with different wording and items. Retired this way:

- run `started_at=2026-10-22T04:52:27+00:00`: `frenzy_somewhere`, `coil_still_worth_farming`,
  `exotic_armor_stacking`, `pvp_then_dungeon_build`
- run `started_at=2026-10-22T06:01:24+00:00`: `envious_assassin_elsewhere`, `two_exotic_weapons`,
  `no_stat_for_trials`

| Retired (regression) | Replacement (held-out) | Class |
| --- | --- | --- |
| `frenzy_somewhere` | `envious_assassin_elsewhere`, then `whirlwind_blade_elsewhere` | perk held only by another character: name the item and its holder |
| `coil_still_worth_farming` | `xur_stock_now` | live-state: source or stated limit |
| `exotic_armor_stacking` | `two_exotic_weapons`, then `class_premise_on_vaulted_armor` (2 turns) | equip rules / item facts: a wrong player premise is corrected, not agreed with |
| `pvp_then_dungeon_build` | `onslaught_build_then_waves` (2 turns) | no curated guide: no borrowed guide |
| `no_stat_for_trials` | `headshots_with_rose` | unreturned stat: absence is not a value |

The 15 now hold six multi-turn scenarios. A scenario that exposes a failure in the next run counts
as seen too: retire and replace it before the following held-out run.

Check design notes:

- Exotic-rule turns use `exotic_limit(...)` (a stated "only one Exotic <kind>", "one Exotic <kind>
  ... at once/at a time/at any given time" with up to 60 characters between, or "can't wear both")
  plus `EXOTIC_WRONG_YES` (a yes to wearing both, "one Exotic per slot"), so a bare "one Exotic"
  does not pass.
- `linked_source_or_limitation` also accepts a stated limit built on the shared `LACK` patterns
  ("can't reliably answer", "unable to check live stock", "aren't available"), but a bare "can't"
  with no limit-related word nearby does not count.
- An unreturned stat must not be reported as any value, zero and "none" included (`STREAK_ASSERTED`
  for streaks; the headshot scenario has its own list). Explaining what a streak is stays allowed.
- `no_contradicted_class_restrictions` (every scenario, every turn) flags "<item> is <Class>-only",
  "locked/restricted to <Class>", "<Class> Exotic" and "<Class>s can wear <item>" when the fixture's
  `class_restriction` differs; negations and corrections ("isn't Titan-only", "Titans can't wear")
  are not flagged. Item truth comes from `profiles.py`, so an item the profile does not own is
  checked against the eval gear table.
- `no_activity_record_conflation` skips a sentence that explicitly denies the credit ("does NOT
  count toward the Triumph").
- `TurnExpectation.tools_any` requires a tool on one turn, so a mechanics follow-up is not graded on
  the build turn's loadout tool or gear names (`onslaught_build_then_waves`).

Expected facts are asserted against `profiles.py` in `tests/test_eval_heldout.py`.

Known issues recorded for later triage (`heldout.KNOWN_ISSUES`): Prismatic first-turn false
refusal; unverified equipment improvements despite empty alternative-search results; planning
fallback and unsupported Haunted Altars reward/solo claims; the Sunshot Exotic replacement check
is likely a false positive; current-source answers should expose the sources they cite.
