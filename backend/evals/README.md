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

Run `started_at=2026-10-22T04:52:27+00:00` exposed four of the original scenarios, so they count as
seen. `heldout_retired.py` keeps them, under their original ids, as regression scenarios
(`--subset heldout_retired`); they are not held-out and not counted toward the 15. Each was replaced
by a new scenario for the same behavior class with different wording, items and activities:

| Retired (regression) | Replacement (held-out) | Class |
| --- | --- | --- |
| `frenzy_somewhere` | `envious_assassin_elsewhere` | perk search across characters |
| `coil_still_worth_farming` | `xur_stock_now` | live-state: source or stated limit |
| `exotic_armor_stacking` | `two_exotic_weapons` (3 turns) | equip rules, wrong "yes" rejected |
| `pvp_then_dungeon_build` | `onslaught_build_then_waves` (2 turns) | no curated guide: no borrowed guide |

Check design notes: exotic-rule turns use `exotic_limit(...)` (a stated "only one Exotic <kind>" or
"can't wear both") plus `EXOTIC_WRONG_YES` (a yes to wearing both, "one Exotic per slot"), so a bare
"one Exotic" no longer passes. `linked_source_or_limitation` also accepts a stated limit built on
the shared `LACK` patterns ("can't reliably answer", "unable to check live stock"), but a bare
"can't" with no limit-related word nearby does not count. Expected facts are asserted against
`profiles.py` in `tests/test_eval_heldout.py`.

Known issues recorded for later triage (`heldout.KNOWN_ISSUES`): Prismatic first-turn false
refusal; unverified equipment improvements despite empty alternative-search results; planning
fallback and unsupported Haunted Altars reward/solo claims; the Sunshot Exotic replacement check
is likely a false positive; current-source answers should expose the sources they cite.
