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

Golden prompts are regression tests, not special cases: never branch production code on them.
A failing property points at a behavior class; fix the class. Judge scores are advisory;
deterministic flags are authoritative for what they cover, and judge "ungrounded" notes
should be checked against `results.json` before acting on them.
