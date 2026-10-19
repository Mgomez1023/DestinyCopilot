"""M0 mechanics baseline: scenario bank shape, informational property, and aggregation."""

import re
from typing import Any

from app.build_analysis import BUILD_TOOL_DEFINITIONS
from app.destiny_knowledge import DESTINY_KNOWLEDGE_TOOL_DEFINITIONS
from app.guardian_tools import GUARDIAN_TOOL_DEFINITIONS
from evals.mechanics import MECHANICS_SCENARIOS, MECHANICS_SPECS, mechanics_baseline
from evals.profiles import PROFILES, profile_facts
from evals.properties import PROPERTIES, evaluate_properties
from evals.rubric import FLAGS
from evals.runtime import Observation, TurnRecord
from evals.scenarios import BANK, SUBSETS

REQUIRED_CATEGORIES = {
    "exotic_effect",
    "armor_slot",
    "armor_class",
    "exotic_conditional",
    "perk_explanation",
    "configurable_vs_fixed",
    "equipped_configuration",
    "mod_compatibility",
    "mod_effect",
    "fragment_effect",
    "damage_source",
    "hypothetical_configuration",
    "multi_item_synergy",
    "unknown_mechanic",
    "patch_sensitive",
}


def test_the_mechanics_bank_has_about_twenty_distinct_scenarios() -> None:
    ids = [value.id for value in MECHANICS_SCENARIOS]

    assert 18 <= len(ids) <= 24
    assert len(set(ids)) == len(ids)
    assert all(value.startswith("mech.") for value in ids)
    assert set(MECHANICS_SPECS) == set(ids)


def test_the_bank_covers_every_audited_mechanics_category() -> None:
    assert {spec.category for spec in MECHANICS_SPECS.values()} >= REQUIRED_CATEGORIES


def test_every_scenario_states_its_evidence_requirement_and_serious_failure() -> None:
    for scenario_id, spec in MECHANICS_SPECS.items():
        assert spec.evidence_required, scenario_id
        assert spec.serious_failure.strip(), scenario_id


def test_scenarios_use_existing_profiles_tools_and_valid_patterns() -> None:
    known_tools = {
        value["name"]
        for value in [
            *GUARDIAN_TOOL_DEFINITIONS,
            *BUILD_TOOL_DEFINITIONS,
            *DESTINY_KNOWLEDGE_TOOL_DEFINITIONS,
        ]
    }
    for scenario in MECHANICS_SCENARIOS:
        assert scenario.profile in PROFILES, scenario.id
        assert scenario.prompt.strip()
        assert set(scenario.expect.tools_any) <= known_tools, scenario.id
        for pattern in scenario.expect.forbidden:
            re.compile(pattern)


def test_the_mechanics_subset_is_registered_and_does_not_grow_the_main_bank() -> None:
    assert SUBSETS["mechanics"] == MECHANICS_SCENARIOS
    assert not {value.id for value in MECHANICS_SCENARIOS} & {value.id for value in BANK}
    assert len(BANK) <= 150


def test_scenarios_that_name_ownership_or_class_set_matching_expectations() -> None:
    by_id = {value.id: value for value in MECHANICS_SCENARIOS}

    assert by_id["mech.owned_only_hypothetical_change"].expect.only_owned is True
    assert by_id["mech.cross_class_hypothetical"].expect.not_owned == ("Verity's Brow",)
    assert by_id["mech.patch_sensitive_claim"].expect.live_sensitive is True


def observe_with_trace(trace: dict[str, Any]) -> tuple[Observation, Any]:
    scenario = MECHANICS_SCENARIOS[0]
    context = PROFILES[scenario.profile]()
    turn = TurnRecord(
        message=scenario.prompt,
        answer="Sunshot fires explosive rounds.",
        source="openai",
        trace=trace,
    )
    return Observation(scenario, context, [turn]), profile_facts(context)


def test_the_mechanics_property_is_informational_and_never_flags() -> None:
    summary = {
        "mode": "trace_only",
        "claims_total": 3,
        "needs_attention": 2,
        "by_status": {"mentioned_without_evidence": 2, "supported": 1},
    }
    obs, facts = observe_with_trace({"grounding": {}, "mechanics_claims": summary})

    result = next(
        value for value in evaluate_properties(obs, facts) if value.name == "mechanics_claims_trace"
    )

    assert result.applies and result.passed and result.flag is None
    assert "2 need attention" in result.detail


def test_the_mechanics_property_is_skipped_without_a_trace() -> None:
    obs, facts = observe_with_trace({"grounding": {}})

    result = next(
        value for value in evaluate_properties(obs, facts) if value.name == "mechanics_claims_trace"
    )

    assert result.applies is False


def test_the_property_adds_no_new_serious_failure_flag() -> None:
    assert len(PROPERTIES) == len({check.__name__ for check in PROPERTIES})
    assert "unsupported_mechanics" not in FLAGS


def test_baseline_aggregation_counts_statuses_across_scenarios() -> None:
    records = [
        {
            "id": "mech.a",
            "turns": [
                {
                    "mechanics_claims": {
                        "claims_total": 3,
                        "needs_attention": 2,
                        "by_status": {"mentioned_without_evidence": 2, "supported": 1},
                    }
                }
            ],
        },
        {
            "id": "mech.b",
            "turns": [{"mechanics_claims": None}, {"mechanics_claims": {"error": "x"}}],
        },
        {
            "id": "mech.c",
            "turns": [
                {
                    "mechanics_claims": {
                        "claims_total": 1,
                        "needs_attention": 0,
                        "by_status": {"supported": 1},
                    }
                }
            ],
        },
    ]

    baseline = mechanics_baseline(records)

    assert baseline["turns"] == 4
    assert baseline["turns_traced"] == 2
    assert baseline["claims_total"] == 4
    assert baseline["needs_attention"] == 2
    assert baseline["by_status"] == {"mentioned_without_evidence": 2, "supported": 2}
    assert baseline["needs_attention_by_scenario"] == {"mech.a": 2}
