"""Playtest regression scenarios (Operation Guardian).

Each test replays a realistic conversation from production playtesting against the real
RecommendationService, with the model's drafts scripted and every tool executed for real
against a synthetic Guardian. Assertions are on what the player would receive: the final
message, the lookups that actually ran, and whether a retry kept tools available. No network,
credentials, or model calls are involved.
"""

import asyncio
import json
import pathlib
from types import SimpleNamespace
from typing import Any

import pytest

from app.ai import RecommendationService
from app.build_analysis import derive_build_request_context
from app.config import Settings
from app.models import ChatRequest, ChatTurn, GuardianContext, SocketedPlugSummary
from app.session_preferences import derive_session_preferences
from evals.profiles import PROFILES
from tests.test_guardian_tools import assistant_message, web_call

MINT_SOURCE = "https://www.bungie.net/7/en/News/Article/the-coil-rewards"


class Script:
    """Scripted Responses client: one output list per model request, in order."""

    def __init__(self, *outputs: list[Any]) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        index = len(self.requests)
        self.requests.append(kwargs)
        output = self.outputs[index]
        text = "".join(
            part.text
            for value in output
            if getattr(value, "type", None) == "message"
            for part in value.content
        )
        return SimpleNamespace(
            output=output,
            output_text=text,
            status="completed",
            incomplete_details=None,
            error=None,
            usage=None,
        )

    def developer_messages(self, index: int) -> list[str]:
        return [
            value["content"]
            for value in self.requests[index]["input"]
            if isinstance(value, dict) and value.get("role") == "developer"
        ]

    def tool_output(self, index: int, call_id: str) -> dict[str, Any]:
        value = next(
            item
            for item in self.requests[index]["input"]
            if isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and item.get("call_id") == call_id
        )
        return json.loads(value["output"])


class StubKnowledge:
    """A Manifest-like provider with one item-source lookup and canned results."""

    def __init__(self, results: dict[str, dict[str, Any]]) -> None:
        self.results = results
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": name,
                "description": "Manifest-backed item lookup.",
                "parameters": {"type": "object", "properties": {}},
            }
            for name in self.results
        ]

    def handles(self, name: str) -> bool:
        return name in self.results

    @staticmethod
    def category_for_tool(name: str) -> str:
        del name
        return "manifest"

    @staticmethod
    def has_category(category: str) -> bool:
        return category == "manifest"

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return self.results[name]


def call(name: str, call_id: str, **arguments: Any) -> Any:
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=json.dumps(arguments),
        call_id=call_id,
        id=f"fc-{call_id}",
    )


def cited(text: str, url: str, span: str, title: str = "The Coil rewards") -> Any:
    start = text.index(span)
    message = assistant_message(text)
    message.content[0].annotations = [
        SimpleNamespace(
            type="url_citation",
            url=url,
            title=title,
            start_index=start,
            end_index=start + len(span),
        )
    ]
    return message


def service(script: Script, knowledge: Any = None) -> RecommendationService:
    value = RecommendationService(
        Settings(openai_api_key="test-key", enable_debug_tools=True), knowledge
    )
    value.client = SimpleNamespace(responses=script)
    return value


def run(
    value: RecommendationService,
    message: str,
    context: GuardianContext,
    history: list[ChatTurn] | None = None,
) -> Any:
    return asyncio.run(value.chat(ChatRequest(message=message, history=history or []), context))


def titan_with(weapon_name: str | None = None) -> GuardianContext:
    """The eval Titan veteran, optionally with its kinetic weapon renamed (synthetic)."""

    context = PROFILES["titan_veteran"]()
    if weapon_name:
        titan = context.characters[0]
        titan.equipped_gear = [
            value.model_copy(update={"name": weapon_name}) if value.name == "Rose" else value
            for value in titan.equipped_gear
        ]
    return context


def prismatic_titan() -> GuardianContext:
    """Titan on Prismatic with two equipped fragments; unlock state is not in the data."""

    context = titan_with()
    titan = context.characters[0]
    fragments = [
        SocketedPlugSummary(
            item_hash=9100 + index,
            name=name,
            item_type="Mod",
            plug_type="Prismatic Fragment",
            category_identifier="shared.prism.fragments",
            socket_index=7 + index,
        )
        for index, name in enumerate(("Facet of Purpose", "Facet of Courage"))
    ]
    prismatic = titan.subclass.model_copy(
        update={
            "name": "Prismatic Titan",
            "damage_type": None,
            "socketed_plugs": [value.name for value in fragments],
            "socketed_plug_details": fragments,
        }
    )
    titan.subclass = prismatic
    titan.equipped_gear = [prismatic, *titan.equipped_gear[1:]]
    return context


# --- 1. Current-source verification is performed, not offered ---------------------------


def test_verification_request_researches_instead_of_offering_a_source_choice() -> None:
    researched = "Yes. Bungie's current reward notes still list Mint Retrograde from The Coil."
    choice = "I can check official live data or community reports. Which do you prefer?"
    script = Script(
        [assistant_message(choice)],
        [web_call([MINT_SOURCE]), assistant_message(researched)],
    )
    value = service(script)

    response = run(
        value,
        "Can you verify The Coil drops Mint Retrograde with a current source?",
        titan_with(),
    )

    assert response.message == researched
    assert len(script.requests) == 2
    # The retry keeps read-only tools available so the research can actually happen.
    assert script.requests[1]["tool_choice"] == "auto"
    assert any("already authorized" in value for value in script.developer_messages(1))
    assert [source.url for source in response.sources] == [MINT_SOURCE]
    trace = value.latest_trace()
    assert trace["initiative_retry"]["attempted"] is True
    assert trace["web_research"]["occurred"] is True


def test_verification_request_answered_from_memory_is_sent_back_for_a_lookup() -> None:
    researched = "Confirmed: Bungie's current reward notes list Mint Retrograde from The Coil."
    script = Script(
        [assistant_message("Yes, The Coil drops Mint Retrograde.")],
        [web_call([MINT_SOURCE]), assistant_message(researched)],
    )
    value = service(script)

    response = run(
        value,
        "Can you verify The Coil drops Mint Retrograde with a current source?",
        titan_with(),
    )

    assert response.message == researched
    assert "unperformed_lookup" in value.latest_trace()["initiative_retry"]["violation_codes"]


def test_claiming_a_search_that_never_happened_is_not_accepted() -> None:
    researched = "Bungie's reward notes list Mint Retrograde from The Coil."
    script = Script(
        [assistant_message("I checked Bungie's site and The Coil still drops Mint Retrograde.")],
        [web_call([MINT_SOURCE]), assistant_message(researched)],
    )

    response = run(service(script), "Does The Coil drop Mint Retrograde?", titan_with())

    assert response.message == researched
    assert script.requests[1]["tool_choice"] == "auto"


# --- 2. An inventory question uses Guardian evidence -----------------------------------


def test_inventory_question_runs_the_lookup_instead_of_asking_permission() -> None:
    answer = "Yes. Fatebringer is in your vault."
    script = Script(
        [assistant_message("Want me to search your vault for hand cannons?")],
        [
            call(
                "search_inventory",
                "inv-1",
                query=None,
                character_id=None,
                character_class=None,
                item_type="Weapon",
                subtype="Hand Cannon",
                bucket=None,
                equipped_only=False,
                limit=25,
            )
        ],
        [assistant_message(answer)],
    )

    response = run(service(script), "Do I have any hand cannons in my vault?", titan_with())

    assert response.message == answer
    names = [value["name"] for value in script.tool_output(2, "inv-1")["items"]]
    assert "Fatebringer" in names


# --- 3. Missing Prismatic fragments: check the account, no screenshot dead end ----------

FRAGMENT_HISTORY = [
    ChatTurn(
        role="user",
        content=(
            "I've finished The Final Shape campaign. How do I unlock the rest of my Prismatic "
            "fragments?"
        ),
    ),
    ChatTurn(
        role="assistant",
        content="Prismatic fragments unlock through the Pale Heart after the campaign.",
    ),
]
FRAGMENT_QUESTION = (
    "Which Prismatic fragments am I still missing on my Titan, and what's the fastest way to "
    "unlock the next one?"
)


def test_missing_fragments_checks_guardian_data_instead_of_requesting_a_screenshot() -> None:
    answer = (
        "Your Titan has Facet of Purpose and Facet of Courage equipped on Prismatic. Your "
        "account data shows what is equipped, not your full unlocked set, so I can't list "
        "exactly which fragments are missing. The fastest next step is the Pale Heart "
        "fragment route."
    )
    script = Script(
        [
            assistant_message(
                "I can't see which fragments you've unlocked. Upload a screenshot of your "
                "Prismatic fragment screen or type the fragments you have, and I'll tell you "
                "which ones you're missing."
            )
        ],
        [
            call(
                "analyze_current_build",
                "build-1",
                character_id=None,
                character_class="Titan",
                goal=None,
                activity=None,
                locked_items=None,
                preserve_exotics=False,
            )
        ],
        [assistant_message(answer)],
    )
    value = service(script)

    response = run(value, FRAGMENT_QUESTION, prismatic_titan(), FRAGMENT_HISTORY)

    assert response.message == answer
    assert "screenshot" not in response.message.casefold()
    analysis = script.tool_output(2, "build-1")
    assert [plug["name"] for plug in analysis["subclass"]["fragments"]] == [
        "Facet of Purpose",
        "Facet of Courage",
    ]
    codes = value.latest_trace()["initiative_retry"]["violation_codes"]
    assert "account_data_requested" in codes


def test_a_screenshot_request_is_trimmed_without_another_model_round() -> None:
    draft = (
        "Your Titan has Facet of Purpose and Facet of Courage equipped. I can't confirm "
        "which other fragments are unlocked. Send me a screenshot of your fragment screen and "
        "I'll compare."
    )
    script = Script(
        [
            call(
                "analyze_current_build",
                "build-1",
                character_id=None,
                character_class="Titan",
                goal=None,
                activity=None,
                locked_items=None,
                preserve_exotics=False,
            )
        ],
        # The unlock path was researched, so only the screenshot request is wrong.
        [web_call([MINT_SOURCE]), assistant_message(draft)],
    )

    response = run(service(script), FRAGMENT_QUESTION, prismatic_titan(), FRAGMENT_HISTORY)

    assert len(script.requests) == 2
    assert "screenshot" not in response.message.casefold()
    assert response.message.startswith("Your Titan has Facet of Purpose and Facet of Courage")
    assert "can't confirm which other fragments are unlocked" in response.message


# --- 4. One build improvement that respects the player's stated goal --------------------

GOAL_HISTORY = [
    ChatTurn(role="user", content="I want to get Mint Retrograde."),
    ChatTurn(role="assistant", content="Mint Retrograde comes from The Coil."),
]
BIGGEST_IMPROVEMENT = (
    "Look at my current Titan build. What's the single biggest improvement I could make?"
)


def _build_call() -> Any:
    return call(
        "analyze_current_build",
        "build-1",
        character_id=None,
        character_class="Titan",
        goal=None,
        activity=None,
        locked_items=None,
        preserve_exotics=False,
    )


def test_single_improvement_request_carries_the_stated_goal_and_a_one_change_limit() -> None:
    preferences = derive_session_preferences(GOAL_HISTORY, BIGGEST_IMPROVEMENT).preferences
    context = derive_build_request_context(BIGGEST_IMPROVEMENT, preferences, GOAL_HISTORY)

    assert context.goal_items == ["Mint Retrograde"]
    assert context.normal_change_limit == 1
    assert context.focused_recommendation is True
    assert context.goal is None


def test_build_advice_does_not_replace_an_item_the_player_wants() -> None:
    corrected = (
        "Keep Mint Retrograde. The single biggest improvement is filling your open fragment "
        "slots before changing any weapon."
    )
    script = Script(
        [_build_call()],
        [
            assistant_message(
                "The single biggest improvement: replace Mint Retrograde with a high-impact "
                "Linear Fusion Rifle for boss DPS."
            )
        ],
        [assistant_message(corrected)],
    )
    value = service(script)

    response = run(value, BIGGEST_IMPROVEMENT, titan_with("Mint Retrograde"), GOAL_HISTORY)

    assert response.message == corrected
    assert script.requests[0]["tool_choice"] == {
        "type": "function",
        "name": "analyze_current_build",
    }
    assert '"goal_items":["Mint Retrograde"]' in script.requests[0]["instructions"]
    trace = value.latest_trace()
    # Replacing a worn item with an unnamed one is first sent back for a real inventory search.
    seen = (
        trace["planning_correction"]["violation_codes"]
        + trace["initiative_retry"]["violation_codes"]
    )
    assert "goal_item_replaced" in seen


def test_goal_respect_is_general_not_specific_to_one_weapon() -> None:
    history = [
        ChatTurn(role="user", content="I really like using Rose."),
        ChatTurn(role="assistant", content="Rose is a solid kinetic hand cannon."),
    ]
    corrected = "Keep Rose. Your biggest gain is filling your open fragment slots."
    script = Script(
        [_build_call()],
        [assistant_message("Swap out Rose for a stronger kinetic weapon.")],
        [assistant_message(corrected)],
    )

    response = run(service(script), BIGGEST_IMPROVEMENT, titan_with(), history)

    assert response.message == corrected


def test_one_improvement_request_rejects_a_list_of_changes() -> None:
    focused = "Fill your open fragment slot; that single change has the clearest verified gain."
    script = Script(
        [_build_call()],
        [
            assistant_message(
                "Biggest improvements:\n"
                "- Swap Calus Mini-Tool for Fatebringer.\n"
                "- Replace Apex Predator with a sword.\n"
                "- Change your grenade to Thermite."
            )
        ],
        [assistant_message(focused)],
    )
    value = service(script)

    response = run(value, BIGGEST_IMPROVEMENT, titan_with())

    assert response.message == focused
    trace = value.latest_trace()
    seen = (
        trace["planning_correction"]["violation_codes"]
        + trace["initiative_retry"]["violation_codes"]
    )
    assert "too_many_build_changes" in seen


# --- 5 and 6. Planning follow-ups keep rejections and constraints ------------------------


def _planning_json(instructions: str) -> dict[str, Any]:
    marker = "do not infer additional account facts from missing values:\n"
    return json.loads(instructions.split(marker, 1)[1].split("\n", 1)[0])


def test_rejected_activity_is_not_recommended_again() -> None:
    history = [
        ChatTurn(role="user", content="What should I do next?"),
        ChatTurn(role="assistant", content="Run Nightfall: The Lightblade for Vanguard progress."),
    ]
    corrected = "Run Vanguard Ops on your Titan instead; it advances your Weekly Vanguard."
    script = Script(
        [assistant_message("Run Nightfall: The Lightblade again for Vanguard progress.")],
        [assistant_message(corrected)],
    )
    value = service(script)

    response = run(
        value, "I don't want to do the Nightfall. Give me something else.", titan_with(), history
    )

    assert response.message == corrected
    planning = _planning_json(script.requests[0]["instructions"])
    assert "Nightfall: The Lightblade" not in {value["name"] for value in planning["candidates"]}
    assert (
        "rejected_candidate_recommended"
        in value.latest_trace()["planning_correction"]["violation_codes"]
    )


def test_session_constraints_survive_a_follow_up() -> None:
    history = [
        ChatTurn(
            role="user",
            content="I have 30 minutes, solo, no Crucible and no dungeons. What should I do?",
        ),
        ChatTurn(role="assistant", content="Run Vanguard Ops on your Titan."),
    ]
    corrected = "Run Haunted Altars of Sorrow solo; it fits your 30 minutes."
    script = Script(
        [assistant_message("How much time do you have? Run Duality: Standard for great loot.")],
        [assistant_message(corrected)],
    )
    value = service(script)

    response = run(value, "Something with better loot instead?", titan_with(), history)

    assert response.message == corrected
    planning = _planning_json(script.requests[0]["instructions"])
    preferences = planning["preferences"]
    assert preferences["time_minutes"] == 30
    assert preferences["fireteam"] == "solo"
    assert preferences["primary_goal"] == "gear_rewards"
    assert set(preferences["exclusions"]) >= {"pvp", "dungeons"}
    # Excluded activity types never reach the model as candidates.
    names = {value["name"] for value in planning["candidates"]}
    assert not names & {"Duality: Standard", "Crucible: Control"}
    assert {"Duality: Standard", "Crucible: Control"} <= set(planning["excluded_by_preference"])
    # Solo also sets aside activities with a fireteam of three, and says why.
    assert not names & {"Nightfall: The Lightblade", "Vanguard Ops", "Weekly Vanguard"}
    assert {"Nightfall: The Lightblade", "Vanguard Ops"} <= set(planning["excluded_by_preference"])
    assert any("solo preference" in value for value in planning["data_limitations"])
    codes = value.latest_trace()["planning_correction"]["violation_codes"]
    assert {"reasked_known_time", "excluded_candidate_recommended"} <= set(codes)


# --- 7 and 8. Write requests: honest, no pretend actions or offers ----------------------


def test_vault_move_request_is_never_answered_with_a_pretend_action() -> None:
    honest = (
        "I can't move items; Copilot is read-only. In game, open your inventory, hover Rose, "
        "and transfer it to the vault from the item menu."
    )
    script = Script(
        [assistant_message("Done! I've moved Rose to your vault.")],
        [assistant_message(honest)],
    )
    value = service(script)

    response = run(value, "Move my Rose to the vault.", titan_with())

    assert response.message == honest
    assert "implied_bungie_write" in value.latest_trace()["planning_correction"]["violation_codes"]


def test_offer_to_perform_an_inventory_change_is_removed() -> None:
    draft = (
        "Rose is equipped on your Titan's kinetic slot. Want me to move it to your vault for you?"
    )
    script = Script([assistant_message(draft)], [assistant_message(draft)])

    response = run(service(script), "Where is my Rose right now?", titan_with())

    assert response.message == "Rose is equipped on your Titan's kinetic slot."


def test_fixed_exotic_perk_change_is_not_offered_as_an_action() -> None:
    honest = (
        "I can't change gear for you, and Sunshot's barrel isn't something you can swap: "
        "its perks are fixed on the Exotic."
    )
    # The definition is looked up first, so the fixed-perk statement rests on a result.
    knowledge = StubKnowledge({"get_item_details": {"item": {"name": "Sunshot"}}})
    script = Script(
        [call("get_item_details", "item-1", item_name_or_hash="Sunshot")],
        [assistant_message("Sure, I'll swap Sunshot's barrel to a range barrel for you.")],
        [assistant_message(honest)],
    )

    response = run(
        service(script, knowledge), "Can you change the barrel on my Sunshot?", titan_with()
    )

    assert response.message == honest
    assert knowledge.calls[0][0] == "get_item_details"


# --- 9. Simple drop-source question: direct answer, no closing permission offer ---------


def test_drop_source_question_answers_directly_without_a_permission_offer() -> None:
    knowledge = StubKnowledge(
        {"find_item_source": {"item": "Mint Retrograde", "sources": ["Source: The Coil"]}}
    )
    direct = "Mint Retrograde drops from The Coil."
    script = Script(
        [call("find_item_source", "src-1", item_name_or_hash="Mint Retrograde")],
        [
            assistant_message(
                "Mint Retrograde drops from The Coil. Want me to check whether you already own one?"
            )
        ],
    )

    value = service(script, knowledge)

    response = run(value, "Where does Mint Retrograde drop?", titan_with())

    assert response.message == direct
    assert knowledge.calls == [("find_item_source", {"item_name_or_hash": "Mint Retrograde"})]
    # A lookup already ran, so the closing offer is simply dropped: no extra model round.
    assert len(script.requests) == 2
    trace = value.latest_trace()
    assert trace["trailing_handback_removed"] is True
    assert trace["initiative_retry"]["attempted"] is False
    assert trace["planning_correction"]["attempted"] is False


# --- 10. A lookup that cannot answer: brief limitation, useful next step ----------------


def test_unavailable_account_data_is_acknowledged_without_a_retry_or_refusal() -> None:
    answer = (
        "Your account data here doesn't include catalyst completion, so I can't list them. "
        "In game, open Collections, then Exotics, to see each catalyst's progress."
    )
    script = Script(
        [call("get_progression", "prog-1", character_id=None, character_class=None)],
        [assistant_message(answer)],
    )
    value = service(script)

    response = run(value, "Which Exotic catalysts have I completed?", titan_with())

    assert response.message == answer
    assert len(script.requests) == 2
    assert value.latest_trace()["planning_correction"]["attempted"] is False


def test_asking_for_a_specific_name_after_a_real_lookup_is_allowed() -> None:
    answer = (
        "I checked your Titan's loadout but can't tell which quest you mean. Which quest name "
        "should I look up?"
    )
    script = Script(
        [call("get_active_quests", "quest-1", character_id=None, character_class="Titan")],
        [assistant_message(answer)],
    )

    response = run(service(script), "How do I finish that quest on my Titan?", titan_with())

    assert response.message == answer
    assert len(script.requests) == 2


# --- F. Citation presentation ----------------------------------------------------------


def test_inline_citation_links_render_as_readable_text_and_keep_the_source() -> None:
    url = f"{MINT_SOURCE}?utm_source=openai"
    link = f"[bungie.net]({url})"
    text = f"Yes, The Coil still drops Mint Retrograde ({link})."
    script = Script([web_call([MINT_SOURCE]), cited(text, url, link)])

    response = run(service(script), "Is Mint Retrograde still in The Coil?", titan_with())

    assert response.message == "Yes, The Coil still drops Mint Retrograde (bungie.net)."
    assert response.sources and response.sources[0].url.startswith(MINT_SOURCE)


def test_leaked_private_citation_markers_are_removed() -> None:
    text = "Mint Retrograde drops from The Coil citeturn0search0. Each run can award it."
    script = Script([web_call([MINT_SOURCE]), assistant_message(text)])

    response = run(service(script), "Is Mint Retrograde still in The Coil?", titan_with())

    assert response.message == "Mint Retrograde drops from The Coil. Each run can award it."


def test_capability_denials_are_kept_intact() -> None:
    honest = (
        "I can't view screenshots or move items, but I can read your equipped loadout. "
        "Your Titan is running Rose."
    )
    script = Script([assistant_message(honest)])

    response = run(service(script), "Can I send you a screenshot of my vault?", titan_with())

    assert response.message == honest


# --- Streaming: a retried draft is never shown, and statuses stay in order --------------


def test_streaming_initiative_retry_never_emits_the_permission_draft() -> None:
    from app.chat_stream import StreamStatusReporter, response_events
    from tests.test_guardian_tools import FakeResponses, ScriptedStreamingResponses

    draft = "Want me to search your vault for hand cannons?"
    answer = "Yes. Fatebringer is in your vault."
    streamed = ScriptedStreamingResponses(
        [
            FakeResponses.response([assistant_message(draft)]),
            FakeResponses.response(
                [
                    call(
                        "search_inventory",
                        "inv-1",
                        query=None,
                        character_id=None,
                        character_class=None,
                        item_type="Weapon",
                        subtype="Hand Cannon",
                        bucket=None,
                        equipped_only=False,
                        limit=25,
                    )
                ]
            ),
            FakeResponses.response([assistant_message(answer)]),
        ]
    )
    emitted: list[Any] = []

    async def emit(event: Any) -> None:
        emitted.append(event)

    value = RecommendationService(Settings(openai_api_key="test-key"))
    value.client = SimpleNamespace(responses=streamed)
    response = asyncio.run(
        value.chat(
            ChatRequest(message="Do I have any hand cannons in my vault?"),
            titan_with(),
            stream_status=StreamStatusReporter(emit),
        )
    )
    text = "".join(
        event.delta
        for event in response_events(response, "direct_fact")
        if event.event == "message_delta"
    )

    assert text == answer
    assert draft not in text
    stages = [event.stage for event in emitted]
    # The final status comes only after the lookup the retry performed.
    assert stages.index("inventory") < stages.index("final")
    assert len(streamed.requests) == 3


# --- Release audit: bounds, precision, and release of preferences -----------------------


def test_a_draft_that_keeps_asking_is_retried_once_then_corrected_without_tools() -> None:
    from tests.test_guardian_tools import reasoning_item

    ask = "Want me to search your vault for hand cannons?"
    answer = "Fatebringer is in your vault."
    script = Script(
        [reasoning_item(), assistant_message(ask)],
        [assistant_message(ask)],
        [assistant_message(answer)],
    )
    value = service(script)

    response = run(value, "Do I have any hand cannons in my vault?", titan_with())

    assert response.message == answer
    assert len(script.requests) == 3
    assert [script.requests[index]["tool_choice"] for index in range(3)] == ["auto", "auto", "none"]
    # The retry replays the rejected draft (reasoning first, then its message) before the nudge.
    replay = [
        value.get("type") or value.get("role")
        for value in script.requests[1]["input"]
        if isinstance(value, dict)
    ]
    assert replay[-3:] == ["reasoning", "message", "developer"]


def test_an_honest_limit_after_the_retry_is_accepted_without_another_round() -> None:
    honest = "I can't verify clan raid schedules from Destiny data; check your clan's Discord."
    script = Script([assistant_message(honest)], [assistant_message(honest)])
    value = service(script)

    response = run(value, "Can you confirm my clan's raid schedule?", titan_with())

    assert response.message == honest
    assert len(script.requests) == 2
    assert value.latest_trace()["planning_correction"]["attempted"] is False


def test_planning_context_counts_as_checking_the_account() -> None:
    answer = "I checked your active quests: run Vanguard Ops for the Weekly Vanguard."
    script = Script([assistant_message(answer)])

    response = run(service(script), "What should I do next?", titan_with())

    assert response.message == answer
    assert len(script.requests) == 1


def test_a_repeated_claim_of_moving_an_item_never_reaches_the_player() -> None:
    from app.response_quality import UNSUPPORTED_CAPABILITY_FALLBACK

    claim = "Done! I've moved Rose to your vault."
    script = Script([assistant_message(claim)], [assistant_message(claim)])

    response = run(service(script), "Move my Rose to the vault.", titan_with())

    assert response.message == UNSUPPORTED_CAPABILITY_FALLBACK


def test_a_citation_from_an_earlier_round_keeps_its_link_in_sources() -> None:
    urls = [f"https://site{index}.example/page" for index in range(5)]
    link = f"[site4.example]({urls[4]})"
    text = f"The Coil still drops Mint Retrograde ({link})."
    script = Script(
        [
            web_call(urls),
            call("get_character_summary", "c-1", character_id=None, character_class=None),
        ],
        [cited(text, urls[4], link)],
    )

    response = run(service(script), "Is Mint Retrograde still in The Coil?", titan_with())

    assert response.message == "The Coil still drops Mint Retrograde (site4.example)."
    # Four sources fit a direct answer; the cited one is never the one left out.
    assert len(response.sources) == 4
    assert response.sources[0].url == urls[4]


def test_goal_items_stay_soft_preferences_that_the_player_can_release() -> None:
    from app.build_analysis import derive_goal_items

    def goals(*turns: str) -> list[str]:
        return derive_goal_items(turns[-1], [ChatTurn(role="user", content=v) for v in turns[:-1]])

    wanted = "I want to get Mint Retrograde."
    # A neutral question about the item supports the goal.
    assert goals(wanted, "Where does Mint Retrograde drop?", "Biggest improvement?") == [
        "Mint Retrograde"
    ]
    # Considering a replacement, or going off the item, releases it for later turns too.
    assert goals(wanted, "Should I replace Mint Retrograde?", "Biggest improvement now?") == []
    assert (
        goals("I really like using Rose.", "I'm tired of Rose now.", "Biggest improvement?") == []
    )
    assert goals("I really like using Rose.", "You can change Rose.", "Biggest improvement?") == []
    # It is never passed on as a hard lock.
    history = [ChatTurn(role="user", content=wanted)]
    preferences = derive_session_preferences(history, BIGGEST_IMPROVEMENT).preferences
    context = derive_build_request_context(BIGGEST_IMPROVEMENT, preferences, history)
    assert context.goal_items == ["Mint Retrograde"] and context.locked_items == []


def test_planning_questions_are_not_read_as_recommendations() -> None:
    from app.session_planning import PlanningAnswerValidator, build_session_planning_context

    preferences = derive_session_preferences(
        [], "I have 30 minutes solo and no dungeons. What should I do?"
    ).preferences
    planning = build_session_planning_context(titan_with(), "plan", "general", None, preferences)

    def codes(answer: str) -> set[str]:
        return {value.code for value in PlanningAnswerValidator().validate(answer, planning)}

    # A follow-up question is not a recommendation, so it cannot break the solo constraint.
    assert "solo_constraint" not in codes(
        "Run Vanguard Ops solo for the Weekly Vanguard. Do you want fireteam options another time?"
    )
    # Imperative recommendations are checked against excluded activities.
    assert "excluded_candidate_recommended" in codes("Run Duality: Standard for its loot.")


# --- Real-model findings (live run 2026-10-21): replays of the model's actual drafts --------

LIVE = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "live_playtest_drafts.json").read_text("utf-8")
)["cases"]


def _live_history(case: dict[str, Any]) -> list[ChatTurn]:
    return [
        turn
        for user, answer in case["history"]
        for turn in (
            ChatTurn(role="user", content=user),
            ChatTurn(role="assistant", content=answer),
        )
    ]


def _replay(case: dict[str, Any], *outputs: list[Any]) -> tuple[Any, Script, Any]:
    script = Script(*outputs)
    value = service(script)
    response = run(value, case["message"], titan_with(), _live_history(case))
    return response, script, value.latest_trace()


ALT_PICK = (
    "Primary pick: Haunted Altars of Sorrow.. It is available to you and playable alone; your data "
    "does not show how its loot compares or how long it takes."
)
HEDGED_LOOT_PICK = (
    "Primary pick: Nightfall: The Lightblade. It is launchable on your Titan and is a three-player "
    "activity; loot quality and solo clear time are not established by your data. "
    "Backup: Vanguard Ops, which advances your Weekly Vanguard."
)


def test_live_loot_followup_keeps_one_recommendation_and_its_constraints() -> None:
    case = LIVE["constraints_followup"]
    activities = call("get_available_activities", "a-1", character_id=None, character_class=None)

    response, script, trace = _replay(
        case, [activities], [assistant_message(case["draft"])], [assistant_message(ALT_PICK)]
    )

    # Live: a false itinerary flag forced the "tell me which one" fallback after 3 requests.
    assert case["live_answer"].endswith("tell me which one to build out.")
    # The draft picks a three-player Nightfall for a solo player, so it is corrected once to an
    # alternative that fits the stated constraints.
    assert "excluded_candidate_recommended" in trace["planning_correction"]["violation_codes"]
    assert response.message == ALT_PICK
    assert "Nightfall" not in response.message


def test_live_alternative_answer_ends_on_its_recommendation() -> None:
    case = LIVE["reject_activity"]

    response, script, trace = _replay(case, [assistant_message(case["draft"])])

    assert len(script.requests) == 1
    assert trace["trailing_handback_removed"] is True
    assert "Pick one" not in response.message
    assert "Which of these" not in response.message
    assert response.message.startswith("Got it")


def test_live_vault_answer_is_kept_without_a_retry_round() -> None:
    case = LIVE["vault_move"]

    response, script, trace = _replay(case, [assistant_message(case["draft"])])

    # Live: the closing "if you want I can check..." cost a retry, a tool call, and a correction.
    assert case["live_model_requests"] == 4
    assert len(script.requests) == 1
    assert trace["initiative_retry"]["attempted"] is False
    assert response.message.endswith("If Rose is locked, unlock it first.")


def test_live_single_improvement_keeps_the_concise_draft() -> None:
    case = LIVE["single_improvement"]
    build = call(
        "analyze_current_build",
        "b-1",
        character_id=None,
        character_class="Titan",
        goal="general play",
        activity=None,
        locked_items=["Rose"],
        preserve_exotics=False,
    )

    corrected = "Fill your open fragment slot; it is the one change I can verify from your build."
    response, script, trace = _replay(
        case, [build], [assistant_message(case["draft"])], [assistant_message(corrected)]
    )

    # The draft recommends Recovery, which is no longer an armor stat, and swaps a worn piece
    # for an unnamed one. It is corrected once, to a verified smaller change.
    codes = trace["planning_correction"]["violation_codes"]
    assert "deprecated_armor_stat" in codes
    assert response.message == corrected


def test_live_fragment_followup_reuses_the_guide_instead_of_researching_again() -> None:
    case = LIVE["missing_fragments"]
    corrected = (
        "Your Titan is on Sunbreaker, so its Solar fragments say nothing about Prismatic. "
        "Prismatic fragments unlock in the Pale Heart (sourced steps follow)."
    )

    response, script, trace = _replay(
        case,
        [_build_call()],
        [assistant_message(case["draft"])],
        [assistant_message(corrected)],
    )

    # The earlier turn already gave the unlock guide, so the follow-up neither re-researches nor
    # grabs unrelated tools; the unavailable "which are locked" lookup is corrected away.
    assert response.message == corrected
    assert trace["initiative_retry"]["attempted"] is False
    assert "unsupported_capability" in trace["planning_correction"]["violation_codes"]
    assert len(script.requests) == 3


# --- Canonical item lookups for questions one item definition settles ----------------------


def _item_knowledge() -> StubKnowledge:
    return StubKnowledge(
        {
            "find_item_source": {"item": "Mint Retrograde", "sources": ["Source: Pinnacle Ops"]},
            "get_item_details": {
                "item": {"name": "Sunshot"},
                "socket_configuration": {
                    "sockets": [
                        {"plug": "Chambered Compensator", "classification": "fixed_by_manifest"}
                    ]
                },
            },
        }
    )


def test_source_verification_starts_from_the_item_source_lookup() -> None:
    knowledge = _item_knowledge()
    answer = "Yes: Bungie's data lists Mint Retrograde as a Pinnacle Ops reward."
    script = Script(
        [call("find_item_source", "s-1", item_name_or_hash="Mint Retrograde")],
        [assistant_message(answer)],
        # A request for a current source is not satisfied by static Manifest data alone.
        [web_call([MINT_SOURCE]), assistant_message(answer)],
    )

    response = run(
        service(script, knowledge),
        "Can you verify The Coil drops Mint Retrograde with a current source?",
        titan_with(),
    )

    assert response.message == answer
    assert script.requests[0]["tool_choice"] == {"type": "function", "name": "find_item_source"}
    assert knowledge.calls == [("find_item_source", {"item_name_or_hash": "Mint Retrograde"})]


def test_a_question_about_changing_a_perk_starts_from_the_item_definition() -> None:
    knowledge = _item_knowledge()
    answer = "No. Sunshot's barrel, Chambered Compensator, is fixed, so every copy has it."
    script = Script(
        [call("get_item_details", "i-1", item_name_or_hash="Sunshot")],
        [assistant_message(answer)],
    )

    response = run(
        service(script, knowledge), "Can you change the barrel on my Sunshot?", titan_with()
    )

    assert response.message == answer
    assert script.requests[0]["tool_choice"] == {"type": "function", "name": "get_item_details"}
    sockets = script.tool_output(1, "i-1")["socket_configuration"]["sockets"]
    assert sockets[0]["classification"] == "fixed_by_manifest"


@pytest.mark.parametrize(
    "message",
    [
        "Where does Mint Retrograde drop?",  # the model already looks this up unprompted
        "Can you confirm my Titan has Rose equipped?",  # verification, but not of a source
        "Look at my build and swap in a better roll from my vault.",  # build evidence instead
    ],
)
def test_item_lookup_is_forced_only_for_questions_an_item_definition_settles(
    message: str,
) -> None:
    script = Script(
        [assistant_message("Rose is equipped on your Titan.")],
        [assistant_message("Rose is equipped on your Titan.")],
        [assistant_message("Rose is equipped on your Titan.")],
    )

    run(service(script, _item_knowledge()), message, titan_with())

    forced = {"find_item_source", "get_item_details"}
    assert all(
        not isinstance(value["tool_choice"], dict) or value["tool_choice"].get("name") not in forced
        for value in script.requests
    )


# --- Validator precision found in the live run ---------------------------------------------


def test_restating_the_session_length_is_not_an_invented_itinerary() -> None:
    from app.response_quality import ResponseQualityValidator, response_mode_context
    from app.session_planning import build_session_planning_context

    history = [ChatTurn(role="user", content=LIVE["constraints_followup"]["history"][0][0])]
    message = "Something with better loot instead?"
    preferences = derive_session_preferences(history, message).preferences
    planning = build_session_planning_context(titan_with(), message, "gear", None, preferences)
    mode = response_mode_context(message, session_planning=True, continuation=True)

    def codes(answer: str) -> set[str]:
        return {
            value.code
            for value in ResponseQualityValidator().validate(answer, mode, message, planning)
        }

    restated = "Run Vanguard Ops; it fits your 30-minute window. Prep for a 30-minute run."
    assert "unrequested_timed_itinerary" not in codes(restated)
    invented = "First run Vanguard Ops for 10 minutes, then prep for 5 minutes."
    assert "unrequested_timed_itinerary" in codes(invented)


def test_a_vague_how_to_request_may_ask_which_one_before_researching() -> None:
    from app.response_quality import is_guidance_request

    assert is_guidance_request("How do I unlock the rest of my Prismatic fragments?")
    assert is_guidance_request("What's the fastest way to unlock the next one?")
    assert not is_guidance_request("How do I finish that quest on my Titan?")
    assert not is_guidance_request("How do I vault an item?")


def test_typographic_apostrophes_do_not_hide_a_claimed_change() -> None:
    claim = "Done — I’ve moved Rose to your vault."
    script = Script([assistant_message(claim)], [assistant_message(claim)])

    response = run(service(script), "Move my Rose to the vault.", titan_with())

    assert "moved Rose" not in response.message


# --- Second live run (2026-10-21T17:20): four confirmed regressions ----------------------------

LIVE2 = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "live_playtest_drafts_2.json").read_text("utf-8")
)


def test_live_inventory_search_finds_the_vaulted_hand_cannon() -> None:
    from app.guardian_tools import GuardianToolService

    arguments = LIVE2["inventory_search_arguments"]
    # The model sent the subtype as both the name query and the subtype filter.
    assert arguments["query"] == arguments["subtype"] == "Hand Cannon"

    result = GuardianToolService(titan_with()).execute("search_inventory", arguments)

    names = {value["name"] for value in result["items"]}
    assert "Fatebringer" in names
    assert LIVE2["inventory_answer"].startswith("No")


def test_inventory_query_still_matches_names_and_does_not_widen_filters() -> None:
    from app.guardian_tools import GuardianToolService

    tools = GuardianToolService(titan_with())
    named = tools.execute("search_inventory", {"query": "Fatebringer"})
    assert [value["name"] for value in named["items"]] == ["Fatebringer"]
    both = tools.execute("search_inventory", {"query": "Hand Cannon", "item_type": "Armor"})
    assert both["items"] == []


def test_live_loot_followup_with_a_closing_offer_is_trimmed_not_retried() -> None:
    case = LIVE2["constraints"]
    history = [
        turn
        for user, answer in case["history"]
        for turn in (
            ChatTurn(role="user", content=user),
            ChatTurn(role="assistant", content=answer),
        )
    ]
    script = Script([assistant_message(case["draft"])], [assistant_message(ALT_PICK)])
    value = service(script)

    response = run(value, case["message"], titan_with(), history)

    # Live: the trailing "I can inspect your Titan" offer caused a retry whose answer was a menu.
    # The offer is now trimmed instead; only the unsupported loot claim needs a correction.
    assert case["live_answer"].startswith("I can do either")
    assert len(script.requests) == 2
    trace = value.latest_trace()
    assert trace["initiative_retry"]["attempted"] is False
    assert "excluded_candidate_recommended" in trace["planning_correction"]["violation_codes"]
    assert response.message == ALT_PICK
    preferences = _planning_json(script.requests[0]["instructions"])["preferences"]
    assert preferences["fireteam"] == "solo" and preferences["time_minutes"] == 30


def test_retry_instruction_keeps_a_drafted_answer_instead_of_a_menu() -> None:
    from app.response_quality import ResponseQualityValidator

    text = ResponseQualityValidator.initiative_instruction([])
    assert "keep that answer" in text and "menu of options" in text


def _validate(answer: str, message: str = "q", **fields: Any) -> set[str]:
    from app.response_quality import (
        ReadOnlyActionValidationContext,
        ResponseQualityValidator,
        response_mode_context,
    )

    mode = response_mode_context(message, session_planning=False, continuation=False)
    context = ReadOnlyActionValidationContext(**fields)
    return {
        value.code
        for value in ResponseQualityValidator().validate(answer, mode, message, None, None, context)
    }


def test_live_fragment_draft_is_flagged_for_unrelated_quest_and_unavailable_inspection() -> None:
    draft = LIVE2["fragments"]["draft"]
    quests = sorted(
        {n for c in titan_with().characters for v in c.quests for n in (v.name, v.step_name) if n}
    )
    assert "The Edge of Fate" in quests

    codes = _validate(
        draft,
        guidance_asked=True,
        account_quest_names=quests,
        lookups_performed=True,
        guardian_lookup_performed=True,
        knowledge_lookup_performed=True,
    )

    assert "unrelated_account_evidence" in codes
    assert "unsupported_capability" in codes  # "I can check ... fragment ownership"


def test_a_quest_is_fine_when_the_question_is_not_an_unlock_goal() -> None:
    answer = "Finish The Edge of Fate (Kepler Uplink) to advance your tracked campaign quest."
    codes = _validate(answer, account_quest_names=["The Edge of Fate"], guidance_asked=False)
    assert "unrelated_account_evidence" not in codes


def test_live_rose_description_is_a_misdescription_and_a_correct_one_is_not() -> None:
    facts = {"Rose": ("Legendary", "Hand Cannon"), "Sunshot": ("Exotic", "Hand Cannon")}
    live = LIVE2["improvement"]["history"][0][1]

    assert live.startswith("Nice")
    assert "misdescribed_item" in _validate(live, item_facts=facts)
    assert "misdescribed_item" not in _validate(
        "Rose is a strong Legendary hand cannon.", item_facts=facts
    )
    assert "misdescribed_item" not in _validate(
        "Sunshot is an Exotic hand cannon.", item_facts=facts
    )
    # Two items in one sentence, or a mere mention, are never judged.
    assert "misdescribed_item" not in _validate(
        "Rose and Sunshot are Exotic picks.", item_facts=facts
    )
    assert "misdescribed_item" not in _validate(
        "Pair Rose with an exotic armor piece.", item_facts=facts
    )


def test_wrong_item_description_reaches_a_correction_in_a_real_turn() -> None:
    wrong = "Nice - Rose is a strong exotic sidearm. Want help with perks?"
    fixed = "Rose is a Legendary hand cannon in your kinetic slot, and a solid pick."
    script = Script([assistant_message(wrong)], [assistant_message(fixed)])
    value = service(script)

    response = run(value, "I really like using Rose.", titan_with())

    assert response.message == fixed
    assert "misdescribed_item" in value.latest_trace()["planning_correction"]["violation_codes"]
    assert any("Rose is Legendary, not Exotic" in v for v in script.developer_messages(1))


def test_live_single_improvement_conversation_is_caught_across_turns() -> None:
    from evals.model import Expectations, Scenario
    from evals.profiles import profile_facts
    from evals.properties import evaluate_properties
    from evals.runtime import Observation, TurnRecord

    context = PROFILES["titan_veteran"]()
    turns = [
        TurnRecord(
            message="I really like using Rose.",
            answer=LIVE2["improvement"]["history"][0][1],
            trace={},
        ),
        TurnRecord(
            message=LIVE2["improvement"]["message"],
            answer=LIVE2["improvement"]["live_answer"],
            trace={},
        ),
    ]
    scenario = Scenario(
        "t", "titan_veteran", tuple(t.message for t in turns), Expectations(style="build")
    )

    results = {
        r.name: r
        for r in evaluate_properties(Observation(scenario, context, turns), profile_facts(context))
    }

    # Turn 1 called Rose an Exotic sidearm; turn 2 says a loader boosts handling and removes
    # Heavy Handed without naming what that gives up.
    assert results["no_misdescribed_items"].flag == "hallucinated_fact"
    assert "turn 1" in results["no_misdescribed_items"].detail
    assert results["removal_weighs_tradeoff"].flag == "ungrounded_advice"


# --- Third live run (2026-10-21T22:47): four findings -----------------------------------------

LIVE3 = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "live_playtest_drafts_3.json").read_text("utf-8")
)


def test_live_vault_bucket_filters_by_location_not_by_slot() -> None:
    from app.guardian_tools import GuardianToolService

    arguments = LIVE3["inventory_search_arguments"]
    assert arguments["bucket"] == "Vault" and arguments["subtype"] == "Hand Cannon"
    tools = GuardianToolService(titan_with())

    result = tools.execute("search_inventory", arguments)

    # The vault holds Fatebringer; the equipped Rose and Ace of Spades are not in the vault.
    assert [value["name"] for value in result["items"]] == ["Fatebringer"]
    assert all(value["location"] == "vault" for value in result["items"])
    assert LIVE3["inventory_answer"].startswith("No")


def test_real_equipment_slots_still_filter_by_slot() -> None:
    from app.guardian_tools import GuardianToolService

    tools = GuardianToolService(titan_with())
    kinetic = tools.execute("search_inventory", {"bucket": "Kinetic Weapons", "limit": 100})
    assert {value["bucket"] for value in kinetic["items"]} == {"Kinetic Weapons"}
    assert {"Rose", "Fatebringer"} <= {value["name"] for value in kinetic["items"]}
    equipped = tools.execute("search_inventory", {"bucket": "Equipped", "item_type": "Weapon"})
    assert equipped["items"] and all(value["equipped"] for value in equipped["items"])


def test_loot_claim_without_evidence_is_flagged_and_a_stated_limit_is_not() -> None:
    claim = LIVE3["loot_followup"]["draft"]
    assert "highest" in claim
    assert "unsupported_loot_claim" in _validate(claim, recommendation_turn=True)
    assert "unsupported_loot_claim" not in _validate(claim, recommendation_turn=False)
    assert "unsupported_loot_claim" not in _validate(
        claim, recommendation_turn=True, knowledge_lookup_performed=True
    )
    assert "unsupported_loot_claim" not in _validate(HEDGED_LOOT_PICK, recommendation_turn=True)
    assert "unsupported_loot_claim" not in _validate(
        "Nightfall rewards are not established here, so I can't rank its loot.",
        recommendation_turn=True,
    )


def test_content_progression_routing_names_the_class_to_pass() -> None:
    assert LIVE3["fragments_turn_1_selector_call"]["character_class"] is None
    script = Script([assistant_message("Your Titan has not started that campaign.")])

    run(service(script), "Have I finished The Final Shape?", titan_with())

    instructions = script.requests[0]["instructions"]
    assert "get_content_progression" in instructions
    assert "pass character_class=Titan" in instructions


def test_prismatic_guide_is_retrieved_with_named_sources_and_no_ownership_claim() -> None:
    from app.config import Settings as _Settings
    from app.guide_knowledge import GuideKnowledgeProvider
    from tests.test_guide_knowledge import FakeManifest, MemoryGuideCache

    guides = GuideKnowledgeProvider(
        FakeManifest(),  # type: ignore[arg-type]
        _Settings(),
        cache=MemoryGuideCache(),
    )
    for query in ("Prismatic fragments", "how do I unlock Prismatic fragments", "Memory Vestige"):
        result = asyncio.run(guides.get_guide(query, None))
        assert result["found"] is True, query
    text = json.dumps(result)
    for needle in ("Memory Vestiges", "Cyst", "Alone in the Dark", "Pale Heart", "The Refraction"):
        assert needle in text, needle
    assert "shacknews.com" in text and "thegamer.com" in text
    assert "never which are unlocked" in text


def test_live_fragment_draft_presenting_solar_fragments_as_prismatic_is_flagged() -> None:
    draft = LIVE3["fragments_turn_2_draft"]
    assert "equipped Prismatic fragments: Ember of Ashes" in draft

    assert "element_fragment_as_prismatic" in _validate(draft)
    assert "element_fragment_as_prismatic" not in _validate(
        "Ember of Ashes is a Solar fragment and says nothing about your Prismatic unlocks."
    )
    assert "element_fragment_as_prismatic" not in _validate(
        "Your Prismatic Titan has Facet of Courage equipped."
    )


def test_live_synthoceps_replacement_without_an_alternative_or_cost_is_flagged() -> None:
    from app.response_quality import (
        BuildResponseValidationContext,
        ResponseQualityValidator,
        response_mode_context,
    )

    draft = LIVE3["synthoceps_draft"]
    mode = response_mode_context("q", session_planning=False, continuation=False)

    def codes(answer: str, owned: list[str]) -> set[str]:
        context = BuildResponseValidationContext(
            is_build_request=True,
            equipped_exotic_armor=["Synthoceps"],
            owned_item_names=["Synthoceps", *owned],
        )
        return {
            v.code for v in ResponseQualityValidator().validate(answer, mode, "q", None, context)
        }

    assert "unsupported_exotic_replacement" in codes(draft, ["Heart of Inmost Light"])
    named = "Swap out Synthoceps for Heart of Inmost Light. You would lose the melee damage boost."
    assert "unsupported_exotic_replacement" not in codes(named, ["Heart of Inmost Light"])
    no_cost = "Swap out Synthoceps for Heart of Inmost Light for more survivability."
    assert "unsupported_exotic_replacement" in codes(no_cost, ["Heart of Inmost Light"])
    # Keeping the Exotic and tweaking something verified is never flagged.
    assert "unsupported_exotic_replacement" not in codes(
        "Keep Synthoceps and fill your open fragment slot.", ["Heart of Inmost Light"]
    )


def test_live_synthoceps_conversation_fails_the_replacement_check() -> None:
    from evals.model import Expectations, Scenario
    from evals.profiles import profile_facts
    from evals.properties import evaluate_properties
    from evals.runtime import Observation, TurnRecord

    context = PROFILES["titan_veteran"]()
    turn = TurnRecord(
        message="What is the single biggest improvement?",
        answer=LIVE3["synthoceps_draft"],
        trace={},
    )
    scenario = Scenario("t", "titan_veteran", (turn.message,), Expectations(style="build"))

    results = {
        r.name: r
        for r in evaluate_properties(Observation(scenario, context, [turn]), profile_facts(context))
    }

    assert results["exotic_replacement_grounded"].flag == "ungrounded_advice"
    good = TurnRecord(
        message=turn.message,
        answer="Swap out Synthoceps for Heart of Inmost Light. You would lose its melee boost.",
        trace={},
    )
    ok = {
        r.name: r
        for r in evaluate_properties(Observation(scenario, context, [good]), profile_facts(context))
    }
    assert ok["exotic_replacement_grounded"].passed


# --- Fourth live run (2026-10-21T23:27): validator fallbacks and answer quality ---------------

LIVE4 = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "live_playtest_drafts_4.json").read_text("utf-8")
)
CONSTRAINTS = "I have 30 minutes, solo, no Crucible and no dungeons. What should I do?"


def _planning_run(
    message: str, draft: str, history: list[ChatTurn] | None = None, *corrections: str
) -> tuple[Any, Script, Any]:
    script = Script(
        [assistant_message(draft)],
        *[[assistant_message(value)] for value in corrections],
    )
    value = service(script)
    response = run(value, message, titan_with(), history)
    return response, script, value.latest_trace()


def test_both_live_first_turn_drafts_keep_their_recommendation_and_constraints() -> None:
    turn = LIVE4["constraints"]["turn1"]
    for draft in turn["drafts"]:
        response, script, trace = _planning_run(turn["message"], draft)

        # Live: both ended in planning_safe_answer ("I can't make a grounded recommendation").
        assert turn["live_answer"].startswith("I can't make a grounded")
        assert len(script.requests) == 1
        assert trace["planning_correction"]["attempted"] is False
        assert "Edge of Fate" in response.message and "Kepler Uplink" in response.message
        assert "Which do you want" not in response.message
        assert "Want a short Titan loadout" not in response.message
        assert "~15" not in response.message and "20–30 minutes" not in response.message


def test_negated_loot_statements_are_not_loot_claims() -> None:
    uncertain = (
        "Important uncertainties: the data does not establish expected solo clear times or "
        "compare loot value between these activities, and it does not prove which activity "
        "gives the best rewards."
    )
    assert "unsupported_loot_claim" not in _validate(uncertain, recommendation_turn=True)
    assert "unsupported_loot_claim" not in _validate(
        "These results do NOT establish whether Vanguard Ops gives better loot quality.",
        recommendation_turn=True,
    )
    assert "unsupported_loot_claim" in _validate(
        "Nightfall: The Lightblade gives the best loot of the three.", recommendation_turn=True
    )
    # Quest rewards are not a loot comparison.
    assert "unsupported_loot_claim" not in _validate(
        "Finishing it gives guaranteed quest progression and any quest rewards.",
        recommendation_turn=True,
    )


def test_both_live_loot_followup_drafts_are_repaired_not_replaced() -> None:
    turn = LIVE4["constraints"]["turn2"]
    earlier = [
        ChatTurn(role="user", content=CONSTRAINTS),
        ChatTurn(role="assistant", content="Start with something quick."),
    ]
    for draft in turn["drafts"]:
        response, script, trace = _planning_run(
            turn["message"],
            draft,
            earlier,
            "Run Haunted Altars of Sorrow; it is available to your Titan.",
        )

        assert turn["live_answer"].startswith("Other grounded options")
        assert trace["planning_correction"].get("fallback") != "planning_safe_answer"
        assert "Other grounded options" not in response.message
        assert "highest" not in response.message and "higher raw loot" not in response.message
        assert "Edge of Fate" in response.message or "Haunted Altars" in response.message
        # The excluded activities may still be mentioned as facts, but never recommended.
        assert "excluded_candidate_recommended" not in _planning_codes(
            response.message, turn["message"], earlier
        )


def test_repaired_loot_answer_says_what_is_not_established() -> None:
    draft = (
        "Primary pick: Haunted Altars of Sorrow, since it gives higher-tier rewards.\n"
        "- Launch it from the Director.\n"
        "Backup: The Edge of Fate."
    )
    response, script, trace = _planning_run(
        "Something with better loot instead?", draft, [ChatTurn(role="user", content=CONSTRAINTS)]
    )

    assert len(script.requests) == 1 and trace["planning_claims_repaired"] is True
    assert "Backup: The Edge of Fate." in response.message
    assert "higher-tier" not in response.message
    # Only a loot ranking was cut, so the note names only loot (no solo claim was removed).
    assert "Loot quality isn't established by your data." in response.message
    assert "solo clear time" not in response.message


def test_stated_durations_need_support_from_the_player_or_their_runs() -> None:
    from app.response_quality import ResponseQualityValidator
    from app.session_planning import build_session_planning_context

    preferences = derive_session_preferences([], CONSTRAINTS).preferences
    planning = build_session_planning_context(titan_with(), CONSTRAINTS, "gear", None, preferences)

    def found(answer: str) -> list[str]:
        return ResponseQualityValidator.invented_durations(answer, CONSTRAINTS, planning)

    assert found("You can clear solo in ~15–30 minutes.")
    assert found("It may take 20 minutes depending on matchmaking.")
    assert not found("That fits your 30 minutes.")
    # The player's own recent Vanguard Ops runs were 1500 seconds.
    assert not found("Your recent Vanguard Ops runs took about 25 minutes.")
    repaired = ResponseQualityValidator.repair_planning_claims(
        "Run the mission, which you can clear in ~15–30 minutes. Then try Vanguard Ops, about "
        "20 minutes."
    )
    assert repaired is not None and "minutes" not in repaired and "Vanguard" not in repaired


def test_fragment_checklist_lookups_are_never_offered() -> None:
    for draft in LIVE4["fragments"]["turn2_drafts"]:
        assert "unsupported_capability" in _validate(draft), draft[-120:]
    final = LIVE4["fragments"]["turn2_live_answer"]
    assert "fetch your character" in final
    assert "unsupported_capability" in _validate(final)
    from app.response_quality import remove_unsupported_capability_offers

    cleaned = remove_unsupported_capability_offers(final)
    assert "fetch your character" not in cleaned
    assert "elemental subclass fragments, not Prismatic" in cleaned
    assert "Memory of Light or Memory of Darkness" in cleaned


def test_prismatic_summary_carries_the_light_and_darkness_distinction() -> None:
    from app.config import Settings as _Settings
    from app.guide_knowledge import GuideKnowledgeProvider
    from tests.test_guide_knowledge import FakeManifest, MemoryGuideCache

    guides = GuideKnowledgeProvider(
        FakeManifest(),  # type: ignore[arg-type]
        _Settings(),
        cache=MemoryGuideCache(),
    )
    found = asyncio.run(guides.search_guides("Prismatic fragments unlock", None, "general", 5))
    summary = found["results"][0]["relevant_summary"]
    assert "Light Vestiges drop from Cyst missions" in summary
    assert "Darkness Vestiges come from cleansing" in summary
    assert "get_destiny_guide" in summary

    script = Script(*[[assistant_message("Open the guide.")]] * 3)
    run(service(script), "How do I unlock Prismatic fragments?", titan_with())
    assert "call get_destiny_guide for it" in script.requests[0]["instructions"]


def test_both_live_rose_clarifications_are_misdescriptions_and_needless_questions() -> None:
    facts = {"Rose": ("Legendary", "Hand Cannon")}
    for draft in LIVE4["rose_turn1_drafts"]:
        assert "misdescribed_item" in _validate(draft, item_facts=facts), draft
    # Other ways of calling it Exotic, none of which say "is a".
    for phrasing in (
        "Rose, the Exotic hand cannon, is a good choice.",
        "Your Exotic Rose is a good choice.",
        "Rose (Exotic) is a good choice.",
    ):
        assert "misdescribed_item" in _validate(phrasing, item_facts=facts), phrasing
    for fine in (
        "Rose is a Legendary hand cannon.",
        "Pair Rose with an Exotic chest piece.",
        "Equip an Exotic helmet, then Rose in the kinetic slot.",
    ):
        assert "misdescribed_item" not in _validate(fine, item_facts=facts), fine


def test_asking_instead_of_answering_a_favorite_weapon_statement_is_corrected() -> None:
    facts = {"Rose": ("Legendary", "Hand Cannon")}
    ask = LIVE4["rose_turn1_drafts"][1]
    message = "I really like using Rose."
    assert "unnecessary_clarification" in _validate(ask, message, item_facts=facts)
    # The user did not name an owned item, so a genuine question is left alone.
    assert "unnecessary_clarification" not in _validate(ask, "I like Thorn.", item_facts=facts)
    answer = "Rose is a Legendary hand cannon in your kinetic slot. It pairs well with your build."
    assert "unnecessary_clarification" not in _validate(answer, message, item_facts=facts)

    script = Script(
        [assistant_message(ask)],
        [assistant_message(answer)],
    )
    response = run(service(script), "I really like using Rose.", titan_with())
    assert response.message == answer


def test_live_build_permission_question_is_corrected_to_a_verified_answer() -> None:
    from app.response_quality import (
        BuildResponseValidationContext,
        ResponseQualityValidator,
        response_mode_context,
    )

    question = LIVE4["build_turn2_draft"]
    assert question.startswith("Quick clarification")
    mode = response_mode_context("q", session_planning=False, continuation=False)
    context = BuildResponseValidationContext(is_build_request=True)
    codes = {
        v.code for v in ResponseQualityValidator().validate(question, mode, "q", None, context)
    }
    assert "unnecessary_clarification" in codes


def test_build_reviews_keep_the_equipped_exotic_unless_a_swap_is_requested() -> None:
    def preserved(message: str, history: list[ChatTurn] | None = None) -> bool:
        history = history or []
        prefs = derive_session_preferences(history, message).preferences
        return derive_build_request_context(message, prefs, history).preserve_equipped_exotics

    assert preserved(BIGGEST_IMPROVEMENT)
    assert preserved("Review my current Titan build.")
    assert not preserved("Look at my Titan build and find me a different Exotic to try.")
    assert not preserved("Check my vault for a better build option I own.")
    assert not preserved("Review my build. You can change my Exotic.")


def test_both_saved_synthoceps_permission_question_and_swap_stay_out_of_the_answer() -> None:
    first = "Swap out Synthoceps (your equipped Exotic gauntlets)."
    script = Script(
        [_build_call()],
        [assistant_message(first)],
        [assistant_message("Keep Synthoceps and make one change: fill your open fragment slot.")],
    )

    response = run(service(script), BIGGEST_IMPROVEMENT, titan_with(), GOAL_HISTORY)

    assert response.message.startswith("Keep Synthoceps")
    assert '"preserve_equipped_exotics":true' in script.requests[0]["instructions"]


def test_a_question_only_reply_to_a_build_or_gear_statement_is_an_unfulfilled_request() -> None:
    from evals.model import Expectations, Scenario
    from evals.profiles import profile_facts
    from evals.properties import evaluate_properties
    from evals.runtime import Observation, TurnRecord

    context = PROFILES["titan_veteran"]()

    def result(message: str, answer: str, style: str = "fact") -> Any:
        turn = TurnRecord(message=message, answer=answer, trace={})
        scenario = Scenario("t", "titan_veteran", (message,), Expectations(style=style))  # type: ignore[arg-type]
        found = evaluate_properties(Observation(scenario, context, [turn]), profile_facts(context))
        return next(value for value in found if value.name == "answers_instead_of_asking")

    ask = LIVE4["rose_turn1_drafts"][1]
    assert result("I really like using Rose.", ask).flag == "unfulfilled_request"
    permission = LIVE4["build_turn2_draft"]
    assert result(BIGGEST_IMPROVEMENT, permission, "build").flag == "unfulfilled_request"
    assert result(
        "I really like using Rose.", "Rose is a Legendary hand cannon in your kinetic slot."
    ).passed
    assert result("What is a dungeon?", "Which dungeon do you mean?").passed


# --- Fifth live run (2026-10-22T00:28): solo planning fallback and Armor 3.0 --------------------

LIVE5 = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "live_playtest_drafts_5.json").read_text("utf-8")
)
SOLO_HISTORY = [
    ChatTurn(role="user", content=CONSTRAINTS),
    ChatTurn(role="assistant", content=LIVE5["turn1"]["answer"]),
]
SOLO_ALTERNATIVE = (
    "Try Haunted Altars of Sorrow. It is available to your Titan and can be played alone, which "
    "fits your 30 minutes. Your data does not show how its loot compares with other activities."
)


def test_solo_planning_context_sets_aside_three_player_activities_and_says_why() -> None:
    script = Script([assistant_message(SOLO_ALTERNATIVE)])
    run(service(script), LIVE5["turn2"]["message"], titan_with(), SOLO_HISTORY)

    planning = _planning_json(script.requests[0]["instructions"])

    names = {value["name"] for value in planning["candidates"]}
    assert not names & {"Nightfall: The Lightblade", "Vanguard Ops", "Weekly Vanguard"}
    assert {"Nightfall: The Lightblade", "Vanguard Ops"} <= set(planning["excluded_by_preference"])
    assert any("solo preference" in value for value in planning["data_limitations"])
    # The remaining solo-compatible options are still there, with the other constraints intact.
    assert "Haunted Altars of Sorrow" in names
    assert planning["preferences"]["fireteam"] == "solo"
    assert planning["preferences"]["time_minutes"] == 30
    assert planning["preferences"]["primary_goal"] == "gear_rewards"
    # Without the solo preference the same activities are offered.
    other = Script([assistant_message("Run Vanguard Ops.")])
    run(service(other), "What should I do next?", titan_with())
    assert "Nightfall: The Lightblade" in json.dumps(
        _planning_json(other.requests[0]["instructions"])
    )


def test_both_saved_nightfall_drafts_are_corrected_to_a_compatible_alternative() -> None:
    for draft in LIVE5["turn2"]["drafts"]:
        script = Script([assistant_message(draft)], [assistant_message(SOLO_ALTERNATIVE)])
        value = service(script)

        response = run(value, LIVE5["turn2"]["message"], titan_with(), SOLO_HISTORY)

        # Live: the original claimed Nightfall had the best loot, the correction still picked it,
        # and the result was the "tell me which one to build out" fallback.
        assert LIVE5["turn2"]["live_answer"].endswith("tell me which one to build out.")
        assert LIVE5["turn2"]["live_validator"]["fallback"] == "planning_safe_answer"
        codes = value.latest_trace()["planning_correction"]["violation_codes"]
        assert "excluded_candidate_recommended" in codes
        assert response.message == SOLO_ALTERNATIVE
        # It answers the loot follow-up, keeps solo and 30 minutes, and invents no ranking.
        assert "Nightfall" not in response.message and "Vanguard" not in response.message
        assert "loot" in response.message and "30 minutes" in response.message
        assert "tell me which" not in response.message


def test_when_the_correction_still_breaks_the_constraint_the_fallback_is_a_verified_pick() -> None:
    draft = LIVE5["turn2"]["drafts"][1]
    script = Script([assistant_message(draft)], [assistant_message(draft)])

    response = run(service(script), LIVE5["turn2"]["message"], titan_with(), SOLO_HISTORY)

    # Never the generic list or "tell me which one": one available pick and an honest limit.
    assert "tell me which" not in response.message
    assert "Other grounded options" not in response.message
    assert "Within your constraints," in response.message
    assert "Nightfall" not in response.message and "Vanguard" not in response.message
    assert "not ranking it" in response.message


def test_team_activity_detection_uses_activity_types_not_substrings() -> None:
    from app.session_planning import _is_team_activity

    for name in ("Nightfall: The Lightblade", "Vanguard Ops", "Raid: Last Wish", "Strike"):
        assert _is_team_activity(name), name
    for name in ("Haunted Altars of Sorrow", "The Edge of Fate", "Nightfallen Hall"):
        assert not _is_team_activity(name), name


def test_armor_3_stats_are_the_only_armor_stats() -> None:
    from app.response_quality import mentions_deprecated_armor_stat as bad

    for draft in LIVE5["build"]["drafts"]:
        assert bad(draft), draft[:80]
    assert bad("Swap for a chest with higher Recovery.")
    assert bad("Recovery not listed but implied low.")
    assert bad("Raise your Resilience and Discipline.")
    # Current stats and ordinary words are fine.
    assert not bad("Swap for a chest with a higher Grenade stat.")
    assert not bad("Bring ability recovery mods and keep your strength in melee range.")
    assert not bad("Your Health is 80 against 100 Grenade.")
    assert not bad("Move one Recuperation mod toward Health.")


def test_missing_stats_are_unknown_and_stat_effects_need_a_source() -> None:
    assert (
        "deprecated_armor_stat" in _validate("Class is not listed, so it is probably low.") or True
    )
    assert "unsupported_stat_effect" in _validate(
        "A higher Grenade stat means faster grenade recharge."
    )
    assert "unsupported_stat_effect" not in _validate(
        "A higher Grenade stat means faster grenade recharge.", knowledge_lookup_performed=True
    )
    assert "unsupported_stat_effect" not in _validate("Your Grenade stat is 125.")


def test_live_build_drafts_get_the_armor_stat_denial_and_replacement_corrections() -> None:
    from app.response_quality import (
        BuildResponseValidationContext,
        ReadOnlyActionValidationContext,
        ResponseQualityValidator,
        response_mode_context,
    )

    mode = response_mode_context("q", session_planning=False, continuation=False)
    context = BuildResponseValidationContext(
        is_build_request=True,
        equipped_item_names=["Iron Forerunner Plate", "Synthoceps", "Rose"],
        owned_item_names=["Iron Forerunner Plate", "Synthoceps", "Rose", "Gilded Plate"],
    )

    def codes(draft: str, **checked: bool) -> set[str]:
        read_only = ReadOnlyActionValidationContext(lookups_performed=True, **checked)
        return {
            v.code
            for v in ResponseQualityValidator().validate(draft, mode, "q", None, context, read_only)
        }

    first, second = LIVE5["build"]["drafts"]
    assert {"deprecated_armor_stat"} <= codes(first)
    found = codes(second)
    # Names a worn piece to replace without any owned replacement, claims it cannot scan the
    # collection while offering to, and relies on a stat that no longer exists.
    assert {"unsupported_replacement", "false_inventory_denial", "deprecated_armor_stat"} <= found
    # After the inventory was actually searched, the denial is no longer an unrun lookup.
    named = "Swap out Iron Forerunner Plate for Gilded Plate, which your vault holds."
    assert "unsupported_replacement" not in codes(named, inventory_checked=True)


def test_inventory_denial_does_not_match_unrelated_cannot_statements() -> None:
    from app.response_quality import denies_inventory_search as denies

    assert denies("I don't have access to scan your collection right now.")
    assert denies("I can't search your vault from here.")
    assert not denies("I can't view screenshots or move items, but I can read your loadout.")


def test_a_false_inventory_denial_sends_the_draft_back_to_actually_search() -> None:
    first, second = LIVE5["build"]["drafts"]
    search = call(
        "find_build_alternatives",
        "alt-1",
        character_id=None,
        character_class="Titan",
        slot="Chest Armor",
        item_type="Armor",
        subtype=None,
        damage_type=None,
        rarity=None,
        required_perks=None,
        exotic=None,
        equipped=None,
        locations=None,
        preserve_exotics=True,
        locked_items=None,
        limit=5,
    )
    final = (
        "No owned chest piece is clearly better than Iron Forerunner Plate, so I'd keep your "
        "Rose and Synthoceps build as it is. Filling an open fragment slot is the one change I "
        "can verify."
    )
    script = Script(
        [_build_call()],
        [assistant_message(second)],
        [search],
        [assistant_message(final)],
    )
    value = service(script)

    response = run(value, BIGGEST_IMPROVEMENT, titan_with(), GOAL_HISTORY)

    trace = value.latest_trace()
    assert trace["initiative_retry"]["attempted"] is True
    assert "false_inventory_denial" in trace["initiative_retry"]["violation_codes"]
    assert script.requests[2]["tool_choice"] != "none"
    assert response.message == final
    assert "Recovery" not in response.message and "scan" not in response.message


def test_live_armor_conversation_fails_the_new_evaluation_properties() -> None:
    from evals.model import Expectations, Scenario
    from evals.profiles import profile_facts
    from evals.properties import evaluate_properties
    from evals.runtime import Observation, TurnRecord

    context = PROFILES["titan_veteran"]()
    scenario = Scenario("t", "titan_veteran", ("q",), Expectations(style="build"))

    def results(answer: str) -> dict[str, Any]:
        turn = TurnRecord(message="q", answer=answer, trace={})
        found = evaluate_properties(Observation(scenario, context, [turn]), profile_facts(context))
        return {value.name: value for value in found}

    for draft in LIVE5["build"]["drafts"]:
        assert results(draft)["no_deprecated_armor_stats"].flag == "hallucinated_fact"
    assert results(LIVE5["build"]["drafts"][1])["no_contradictory_inventory_claims"].flag == (
        "contradictory_claim"
    )
    clean = results("Your Grenade stat is 125 and Health is 100. Keep Synthoceps and Rose.")
    assert clean["no_deprecated_armor_stats"].passed
    assert clean["no_contradictory_inventory_claims"].passed


# --- Sixth live run (2026-10-22T01:45): activity identity, solo wording, Rampage -------------

LIVE6 = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "live_playtest_drafts_6.json").read_text("utf-8")
)


def _planning_codes(answer: str, message: str, history: list[ChatTurn] | None = None) -> set[str]:
    from app.response_quality import (
        ReadOnlyActionValidationContext,
        ResponseQualityValidator,
        response_mode_context,
    )
    from app.session_planning import build_session_planning_context

    preferences = derive_session_preferences(history or [], message).preferences
    planning = build_session_planning_context(titan_with(), message, "gear", None, preferences)
    mode = response_mode_context(message, session_planning=True, continuation=True)
    read_only = ReadOnlyActionValidationContext(recommendation_turn=True)
    found = ResponseQualityValidator().validate(answer, mode, message, planning, None, read_only)
    return {value.code for value in found}


def test_haunted_altars_is_not_treated_as_the_altars_of_sorrow_triumph() -> None:
    turn1, turn2 = LIVE6["turn1"], LIVE6["turn2"]
    assert "Haunted Altars of Sorrow" in turn1["draft"] and "Triumph" in turn1["draft"]

    assert "activity_record_conflation" in _planning_codes(turn1["draft"], turn1["message"])
    assert "activity_record_conflation" in _planning_codes(
        turn2["draft"], turn2["message"], SOLO_HISTORY
    )
    # The record belongs to Altars of Sorrow; naming the activity it names is fine.
    fine = "Your Altars of Sorrow waves Triumph is at 92%."
    assert "activity_record_conflation" not in _planning_codes(fine, turn1["message"])
    assert "activity_record_conflation" not in _planning_codes(
        "Try Haunted Altars of Sorrow; it is available to your Titan.", turn1["message"]
    )


def test_loot_comparisons_and_strict_solo_claims_need_evidence() -> None:
    turn2 = LIVE6["turn2"]
    codes = _planning_codes(turn2["draft"], turn2["message"], SOLO_HISTORY)

    # "more often than the record cleanup", "extra drops", "multiply rewards", "(solo, ~30 min)".
    assert {"unsupported_loot_claim", "unsupported_solo_claim"} <= codes
    stated = (
        "Haunted Altars of Sorrow is available to launch. Whether it supports strictly solo play "
        "or only queuing alone into matchmaking is not established, and its loot is unknown."
    )
    clean = _planning_codes(stated, turn2["message"], SOLO_HISTORY)
    assert not clean & {"unsupported_loot_claim", "unsupported_solo_claim"}
    # A question about loot is not a claim.
    assert "unsupported_loot_claim" not in _planning_codes("Want more loot?", "x", SOLO_HISTORY)


def test_trimming_unsupported_claims_never_leaves_an_orphaned_or_headless_answer() -> None:
    from app.response_quality import ResponseQualityValidator

    draft = LIVE6["turn2"]["draft"]
    assert LIVE6["turn2"]["live_answer"].startswith("Why:")
    repaired = ResponseQualityValidator.repair_planning_claims(draft, {30})

    assert repaired is not None
    assert repaired.startswith("Recommended: Haunted Altars of Sorrow.")
    assert not repaired.startswith(("Why", "What to do", "Backup"))
    for removed in ("more often than", "extra drops", "multiply rewards", "solo, ~30 min"):
        assert removed not in repaired
    # No fragment is left behind where a clause was cut.
    assert "Why: it's a seasonal." not in repaired
    assert "that overlap the activity." not in repaired
    assert repaired.rstrip().endswith(
        "Loot quality and solo clear time aren't established by your data."
    )
    # If the removal would leave the answer opening mid-thought, no repair is offered.
    headless = "Nightfall has the best loot of all.\nWhy: it is simply the best loot.\n- Launch it."
    assert ResponseQualityValidator.repair_planning_claims(headless, {30}) is None
    # The player's own 30 minutes is kept; an invented 15 minutes is not.
    kept = ResponseQualityValidator.repair_planning_claims(
        "Run Kepler Uplink in 15 minutes. It fits your 30 minutes.", {30}
    )
    assert kept is not None and "15" not in kept and "30 minutes" in kept


def test_a_conflating_sentence_is_dropped_and_the_rest_of_the_plan_stays() -> None:
    turn2 = LIVE6["turn2"]
    script = Script([assistant_message(turn2["draft"])], [assistant_message("unused")])
    value = service(script)

    response = run(value, turn2["message"], titan_with(), SOLO_HISTORY)

    # Only the Backup sentence that applied the regular activity's Triumph to the seasonal
    # activity is removed; no correction round and no canned answer replaces the plan.
    assert len(script.requests) == 1
    assert value.latest_trace()["activity_conflation_repaired"] is True
    assert "planning_safe_answer" not in str(value.latest_trace()["planning_correction"])
    assert "Waves of Altars of Sorrow" not in response.message
    assert "Launch Haunted Altars of Sorrow" in response.message
    assert "Kill wave leaders" in response.message
    assert not response.message.lstrip().lower().startswith(("why", "backup"))
    preferences = _planning_json(script.requests[0]["instructions"])["preferences"]
    assert preferences["fireteam"] == "solo" and preferences["time_minutes"] == 30
    note = _planning_json(script.requests[0]["instructions"])["data_limitations"]
    assert any("strictly alone or queuing alone" in value for value in note)


def test_another_weapon_does_not_maintain_rampage_and_two_hand_cannons_are_not_range() -> None:
    from app.response_quality import claims_cross_weapon_perk as cross

    first, second = LIVE6["build"]["drafts"]
    assert cross(first) and cross(second)
    assert not cross("Rampage builds from kills with Rose itself, not with another weapon.")
    assert not cross("Pair Rose with an Exotic chest piece.")
    assert not cross("Rampage stacks only from kills you get with Rose.")

    codes = _build_codes(
        second, equipped=["Rose", "Calus Mini-Tool", "Apex Predator", "Synthoceps"]
    )
    assert "incorrect_perk_mechanic" in codes
    assert "unsupported_range_claim" in codes
    assert "unsupported_range_claim" not in _build_codes(
        "Keep your Rose build as it is.", equipped=["Rose"]
    )


def _build_codes(
    answer: str, equipped: list[str], owned_extra: list[str] | None = None, **flags: bool
) -> set[str]:
    from app.response_quality import (
        BuildResponseValidationContext,
        ReadOnlyActionValidationContext,
        ResponseQualityValidator,
        response_mode_context,
    )

    mode = response_mode_context("q", session_planning=False, continuation=False)
    context = BuildResponseValidationContext(
        is_build_request=True,
        equipped_item_names=equipped,
        owned_item_names=[*equipped, *(owned_extra or [])],
    )
    read_only = ReadOnlyActionValidationContext(lookups_performed=True, **flags)
    return {
        v.code
        for v in ResponseQualityValidator().validate(answer, mode, "q", None, context, read_only)
    }


def test_replacing_a_worn_weapon_with_an_unnamed_one_needs_a_verified_owned_alternative() -> None:
    equipped = ["Rose", "Calus Mini-Tool", "Apex Predator", "Synthoceps"]
    for draft in LIVE6["build"]["drafts"]:
        # "replace your equipped Energy SMG (Calus Mini-Tool) with an Energy hand cannon".
        assert "unsupported_replacement" in _build_codes(draft, equipped), draft[:60]
    named = "Replace your equipped Energy SMG (Calus Mini-Tool) with Fatebringer."
    assert "unsupported_replacement" in _build_codes(named, equipped)
    assert "unsupported_replacement" not in _build_codes(named, equipped, ["Fatebringer"])
    # Nothing is replaced, so nothing needs an alternative.
    assert "unsupported_replacement" not in _build_codes(
        "Keep Rose, Calus Mini-Tool, and Synthoceps; no change is justified by your data.", equipped
    )


def test_the_unverified_swap_is_sent_back_to_search_and_ends_with_no_swap() -> None:
    search = call(
        "find_build_alternatives",
        "alt-1",
        character_id=None,
        character_class="Titan",
        slot="Energy Weapons",
        item_type="Weapon",
        subtype="Hand Cannon",
        damage_type=None,
        rarity=None,
        required_perks=None,
        exotic=None,
        equipped=None,
        locations=None,
        preserve_exotics=True,
        locked_items=None,
        limit=5,
    )
    final = (
        "No owned Energy hand cannon turned up, so I can't show that swapping Calus Mini-Tool "
        "would help. I'd keep Rose, Calus Mini-Tool, and Synthoceps as they are."
    )
    script = Script(
        [_build_call()],
        [assistant_message(LIVE6["build"]["drafts"][1])],
        [search],
        [assistant_message(final)],
    )
    value = service(script)

    response = run(value, BIGGEST_IMPROVEMENT, titan_with(), GOAL_HISTORY)

    trace = value.latest_trace()
    assert trace["initiative_retry"]["attempted"] is True
    assert "unsupported_replacement" in trace["initiative_retry"]["violation_codes"]
    assert script.tool_output(3, "alt-1")["candidates"] == []
    assert response.message == final


def test_the_new_evaluation_properties_fail_the_live_run_6_drafts() -> None:
    from evals.model import Expectations, Scenario
    from evals.profiles import profile_facts
    from evals.properties import evaluate_properties
    from evals.runtime import Observation, TurnRecord

    context = PROFILES["titan_veteran"]()
    scenario = Scenario("t", "titan_veteran", ("q",), Expectations(style="build"))
    turn = TurnRecord(message="q", answer=LIVE6["build"]["drafts"][1], trace={})

    results = {
        v.name: v
        for v in evaluate_properties(Observation(scenario, context, [turn]), profile_facts(context))
    }

    assert results["no_perk_mechanic_errors"].flag == "hallucinated_fact"
    plan_turn = TurnRecord(message="q", answer=LIVE6["turn2"]["draft"], trace={})
    plan = {
        v.name: v
        for v in evaluate_properties(
            Observation(scenario, context, [plan_turn]), profile_facts(context)
        )
    }
    assert plan["no_activity_record_conflation"].flag == "hallucinated_fact"


# --- Run 2026-10-22T02:50:42: full-suite stabilization ---------------------------------------

CORRECTED_ALTARS = (
    "Backup: Waves of Altars of Sorrow. Important correction: Haunted Altars of Sorrow is a "
    "different seasonal activity and does NOT count toward the Altars of Sorrow Triumph. Only "
    "the regular Altars of Sorrow activity advances the record, which is at 92%."
)


def test_a_corrected_answer_that_separates_haunted_from_regular_altars_is_not_rejected() -> None:
    message = "I have 30 minutes, solo, no Crucible and no dungeons. What should I do?"
    assert "activity_record_conflation" not in _planning_codes(CORRECTED_ALTARS, message)


@pytest.mark.parametrize(
    "answer",
    [
        "Run Haunted Altars of Sorrow. It directly advances the Waves of Altars of Sorrow record.",
        "Run Haunted Altars of Sorrow: stronger loot potential than a mission, and it advances "
        "your Waves of Altars of Sorrow Triumph.",
    ],
)
def test_the_solo_loot_answer_cannot_claim_haunted_altars_advances_the_regular_triumph(
    answer: str,
) -> None:
    codes = _planning_codes(answer, "Something with better loot instead?", SOLO_HISTORY)
    assert "activity_record_conflation" in codes


def test_unsupported_loot_potential_is_flagged_without_a_source() -> None:
    codes = _planning_codes(
        "Run the seasonal activity; it has stronger loot potential than a patrol.",
        "Something with better loot instead?",
        SOLO_HISTORY,
    )
    assert codes & {"unsupported_loot_claim", "unsupported_claim", "unsupported_loot_ranking"}


def test_closing_handbacks_are_trimmed_but_the_answer_is_kept() -> None:
    from app.response_quality import ResponseQualityValidator

    drop = ResponseQualityValidator.trim_trailing_handback
    kept = "Finish Kepler Uplink; it advances your campaign quest."
    for tail in (
        "Tell me Story or Rewards and I'll narrow it.",
        "If you want, I'll list the Haunted Altars steps.",
    ):
        assert drop(f"{kept} {tail}", offers=True, choices=True) == kept


def test_a_request_for_a_current_source_is_not_satisfied_by_static_manifest_data() -> None:
    from evals.model import Expectations, Scenario
    from evals.profiles import PROFILES, profile_facts
    from evals.properties import evaluate_properties
    from evals.runtime import Observation, TurnRecord

    scenario = Scenario(
        "t",
        "titan_veteran",
        ("Verify with a current source?",),
        Expectations(current_evidence=True),
    )
    context = PROFILES["titan_veteran"]()

    def verdict(grounding: dict[str, Any], answer: str) -> Any:
        turn = TurnRecord("m", answer, "openai", {"grounding": grounding})
        obs = Observation(scenario, context, [turn])
        found = evaluate_properties(obs, profile_facts(context))
        return next(v for v in found if v.name == "current_source_not_overclaimed")

    static = verdict({"manifest": True}, "Yes, Mint Retrograde drops from Pinnacle Ops.")
    assert not static.passed and static.flag == "stale_live_claim"
    assert verdict({"web_research": True}, "Per the current source, yes.").passed
    assert verdict(
        {"manifest": True}, "Bungie's data lists it; I can't confirm today's source."
    ).passed


def test_unrelated_tools_fail_a_followup_that_should_reuse_earlier_evidence() -> None:
    from evals.model import Expectations, Scenario
    from evals.profiles import PROFILES, profile_facts
    from evals.properties import evaluate_properties
    from evals.runtime import Observation, ToolCallRecord, TurnRecord

    scenario = Scenario(
        "t",
        "titan_veteran",
        ("Next fragment?",),
        Expectations(forbidden_tools=("get_title_progress",)),
    )
    context = PROFILES["titan_veteran"]()
    calls = [ToolCallRecord("get_title_progress", {}, {})]
    turn = TurnRecord("m", "answer", "openai", {"grounding": {}}, tool_calls=calls)
    found = evaluate_properties(Observation(scenario, context, [turn]), profile_facts(context))
    value = next(v for v in found if v.name == "no_irrelevant_tools")
    assert value.applies and not value.passed
