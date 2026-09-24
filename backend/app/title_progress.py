import re

from pydantic import BaseModel, Field

from app.models import ChatTurn


class TitleRequestContext(BaseModel):
    title_intent_present: bool = False
    intent_resolved: bool = False
    retrieval_required: bool = False
    requested_limit: int = Field(default=5, ge=1, le=10)
    is_followup: bool = False


_EXPLICIT_TITLE = re.compile(
    r"\b(?:my|account|guardian)\b.{0,100}\b(?:(?:triumph|record)\s+titles?|seals?)\b|"
    r"\b(?:titles?|seals?)\b.{0,70}\b(?:closest|nearest|completion|progress|finish next)\b|"
    r"\b(?:closest|nearest)\b.{0,70}\b(?:titles?|seals?)\b|"
    r"\b(?:which|what)\b.{0,40}\b(?:triumph|record)\s+titles?\b.{0,70}"
    r"\b(?:should i|finish|go for)\b",
    re.IGNORECASE,
)
_ACCOUNT_TITLE_REQUEST = re.compile(
    r"\b(?:my|account|guardian|progress|easiest|closest|nearest)\b.{0,100}\btitles?\b|"
    r"\btitles?\b.{0,100}\b(?:my|account|guardian|progress|easiest|closest|nearest)\b",
    re.IGNORECASE,
)
_SINGLE_RESULT = re.compile(
    r"\b(?:single|one)\s+(?:title|seal)\b|\b(?:easiest|closest|nearest)\b",
    re.IGNORECASE,
)
_AFFIRMATIVE_OR_ACTION = re.compile(
    r"^\s*(?:yes|yes[,!]?\s+(?:please|those are what i mean)|yep|yeah|sure|go ahead|"
    r"do it|do that|do the recommended(?: one| analysis)?|proceed)\s*[.!]?\s*$",
    re.IGNORECASE,
)
_READ_ONLY_TITLE_PROPOSAL = re.compile(
    r"\b(?:title|seal|triumph|record)\b.{0,180}\b(?:check|scan|fetch|retrieve|look up|"
    r"review|analy[sz]e|compare|recommend)\b|"
    r"\b(?:check|scan|fetch|retrieve|look up|review|analy[sz]e|compare|recommend)\b"
    r".{0,180}\b(?:title|seal|triumph|record)\b",
    re.IGNORECASE,
)
_TITLE_CLARIFICATION = re.compile(r"\b(?:triumph|record)\s+titles?\b|\bseals?\b", re.IGNORECASE)


def derive_title_request_context(message: str, history: list[ChatTurn]) -> TitleRequestContext:
    explicit_current = bool(_EXPLICIT_TITLE.search(message))
    account_title_current = bool(_ACCOUNT_TITLE_REQUEST.search(message))
    requested_limit = 1 if _SINGLE_RESULT.search(message) else 5

    if explicit_current:
        return TitleRequestContext(
            title_intent_present=True,
            intent_resolved=True,
            retrieval_required=True,
            requested_limit=requested_limit,
        )

    recent = history[-6:]
    prior_user_requests = [
        turn.content
        for turn in recent
        if turn.role == "user"
        and (_ACCOUNT_TITLE_REQUEST.search(turn.content) or _EXPLICIT_TITLE.search(turn.content))
    ]
    last_assistant = next(
        (turn.content for turn in reversed(recent) if turn.role == "assistant"), ""
    )
    followup = bool(
        _AFFIRMATIVE_OR_ACTION.fullmatch(message)
        and prior_user_requests
        and (
            _READ_ONLY_TITLE_PROPOSAL.search(last_assistant)
            or ("?" in last_assistant and _TITLE_CLARIFICATION.search(last_assistant))
        )
    )
    if followup:
        prior_limit = 1 if any(_SINGLE_RESULT.search(value) for value in prior_user_requests) else 5
        return TitleRequestContext(
            title_intent_present=True,
            intent_resolved=True,
            retrieval_required=True,
            requested_limit=prior_limit,
            is_followup=True,
        )

    return TitleRequestContext(
        title_intent_present=account_title_current,
        intent_resolved=False,
        retrieval_required=False,
        requested_limit=requested_limit,
    )
