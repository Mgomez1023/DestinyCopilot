"""Absence of data must be reported as unavailable, never as a zero.

"No recent matches came back" is not "your streak is 0": a tool that returned nothing about a
measure cannot say that measure is nothing. This module finds a sentence that states a zero or
"none" value for a measure none of this turn's tool results mention, and says so plainly instead.
It reads wording only; it knows no particular measure.
"""

import re
from typing import Any

MAX_RESULT_DEPTH = 6
MAX_TERMS = 4000
MIN_TERM_CHARS = 4

_ZERO = r"(?:0|zero|none|nothing|nil)"
_METRIC = r"(?P<metric>[A-Za-z][A-Za-z/'-]{2,24}(?:\s[A-Za-z][A-Za-z/'-]{2,24})?)"
_ZERO_VALUE_PATTERNS = (
    # "your streak is 0", "current win streak stands at zero", "kills: 0", "a streak of 0"
    re.compile(
        rf"\b{_METRIC}\s*(?:is|are|stands at|sits at|is currently|is at|equals|of|=|:)\s*"
        rf"(?:currently\s+)?{_ZERO}\b(?!\.\d|\s*(?:%|percent))",
        re.IGNORECASE,
    ),
    # "0 wins", "zero matches played"
    re.compile(r"\b(?:0|zero)\s+(?P<metric>[A-Za-z][A-Za-z'-]{2,24})", re.IGNORECASE),
)
# A sentence that reports the absence correctly is exactly what we want, so it is never flagged.
_STATES_ABSENCE = re.compile(
    r"\b(?:no data|not (?:available|tracked|exposed|returned|shown|provided)|unavailable|"
    r"unknown|can'?t (?:see|tell|find|determine|confirm|verify)|cannot (?:see|tell|find|"
    r"determine|confirm|verify)|couldn'?t (?:find|see|confirm|verify)|don'?t have|doesn'?t "
    r"(?:show|include|expose|return|say)|did(?:n'?t| not) (?:return|show|include)|not "
    r"established|no way to)\b",
    re.IGNORECASE,
)
# The sentence attributes the measure to the player: "your ...", "you have/you've got ...",
# "you're on/at ...", or "<your character> has ...".
_PLAYER_FRAMED = re.compile(
    r"\byour\b|\byou(?:'ve| have| has)\b|\byou(?:'re| are) (?:currently )?(?:on|at)\b|"
    r"\b(?:the|that) (?:Titan|Hunter|Warlock|Guardian|character) (?:has|have)\b",
    re.IGNORECASE,
)
_NOT_MEASURES = frozenset(
    [
        "this",
        "that",
        "these",
        "those",
        "with",
        "from",
        "have",
        "that",
        "they",
        "them",
        "what",
        "when",
        "then",
        "than",
        "been",
        "were",
        "times",
        "time",
        "minutes",
        "minute",
        "hours",
        "hour",
        "seconds",
        "second",
        "percent",
        "more",
        "less",
        "much",
        "many",
        "other",
        "else",
        "just",
        "only",
        "also",
        "still",
        "very",
        "real",
    ]
)
_TOKEN = re.compile(r"[a-z]{4,}")


def _stem(word: str) -> str:
    word = word.casefold()
    for suffix in ("ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_TERM_CHARS:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def collect_evidence_terms(result: Any, into: set[str]) -> None:
    """Add the words of every key and text value a tool returned, stemmed."""

    def walk(value: Any, depth: int) -> None:
        if depth > MAX_RESULT_DEPTH or len(into) >= MAX_TERMS:
            return
        if isinstance(value, dict):
            for key, inner in value.items():
                into.update(_stem(token) for token in _TOKEN.findall(str(key).casefold()))
                walk(inner, depth + 1)
        elif isinstance(value, list):
            for inner in value:
                walk(inner, depth + 1)
        elif isinstance(value, str):
            into.update(_stem(token) for token in _TOKEN.findall(value.casefold()))

    walk(result, 0)


_DETERMINERS = frozenset(["your", "the", "current", "a", "an", "their", "my", "its", "our"])


def _display(metric: str) -> str:
    words = [word for word in metric.split() if word.casefold() not in _DETERMINERS]
    return " ".join(words) or metric.strip()


def _measure_unreturned(metric: str, terms: set[str]) -> bool:
    """Whether the measure's head word (its last word) appears in nothing a tool returned."""

    words = re.findall(r"[a-z]+", _display(metric).casefold())
    head = words[-1] if words else ""
    if len(head) < MIN_TERM_CHARS or head in _NOT_MEASURES:
        return False
    stem = _stem(head)
    return not (
        stem in terms or any(stem.startswith(term) or term.startswith(stem) for term in terms)
    )


def absent_measure_values(answer: str, terms: set[str]) -> list[tuple[str, str]]:
    """(sentence, measure) pairs where a zero/none value is stated for an unreturned measure."""

    if not terms:
        return []
    found: list[tuple[str, str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer.replace("’", "'")):
        if not sentence.strip() or sentence.rstrip().endswith("?"):
            continue
        # Only the player's own values can be missing data; "Duality has 0 champions" or
        # "rockets have no falloff" are statements about the game, not about this account.
        if not _PLAYER_FRAMED.search(sentence) or _STATES_ABSENCE.search(sentence):
            continue
        for pattern in _ZERO_VALUE_PATTERNS:
            match = pattern.search(sentence)
            if match and _measure_unreturned(match.group("metric"), terms):
                found.append((sentence.strip(), _display(match.group("metric"))))
                break
    return found


def replace_absent_values(answer: str, terms: set[str]) -> str | None:
    """The answer with each such sentence replaced by a plain statement that it is unavailable."""

    found = absent_measure_values(answer, terms)
    if not found:
        return None
    text = answer
    for sentence, metric in found:
        text = text.replace(sentence, f"I can't see your {metric} in the available data.")
    return text
