"""Local, non-destructive repairs: change only what a validator flagged, never add content.

A validator finding is fixed by moving, trimming, or removing the offending sentences of the
draft itself. A model rewrite is a last resort, and one that damages the draft (drops its
sources, the names it took from tool results, or most of its sentences) is discarded.
"""

import re
from collections.abc import Callable, Iterable

from app.response_quality import (
    _CITED_FACT,
    _READ_ONLY_OFFER,
    _WHICH_TARGET_QUESTION,
    _plain,
    _rejoin,
    _sentence_parts,
    extreme_max_words,
    strip_filler_opener,
)
from app.session_planning import contains_recommendation

# Findings about presentation only. They are repaired here and never sent back to the model.
LOCAL_STYLE_CODES = frozenset(
    {
        "answer_not_first",
        "extreme_verbosity",
        "mechanical_structure",
        "repeated_content",
        "too_many_followups",
        "unnecessary_clarification",
        "unnecessary_read_only_permission",
    }
)
# A draft that is only a question has no content to repair locally; it needs an answer.
QUESTION_CODES = frozenset(
    {"too_many_followups", "unnecessary_clarification", "unnecessary_read_only_permission"}
)
# Constraint findings fixed by removing the sentence that recommends the excluded thing.
CONSTRAINT_CODES = frozenset(
    {
        "locked_item_replaced",
        "solo_constraint",
        "group_constraint",
        "activity_mode_constraint",
        "chill_constraint",
        "excluded_candidate_recommended",
        "rejected_candidate_recommended",
    }
)

_OPENERS = (
    "based on the available data",
    "based on the data",
    "primary recommendation",
    "short answer",
    "there are several things to consider",
)
_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_URL_OR_DATE = re.compile(
    rf"https?://|\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b{_MONTH}\.?\s+\d{{1,2}}\b", re.IGNORECASE
)
_LIMIT_STATEMENT = re.compile(
    r"\b(?:can't|cannot|couldn't|unknown|not established|isn't established|unverified|"
    r"not confirmed|no data)\b",
    re.IGNORECASE,
)
# A line that explains or details the pick before it, and goes with it when that pick goes.
_DETAIL_LABEL = re.compile(
    r"^\W*(?:why|reason|because|what to do|how|steps?|plan|details?)\b[^.\n]{0,40}(?::|$)",
    re.IGNORECASE,
)
_BACKUP_LABEL = re.compile(
    r"^(?P<lead>\W*)(?:\w+\s+){0,2}?(?:backup|alternative|alternatively|fallback|"
    r"second (?:option|choice)|plan b|as a backup)\b[^:—–\n]{0,60}[:—–-]\s*",
    re.IGNORECASE,
)
_HEADING_ONLY = re.compile(r"^\s*(?:#{1,4}\s+|\*\*)[^.!?\n]{1,60}?(?:\*\*)?\s*:?\s*$")
_MARKUP = re.compile(r"^\s*#{1,4}\s+|\*\*")


def _units(answer: str) -> tuple[list[str], list[str]]:
    parts = _sentence_parts(answer)
    return parts, parts[::2]


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def is_constraint_code(code: str) -> bool:
    """A finding fixed by removing the sentence that breaks a stated preference."""

    return code in CONSTRAINT_CODES or code.startswith("reasked_")


def _is_question(sentence: str) -> bool:
    return sentence.rstrip().endswith("?")


def _result(text: str, original: str) -> str | None:
    text = text.strip()
    return text if text and text != original.strip() and _words(text) >= 3 else None


# --- Presentation-only repairs ----------------------------------------------------------------


def move_answer_first(answer: str) -> str | None:
    """Put the first recommending sentence first; drop a bare filler lead-in that precedes it."""

    stripped = strip_filler_opener(answer)
    if stripped is not None:
        return stripped
    parts, units = _units(answer)
    if not units or not _plain(units[0]).casefold().startswith(_OPENERS):
        return None
    for index, unit in enumerate(units[1:], start=1):
        plain = _plain(unit)
        if (
            _is_question(plain)
            or plain.casefold().startswith(_OPENERS)
            or not contains_recommendation(plain)
        ):
            continue
        filler = _words(units[0]) <= 12
        remaining = _rejoin(
            parts, [None if i in {index, 0 if filler else -1} else u for i, u in enumerate(units)]
        )
        joiner = "\n" if "\n" in answer else " "
        return _result(f"{unit.strip()}{joiner}{remaining}".strip(), answer)
    return None


def drop_extra_questions(answer: str, *, permission: bool, clarification: bool) -> str | None:
    """Drop trailing questions beyond one, and permission/offer questions.

    Only questions go, and only when a statement remains: a draft that is nothing but a
    question is left as it is, since nothing here may add content.
    """

    parts, units = _units(answer)
    drop: set[int] = set()
    questions = [i for i, u in enumerate(units) if _is_question(_plain(u))]
    for index in questions:
        plain = _plain(units[index])
        if permission and _READ_ONLY_OFFER.search(_WHICH_TARGET_QUESTION.sub(" ", plain)):
            drop.add(index)
    for index in questions:
        if index in drop:
            continue
        # "Want me to check X? Or Y?": the second question belongs to the dropped ask.
        follows_ask = index - 1 in drop and re.match(r"\s*or\b", _plain(units[index]), re.I)
        if clarification or follows_ask:
            drop.add(index)
    kept_questions = [i for i in questions if i not in drop]
    while len(kept_questions) > 1:
        drop.add(kept_questions.pop())  # the last ones go first
    if not drop:
        return None
    remaining = [u for i, u in enumerate(units) if i not in drop]
    if not any(_words(u) >= 4 and not _is_question(_plain(u)) for u in remaining):
        return None
    return _result(_rejoin(parts, [None if i in drop else u for i, u in enumerate(units)]), answer)


def drop_repeated_sentences(answer: str) -> str | None:
    parts, units = _units(answer)
    seen: set[str] = set()
    replaced: list[str | None] = []
    for unit in units:
        key = re.sub(r"[^a-z0-9]+", " ", unit.casefold()).strip()
        if _words(key) >= 6 and key in seen:
            replaced.append(None)
            continue
        seen.add(key)
        replaced.append(unit)
    return _result(_rejoin(parts, replaced), answer) if None in replaced else None


def trim_excess(answer: str, max_words: int) -> str | None:
    """Remove whole lines from the end until the answer fits.

    The first line, any line with a link or date, and any stated limit stay. Nothing is cut
    mid-sentence, and a list header left with nothing under it goes too.
    """

    lines = answer.strip().split("\n")
    if _words(answer) <= max_words:
        return None
    index = len(lines) - 1
    while index > 0 and _words("\n".join(lines)) > max_words:
        line = lines[index]
        if line.strip() and not _URL_OR_DATE.search(line) and not _LIMIT_STATEMENT.search(line):
            del lines[index]
        index -= 1
    while len(lines) > 1 and (not lines[-1].strip() or lines[-1].rstrip().endswith(":")):
        lines.pop()
    # Trimming can strand a label ("Backup (...)", "Why:") whose lines went; drop those too.
    text = clean_layout("\n".join(lines))
    return _result(text, answer) if _words(text) < _words(answer) else None


def flatten_headings(answer: str) -> str | None:
    """Drop heading-only lines and heading markup; the text under them is untouched."""

    out: list[str] = []
    for line in answer.split("\n"):
        if _HEADING_ONLY.match(line) and not re.search(r"[.!?]\s*$", line.strip("* ")):
            continue
        out.append(_MARKUP.sub("", line) if line.lstrip().startswith(("#", "**")) else line)
    return _result("\n".join(out), answer)


def repair_style(
    answer: str,
    codes: Iterable[str],
    *,
    mode,  # noqa: ANN001  ResponseModeContext
    permission_allowed: bool,
) -> tuple[str, list[dict[str, str]]]:
    """Apply the local fix for each presentation finding. Returns the text and what was done."""

    wanted = set(codes) & LOCAL_STYLE_CODES
    text, done = answer, []

    def apply(code: str, action: str, fixed: str | None) -> None:
        nonlocal text
        if fixed is not None:
            text = fixed
            done.append({"validator": code, "action": action, "result": "applied"})
        else:
            done.append({"validator": code, "action": action, "result": "nothing_to_change"})

    if "answer_not_first" in wanted:
        apply("answer_not_first", "move_answer_first", move_answer_first(text))
    if "repeated_content" in wanted:
        apply("repeated_content", "drop_repeated_sentences", drop_repeated_sentences(text))
    if wanted & {
        "too_many_followups",
        "unnecessary_clarification",
        "unnecessary_read_only_permission",
    }:
        permission = "unnecessary_read_only_permission" in wanted and permission_allowed
        apply(
            "too_many_followups"
            if "too_many_followups" in wanted
            else "unnecessary_clarification"
            if "unnecessary_clarification" in wanted
            else "unnecessary_read_only_permission",
            "drop_extra_questions",
            drop_extra_questions(
                text,
                permission=permission,
                clarification="unnecessary_clarification" in wanted,
            ),
        )
    if "mechanical_structure" in wanted:
        apply("mechanical_structure", "flatten_headings", flatten_headings(text))
    if "extreme_verbosity" in wanted:
        apply("extreme_verbosity", "trim_excess", trim_excess(text, extreme_max_words(mode)))
    if done and text != answer:
        text = clean_layout(text)
    return text, done


# --- Constraint repairs -----------------------------------------------------------------------


def offending_sentences(
    answer: str,
    codes: Iterable[str],
    validate: Callable[[str], Iterable[str]],
) -> set[str]:
    """Sentences that raise one of `codes` when checked on their own.

    `validate` returns the violation codes for a piece of text. These checks read one sentence
    at a time, so a sentence that fails alone is the one breaking the rule.
    """

    wanted = set(codes)
    return {
        _plain(unit).strip()
        for unit in _units(answer)[1]
        if unit.strip() and wanted & set(validate(unit))
    }


_ITEM_LINE = re.compile(r"\s*(?:[-*•]|\d{1,2}[.)])\s")


def _layout(parts: list[str]) -> list[tuple[int, int] | None]:
    """(line, paragraph) of each sentence; None for an empty piece."""

    line = para = newlines = 0
    placed: list[tuple[int, int] | None] = []
    for index in range(0, len(parts), 2):
        if parts[index].strip():
            if newlines >= 1:
                line += 1
            if newlines >= 2:
                para += 1
            newlines = 0
            placed.append((line, para))
        else:
            placed.append(None)
        if index + 1 < len(parts):
            newlines += parts[index + 1].count("\n")
    return placed


def _drop_indices(parts: list[str], flagged: set[str]) -> tuple[set[int], bool]:
    """Sentence indices to remove for the flagged sentences, and whether the first pick is among
    them.

    A flagged sentence that opens a bullet or labelled line takes its whole line with it (so no
    fragment is left), and the bullets or labelled lines directly under a removed line go too.
    When the first recommendation is removed and a labelled backup follows, everything between
    them was detail for the removed pick and goes as well.
    """

    units = parts[::2]
    place = _layout(parts)
    drop = {i for i, u in enumerate(units) if _plain(u).strip() in flagged and place[i]}
    if not drop:
        return set(), False

    def line_of(i: int) -> int:
        return place[i][0] if place[i] else -1

    def opens_line(i: int) -> bool:
        return not any(
            j != i and place[j] and line_of(j) == line_of(i) and j < i for j in range(len(units))
        )

    for i in sorted(drop):
        if opens_line(i) and (_ITEM_LINE.match(units[i]) or _DETAIL_LABEL.match(_plain(units[i]))):
            drop.update(j for j in range(len(units)) if place[j] and line_of(j) == line_of(i))
    # Lines directly under a removed heading or pick that only detail it. The bullets after a
    # removed bullet are its siblings, not its detail, and stay.
    for i in sorted(drop):
        line, para = place[i] or (-1, -1)
        head = next(j for j in range(len(units)) if place[j] and line_of(j) == line)
        if _ITEM_LINE.match(units[head]):
            continue
        nxt = line + 1
        while True:
            members = [j for j in range(len(units)) if place[j] and line_of(j) == nxt]
            if not members or place[members[0]][1] != para:
                break
            first = units[members[0]]
            if not (_ITEM_LINE.match(first) or _DETAIL_LABEL.match(_plain(first))):
                break
            drop.update(members)
            nxt += 1
    recommending = [i for i, u in enumerate(units) if contains_recommendation(_plain(u))]
    primary_removed = bool(recommending) and recommending[0] in drop
    if primary_removed:
        last = recommending[0]
        backup = next(
            (
                i
                for i in range(last + 1, len(units))
                if place[i] and i not in drop and _BACKUP_LABEL.match(units[i]) and opens_line(i)
            ),
            None,
        )
        if backup is not None:
            first_line = min(line_of(i) for i in drop)
            drop.update(
                j
                for j in range(len(units))
                if place[j] and first_line <= line_of(j) < line_of(backup)
            )
    return drop, primary_removed


def dependents_of(answer: str, flagged: set[str]) -> set[str]:
    """The flagged sentences plus the lines that only detail a removed pick."""

    parts = _sentence_parts(answer)
    drop, _ = _drop_indices(parts, flagged)
    return {_plain(parts[2 * i]).strip() for i in drop}


def has_statement(answer: str) -> bool:
    """Whether the answer says something besides asking: a non-question sentence of 4+ words."""

    return any(
        _words(u) >= 4 and not _is_question(_plain(u)) for u in _units(answer)[1] if u.strip()
    )


def locally_handled(codes: Iterable[str], answer: str) -> set[str]:
    """The presentation findings that are repaired in place and never sent back to the model."""

    handled = set(codes) & LOCAL_STYLE_CODES
    return handled if has_statement(answer) else handled - QUESTION_CODES


_LABEL_WORDS = re.compile(
    r"^(?:\w+\s+){0,2}?(?:plan|backup|alternatives?|fallback|primary(?:\s+pick)?|options?|steps?|"
    r"checklist|tips?|notes?|why|how|what to do)\b[^.!?:]{0,40}$",
    re.IGNORECASE,
)


def _is_markdown_heading(line: str) -> bool:
    text = line.strip()
    return bool(re.match(r"#{1,4}\s+\S", text) or re.fullmatch(r"\*\*[^*]+\*\*:?", text))


def _is_heading(line: str) -> bool:
    """A label that introduces what follows: "Options:", "Plan", "Backup (if ...)", "## Steps".

    A short plain closing line ("Have fun") is not a label, so it is never treated as one.
    """

    text = _plain(line).strip().strip("*#").strip()
    if not text or _ITEM_LINE.match(line) or _words(text) > 12:
        return False
    return (
        (_is_markdown_heading(line) and not re.search(r"[.!?]$", text))
        or text.endswith(":")
        or bool(_LABEL_WORDS.match(text))
    )


def drop_dependents(parts: list[str], replaced: list[str | None]) -> list[str | None]:
    """`replaced` with the lines that depended on removed sentences removed too.

    A removed sentence that opened a bullet or labelled line takes the rest of that line, and
    the "Why:"/"How:" or bullet lines under a removed heading or pick; a list header left with
    nothing under it goes as well.
    """

    units = parts[::2]
    removed = {
        _plain(unit).strip()
        for unit, kept in zip(units, replaced, strict=True)
        if kept is None and unit.strip()
    }
    if not removed:
        return replaced
    drop, _ = _drop_indices(parts, removed)
    out = [None if i in drop else kept for i, kept in enumerate(replaced)]
    gone = drop | {i for i, kept in enumerate(replaced) if kept is None}
    return _tidy_after_removal(units, out, gone)


def clean_layout(text: str) -> str:
    """No blank bullets, no heading or label left with nothing under it, no runs of blank lines.

    A label counts as empty at the end of the answer, before another label, and (unless it is a
    markdown heading) when a blank line separates it from plain text, because its bullets went.
    """

    lines = [line.rstrip() for line in text.split("\n")]
    lines = [line for line in lines if not re.fullmatch(r"\s*(?:[-*•]|\d{1,2}[.)])\s*", line)]
    kept: list[str] = []
    for index, line in enumerate(lines):
        if line.strip() and _is_heading(line):
            rest = lines[index + 1 :]
            after = next((value for value in rest if value.strip()), None)
            gap = bool(rest) and not rest[0].strip()
            if (
                after is None
                or (_is_heading(after) and not _ITEM_LINE.match(after))
                or (gap and not _ITEM_LINE.match(after) and not _is_markdown_heading(line))
            ):
                continue
        kept.append(line)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return cleaned or text.strip()


def _tidy_after_removal(
    units: list[str], replaced: list[str | None], dropped: set[int]
) -> list[str | None]:
    """Drop a list header left with nothing under it, and the "Or" that led a bullet whose
    sibling was removed."""

    out = list(replaced)
    for i, unit in enumerate(out):
        if unit is None:
            continue
        following = next((out[j] for j in range(i + 1, len(out)) if out[j] is not None), None)
        had_items = any(
            j in dropped and re.match(r"\s*(?:[-*•]|\d{1,2}[.)])\s", units[j])
            for j in range(i + 1, min(len(units), i + 8))
        )
        is_item = bool(re.match(r"\s*(?:[-*•]|\d{1,2}[.)])\s", unit))
        if (
            _is_heading(unit)
            and had_items
            and (following is None or not re.match(r"\s*(?:[-*•]|\d{1,2}[.)])\s", following))
        ):
            out[i] = None
        elif is_item and i > 0 and (i - 1) in dropped:
            out[i] = re.sub(
                r"^(\s*(?:[-*•]|\d{1,2}[.)])\s+)or\s+(\w)",
                lambda m: m.group(1) + m.group(2).upper(),
                unit,
                count=1,
                flags=re.IGNORECASE,
            )
    # An offer that answered a removed question ("I'll give you a loadout for either") is moot.
    live = [i for i, unit in enumerate(out) if unit is not None]
    prior = [j for j in range(live[-1]) if units[j].strip()][-1:] if live else []
    if live and prior and prior[0] in dropped:
        tail = _plain(out[live[-1]] or "")
        if re.match(r"\W*(?:i'll|i will|i can|i could|let me know)\b", tail, re.IGNORECASE):
            out[live[-1]] = None
    return out


def remove_flagged_sentences(
    answer: str,
    flagged: set[str],
    acceptable_pick: Callable[[str], bool] | None = None,
) -> str | None:
    """The answer without the flagged sentences, keeping the rest.

    If the first recommendation was removed, the lines that only explained it go with it and a
    labelled backup becomes the pick, provided `acceptable_pick(body)` agrees (a backup that
    breaks the same constraint is no replacement). Returns None when nothing was flagged, when
    what is left opens mid-thought, or when no recommendation remains.
    """

    parts, units = _units(answer)
    drop, primary_removed = _drop_indices(parts, flagged)
    if not drop:
        return None
    replaced: list[str | None] = [None if i in drop else u for i, u in enumerate(units)]
    if primary_removed:
        for i, unit in enumerate(replaced):
            label = _BACKUP_LABEL.match(unit) if unit else None
            if label and i not in drop:
                body = unit[label.end() :].strip()
                if body and _words(body) >= 3:
                    if acceptable_pick is not None and not acceptable_pick(body):
                        return None
                    replaced[i] = label.group("lead") + body[:1].upper() + body[1:]
                break
    replaced = _tidy_after_removal(units, replaced, drop)
    text = clean_layout(_rejoin(parts, replaced))
    if not text or _words(text) < 3:
        return None
    if re.match(r"^\W*(?:why|what to do|how|note|backup)\b[^.\n]{0,25}:", _plain(text), re.I):
        return None
    if not contains_recommendation(_plain(text)):
        return None
    return text


# Lines that ask, offer, or hand the choice back carry no information a rewrite could lose.
_NOT_INFORMATION = re.compile(
    r"\b(?:i can|i could|i'll|i will|want me to|would you like|do you want|which (?:one |would )|"
    r"pick one|let me know|if you want|if you'd like|quick options?|two options)\b",
    re.IGNORECASE,
)


def _is_information(sentence: str) -> bool:
    plain = _plain(sentence).strip()
    return (
        _words(plain) >= 3
        and not _is_question(plain)
        and not plain.endswith(":")
        and not _NOT_INFORMATION.search(plain)
    )


# --- The damage guard for model rewrites ------------------------------------------------------


def _covered(sentence: str, rewrite_words: set[str]) -> bool:
    """Whether most of a sentence's content words survive in the rewrite."""

    words = {w for w in re.findall(r"[a-z0-9']+", sentence.casefold()) if len(w) >= 4}
    return not words or len(words & rewrite_words) / len(words) >= 0.6


def rewrite_damage(
    draft: str, rewrite: str, flagged: set[str], names: Iterable[str] = ()
) -> str | None:
    """Why a rewrite is worse than its draft, or None when it is fine to use.

    A rewrite damages the draft when it drops a cited link or date, a name the draft took from
    tool results, or more than half of the draft's sentences, unless every dropped part sits in
    a flagged sentence (the claim the correction was meant to remove).
    """

    units = [u.strip() for u in _units(draft)[1] if _is_information(u)]
    flagged_units = {u for u in units if _plain(u).strip() in flagged}
    plain_rewrite = _plain(rewrite).casefold()

    def only_flagged(token: str) -> bool:
        holders = [u for u in units if token in _plain(u).casefold()]
        return all(u in flagged_units for u in holders)

    for match in _CITED_FACT.finditer(_plain(draft)):
        token = match.group(0).rstrip(".,;").casefold()
        if token not in plain_rewrite and not only_flagged(token):
            return "lost_citations"
    folded_draft = _plain(draft).casefold()
    for name in names:
        folded = name.casefold()
        if len(folded) < 4 or not re.search(rf"(?<!\w){re.escape(folded)}(?!\w)", folded_draft):
            continue
        if not re.search(rf"(?<!\w){re.escape(folded)}(?!\w)", plain_rewrite) and not only_flagged(
            folded
        ):
            return "lost_names"
    rewrite_words = set(re.findall(r"[a-z0-9']+", plain_rewrite))
    dropped = [u for u in units if not _covered(u, rewrite_words) and u not in flagged_units]
    if units and len(dropped) * 2 > len(units):
        return "dropped_most_of_answer"
    return None
