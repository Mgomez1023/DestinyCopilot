"""Presentation cleanup for web-search citation artifacts in final answer text.

The chat UI renders plain text, and two artifacts from the Responses API show up there raw:

* Inline citations the web_search tool writes as Markdown links, such as
  "([bungie.net](https://www.bungie.net/...?utm_source=openai))". The API marks each one
  with a url_citation annotation, so exactly those annotated spans are rewritten to their
  link text ("(bungie.net)"). The URL itself stays in the response's Sources list.
* Private-use citation markers ("citeturn0search0") that the model can emit
  when a citation is not converted into an annotation, for example in a later stateless
  round. They are never readable text, so they are removed with the space before them.

Nothing else in the text is touched: unannotated links and ordinary brackets stay as written.
"""

import re
from collections.abc import Sequence
from typing import Any

_MARKDOWN_LINK = re.compile(r"^\[(?P<label>[^\]\n]{1,120})\]\((?P<url>https?://[^\s)]+)\)$")
_PRIVATE_MARKER = re.compile(r"[ \t]*[^\n]{0,200}")
_STRAY_PRIVATE = re.compile(r"[-]")
# The same markers after their private-use delimiters were lost: "citeturn0search3".
_BARE_MARKER = re.compile(r"[ \t]*\bcite(?:turn\d+[a-z]+\d+)+\b")
_BRACKET_MARKER = re.compile(r"[ \t]*【(?:[^】\n]{0,80}†[^】\n]{0,80}|turn\d+[a-z]+\d+)】")


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def citation_spans(output_items: list[Any]) -> list[tuple[str, str]]:
    """(annotated Markdown link, readable replacement) pairs from url_citation annotations."""

    spans: list[tuple[str, str]] = []
    for item in output_items or []:
        if _field(item, "type") != "message":
            continue
        for content in _field(item, "content") or []:
            text = _field(content, "text")
            if not isinstance(text, str):
                continue
            for annotation in _field(content, "annotations") or []:
                if _field(annotation, "type") != "url_citation":
                    continue
                start = _field(annotation, "start_index")
                end = _field(annotation, "end_index")
                if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end):
                    continue
                span = text[start:end]
                wrapped = span.startswith("(") and span.endswith(")")
                match = _MARKDOWN_LINK.match(span[1:-1] if wrapped else span)
                if match:
                    label = match.group("label")
                    spans.append((span, f"({label})" if wrapped else label))
    return spans


def clean_citation_artifacts(text: str, spans: Sequence[tuple[str, str]] = ()) -> str:
    cleaned = text
    for span, replacement in spans:
        cleaned = cleaned.replace(span, replacement)
    for pattern in (_PRIVATE_MARKER, _BARE_MARKER, _BRACKET_MARKER):
        cleaned = pattern.sub("", cleaned)
    return _STRAY_PRIVATE.sub("", cleaned).strip()
