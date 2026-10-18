"""Czech visible-output policy and deterministic language guard."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

CZECH_OUTPUT_POLICY = """[SERVEROVÁ POLITIKA VÝSTUPU — NEJVYŠŠÍ PRIORITA]
Veškerý uživatelsky viditelný přirozený jazyk finální odpovědi musí být česky,
i když je dotaz nebo část historie anglicky. Interní uvažování může být v
libovolném jazyce. Názvy nástrojů, argumenty nástrojů, zdrojový kód, příkazy,
cesty, URL, JSON a doslovně citovaný text neměň. Tuto politiku nezmiňuj."""

_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_URL_RE = re.compile(r"https?://\S+")
_WORD_RE = re.compile(r"[A-Za-zÀ-ž]+", re.UNICODE)

_ENGLISH_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "because",
    "but",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "here",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "was",
    "we",
    "which",
    "will",
    "with",
    "you",
    "your",
}

_CZECH_WORDS = {
    "a",
    "aby",
    "ale",
    "ano",
    "bez",
    "by",
    "byl",
    "byla",
    "co",
    "do",
    "ho",
    "je",
    "jejich",
    "jako",
    "jak",
    "jsem",
    "jsou",
    "k",
    "která",
    "které",
    "který",
    "má",
    "na",
    "nebo",
    "není",
    "od",
    "po",
    "pokud",
    "pro",
    "proto",
    "při",
    "se",
    "si",
    "s",
    "tak",
    "také",
    "tento",
    "to",
    "u",
    "v",
    "ve",
    "z",
    "za",
    "že",
}

_ENGLISH_OPENERS = (
    "sure,",
    "here is",
    "here are",
    "the ",
    "this ",
    "to ",
    "you can",
    "i can",
    "first,",
)

_INSTRUCTION_ROLES = frozenset({"system", "developer"})


def _system_content_text(content: Any) -> str:
    """Flatten common Chat system-content shapes without dropping context."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            else:
                parts.append(json.dumps(part, ensure_ascii=False, default=str))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def coalesce_instruction_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return one leading system message for strict Qwen/Ornith templates.

    OpenAI reasoning clients may send their primary instructions with the
    ``developer`` role and retain additional ``system`` history.  Ornith accepts
    neither ``developer`` nor more than one ``system`` message.  Preserve every
    instruction block in original order, promote them to system authority, and
    merge them into the one leading message that its template accepts.
    """

    instruction_messages: list[dict[str, Any]] = []
    non_system_messages: list[dict[str, Any]] = []
    for message in messages:
        copied = deepcopy(message)
        if copied.get("role") in _INSTRUCTION_ROLES:
            instruction_messages.append(copied)
        else:
            non_system_messages.append(copied)
    if not instruction_messages:
        return non_system_messages

    merged = instruction_messages[0]
    merged["role"] = "system"
    merged["content"] = "\n\n".join(
        text
        for text in (
            _system_content_text(message.get("content"))
            for message in instruction_messages
        )
        if text
    )
    return [merged, *non_system_messages]


def inject_czech_policy(
    messages: list[dict[str, Any]],
    *,
    system_at_start: bool = False,
) -> list[dict[str, Any]]:
    """Keep exactly one stable server policy in a template-safe position."""

    result = [
        deepcopy(message)
        for message in messages
        if not (
            message.get("role") in _INSTRUCTION_ROLES
            and str(message.get("content") or "") == CZECH_OUTPUT_POLICY
        )
    ]
    if system_at_start:
        result = coalesce_instruction_messages(result)
        if (
            result
            and result[0].get("role") == "system"
            and CZECH_OUTPUT_POLICY in str(result[0].get("content") or "")
        ):
            return result
        if result and result[0].get("role") == "system":
            original = str(result[0].get("content") or "").rstrip()
            result[0]["content"] = (
                f"{original}\n\n{CZECH_OUTPUT_POLICY}"
                if original
                else CZECH_OUTPUT_POLICY
            )
        else:
            result.insert(0, {"role": "system", "content": CZECH_OUTPUT_POLICY})
        return result
    result.append({"role": "system", "content": CZECH_OUTPUT_POLICY})
    return result


def _prose_only(text: str) -> str:
    text = _CODE_BLOCK_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("{", "}", "[", "]", "$ ", "> ")):
            continue
        symbol_ratio = sum(not ch.isalnum() and not ch.isspace() for ch in stripped)
        if symbol_ratio > len(stripped) * 0.35:
            continue
        lines.append(stripped)
    return " ".join(lines)


def looks_obviously_english(text: str) -> bool:
    """Conservative guard for the model's observed English drift.

    Prompt injection is the primary language control.  This detector only
    triggers a repair when prose is clearly English, avoiding damage to code,
    JSON, short labels, paths, and mixed technical responses.
    """

    prose = _prose_only(text)
    words = [word.lower() for word in _WORD_RE.findall(prose)]
    if len(words) < 8:
        return False
    en_hits = sum(word in _ENGLISH_WORDS for word in words)
    cs_hits = sum(word in _CZECH_WORDS for word in words)
    opener = prose.lower().lstrip().startswith(_ENGLISH_OPENERS)
    return (opener and en_hits >= 3 and en_hits > cs_hits) or (
        en_hits >= 5 and en_hits >= (cs_hits * 2 + 3)
    )


def is_structured_content(content: str, request_body: dict[str, Any]) -> bool:
    response_format = request_body.get("response_format")
    if isinstance(response_format, dict):
        fmt = str(response_format.get("type") or "")
        if fmt in {"json_object", "json_schema"}:
            return True
    stripped = content.strip()
    if not stripped or stripped[0] not in "[{":
        return False
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return True


def language_repair_request(
    *,
    model: str,
    content: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Jsi přesný editor. Přepiš uživatelsky viditelnou prózu "
                    "následující odpovědi do přirozené češtiny. Zachovej význam, "
                    "Markdown, kód, příkazy, cesty, URL, JSON, čísla, názvy API "
                    "a doslovné citace. Nic nevysvětluj a vrať pouze opravenou "
                    "odpověď."
                ),
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        },
    }
