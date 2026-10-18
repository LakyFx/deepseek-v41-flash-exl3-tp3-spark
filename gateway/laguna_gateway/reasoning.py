"""Compatibility helpers for Poolside interleaved reasoning and tool turns."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .cache import TTLCache


def _normalise_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def tool_call_signature(tool_calls: Any) -> str | None:
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    canonical: list[dict[str, Any]] = []
    for raw in tool_calls:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function")
        if not isinstance(function, dict):
            function = {}
        canonical.append(
            {
                "id": str(raw.get("id") or ""),
                "type": str(raw.get("type") or "function"),
                "name": str(function.get("name") or ""),
                "arguments": _normalise_arguments(function.get("arguments")),
            }
        )
    if not canonical:
        return None
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def restore_reasoning(
    messages: list[dict[str, Any]],
    cache: TTLCache[str, str],
    *,
    field: str = "reasoning",
) -> int:
    restored = 0
    for message in messages:
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        if message.get("reasoning") or message.get("reasoning_content"):
            continue
        signature = tool_call_signature(message.get("tool_calls"))
        if signature is None:
            continue
        reasoning = cache.get(signature)
        if reasoning:
            message[field] = reasoning
            restored += 1
    return restored


def capture_reasoning(
    response_data: dict[str, Any],
    cache: TTLCache[str, str],
) -> bool:
    choices = response_data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return False
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    signature = tool_call_signature(message.get("tool_calls"))
    if not isinstance(reasoning, str) or not reasoning or signature is None:
        return False
    cache.put(signature, reasoning)
    return True


def add_reasoning_compatibility_fields(response_data: dict[str, Any]) -> None:
    choices = response_data.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            message.setdefault("reasoning_content", reasoning)


def strip_reasoning_fields(response_data: dict[str, Any]) -> int:
    """Remove private chain-of-thought fields from a public completion payload.

    Tool-turn reasoning is cached before this runs, so later continuations can
    restore the exact upstream field without exposing it to Botio's HTTP/UI
    response contract.
    """

    removed = 0
    choices = response_data.get("choices")
    if not isinstance(choices, list):
        return removed
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        for field in ("reasoning", "reasoning_content", "thinking"):
            if field in message:
                message.pop(field, None)
                removed += 1
    return removed
