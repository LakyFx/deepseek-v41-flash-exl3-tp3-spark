"""Represent Chat history as evidence for the shared validated checkpoint engine.

This conversion is used only for historical evidence. The live Chat messages,
tool schemas, role order and tool-call IDs are forwarded in their native form.
"""

from copy import deepcopy
from typing import Any


def evidence_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": deepcopy(message.get("content")),
                }
            )
            continue
        content = message.get("content")
        if content or not message.get("tool_calls"):
            result.append(
                {"type": "message", "role": role, "content": deepcopy(content)}
            )
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            result.append(
                {
                    "type": "function_call",
                    "call_id": str(call.get("id") or ""),
                    "name": function.get("name"),
                    "arguments": function.get("arguments"),
                }
            )
    return result
