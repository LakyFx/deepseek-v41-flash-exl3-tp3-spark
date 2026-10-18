"""Lossless-enough OpenAI Responses <-> Chat Completions compatibility.

Pulsar intentionally exposes the much smaller Chat Completions surface.  This
module keeps the protocol translation outside the model runtime so Codex can
still use ``/v1/responses`` without teaching the inference engine about HTTP
protocol variants. Only function tools are translated. Multimodal parts are
preserved when the caller explicitly enables the adapter for a multimodal
backend; text-only backends continue to fail closed instead of silently
discarding media.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for part in content:
        if isinstance(part, str):
            chunks.append(part)
            continue
        if not isinstance(part, dict):
            continue
        kind = str(part.get("type") or "")
        if kind in {
            "input_image",
            "image",
            "image_url",
            "input_video",
            "video",
            "video_url",
        }:
            raise ValueError(
                "the selected Responses adapter is text-only and cannot accept media input"
            )
        if kind in {"input_text", "output_text", "summary_text", "text"}:
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunk for chunk in chunks if chunk)


def _media_url(value: Any, *, kind: str) -> dict[str, Any]:
    if isinstance(value, dict):
        result = deepcopy(value)
    elif isinstance(value, str) and value:
        result = {"url": value}
    else:
        raise ValueError(f"{kind} input is missing its URL")
    if not isinstance(result.get("url"), str) or not result["url"]:
        raise ValueError(f"{kind} input is missing its URL")
    return result


def _chat_content_from_response(content: Any, *, allow_multimodal: bool) -> Any:
    """Translate Responses message content into Chat content parts."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        kind = str(part.get("type") or "")
        if kind in {"input_text", "output_text", "summary_text", "text"}:
            value = part.get("text")
            if isinstance(value, str):
                parts.append({"type": "text", "text": value})
            continue
        if kind in {"input_image", "image", "image_url"}:
            if not allow_multimodal:
                raise ValueError(
                    "the selected Responses adapter is text-only and cannot accept media input"
                )
            image = _media_url(part.get("image_url") or part.get("url"), kind="image")
            if part.get("detail") is not None:
                image.setdefault("detail", part["detail"])
            # vLLM's older Chat schema rejects Responses/Codex "original".
            # Only normalize metadata; never decode/resize/re-encode image bytes.
            if image.get("detail") == "original":
                image["detail"] = "high"
            parts.append({"type": "image_url", "image_url": image})
            continue
        if kind in {"input_video", "video", "video_url"}:
            if not allow_multimodal:
                raise ValueError(
                    "the selected Responses adapter is text-only and cannot accept media input"
                )
            video = _media_url(part.get("video_url") or part.get("url"), kind="video")
            parts.append({"type": "video_url", "video_url": video})
            continue
        raise ValueError(f"unsupported Responses content part type: {kind}")
    return parts


def _tool_content_from_response(content: Any, *, allow_multimodal: bool) -> Any:
    """Preserve legacy text outputs; translate only explicitly typed media."""
    media_types = {
        "input_image", "image", "image_url",
        "input_video", "video", "video_url",
        "input_audio", "audio", "audio_url",
    }
    has_media = isinstance(content, list) and any(
        isinstance(part, dict) and part.get("type") in media_types
        for part in content
    )
    if has_media:
        return _chat_content_from_response(content, allow_multimodal=allow_multimodal)
    return _text_from_content(content)


def responses_input_media_counts(payload: dict[str, Any]) -> dict[str, int]:
    """Count media anywhere in a Responses input item tree.

    Codex can return images from tools inside
    ``custom_tool_call_output.output`` rather than a regular message.  Those
    blocks are still model input and must participate in routing and admission
    accounting, but their base64 payload must never be counted as text.
    """

    counts = {"image": 0, "video": 0}

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        kind = str(value.get("type") or "")
        if kind in {"input_image", "image", "image_url", "computer_screenshot"}:
            counts["image"] += 1
        elif kind in {"input_video", "video", "video_url"}:
            counts["video"] += 1
        for child in value.values():
            visit(child)

    visit(payload.get("input"))
    return counts


def _flush_calls(messages: list[dict[str, Any]], calls: list[dict[str, Any]]) -> None:
    if calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": calls[:]})
        calls.clear()


def _messages_from_input(
    payload: dict[str, Any], *, allow_multimodal: bool
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})

    raw_input = payload.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
        return messages
    if not isinstance(raw_input, list):
        raise TypeError("Responses input must be a string or an item list")

    pending_calls: list[dict[str, Any]] = []
    for item in raw_input:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "message")
        if kind == "function_call":
            pending_calls.append(
                {
                    "id": str(item.get("call_id") or item.get("id") or "call_unknown"),
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or ""),
                        "arguments": str(item.get("arguments") or "{}"),
                    },
                }
            )
            continue
        _flush_calls(messages, pending_calls)
        if kind == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or "call_unknown"),
                    "content": _tool_content_from_response(
                        item.get("output"), allow_multimodal=allow_multimodal
                    ),
                }
            )
        elif kind == "message":
            role = str(item.get("role") or "user")
            if role not in {"system", "developer", "user", "assistant", "tool"}:
                role = "user"
            message: dict[str, Any] = {
                "role": role,
                "content": _chat_content_from_response(
                    item.get("content"), allow_multimodal=allow_multimodal
                ),
            }
            if role == "tool" and item.get("tool_call_id"):
                message["tool_call_id"] = str(item["tool_call_id"])
            messages.append(message)
        elif kind == "reasoning":
            # Responses reasoning items are opaque continuity artefacts.  A
            # public summary is safe to replay; encrypted/private state is not.
            summary = _text_from_content(item.get("summary"))
            if summary:
                messages.append({"role": "assistant", "content": summary})
        elif kind == "item_reference":
            raise ValueError("item_reference requires native Responses state")
        else:
            raise ValueError(f"unsupported Responses input item type: {kind}")
    _flush_calls(messages, pending_calls)
    if not messages:
        raise ValueError("Responses input did not contain any usable messages")
    return messages


def _chat_tools(raw_tools: Any) -> list[dict[str, Any]]:
    if raw_tools is None:
        return []
    if not isinstance(raw_tools, list):
        raise TypeError("Responses tools must be a list")
    tools: list[dict[str, Any]] = []
    for item in raw_tools:
        if not isinstance(item, dict) or item.get("type") != "function":
            raise ValueError("DeepSeek Pulsar supports function tools only")
        function = {
            "name": str(item.get("name") or ""),
            "description": str(item.get("description") or ""),
            "parameters": deepcopy(item.get("parameters") or {"type": "object"}),
        }
        if item.get("strict") is not None:
            function["strict"] = bool(item["strict"])
        tools.append({"type": "function", "function": function})
    return tools


def responses_to_chat(
    payload: dict[str, Any], *, allow_multimodal: bool = False
) -> dict[str, Any]:
    """Translate a validated Responses request into Chat Completions."""

    if payload.get("previous_response_id"):
        raise ValueError("previous_response_id requires native Responses state")
    body: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": _messages_from_input(payload, allow_multimodal=allow_multimodal),
        "stream": bool(payload.get("stream", False)),
        "max_tokens": int(payload.get("max_output_tokens") or 1),
    }
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        effort = str(reasoning["effort"])
        body["reasoning_effort"] = effort
        body["thinking"] = effort not in {"none", "off", "disabled", "false"}
    tools = _chat_tools(payload.get("tools"))
    if tools:
        body["tools"] = tools
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        body["tool_choice"] = {
            "type": "function",
            "function": {"name": str(tool_choice.get("name") or "")},
        }
    elif tool_choice is not None:
        body["tool_choice"] = deepcopy(tool_choice)
    for key in (
        "parallel_tool_calls",
        "temperature",
        "top_p",
        "seed",
        "mm_processor_kwargs",
        "media_io_kwargs",
        "thinking_token_budget",
    ):
        if payload.get(key) is not None:
            body[key] = deepcopy(payload[key])
    text = payload.get("text")
    if isinstance(text, dict) and isinstance(text.get("format"), dict):
        fmt = deepcopy(text["format"])
        if fmt.get("type") == "json_schema" and "schema" in fmt:
            fmt = {
                "type": "json_schema",
                "json_schema": {
                    "name": str(fmt.get("name") or "response"),
                    "schema": fmt["schema"],
                    "strict": bool(fmt.get("strict", False)),
                },
            }
        body["response_format"] = fmt
    return body


def _usage_from_chat(data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    result: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
    }
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        result["output_tokens_details"] = deepcopy(details)
    return result


def chat_to_response(data: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate a non-stream Chat completion into a Responses envelope."""

    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    if not isinstance(message, dict):
        message = {}
    chat_id = str(data.get("id") or f"chatcmpl-{time.time_ns()}")
    response_id = chat_id if chat_id.startswith("resp_") else f"resp_{chat_id}"
    output: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        output.append(
            {
                "type": "reasoning",
                "id": f"rs_{chat_id}",
                "summary": [{"type": "summary_text", "text": reasoning}],
            }
        )
    content = message.get("content")
    if isinstance(content, str) and content:
        output.append(
            {
                "type": "message",
                "id": f"msg_{chat_id}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        )
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for index, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue
            function = (
                call.get("function") if isinstance(call.get("function"), dict) else {}
            )
            call_id = str(call.get("id") or f"call_{chat_id}_{index}")
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{chat_id}_{index}",
                    "call_id": call_id,
                    "name": str(function.get("name") or ""),
                    "arguments": str(function.get("arguments") or "{}"),
                    "status": "completed",
                }
            )
    finish_reason = (
        str(choice.get("finish_reason") or "stop")
        if isinstance(choice, dict)
        else "stop"
    )
    status = "incomplete" if finish_reason == "length" else "completed"
    response: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(data.get("created") or time.time()),
        "status": status,
        "error": None,
        "incomplete_details": (
            {"reason": "max_output_tokens"} if status == "incomplete" else None
        ),
        "model": str(data.get("model") or model),
        "output": output,
        "parallel_tool_calls": True,
        "usage": _usage_from_chat(data),
    }
    return response


def _sse(event: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()


@dataclass(slots=True)
class ChatResponsesStreamAdapter:
    """Stateful Chat SSE to Responses SSE converter."""

    model: str
    response_id: str = field(default_factory=lambda: f"resp_{time.time_ns()}")
    created_at: int = field(default_factory=lambda: int(time.time()))
    output: list[dict[str, Any]] = field(default_factory=list)
    reasoning_item: dict[str, Any] | None = None
    reasoning_index: int | None = None
    reasoning_text: str = ""
    message: dict[str, Any] | None = None
    message_index: int | None = None
    text: str = ""
    tool_items: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: str = "stop"
    usage: dict[str, Any] = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )

    def started(self) -> list[bytes]:
        shell = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "in_progress",
            "model": self.model,
            "output": [],
        }
        return [
            _sse({"type": "response.created", "sequence_number": 0, "response": shell}),
            _sse(
                {
                    "type": "response.in_progress",
                    "sequence_number": 1,
                    "response": shell,
                }
            ),
        ]

    def _ensure_message(self) -> list[bytes]:
        if self.message is not None:
            return []
        self.message_index = len(self.output)
        self.message = {
            "type": "message",
            "id": f"msg_{self.response_id}",
            "status": "in_progress",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "", "annotations": []}],
        }
        self.output.append(self.message)
        return [
            _sse(
                {
                    "type": "response.output_item.added",
                    "output_index": self.message_index,
                    "item": deepcopy(self.message),
                }
            ),
            _sse(
                {
                    "type": "response.content_part.added",
                    "item_id": self.message["id"],
                    "output_index": self.message_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }
            ),
        ]

    def _ensure_reasoning(self) -> list[bytes]:
        if self.reasoning_item is not None:
            return []
        self.reasoning_index = len(self.output)
        self.reasoning_item = {
            "type": "reasoning",
            "id": f"rs_{self.response_id}",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": ""}],
            "encrypted_content": None,
            "status": "in_progress",
        }
        self.output.append(self.reasoning_item)
        return [
            _sse(
                {
                    "type": "response.output_item.added",
                    "output_index": self.reasoning_index,
                    "item": deepcopy(self.reasoning_item),
                }
            ),
            _sse(
                {
                    "type": "response.reasoning_part.added",
                    "item_id": self.reasoning_item["id"],
                    "output_index": self.reasoning_index,
                    "content_index": 0,
                    "part": {"type": "reasoning_text", "text": ""},
                }
            ),
        ]

    def feed(self, payload: dict[str, Any]) -> list[bytes]:
        if payload.get("id") and self.response_id.startswith("resp_"):
            # Keep one stable Responses ID; upstream ID is intentionally not
            # substituted after response.created has already been emitted.
            pass
        usage = payload.get("usage")
        if isinstance(usage, dict):
            self.usage = _usage_from_chat({"usage": usage})
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return []
        choice = choices[0] if isinstance(choices[0], dict) else {}
        if choice.get("finish_reason"):
            self.finish_reason = str(choice["finish_reason"])
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        events: list[bytes] = []
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            events.extend(self._ensure_reasoning())
            self.reasoning_text += reasoning
            assert self.reasoning_item is not None
            assert self.reasoning_index is not None
            self.reasoning_item["content"][0]["text"] = self.reasoning_text
            events.append(
                _sse(
                    {
                        "type": "response.reasoning_text.delta",
                        "item_id": self.reasoning_item["id"],
                        "output_index": self.reasoning_index,
                        "content_index": 0,
                        "delta": reasoning,
                    }
                )
            )
        text = delta.get("content")
        if isinstance(text, str) and text:
            events.extend(self._ensure_message())
            self.text += text
            assert self.message is not None and self.message_index is not None
            self.message["content"][0]["text"] = self.text
            events.append(
                _sse(
                    {
                        "type": "response.output_text.delta",
                        "item_id": self.message["id"],
                        "output_index": self.message_index,
                        "content_index": 0,
                        "delta": text,
                    }
                )
            )
        raw_calls = delta.get("tool_calls")
        if isinstance(raw_calls, list):
            for raw in raw_calls:
                if not isinstance(raw, dict):
                    continue
                index = int(raw.get("index") or 0)
                function = (
                    raw.get("function") if isinstance(raw.get("function"), dict) else {}
                )
                item = self.tool_items.get(index)
                if item is None:
                    output_index = len(self.output)
                    item = {
                        "type": "function_call",
                        "id": f"fc_{self.response_id}_{index}",
                        "call_id": str(
                            raw.get("id") or f"call_{self.response_id}_{index}"
                        ),
                        "name": str(function.get("name") or ""),
                        "arguments": "",
                        "status": "in_progress",
                        "_output_index": output_index,
                    }
                    self.tool_items[index] = item
                    self.output.append(item)
                    public = {k: v for k, v in item.items() if not k.startswith("_")}
                    events.append(
                        _sse(
                            {
                                "type": "response.output_item.added",
                                "output_index": output_index,
                                "item": public,
                            }
                        )
                    )
                elif function.get("name"):
                    item["name"] += str(function["name"])
                if raw.get("id"):
                    item["call_id"] = str(raw["id"])
                arguments = function.get("arguments")
                if isinstance(arguments, str) and arguments:
                    item["arguments"] += arguments
                    events.append(
                        _sse(
                            {
                                "type": "response.function_call_arguments.delta",
                                "item_id": item["id"],
                                "output_index": item["_output_index"],
                                "delta": arguments,
                            }
                        )
                    )
        return events

    def completed(self) -> list[bytes]:
        events: list[bytes] = []
        if self.reasoning_item is not None and self.reasoning_index is not None:
            self.reasoning_item["status"] = "completed"
            reasoning_part = deepcopy(self.reasoning_item["content"][0])
            events.extend(
                [
                    _sse(
                        {
                            "type": "response.reasoning_text.done",
                            "item_id": self.reasoning_item["id"],
                            "output_index": self.reasoning_index,
                            "content_index": 0,
                            "text": self.reasoning_text,
                        }
                    ),
                    _sse(
                        {
                            "type": "response.reasoning_part.done",
                            "item_id": self.reasoning_item["id"],
                            "output_index": self.reasoning_index,
                            "content_index": 0,
                            "part": reasoning_part,
                        }
                    ),
                    _sse(
                        {
                            "type": "response.output_item.done",
                            "output_index": self.reasoning_index,
                            "item": deepcopy(self.reasoning_item),
                        }
                    ),
                ]
            )
        if self.message is not None and self.message_index is not None:
            self.message["status"] = "completed"
            events.extend(
                [
                    _sse(
                        {
                            "type": "response.output_text.done",
                            "item_id": self.message["id"],
                            "output_index": self.message_index,
                            "content_index": 0,
                            "text": self.text,
                        }
                    ),
                    _sse(
                        {
                            "type": "response.content_part.done",
                            "item_id": self.message["id"],
                            "output_index": self.message_index,
                            "content_index": 0,
                            "part": deepcopy(self.message["content"][0]),
                        }
                    ),
                    _sse(
                        {
                            "type": "response.output_item.done",
                            "output_index": self.message_index,
                            "item": deepcopy(self.message),
                        }
                    ),
                ]
            )
        for index in sorted(self.tool_items):
            item = self.tool_items[index]
            item["status"] = "completed"
            output_index = int(item["_output_index"])
            public = {k: v for k, v in item.items() if not k.startswith("_")}
            self.output[output_index] = public
            events.extend(
                [
                    _sse(
                        {
                            "type": "response.function_call_arguments.done",
                            "item_id": item["id"],
                            "output_index": output_index,
                            "arguments": item["arguments"],
                        }
                    ),
                    _sse(
                        {
                            "type": "response.output_item.done",
                            "output_index": output_index,
                            "item": public,
                        }
                    ),
                ]
            )
        status = "incomplete" if self.finish_reason == "length" else "completed"
        response = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "error": None,
            "incomplete_details": (
                {"reason": "max_output_tokens"} if status == "incomplete" else None
            ),
            "model": self.model,
            "output": self.output,
            "parallel_tool_calls": True,
            "usage": self.usage,
        }
        event_type = (
            "response.incomplete" if status == "incomplete" else "response.completed"
        )
        events.extend(
            [_sse({"type": event_type, "response": response}), b"data: [DONE]\n\n"]
        )
        return events
