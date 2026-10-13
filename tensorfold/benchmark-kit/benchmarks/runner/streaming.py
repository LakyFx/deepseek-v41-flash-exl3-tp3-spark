"""Pure OpenAI SSE assembly, preserving reasoning, tools, usage and timing.

Chunk arrival gaps are explicitly not claimed to be per-token latency.
No network call is made by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json


@dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    tools: dict = field(default_factory=dict)
    usage: dict | None = None
    response_id: str | None = None
    finish_reason: str | None = None
    done: bool = False
    first_any_delta: float | None = None
    first_visible_delta: float | None = None
    delta_times: list[float] = field(default_factory=list)
    frames: int = 0
    bytes_received: int = 0

    def accept(self, payload: bytes, when: float) -> None:
        if self.done:
            raise ValueError("SSE data after DONE")
        if payload == b"[DONE]":
            self.done = True
            return
        self.frames += 1
        value = json.loads(payload)
        if not isinstance(value, dict) or "error" in value:
            raise ValueError("invalid inference frame")
        if value.get("id"):
            if self.response_id is not None and value["id"] != self.response_id:
                raise ValueError("SSE response identity changed")
            self.response_id = value["id"]
        if value.get("usage") is not None:
            self.usage = value["usage"]
        choices = value.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("one completion choice required")
        if not choices:
            return
        choice = choices[0]
        if choice.get("index", 0) != 0:
            raise ValueError("unexpected completion index")
        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            raise ValueError("invalid delta")
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise ValueError("nontext completion delta")
        calls = delta.get("tool_calls") or []
        if content or reasoning or calls:
            if self.first_any_delta is None:
                self.first_any_delta = when
            self.delta_times.append(when)
        if content or calls:
            if self.first_visible_delta is None:
                self.first_visible_delta = when
        self.content += content
        self.reasoning += reasoning
        for call in calls:
            index = call.get("index")
            if type(index) is not int or index < 0 or index >= 16:
                raise ValueError("bounded tool call index required")
            target = self.tools.setdefault(index, {"id": None, "type": "function",
                                                   "function": {"name": "", "arguments": ""}})
            if call.get("type") not in (None, "function"):
                raise ValueError("unsupported tool call type")
            if call.get("id"):
                if target["id"] not in (None, call["id"]):
                    raise ValueError("tool call identity changed")
                target["id"] = call["id"]
            function = call.get("function", {})
            for key in ("name", "arguments"):
                piece = function.get(key) or ""
                if not isinstance(piece, str):
                    raise ValueError("invalid tool call delta")
                target["function"][key] += piece
        if choice.get("finish_reason") is not None:
            self.finish_reason = choice["finish_reason"]

    def message(self) -> dict:
        value = {"role": "assistant", "content": self.content or None}
        if self.reasoning:
            value["reasoning_content"] = self.reasoning
        if self.tools:
            calls = [self.tools[index] for index in sorted(self.tools)]
            ids = [call["id"] for call in calls]
            if any(not item for item in ids) or len(ids) != len(set(ids)):
                raise ValueError("complete distinct tool IDs required")
            if any(not call["function"]["name"] for call in calls):
                raise ValueError("complete tool names required")
            value["tool_calls"] = calls
        return value

    def usage_counts(self) -> dict:
        if not self.done or not isinstance(self.usage, dict) or self.finish_reason is None:
            raise ValueError("complete SSE/finish/usage required")
        prompt, completion = self.usage.get("prompt_tokens"), self.usage.get("completion_tokens")
        details = self.usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        if any(type(x) is not int or x < 0 for x in (prompt, completion, cached)) or cached > prompt:
            raise ValueError("explicit valid prompt/completion/cache usage required")
        return {"prompt_tokens": prompt, "completion_tokens": completion,
                "cached_tokens": cached, "computed_prompt_tokens": prompt - cached}


def parse_sse(lines, *, on_first_delta=None) -> Reply:
    """Input is (monotonic timestamp, raw line), including frame blank lines."""
    reply = Reply()
    frame = []
    when = None
    previous_time = None
    def accept(payload, timestamp):
        had_delta = reply.first_any_delta is not None
        reply.accept(payload, timestamp)
        if not had_delta and reply.first_any_delta is not None and on_first_delta is not None:
            on_first_delta(reply.first_any_delta, reply.response_id)
    for timestamp, line in lines:
        if previous_time is not None and timestamp < previous_time:
            raise ValueError("nonmonotonic SSE time")
        previous_time = timestamp
        reply.bytes_received += len(line)
        if len(line) > 2 * 1024**2 or reply.bytes_received > 32 * 1024**2:
            raise ValueError("SSE size limit")
        text = line.rstrip(b"\r\n")
        if not text:
            if frame:
                accept(b"\n".join(frame), when)
                frame = []
                if reply.done:
                    break
            continue
        if text.startswith(b"data:"):
            frame.append(text[5:].lstrip(b" "))
            when = timestamp
    if frame and not reply.done:
        accept(b"\n".join(frame), when)
    if not reply.done:
        raise ValueError("SSE closed without DONE")
    return reply
