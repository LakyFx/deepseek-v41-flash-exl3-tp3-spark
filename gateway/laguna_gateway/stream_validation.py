"""Validate Chat SSE termination without retaining a transcript."""

import json


class ChatStreamValidator:
    def __init__(self, max_event_bytes: int = 8 * 1024 * 1024) -> None:
        self.pending = b""
        self.finished = False
        self.done = False
        self.max_event_bytes = max_event_bytes

    def feed(self, chunk: bytes) -> None:
        self.pending = (self.pending + chunk).replace(b"\r\n", b"\n")
        while b"\n\n" in self.pending:
            block, self.pending = self.pending.split(b"\n\n", 1)
            if len(block) > self.max_event_bytes:
                raise ValueError("upstream SSE event exceeds the size limit")
            data = b"\n".join(
                line[5:].lstrip()
                for line in block.splitlines()
                if line.startswith(b"data:")
            )
            if not data:
                continue
            if self.done:
                raise ValueError("upstream sent model data after DONE")
            if data == b"[DONE]":
                if not self.finished:
                    raise ValueError("upstream ended without a finish reason")
                self.done = True
                continue
            try:
                payload = json.loads(data)
            except (ValueError, UnicodeError) as exc:
                raise ValueError("upstream returned malformed SSE JSON") from exc
            if not isinstance(payload, dict) or payload.get("error"):
                raise ValueError("upstream returned a streaming error")
            for choice in payload.get("choices") or []:
                if not isinstance(choice, dict):
                    raise TypeError("upstream returned an invalid streaming choice")
                finish = choice.get("finish_reason")
                if finish:
                    if finish not in {"stop", "length", "tool_calls", "content_filter"}:
                        raise ValueError(
                            "upstream returned an unsuccessful finish reason"
                        )
                    self.finished = True
        if len(self.pending) > self.max_event_bytes:
            raise ValueError("upstream SSE event exceeds the size limit")

    def finish(self) -> None:
        if not self.done or not self.finished or self.pending.strip():
            raise ValueError("upstream stream closed before a complete terminal event")


class ResponsesStreamValidator:
    def __init__(self, max_event_bytes: int = 8 * 1024 * 1024) -> None:
        self.pending = b""
        self.terminal: str | None = None
        self.max_event_bytes = max_event_bytes

    def feed(self, chunk: bytes) -> None:
        self.pending = (self.pending + chunk).replace(b"\r\n", b"\n")
        while b"\n\n" in self.pending:
            block, self.pending = self.pending.split(b"\n\n", 1)
            if len(block) > self.max_event_bytes:
                raise ValueError("upstream Responses event exceeds the size limit")
            data = b"\n".join(
                line[5:].lstrip()
                for line in block.splitlines()
                if line.startswith(b"data:")
            )
            if not data:
                continue
            if data == b"[DONE]":
                if self.terminal is None:
                    raise ValueError("upstream Responses ended without terminal state")
                continue
            if self.terminal is not None:
                raise ValueError("upstream Responses sent data after terminal state")
            try:
                payload = json.loads(data)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(
                    "upstream Responses returned malformed SSE JSON"
                ) from exc
            if (
                not isinstance(payload, dict)
                or payload.get("error")
                or payload.get("type") == "error"
            ):
                raise ValueError("upstream Responses returned a streaming error")
            kind = payload.get("type")
            if kind in {"response.completed", "response.incomplete", "response.failed"}:
                response = payload.get("response") or {}
                if (
                    not isinstance(response, dict)
                    or response.get("status") != kind.split(".")[1]
                ):
                    raise ValueError(
                        "upstream Responses terminal status is inconsistent"
                    )
                self.terminal = kind
        if len(self.pending) > self.max_event_bytes:
            raise ValueError("upstream Responses event exceeds the size limit")

    def finish(self) -> None:
        if self.terminal is None or self.pending.strip():
            raise ValueError("upstream Responses stream closed without terminal state")
