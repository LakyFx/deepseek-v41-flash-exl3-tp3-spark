"""Preserve Responses custom tools over a function-only native renderer.

Only declared custom tools are wrapped. Function arguments are validated before
any completed custom input reaches the client; native reasoning stays native.
"""

import asyncio
import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path


class CustomTools:
    def __init__(
        self, body, *, normalize_stream_ids=False, grammar_gate=None, chat_adapter=False
    ):
        self.original_tools = deepcopy(body.get("tools") or [])
        if not isinstance(self.original_tools, list):
            raise TypeError("tools must be an array")
        self.original_choice = deepcopy(body.get("tool_choice"))
        self.specs = {}
        self.chat_adapter = chat_adapter
        self.chat_names = {}
        for tool in self.original_tools:
            if not isinstance(tool, dict):
                raise TypeError("tool declaration must be an object")
            namespace = tool.get("name") if tool.get("type") == "namespace" else None
            children = tool.get("tools", []) if namespace else [tool]
            if not isinstance(children, list) or any(
                not isinstance(child, dict) for child in children
            ):
                raise ValueError("namespace tools must be an array of objects")
            for child in children:
                if namespace:
                    alias = self.namespace_alias(namespace, child["name"])
                    self.chat_names[alias] = (namespace, child["name"])
                if child.get("type") == "custom":
                    key = (namespace, child["name"])
                    if key in self.specs:
                        raise ValueError("duplicate custom tool declaration")
                    self.specs[key] = child
        self.items = {}
        self.arguments = {}
        self.validated_inputs = {}
        self.sequence = 0
        self.normalize_stream_ids = normalize_stream_ids
        self.completed_items = {}
        self.grammar_gate = grammar_gate or asyncio.Semaphore(2)
        self.argument_bytes = 0
        self.incomplete_ids = set()

    @staticmethod
    def namespace_alias(namespace, name):
        digest = hashlib.sha256(json.dumps([namespace, name]).encode()).hexdigest()[:40]
        return "gw_ns_" + digest

    def canonical_item(self, item):
        if (
            self.chat_adapter
            and item.get("type") == "function_call"
            and item.get("name") in self.chat_names
        ):
            namespace, name = self.chat_names[item["name"]]
            return {**item, "namespace": namespace, "name": name}
        return item

    @staticmethod
    def incomplete_input(raw):
        """Expose only a decodable raw-input prefix, never a completed call."""
        match = re.match(r'^\s*\{\s*"input"\s*:\s*(".*)', raw, re.DOTALL)
        if not match:
            return ""
        value = match[1]
        try:
            result, _ = json.JSONDecoder().raw_decode(value)
            result.encode("utf-8")
            return result
        except (json.JSONDecodeError, UnicodeError):
            for trim in range(min(12, len(value)) + 1):
                try:
                    candidate = (value[:-trim] if trim else value) + '"'
                    result = json.loads(candidate)
                    result.encode("utf-8")
                    return result
                except (ValueError, UnicodeError):
                    continue
        return ""

    def stable_response_ids(self, response):
        """Native vLLM rebuilds final output IDs; keep identities already emitted."""
        result = deepcopy(response)
        output = result.get("output", [])
        if result.get("status") == "completed" and any(
            index < 0 or index >= len(output) for index in self.completed_items
        ):
            raise ValueError(
                "native final output does not match completed stream items"
            )
        for index, item in enumerate(output):
            previous = self.completed_items.get(index)
            if previous is None:
                continue
            if item.get("type") != previous.get("type"):
                raise ValueError("native final output changed streamed item type")
            if item.get("type") == "function_call":
                for field in ("name", "namespace", "arguments"):
                    if item.get(field) != previous.get(field):
                        raise ValueError(
                            "native final tool call changed streamed content"
                        )
            for field in ("id", "call_id"):
                if previous.get(field) is not None:
                    item[field] = previous[field]
        return result

    def request(self, body):
        result = deepcopy(body)

        def declaration(tool):
            if tool.get("type") == "namespace":
                tool["tools"] = [declaration(child) for child in tool["tools"]]
            elif tool.get("type") == "custom":
                description = tool.get("description", "")
                fmt = tool.get("format") or {"type": "text"}
                if fmt.get("type") == "grammar":
                    description += (
                        "\nThe input string must follow this grammar:\n"
                        + fmt["definition"]
                    )
                tool = {
                    "type": "function",
                    "name": tool["name"],
                    "description": description
                    + "\nTransport: put the complete raw tool input in the input string field; preserve newlines exactly.",
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            return tool

        if "tools" in result:
            result["tools"] = [declaration(tool) for tool in result.get("tools") or []]
        choice = result.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") == "custom":
            choice["type"] = "function"
        if self.chat_adapter:
            flattened = []
            for tool in result.get("tools", []):
                if tool.get("type") != "namespace":
                    if tool.get("name") in self.chat_names:
                        raise ValueError(
                            "namespace tool alias collides with a declared tool"
                        )
                    flattened.append(tool)
                    continue
                for child in tool["tools"]:
                    child["description"] = (
                        f"Namespace {tool['name']}: {tool.get('description') or ''}. Tool {child['name']}. "
                        + str(child.get("description") or "")
                    )
                    child["name"] = self.namespace_alias(tool["name"], child["name"])
                    flattened.append(child)
            if "tools" in result:
                result["tools"] = flattened
            if isinstance(result.get("input"), list):
                for history in result["input"]:
                    if (
                        isinstance(history, dict)
                        and history.get("type") == "function_call"
                        and history.get("namespace")
                    ):
                        history["name"] = self.namespace_alias(
                            history.pop("namespace"), history["name"]
                        )
            if (
                isinstance(choice, dict)
                and choice.get("type") == "function"
                and choice.get("namespace")
            ):
                choice["name"] = self.namespace_alias(
                    choice.pop("namespace"), choice["name"]
                )
        return result

    async def validate(self, spec, value=None):
        fmt = spec.get("format") or {"type": "text"}
        if fmt.get("type") == "text":
            return
        if fmt.get("type") != "grammar" or fmt.get("syntax") not in {"lark", "regex"}:
            raise ValueError("unsupported custom tool input format")
        data = json.dumps({"format": fmt, "input": value}).encode()
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("custom grammar validation input exceeds size limit")
        async with self.grammar_gate:
            await self._grammar_worker(data)

    async def _grammar_worker(self, data):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).with_name("custom_tools_grammar.py")),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            async with asyncio.timeout(5):
                output, _ = await process.communicate(data)
            if process.returncode or output != b"valid":
                raise ValueError("custom tool input or grammar failed validation")
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    async def validate_declarations(self):
        for spec in self.specs.values():
            await self.validate(spec)

    async def item(self, item, *, added=False, incomplete=False):
        item = self.canonical_item(item)
        key = (item.get("namespace"), item.get("name"))
        if item.get("type") != "function_call" or key not in self.specs:
            return item
        result = deepcopy(item)
        raw = result.pop("arguments", "")
        value = ""
        if incomplete:
            value = self.incomplete_input(raw)
            result["status"] = "incomplete"
        elif not added:
            envelope = json.loads(raw)
            if (
                not isinstance(envelope, dict)
                or set(envelope) != {"input"}
                or not isinstance(envelope["input"], str)
            ):
                raise ValueError("invalid custom tool function envelope")
            value = envelope["input"]
            value.encode("utf-8")
            item_id = result.get("id")
            if item_id in self.validated_inputs:
                if self.validated_inputs[item_id] != value:
                    raise ValueError(
                        "custom tool final input disagrees with emitted input"
                    )
            else:
                await self.validate(self.specs[key], value)
                if item_id is not None:
                    self.validated_inputs[item_id] = value
        result["type"] = "custom_tool_call"
        result["input"] = value
        return result

    async def response(self, data):
        result = deepcopy(data)
        if result.get("status") == "completed" and self.incomplete_ids:
            raise ValueError(
                "native completed response contains unfinished custom input"
            )
        result["output"] = [
            await self.item(
                item,
                incomplete=result.get("status")
                in {"incomplete", "failed", "cancelled"},
            )
            for item in result.get("output", [])
        ]
        if "tools" in result:
            result["tools"] = deepcopy(self.original_tools)
        if "tool_choice" in result and self.original_choice is not None:
            result["tool_choice"] = deepcopy(self.original_choice)
        return result

    async def frame(self, frame):
        """Consume one complete SSE frame; custom JSON deltas remain internal."""
        if not self.specs and not self.normalize_stream_ids:
            return [frame]
        raw = b"\n".join(
            line[5:].lstrip()
            for line in frame.splitlines()
            if line.startswith(b"data:")
        )
        if not raw or raw == b"[DONE]":
            return [frame]
        event = json.loads(raw)
        kind = event.get("type", "")
        events = [event]
        if self.normalize_stream_ids and kind == "response.output_item.done":
            self.completed_items[event["output_index"]] = deepcopy(event["item"])
        if kind in {"response.output_item.added", "response.output_item.done"}:
            original = self.canonical_item(event["item"])
            event["item"] = original
            key = (original.get("namespace"), original.get("name"))
            if original.get("type") == "function_call" and key in self.specs:
                item_id = original["id"]
                self.items[item_id] = original
                event["item"] = await self.item(
                    original,
                    added=kind.endswith(".added"),
                    incomplete=item_id in self.incomplete_ids
                    or original.get("status") == "incomplete",
                )
        elif (
            kind.startswith("response.function_call_arguments.")
            and event.get("item_id") in self.items
        ):
            item_id = event["item_id"]
            if kind.endswith(".delta"):
                self.argument_bytes += len(event["delta"].encode("utf-8"))
                if self.argument_bytes > 8 * 1024 * 1024:
                    raise ValueError("custom tool stream exceeds argument size limit")
                self.arguments.setdefault(item_id, []).append(event["delta"])
                return []
            if kind.endswith(".done"):
                raw_arguments = event["arguments"]
                fragments = self.arguments.pop(item_id, None)
                if fragments is not None and "".join(fragments) != raw_arguments:
                    raise ValueError(
                        "custom tool argument deltas disagree with final input"
                    )
                item = {**self.items[item_id], "arguments": raw_arguments}
                try:
                    mapped = await self.item(item)
                except json.JSONDecodeError:
                    self.incomplete_ids.add(item_id)
                    return []
                base = {key: event[key] for key in ("item_id", "output_index")}
                events = [
                    {
                        **base,
                        "type": "response.custom_tool_call_input.delta",
                        "delta": mapped["input"],
                    },
                    {
                        **base,
                        "type": "response.custom_tool_call_input.done",
                        "input": mapped["input"],
                    },
                ]
        if isinstance(event.get("response"), dict):
            if self.normalize_stream_ids and kind in {
                "response.completed",
                "response.failed",
                "response.incomplete",
            }:
                event["response"] = self.stable_response_ids(event["response"])
            event["response"] = await self.response(event["response"])
        frames = []
        for current in events:
            current["sequence_number"] = self.sequence
            self.sequence += 1
            frames.append(
                b"event: "
                + current["type"].encode()
                + b"\ndata: "
                + json.dumps(
                    current, ensure_ascii=False, separators=(",", ":")
                ).encode()
                + b"\n\n"
            )
        return frames
