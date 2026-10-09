"""Deterministic tool environments for short, scored Hermes-like workflows.

All fixtures are authored for the benchmark. No live application, shell or
external network tool is available. Model requests retain main/max reasoning.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field


def function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}}


FINISH = function("finish_task", "Submit the final verified task result.",
                  {"result": {"type": "object"}}, ["result"])
READ = function("read_file", "Read one file from the isolated task fixture.",
                {"path": {"type": "string"}}, ["path"])
REPLACE = function("replace_file", "Replace one Python fixture source file.",
                   {"path": {"type": "string"}, "content": {"type": "string"}},
                   ["path", "content"])
TEST = function("run_tests", "Run the fixture's semantic tests; no shell arguments.", {}, [])
QUERY = function("query_records", "Read a named immutable data page, indexed from zero.",
                 {"collection": {"type": "string"}, "page": {"type": "integer", "minimum": 0}},
                 ["collection", "page"])
DOC = function("read_document", "Read one named deployment document.",
               {"name": {"type": "string"}}, ["name"])

BUGGY_CODE = '''def merge_events(events):
    """Newest event per request_id, ordered by id; ties use last occurrence.

    Keep distinct ids even if timestamps match. Do not mutate input events.
    """
    latest = {}
    for event in events:
        latest[event["timestamp"]] = event
    return sorted(latest.values(), key=lambda event: event["request_id"])
'''

GOLDEN_CODE = '''def merge_events(events):
    latest = {}
    for event in events:
        previous = latest.get(event["request_id"])
        if previous is None or event["timestamp"] >= previous["timestamp"]:
            latest[event["request_id"]] = dict(event)
    return sorted(latest.values(), key=lambda event: event["request_id"])
'''

CODE_INPUTS = [
    [],
    [{"request_id": "a", "timestamp": 1, "status": "done"}],
    [{"request_id": "b", "timestamp": 5, "status": "done"},
     {"request_id": "a", "timestamp": 5, "status": "running"}],
    [{"request_id": "a", "timestamp": 9, "status": "done"},
     {"request_id": "a", "timestamp": 4, "status": "queued"}],
    [{"request_id": "a", "timestamp": 9, "status": "running"},
     {"request_id": "a", "timestamp": 9, "status": "done"}],
    [{"request_id": "c", "timestamp": -1, "status": "queued"},
     {"request_id": "b", "timestamp": 0, "status": "done"},
     {"request_id": "c", "timestamp": 2, "status": "failed"}],
]

USAGE_PAGES = [
    [{"id": "r1", "time": 1, "backend": "ds41", "status": "running", "prompt": 8000, "completion": 20},
     {"id": "r2", "time": 2, "backend": "ds41", "status": "done", "prompt": 16000, "completion": 400},
     {"id": "r3", "time": 3, "backend": "other", "status": "done", "prompt": 9000, "completion": 500}],
    [{"id": "r1", "time": 4, "backend": "ds41", "status": "done", "prompt": 8000, "completion": 200},
     {"id": "r4", "time": 5, "backend": "ds41", "status": "failed", "prompt": 2000, "completion": 50},
     {"id": "r2", "time": 2, "backend": "ds41", "status": "done", "prompt": 16000, "completion": 400},
     {"id": "r5", "time": 6, "backend": "ds41", "status": "done", "prompt": 32000, "completion": 600}],
]

DOCUMENTS = {
    "live": {"observed": "2026-10-08T10:00:00Z", "version": "0.8.9X", "tp": 3,
             "kv_tokens": 3299563, "reasoning": "max", "adaptive": False,
             "image": "original-image", "source_mounts": "x-release"},
    "candidate": {"observed": "2026-10-08T11:00:00Z", "version": "0.8.9X11",
                  "adaptive": True, "tested": False, "deployed": False,
                  "description": "Prepared proposal; this is not the observed running version."},
    "rollback": {"version": "0.8.9X", "restore_units": "captured-unit-bytes",
                 "restore_source": "x-release", "restore_image": "original-image",
                 "kv_restore": False},
}


@dataclass
class Task:
    name: str
    family: str
    instruction: str
    tools: list[dict]
    files: dict = field(default_factory=dict)
    documents: dict = field(default_factory=dict)
    pages: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    finished: bool = False
    correct: bool = False
    final_result: dict | None = None
    last_tests: dict | None = None

    def invoke(self, name: str, arguments: dict, evaluate_code) -> dict:
        allowed = {tool["function"]["name"] for tool in self.tools}
        if name not in allowed or not isinstance(arguments, dict) or self.finished:
            raise ValueError("invalid fixture tool call")
        spec = next(tool["function"]["parameters"] for tool in self.tools
                    if tool["function"]["name"] == name)
        if set(arguments) != set(spec["required"]):
            raise ValueError("tool arguments do not match schema")
        self.calls.append({"name": name, "arguments": copy.deepcopy(arguments)})
        if name == "read_file":
            if arguments["path"] not in self.files:
                raise ValueError("unknown fixture file")
            return {"path": arguments["path"], "content": self.files[arguments["path"]]}
        if name == "replace_file":
            path, content = arguments["path"], arguments["content"]
            if path not in self.files or not isinstance(content, str) or len(content.encode()) > 12000:
                raise ValueError("invalid fixture replacement")
            self.files[path] = content
            self.last_tests = None
            return {"written": path}
        if name == "run_tests":
            self.last_tests = evaluate_code(self.files["events.py"])
            return self.last_tests
        if name == "query_records":
            page = arguments["page"]
            if arguments["collection"] != "usage" or type(page) is not int or not 0 <= page < len(self.pages):
                raise ValueError("unknown fixture records page")
            return {"records": copy.deepcopy(self.pages[page]), "page": page,
                    "next_page": page + 1 if page + 1 < len(self.pages) else None}
        if name == "read_document":
            if arguments["name"] not in self.documents:
                raise ValueError("unknown fixture document")
            return copy.deepcopy(self.documents[arguments["name"]])
        if name == "finish_task":
            value = arguments["result"]
            if not isinstance(value, dict):
                raise ValueError("final result must be an object")
            self.final_result = copy.deepcopy(value)
            self.finished = True
            self.correct = self._grade(value)
            return {"accepted": self.correct}
        raise AssertionError("unhandled fixture tool")

    def _grade(self, value: dict) -> bool:
        before_finish = self.calls[:-1]
        if self.family == "code":
            return bool(self.last_tests and self.last_tests.get("passed") is True
                        and any(call["name"] == "replace_file" for call in before_finish)
                        and value.get("tests_passed") is True)
        if self.family == "tool_data":
            read_pages = {call["arguments"]["page"] for call in before_finish
                          if call["name"] == "query_records"}
            expected = {"requests": 3, "prompt_tokens": 56000, "completion_tokens": 1200}
            return read_pages == {0, 1} and set(value) == set(expected) and all(
                type(value[key]) is int and value[key] == wanted for key, wanted in expected.items())
        if self.family == "analysis":
            read_docs = {call["arguments"]["name"] for call in before_finish
                         if call["name"] == "read_document"}
            expected = {"live_version": "0.8.9X", "live_adaptive": False,
                        "restore_kv": False, "live_kv_tokens": 3299563}
            return read_docs == set(DOCUMENTS) and set(value) == set(expected) and all(
                type(value[key]) is type(wanted) and value[key] == wanted
                for key, wanted in expected.items())
        raise ValueError("unknown task family")


def make_tasks() -> list[Task]:
    return [
        Task("code-event-dedup", "code",
             "Fix events.py. merge_events must keep the newest event per request_id, preserve distinct "
             "ids at equal timestamps, use the last occurrence on ties, sort by request_id and never "
             "mutate its input. Read the file, replace the corrected complete file, run tests and call "
             "finish_task with result {tests_passed: true} only after they pass. Keep the patch concise.",
             [READ, REPLACE, TEST, FINISH], files={"events.py": BUGGY_CODE}),
        Task("tool-usage-accounting", "tool_data",
             "Audit all usage pages using query_records(collection=usage,page=0), then follow next_page. "
             "For each id use only its newest event. Count only status=done and backend=ds41. "
             "Duplicate identical events count once. Submit finish_task result with exactly the integer "
             "fields requests, prompt_tokens and completion_tokens. Do not add together retry snapshots.",
             [QUERY, FINISH], pages=copy.deepcopy(USAGE_PAGES)),
        Task("analysis-deployment-evidence", "analysis",
             "Read live, candidate and rollback documents. Distinguish observed deployment from an "
             "untested proposal even when the proposal has a later timestamp. Submit finish_task with "
             "exact fields live_version(string), live_adaptive(boolean), restore_kv(boolean), "
             "live_kv_tokens(integer). Do not make any deployment or cache request.",
             [DOC, FINISH], documents=copy.deepcopy(DOCUMENTS)),
    ]


def initial_body(task: Task, *, context: str = "", output_tokens: int = 1536) -> dict:
    return {"model": "deepseek-v4.1-flash", "reasoning_effort": "max",
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"},
            "temperature": 0.6, "seed": 42, "max_tokens": output_tokens,
            "tools": copy.deepcopy(task.tools), "tool_choice": "auto",
            "messages": [{"role": "system", "content":
                "You are an engineering assistant working on a local TP3 inference service. "
                "Use only the task tools, verify results, then submit finish_task. "
                "The reference dossier is context; current task tools are authoritative.\n" + context},
                {"role": "user", "content": task.instruction}]}
