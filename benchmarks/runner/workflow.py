"""Bounded interactive task runner, with an injected inference client.

Importing this module or constructing tasks never contacts the model.
Tool retries, reasoning-only answers and failed tasks remain measured work.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

from .fixture_eval import evaluate
from .fixtures import Task, initial_body


def cache_salt(campaign: str, name: str, concurrency: int, slot: int) -> str:
    # Deliberately exclude the version for paired cache/workload conditions.
    return hashlib.sha256(f"{campaign}:workflow:{name}:{concurrency}:{slot}".encode()).hexdigest()


def run_task(task: Task, client, *, campaign: str, concurrency: int, slot: int,
             deadline: float, context: str = "", max_turns: int = 5,
             output_tokens: int = 1536) -> dict:
    body = initial_body(task, context=context, output_tokens=output_tokens)
    body["cache_salt"] = cache_salt(campaign, task.name, concurrency, slot)
    started = time.monotonic()
    requests = []
    attempts = 0
    failure = None
    try:
        for turn in range(max_turns):
            if time.monotonic() >= deadline:
                raise TimeoutError("workflow deadline")
            attempts += 1
            call = client.complete(copy.deepcopy(body), deadline=deadline)
            reply = call["reply"]
            usage = reply.usage_counts()
            message = reply.message()
            requests.append({"turn": turn, "request_id": call["request_id"],
                             "body_sha256": call["body_sha256"], "usage": usage,
                             "started_monotonic": call["started_monotonic"],
                             "ended_monotonic": call["ended_monotonic"],
                             "first_any_delta_monotonic": reply.first_any_delta,
                             "first_visible_delta_monotonic": reply.first_visible_delta,
                             "sse_chunk_times": reply.delta_times,
                             "finish_reason": reply.finish_reason,
                             "assistant": message})
            body["messages"].append(message)
            calls = message.get("tool_calls", [])
            if not calls:
                # Empty visible output is never credited as a fast success.
                failure = "assistant_did_not_call_task_tools"
                break
            for tool in calls:
                arguments = json.loads(tool["function"]["arguments"])
                outcome = task.invoke(tool["function"]["name"], arguments, evaluate)
                body["messages"].append({"role": "tool", "tool_call_id": tool["id"],
                                         "content": json.dumps(outcome, ensure_ascii=False)})
                if task.finished:
                    if tool is not calls[-1]:
                        raise ValueError("tool call after task finish")
                    break
            if task.finished:
                break
        else:
            failure = "workflow_turn_limit"
    except (ValueError, KeyError, TypeError, TimeoutError, OSError) as exc:
        failure = type(exc).__name__ + ":" + str(exc)[:300]
    ended = time.monotonic()
    observed = {key: sum(row["usage"][key] for row in requests) for key in
                ("prompt_tokens", "completion_tokens", "computed_prompt_tokens", "cached_tokens")}
    coverage = len(requests) == attempts
    return {"case_id": task.name, "family": task.family, "regime": f"C{concurrency}",
            "correct": task.correct and task.finished and failure is None,
            "finished": task.finished, "error": failure,
            "started_monotonic": started, "ended_monotonic": ended,
            "elapsed_seconds": ended - started, "tool_calls": len(task.calls),
            "turns": len(requests), "requests": requests,
            "request_attempts": attempts, "usage_complete": coverage,
            "observed_usage": observed,
            "fixture_calls": task.calls, "final_result": task.final_result,
            "prompt_tokens": observed["prompt_tokens"] if coverage else None,
            "completion_tokens": observed["completion_tokens"] if coverage else None,
            "computed_prompt_tokens": observed["computed_prompt_tokens"] if coverage else None,
            "cached_tokens": observed["cached_tokens"] if coverage else None}
