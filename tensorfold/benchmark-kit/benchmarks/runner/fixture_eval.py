"""Semantic evaluator for the pure-function coding fixture, without inference."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

try:
    from .fixtures import CODE_INPUTS
except ImportError:
    if __name__ == "__main__" and sys.argv[1:] == ["--worker"]:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fixtures import CODE_INPUTS

ALLOWED_NODES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.AugAssign, ast.AnnAssign, ast.If, ast.For, ast.Expr, ast.Pass, ast.Break,
    ast.Continue, ast.Name, ast.Load, ast.Store, ast.Constant, ast.Dict, ast.List,
    ast.Tuple, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
    ast.comprehension, ast.Subscript, ast.Slice, ast.Attribute, ast.Call,
    ast.keyword, ast.Lambda, ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp,
    ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd, ast.IfExp,
}
ALLOWED_ATTRIBUTES = {"get", "values", "items", "keys", "copy", "append", "extend",
                      "add", "setdefault", "update", "sort"}
BUILTINS = {name: getattr(__import__("builtins"), name) for name in (
    "max", "min", "len", "sorted", "range", "enumerate", "zip", "isinstance",
    "dict", "list", "tuple", "set", "abs", "any", "all", "sum", "bool", "int",
    "str", "float", "reversed")}


def validate(source: str) -> ast.Module:
    if not isinstance(source, str) or len(source.encode()) > 12000:
        raise ValueError("fixture source size limit")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise ValueError("unsupported pure-function syntax: " + type(node).__name__)
        if isinstance(node, ast.Attribute) and node.attr not in ALLOWED_ATTRIBUTES:
            raise ValueError("unsupported fixture attribute")
        if isinstance(node, (ast.Name, ast.arg)):
            name = node.id if isinstance(node, ast.Name) else node.arg
            if "__" in name:
                raise ValueError("invalid fixture name")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "merge_events":
        raise ValueError("one merge_events function required")
    for node in tree.body:
        if node is not functions[0] and not (isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            raise ValueError("no module-level execution")
    if functions[0].decorator_list:
        raise ValueError("fixture decorators unsupported")
    return tree


def oracle(events: list[dict]) -> list[dict]:
    # Independent grouping/order implementation, not the proposed fix.
    names = sorted({event["request_id"] for event in events})
    answer = []
    for name in names:
        selected = max(((event["timestamp"], index, event) for index, event in enumerate(events)
                        if event["request_id"] == name), key=lambda item: item[:2])
        answer.append(copy.deepcopy(selected[2]))
    return answer


def worker(source: str) -> dict:
    if sys.platform != "win32":
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**2, 128 * 1024**2))
    tree = validate(source)
    namespace = {"__builtins__": BUILTINS}
    exec(compile(tree, "events.py", "exec"), namespace)
    failures = []
    for index, events in enumerate(CODE_INPUTS):
        values = copy.deepcopy(events)
        actual = namespace["merge_events"](values)
        if actual != oracle(events) or values != events:
            failures.append(index)
    return {"passed": not failures, "tests": len(CODE_INPUTS), "failed_cases": failures}


def evaluate(source: str) -> dict:
    try:
        validate(source)
        result = subprocess.run([sys.executable, "-I", "-S", str(Path(__file__).resolve()),
                                 "--worker"], input=json.dumps({"source": source}),
                                text=True, capture_output=True, timeout=4)
        if result.returncode != 0:
            return {"passed": False, "error": "fixture evaluator failed"}
        return json.loads(result.stdout)
    except (ValueError, SyntaxError, subprocess.TimeoutExpired) as exc:
        return {"passed": False, "error": type(exc).__name__}


if __name__ == "__main__" and sys.argv[1:] == ["--worker"]:
    # Isolated mode removes the script directory from sys.path; this fixed
    # directory is needed only for the benchmark's authored fixtures module.
    try:
        print(json.dumps(worker(json.load(sys.stdin)["source"])))
    except Exception:
        print(json.dumps({"passed": False, "error": "fixture execution error"}))
