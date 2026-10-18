"""Short-lived, bounded grammar worker. No gateway or model imports."""

import json
import re
import sys


def main():
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (384 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    data = json.loads(sys.stdin.buffer.read(8 * 1024 * 1024 + 1))
    fmt = data["format"]
    definition = fmt["definition"]
    if not isinstance(definition, str) or len(definition) > 65536:
        raise ValueError("grammar size limit")
    value = data["input"]
    if fmt["syntax"] == "regex":
        import regex

        grammar = regex.compile(definition)
        if value is not None and grammar.fullmatch(value, timeout=2) is None:
            raise ValueError("input does not match regex")
    else:
        from lark import Lark

        # Only Lark's packaged common terminals may be imported. Never resolve
        # caller-selected files or modules from the gateway filesystem.
        for imported in re.findall(r"%import\s+([^\s]+)", definition):
            if not imported.startswith("common.") or ".." in imported:
                raise ValueError("external grammar imports are forbidden")
        # Codex's published apply_patch grammar uses /(.*)/ for optional line
        # contents. Python Lark forbids zero-width lexer tokens. Express the
        # same language as an optional nonempty token, leaving quoted literals
        # untouched. This is a grammar normalization, never an input rewrite.
        grammar_tokens = re.compile(
            r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/(?:\\.|[^/\\\n])*/[a-z]*'
        )
        definition = grammar_tokens.sub(
            lambda match: {"/(.*)/": "/(.+)/?", "/.*/": "/.+/?"}.get(
                match.group(), match.group()
            ),
            definition,
        )
        grammar = Lark(definition, parser="earley", lexer="dynamic")
        if value is not None:
            grammar.parse(value)
    sys.stdout.buffer.write(b"valid")


if __name__ == "__main__":
    main()
