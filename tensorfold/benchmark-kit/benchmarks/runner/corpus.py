"""Hash-bound loading of prepared private corpora; no network or inference."""
import hashlib
import json


def checked(root, filename, expected):
    path = root / filename
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("unsafe corpus path")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("frozen corpus bytes changed: " + filename)
    return raw


def load_workloads(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["schema"] != "dgx.frozen-workload-corpus.v1" or len(value["cases"]) != 16:
        raise ValueError("frozen sixteen-case workload corpus required")
    rows = {}
    for metadata in value["cases"]:
        row = dict(metadata)
        row["body"] = json.loads(checked(path.parent, row["request_file"], row["request_sha256"]))
        context_raw = checked(path.parent, row["context_file"], row["context_sha256"])
        row["context"] = json.loads(context_raw) if row.get('context_encoding') == 'json-string' else context_raw.decode()
        proof = json.loads(checked(path.parent, row["tokenization_receipt_file"], row["tokenization_receipt_sha256"]))
        if proof["count"] != row["measured_prompt_tokens"] or proof["request_sha256"] != row["request_sha256"]:
            raise ValueError("workload tokenization proof differs")
        if row["id"] in rows:
            raise ValueError("duplicate workload id")
        rows[row["id"]] = row
    return rows


def load_legacy(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    historical = hashlib.sha256(raw).hexdigest() == "bd4a01e9deae8808ca14ce59045202466e094e7bcf078cccc6b467e193957244"
    if not historical and value.get('schema') != 'dgx.public-legacy-surrogate.v1':
        raise ValueError('historical corpus identity or public surrogate schema required')
    groups = {"code": [], "prose": []}
    for metadata in value["cases"]:
        body = json.loads(checked(path.parent, metadata["request_file"], metadata["sha256"]))
        proof = json.loads(checked(path.parent, metadata["tokenizer_receipt_file"], metadata["tokenizer_receipt_sha256"]))
        if proof["verified_prompt_tokens"] != metadata["prompt_tokens"]:
            raise ValueError("legacy count receipt differs")
        if metadata["effective_profile"] != "max" or body.get("chat_template_kwargs", {}).get("reasoning_effort") != "max":
            raise ValueError("legacy effective main profile changed")
        groups[metadata["category"]].append({"id": metadata["id"], "body": body,
            "measured_prompt_tokens": metadata["prompt_tokens"],
            "tokenization_receipt_sha256": metadata["tokenizer_receipt_sha256"]})
    if any(len(rows) != 4 for rows in groups.values()):
        raise ValueError("four original code and four prose entries required")
    return groups
