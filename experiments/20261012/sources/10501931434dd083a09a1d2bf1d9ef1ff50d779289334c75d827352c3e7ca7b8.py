"""Load an exact precompiled RoCE specialization; no runtime compilation.

The selected manifest is bound into an immutable candidate environment. A
missing object, dependency mismatch, or corrupt binary fails startup. Loading
does not invoke the collective; the existing Sparknet lifecycle owns that.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def resolve(root, digest, name, key, *, version=importlib.metadata.version):
    root=Path(root).resolve()
    raw=(root/"AOT-MANIFEST.json").read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError("RoCE AOT manifest changed")
    manifest=json.loads(raw)
    if manifest.get("schema")!="dgx.cute-aot-objects.v1" or manifest.get("arch")!="sm_121a":
        raise ValueError("RoCE AOT target differs")
    expected=manifest.get("dependency_versions",{})
    if set(expected)!={"nvidia-cutlass-dsl","cuda-bindings","torch"}:
        raise ValueError("RoCE AOT toolchain receipt missing")
    for package,value in expected.items():
        if version(package)!=value:
            raise ValueError("RoCE AOT dependency changed: "+package)
    matches=[r for r in manifest.get("objects",[]) if r.get("name")==name and r.get("cache_key")==list(key)]
    if len(matches)!=1:
        raise ValueError("RoCE AOT specialization missing or ambiguous")
    row=matches[0]
    prefix="dgx_"+hashlib.sha256(json.dumps([name,list(key)],separators=(",",":")).encode()).hexdigest()[:24]
    if row.get("prefix")!=prefix or row.get("file")!="objects/"+prefix+".o":
        raise ValueError("RoCE AOT object identity differs")
    path=(root/row["file"]).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=row.get("sha256"):
        raise ValueError("RoCE AOT object changed")
    return path,prefix


@functools.cache
def load_required(name,cache_key):
    root=os.environ.get("DGX_CAMPAIGN_R_AOT_ROOT")
    digest=os.environ.get("DGX_CAMPAIGN_R_AOT_SHA256")
    if not root or not digest:
        raise ValueError("RoCE AOT binding required; runtime compilation disabled")
    path,prefix=resolve(root,digest,name,cache_key)
    from cutlass.cute.runtime import load_module
    module=load_module(str(path))
    fn=module[prefix]
    if not fn.load_from_binary or fn.engine is None or fn.capi_func is None:
        raise ValueError("RoCE AOT object did not load")
    # Retain the module alongside its function for the process lifetime.
    fn._dgx_campaign_external_module=module
    return fn
