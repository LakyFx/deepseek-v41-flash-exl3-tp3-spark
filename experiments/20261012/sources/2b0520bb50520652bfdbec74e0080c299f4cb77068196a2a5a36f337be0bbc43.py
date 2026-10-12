"""Bounded metadata only: feature counters, no prompts, IDs or generated text."""
import json
import os
from pathlib import Path
import time

_counts = {}
_last = 0.0

def observe(event, **values):
    global _last
    _counts[event] = _counts.get(event, 0) + 1
    now = time.monotonic()
    if now - _last < 5:
        return
    _last = now
    root = os.environ.get('DGX_V089X_PROFILE_DIR')
    if not root:
        return
    path = Path(root) / ('features-rank'+os.environ.get('DGX_DIAG_RANK','unknown')+'.jsonl')
    try:
        if path.exists() and path.stat().st_size >= 10*1024*1024:
            return
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'schema':'dgx.v089x.features.v1','time':time.time(),
                'counts':dict(_counts),'event':event,
                'values':{k:v for k,v in values.items() if isinstance(v,(int,float))}})+'\n')
    except OSError:
        pass
