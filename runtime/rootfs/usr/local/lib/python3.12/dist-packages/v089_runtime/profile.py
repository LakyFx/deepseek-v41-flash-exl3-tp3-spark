"""Sampled host-phase timings, no GPU synchronisation or inference request."""
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import time

_last={}
_loaded=time.monotonic()

def observe_phase(name, token_arg=None, batch_arg=None, rows_arg=None):
    def decorate(fn):
        @wraps(fn)
        def call(*args,**kwargs):
            # Cheap bypass when diagnostics were not requested.
            if not os.environ.get('DGX_V089_PROFILE_DIR'):
                return fn(*args,**kwargs)
            metadata={}
            if token_arg is not None:
                value=kwargs.get('num_tokens',args[token_arg] if len(args)>token_arg else None)
                if isinstance(value,int):metadata['num_tokens']=value
            if batch_arg is not None and len(args)>batch_arg:
                value=getattr(args[batch_arg],'num_tokens',None)
                if isinstance(value,int):metadata['num_tokens']=value
            if rows_arg is not None and len(args)>rows_arg:
                metadata['hash_rows']=sum(rel.numel() for table,rel,owned in args[rows_arg])
            with host_phase(name,metadata=metadata):
                return fn(*args,**kwargs)
        return call
    return decorate

@contextmanager
def host_phase(name,*,metadata=None):
    started=time.perf_counter()
    yield
    elapsed=time.perf_counter()-started
    now=time.monotonic()
    # Keep separate samples for prefill and decode rather than letting frequent
    # decode calls suppress the long-input observations we need.
    tokens=(metadata or {}).get('num_tokens')
    bucket='prefill' if tokens is not None and tokens>64 else 'small' if tokens is not None else 'unknown'
    key=(name,bucket)
    if now-_last.get(key,-float('inf'))<5:
        return
    _last[key]=now
    root=os.environ.get('DGX_V089_PROFILE_DIR')
    if not root:
        return
    path=Path(root)/f'host-rank{os.environ.get("DGX_DIAG_RANK","unknown")}.jsonl'
    try:
        if path.is_file() and path.stat().st_size>10*1024*1024:
            return  # bounded, no deletion/rotation of user evidence
        record={'schema':'dgx.v089.host-phase.v1','time':time.time(),'phase':name,
                'elapsed_s':elapsed,'pid':os.getpid(),'bucket':bucket,'module_age_s':now-_loaded}
        # Only numeric/safe classifications, never tokens, prompt text or output.
        record['metadata']={key:value for key,value in (metadata or {}).items()
                            if key in ('num_tokens','local_heads','layers','hash_rows') and isinstance(value,(int,float))}
        with path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
    except OSError:
        pass  # diagnostics must not fail inference
