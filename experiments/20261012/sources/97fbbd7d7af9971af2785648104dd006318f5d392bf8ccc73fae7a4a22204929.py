"""Opt-in CUDA workspace reclamation at worker step boundaries, never live KV."""
import functools, logging, os, time
from pathlib import Path
import torch
from vllm.logger import init_logger
log=init_logger("vllm.dgx_workspace_memory")
SLACK=256*1024**2
LOW=250*1024**2

def available():
    return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024

def maintain(runner, phase, force=False):
    if os.environ.get('DGX_WORKSPACE_OBSERVE')!='1':return
    now=time.monotonic()
    if not force and now-getattr(runner,'_dgx_memory_check_at',0)<1:return
    runner._dgx_memory_check_at=now
    with torch.cuda.device(runner.device):
        if torch.cuda.is_current_stream_capturing():return
        free=available();allocated=torch.cuda.memory_allocated(runner.device)
        reserved=torch.cuda.memory_reserved(runner.device)
        released=0
        if os.environ.get('DGX_WORKSPACE_RECLAIM')=='1' and reserved-allocated>SLACK and free<LOW:
            torch.cuda.empty_cache()
            released=reserved-torch.cuda.memory_reserved(runner.device)
        if released or now-getattr(runner,'_dgx_memory_log_at',0)>=5:
            runner._dgx_memory_log_at=now
            log.info('DGX workspace phase=%s available=%d allocated=%d reserved=%d released=%d available_after=%d',phase,free,allocated,reserved,released,available())

def reclaim_execute(fn):
    import inspect
    signature=inspect.signature(fn)
    @functools.wraps(fn)
    def wrapped(self, scheduler_output, *args, **kwargs):
        # Signature binds also cover positional dummy/profile flags.
        bound=signature.bind(self,scheduler_output,*args,**kwargs)
        if bound.arguments.get('dummy_run',False) or bound.arguments.get('is_profile',False):
            return fn(self,scheduler_output,*args,**kwargs)
        prefilling=any(n>1+self.num_speculative_steps for n in scheduler_output.num_scheduled_tokens.values())
        maintain(self,'before_prefill' if prefilling else 'before_decode',force=prefilling)
        result=fn(self,scheduler_output,*args,**kwargs)
        # fn stack is unwound. Returned tensors and execute_model_state stay live.
        maintain(self,'after_prefill' if prefilling else 'after_decode',force=prefilling)
        return result
    return wrapped

def reclaim_sample(fn):
    @functools.wraps(fn)
    def wrapped(self,*args,**kwargs):
        result=fn(self,*args,**kwargs)
        maintain(self,'after_sample')
        return result
    return wrapped
