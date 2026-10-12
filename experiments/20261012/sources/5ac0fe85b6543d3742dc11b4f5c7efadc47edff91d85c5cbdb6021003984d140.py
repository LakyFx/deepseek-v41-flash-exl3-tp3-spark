"""v087 diagnostic wrapper; unchanged v086 reclamation delegates."""
import functools, hashlib, inspect, os, time
import torch
from diaglog import emit, ROOT
from workspace_baseline import maintain, available
from workspace_baseline import reclaim_execute as baseline_execute
from workspace_baseline import reclaim_sample as baseline_sample

_KEYS=("allocated_bytes.all.current","reserved_bytes.all.current","active_bytes.all.current",
       "inactive_split_bytes.all.current","allocated_bytes.all.peak","reserved_bytes.all.peak",
       "num_alloc_retries","num_ooms","num_device_alloc","num_device_free")
_states={}

def stats(runner):
    d=torch.cuda.memory_stats(runner.device)
    return {k:d.get(k) for k in _KEYS}

def safe(fn,*args,**kwargs):
    try:return fn(*args,**kwargs)
    except Exception as exc:emit("diagnostic_error",operation=fn.__name__,error_type=type(exc).__name__)

def state(runner):
    key=(os.getpid(),id(runner))
    if key not in _states:
        _states[key]={"seq":0,"last":0,"snapshot_at":0,"retry":None,"oom":None,"history":False}
        # Bounded allocator history; Python callsites only, no tensor values.
        # Optional API failure explicitly recorded, never fail inference.
        try:
            torch.cuda.memory._record_memory_history(enabled="all",context="alloc",stacks="python",max_entries=2048,device=runner.device)
            _states[key]["history"]=True
            emit("history_enabled",max_entries=2048)
        except Exception as exc:emit("history_unavailable",error_type=type(exc).__name__)
    return _states[key]

def incident(runner,st,reason):
    if time.monotonic()-st["snapshot_at"]<60:return
    st["snapshot_at"]=time.monotonic()
    if not st["history"]:return
    snap=torch.cuda.memory._snapshot()
    # Only bounded allocator events and Python frames, no full segment dump.
    traces=[]
    for device,events in enumerate(snap.get("device_traces",[])):
        for e in events[-2048:]:
            traces.append(dict(device=device,**{k:v for k,v in e.items() if k in ("action","size","addr","stream","device_free","time_us","frames")}))
    # Chunk output records to bound queue item size. Snapshot construction can
    # itself cost CPU; cooldown60s and latency logged, never GPU synchronize.
    for off in range(0,len(traces),16):emit("allocator_history",reason=reason,events=traces[off:off+16])

def boundary(runner,st,kind,**fields):
    s=stats(runner)
    emit(kind,seq=st["seq"],memory=s,**fields)
    retry=s.get("num_alloc_retries");oom=s.get("num_ooms")
    trigger=ROOT/"allocation-trigger"
    trigger_id=trigger.read_text() if trigger.exists() else None
    changed=(st["retry"] is not None and (retry!=st["retry"] or oom!=st["oom"]))
    external=trigger_id is not None and trigger_id!=st.get("trigger")
    if changed or external:
        t=time.monotonic();safe(incident,runner,st,"driver_warning" if external else "allocator_retry")
        emit("incident_capture",elapsed_cpu_s=time.monotonic()-t)
    st.update(retry=retry,oom=oom,trigger=trigger_id)

def reclaim_execute(fn):
    original=baseline_execute(fn);signature=inspect.signature(fn)
    @functools.wraps(fn)
    def wrapped(self,scheduler_output,*args,**kwargs):
        bound=signature.bind(self,scheduler_output,*args,**kwargs)
        if bound.arguments.get("dummy_run",False) or bound.arguments.get("is_profile",False):
            return original(self,scheduler_output,*args,**kwargs)
        st=safe(state,self)
        if st is None:return original(self,scheduler_output,*args,**kwargs)
        st["seq"]+=1
        prefilling=any(n>1+self.num_speculative_steps for n in scheduler_output.num_scheduled_tokens.values())
        sampled=prefilling or time.monotonic()-st["last"]>=1
        data={"prefill_heuristic":prefilling,"scheduled_tokens":scheduler_output.total_num_scheduled_tokens,
              "scheduled_requests":len(scheduler_output.num_scheduled_tokens),
              "tokens_per_request":sorted(scheduler_output.num_scheduled_tokens.values()),
              "gpu_elapsed":None,"measurement":"CPU_call_includes_possible_wait_not_GPU_time"}
        if sampled:
            st["last"]=time.monotonic();safe(boundary,self,st,"execute_begin",**data)
        t=time.monotonic()
        try:return original(self,scheduler_output,*args,**kwargs)
        except BaseException as exc:
            emit("execute_exception",seq=st["seq"],error_type=type(exc).__name__)
            safe(incident,self,st,"execute_exception");raise
        finally:
            if sampled:safe(boundary,self,st,"execute_end",host_elapsed_s=time.monotonic()-t,**data)
    return wrapped

def reclaim_sample(fn):
    original=baseline_sample(fn)
    @functools.wraps(fn)
    def wrapped(self,*args,**kwargs):
        t=time.monotonic()
        try:return original(self,*args,**kwargs)
        finally:
            st=_states.get((os.getpid(),id(self)))
            if st and time.monotonic()-st.get("sample_at",0)>=1:
                st["sample_at"]=time.monotonic()
                safe(boundary,self,st,"sample_end",host_elapsed_s=time.monotonic()-t)
    return wrapped
