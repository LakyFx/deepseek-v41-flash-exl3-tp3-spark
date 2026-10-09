"""Bounded async JSONL. Dedicated root only; no prompts/tensors/credentials."""
import json, os, queue, threading, time
from pathlib import Path
ROOT=Path(os.environ.get("DGX_DIAG_DIR", "/dgx-diag"))
RETENTION=48*3600
_q=None
_pid=None
_dropped=0

def prune(now=None):
    now=time.time() if now is None else now
    # Minute buckets are removed when their START passes 48h: never retain
    # records older than48h beyond the collector's 10s cleanup cadence.
    for p in ROOT.glob("v087-*.jsonl"):
        try:
            bucket=int(p.name.split("-")[1])
            if bucket <= int((now-RETENTION)//60):
                if p.is_file() and not p.is_symlink(): p.unlink()
        except (ValueError,OSError): pass

def _writer(q):
    global _dropped
    last=0
    while True:
        try:
            row=q.get(timeout=5)
            ROOT.mkdir(parents=True,exist_ok=True)
            row["dropped_records"]=_dropped
            path=ROOT/("v087-%d-%d.jsonl" % (int(row["time"]//60),os.getpid()))
            with path.open("a",encoding="utf-8") as f:
                f.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
        except queue.Empty: pass
        except Exception: _dropped+=1
        now=time.time()
        if now-last>=10: prune(now);last=now

def emit(kind, **fields):
    global _q,_pid,_dropped
    try:
        if _pid!=os.getpid():
            _pid=os.getpid();_q=queue.Queue(maxsize=256)
            threading.Thread(target=_writer,args=(_q,),daemon=True,name="v087-jsonl").start()
        row=dict(schema="dgx.diag.v087",kind=kind,time=time.time(),monotonic_ns=time.monotonic_ns(),pid=os.getpid(),rank=os.environ.get("DGX_DIAG_RANK"),**fields)
        _q.put_nowait(row)
    except Exception: _dropped+=1
