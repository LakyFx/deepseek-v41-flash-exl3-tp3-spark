"""Opt-in 1.2.6a sidecar. Never import or launch CUDA; never alter NVMe APST.

Only the native process's block-read counter renews the activity lease. Reading
our own /proc/io or diskstats would let keepalive reads keep themselves awake.
One read-only, aligned O_DIRECT read per period, no buffered fallback. A hung
drive can block this isolated process, not an engine thread; the future unit's
TimeoutStopSec/MemoryMax/TasksMax bound the sidecar independently.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time

BLOCK = 4096


class Activity:
    def __init__(self, period_ms=25, active_ms=2000):
        if not 10 <= period_ms <= 80 or not period_ms <= active_ms <= 2000:
            raise ValueError('period 10..80 ms, active lease at most 2000 ms')
        self.period_ns = period_ms * 1_000_000
        self.active_ns = active_ms * 1_000_000
        self.last_counter = None
        self.until = 0
        self.next_read = 0

    def poll(self, now, native_read_bytes):
        if self.last_counter is not None and native_read_bytes < self.last_counter:
            raise RuntimeError('Native I/O counter reset; refuse PID ambiguity')
        if self.last_counter is not None and native_read_bytes > self.last_counter:
            self.until = now + self.active_ns
        self.last_counter = native_read_bytes
        if now < self.until and now >= self.next_read:
            # No catch-up bursts after scheduling delays.
            self.next_read = now + self.period_ns
            return True
        return False


def process_start(proc):
    # comm may contain spaces/parentheses; starttime is field22, index19 after ')'.
    return int((proc/'stat').read_text().rsplit(')', 1)[1].split()[19])


def read_counter(proc):
    return int(dict(line.split(':', 1) for line in (proc/'io').read_text().splitlines())['read_bytes'])


def select_files(proc, engram_root):
    root = engram_root.resolve(strict=True)
    by_device = {}
    for entry in sorted((proc/'fd').iterdir(), key=lambda p:int(p.name)):
        try:
            target = entry.resolve(strict=True)
            if not target.is_relative_to(root) or target.suffix != '.safetensors':
                continue
            info = target.stat()
            if stat.S_ISREG(info.st_mode) and info.st_size >= 2*BLOCK:
                # Engram may span devices. Keep one fd per actual filesystem
                # device; reject non-NVMe backing separately in qualification.
                by_device.setdefault(info.st_dev, target)
        except (OSError, RuntimeError):
            continue
    return list(by_device.values())


class DirectReader:
    def __init__(self, path):
        self.fd = None
        self.buf = ctypes.c_void_p()
        self.lib = ctypes.CDLL(None, use_errno=True)
        self.lib.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t,ctypes.c_size_t]
        self.lib.posix_memalign.restype = ctypes.c_int
        self.lib.pread.argtypes = [ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int64]
        self.lib.pread.restype = ctypes.c_ssize_t
        self.lib.free.argtypes = [ctypes.c_void_p]
        self.lib.free.restype = None
        if self.lib.posix_memalign(ctypes.byref(self.buf),BLOCK,BLOCK):
            raise MemoryError('aligned keepalive buffer')
        try:
            self.fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_DIRECT)
            # Rank-local Engram wrappers are sparse. A fixed4KiB offset is a
            # hole on followers and would not reliably submit an NVMe read.
            # Refuse unsupported extent queries rather than silently fake I/O.
            probe=BLOCK
            for _ in range(32):
                data=os.lseek(self.fd,probe,os.SEEK_DATA)
                hole=os.lseek(self.fd,data,os.SEEK_HOLE)
                aligned=(data+BLOCK-1)//BLOCK*BLOCK
                if aligned+BLOCK<=hole:
                    self.offset=aligned
                    break
                if hole<=probe:raise RuntimeError('Non-monotonic Engram extent map')
                probe=hole
            else:raise RuntimeError('No safe allocated Engram extent in bounded search')
        except BaseException:
            self.close();raise

    def read(self):
        n=self.lib.pread(self.fd,self.buf,BLOCK,self.offset)
        if n != BLOCK:
            raise OSError(ctypes.get_errno(),f'O_DIRECT keepalive returned {n}')

    def close(self):
        if self.fd is not None:os.close(self.fd);self.fd=None
        if self.buf.value:self.lib.free(self.buf);self.buf=ctypes.c_void_p()


def run(args):
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Linux sidecar only')
    activity=Activity(args.period_ms,args.active_ms)
    if args.native_unit:
        if args.native_unit not in [f'dsv41-tf351-r{i}.service' for i in range(3)]:
            raise ValueError('Not a designated TensorFold native unit')
        args.pid=int(subprocess.check_output(['systemctl','show',args.native_unit,'--property=MainPID','--value'],text=True,timeout=5).strip())
    if args.pid is None or args.pid<=1:raise ValueError('No running native PID')
    proc=Path('/proc')/str(args.pid)
    start=process_start(proc)
    executable=(proc/'exe').resolve(strict=True)
    with executable.open('rb') as f:exe_sha=hashlib.file_digest(f,'sha256').hexdigest()
    if exe_sha != args.exe_sha256:
        raise RuntimeError('Unexpected native binary; sidecar refused')
    if str(args.engram_root.resolve(strict=True)).startswith('/proc/'):
        raise RuntimeError('Engram root must be a real immutable asset directory')
    paths=select_files(proc,args.engram_root)
    if not paths or len(paths)>2:
        raise RuntimeError('Expected one or two Engram backing devices already open by native process')
    for path in paths:
        device=Path('/sys/dev/block')/f'{os.major(path.stat().st_dev)}:{os.minor(path.stat().st_dev)}'
        if not any(part.startswith('nvme') for part in device.resolve(strict=True).parts):
            raise RuntimeError('Engram backing is not a directly identified local NVMe device')
    readers=[];stopping=False;reads=0;errors=0
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for path in paths:readers.append(DirectReader(path))
        print(json.dumps({'event':'keepalive_start','pid':args.pid,'start_ticks':start,'period_ms':args.period_ms,'active_ms':args.active_ms,'devices':len(readers),'allocated_offsets':[r.offset for r in readers]}),flush=True)
        while not stopping:
            if process_start(proc)!=start:raise RuntimeError('Native PID identity changed')
            if activity.poll(time.monotonic_ns(),read_counter(proc)):
                for reader in readers:
                    reader.read();reads+=1
            time.sleep(activity.period_ns/1e9)
    except (OSError,RuntimeError) as exc:
        errors+=1
        print(json.dumps({'event':'keepalive_disabled','error':str(exc)}),flush=True)
        return 1
    finally:
        for reader in readers:reader.close()
        print(json.dumps({'event':'keepalive_stop','reads':reads,'bytes':reads*BLOCK,'errors':errors}),flush=True)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    identity=p.add_mutually_exclusive_group(required=True)
    identity.add_argument('--pid',type=int);identity.add_argument('--native-unit')
    p.add_argument('--exe-sha256',required=True)
    p.add_argument('--engram-root',type=Path,required=True)
    p.add_argument('--period-ms',type=int,default=25);p.add_argument('--active-ms',type=int,default=2000)
    raise SystemExit(run(p.parse_args()))
