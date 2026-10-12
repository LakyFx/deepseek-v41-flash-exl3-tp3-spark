"""Exact CPU-native disk gather with bounded decode eligibility and fallback."""
import ctypes
from pathlib import Path

class NativeGather:
    def __init__(self, original, library, *, max_tokens=64, verify_calls=8, torch_module=None):
        if max_tokens < 1 or max_tokens > 64 or verify_calls < 1:
            raise ValueError('invalid qualification/row bound')
        import torch
        self.torch = torch_module or torch
        self.original=original
        self.max_tokens=max_tokens
        self.verify_left=verify_calls
        self.enabled=False
        self.reason=None
        self.calls=self.fallback_calls=self.unique_rows=0
        self.lib=None
        try:
            self._load(library)
        except Exception as exc:
            self.reason=f'{type(exc).__name__}: {exc}'
            print('DGX_V089_NATIVE_DISARMED '+self.reason,flush=True)

    def _load(self, library):
        self.lib=ctypes.CDLL(str(Path(library).resolve()))
        self.lib.dgx_engram_abi.restype=ctypes.c_int
        if self.lib.dgx_engram_abi()!=1:
            raise ValueError('native Engram ABI mismatch')
        self.lib.dgx_engram_gather.restype=ctypes.c_int
        self.lib.dgx_engram_gather.argtypes=[ctypes.c_int,ctypes.c_int64,ctypes.c_int,ctypes.c_int64,
            ctypes.c_int64,ctypes.c_int,ctypes.c_int,ctypes.c_int,
            ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        t=self.torch
        w=t.arange(256,dtype=t.int16,device='cpu').to(t.uint8).view(t.float8_e4m3fn).to(t.float32)
        scale=(t.arange(256,dtype=t.int32,device='cpu')<<23).view(t.float32)
        self.lut=(scale[:,None]*w[None,:]).to(t.bfloat16).contiguous()
        self.enabled=True

    def __call__(self,requests,*,num_tokens=None,stock=None):
        # The combined prefill/native path supplies this batch's token count.
        # Never retain a callback closing over an earlier batch's n.
        fallback = self.original if stock is None else stock
        if not self.enabled or num_tokens is None or not 0<num_tokens<=self.max_tokens:
            self.fallback_calls+=1
            return fallback(requests)
        t=self.torch
        outs=[]
        try:
            for table, rel, owned in requests:
                if (rel.device.type!='cpu' or owned.device.type!='cpu' or rel.dtype!=t.int64
                    or owned.dtype!=t.bool or rel.ndim!=1 or owned.shape!=rel.shape
                    or table.dim!=256 or table.sb!=8 or rel.numel()>16384):
                    self.fallback_calls+=1
                    return fallback(requests)
                # A small output buffer replaces the original full FP32 dequant
                # temporary. Not a retained model-row cache or a new KV arena.
                ids=rel.contiguous(); mask=owned.to(t.uint8).contiguous()
                out=t.empty((rel.numel(),table.dim),dtype=t.bfloat16,device='cpu')
                stats=(ctypes.c_int64*3)()
                rc=self.lib.dgx_engram_gather(table.w_fd,table.w_off,table.s_fd,table.s_off,
                    table.num_rows,table.dim,table.sb,rel.numel(),ids.data_ptr(),mask.data_ptr(),
                    self.lut.data_ptr(),out.data_ptr(),stats)
                if rc:
                    raise OSError(-rc,'native disk gather failed')
                outs.append(out);self.unique_rows+=int(stats[1])
            if self.verify_left:
                ref=fallback(requests)
                if len(ref)!=len(outs) or any(not t.equal(a.view(t.uint16),b.view(t.uint16)) for a,b in zip(ref,outs)):
                    raise ValueError('native/stock raw BF16 bits differ')
                self.verify_left-=1
                if self.verify_left==0:
                    print('DGX_V089_NATIVE_QUALIFIED',flush=True)
            self.calls+=1
            return outs
        except Exception as exc:
            self.enabled=False;self.reason=f'{type(exc).__name__}: {exc}'
            print('DGX_V089_NATIVE_DISARMED '+self.reason,flush=True)
            self.fallback_calls+=1
            return fallback(requests)  # refill the entire step, never partial
