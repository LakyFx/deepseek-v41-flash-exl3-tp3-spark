"""Small-batch C conversion with the original shared parallel disk reader.

Reads are dispatched once across every table, just as in the attested stock
path. Native conversion is synchronous; GPU staging remains the caller's job.
"""
import ctypes
from pathlib import Path


class ParallelGather:
    def __init__(self, original, library, *, parallel_read, io_threads,
                 max_tokens=64, verify_calls=8, torch_module=None):
        if io_threads != 32 or not 1 <= max_tokens <= 64 or verify_calls < 1:
            raise ValueError('A1 requires the original 32-worker reader and bounded qualification')
        import torch
        self.torch = torch_module or torch
        self.original = original
        self.parallel_read = parallel_read
        self.io_threads = io_threads
        self.max_tokens = max_tokens
        self.verify_left = verify_calls
        self.enabled = False
        self.reason = None
        self.calls = self.fallback_calls = self.unique_rows = 0
        try:
            self.lib = ctypes.CDLL(str(Path(library).resolve()))
            self.lib.dgx_engram_dequant_abi.restype = ctypes.c_int
            if self.lib.dgx_engram_dequant_abi() != 1:
                raise ValueError('parallel Engram conversion ABI mismatch')
            self.convert = self.lib.dgx_engram_dequant
            self.convert.restype = ctypes.c_int
            self.convert.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 6
            t = self.torch
            w = t.arange(256, dtype=t.int16, device='cpu').to(t.uint8).view(t.float8_e4m3fn).to(t.float32)
            scale = (t.arange(256, dtype=t.int32, device='cpu') << 23).view(t.float32)
            self.lut = (scale[:, None] * w[None, :]).to(t.bfloat16).contiguous()
            self.enabled = True
        except Exception as exc:
            self._disarm(exc)

    def _disarm(self, exc):
        self.enabled = False
        self.reason = f'{type(exc).__name__}: {exc}'
        print('DGX_V089_NATIVE_DISARMED A1 ' + self.reason, flush=True)

    def __call__(self, requests, *, num_tokens=None, stock=None):
        fallback = self.original if stock is None else stock
        if not self.enabled or num_tokens is None or not 0 < num_tokens <= self.max_tokens:
            self.fallback_calls += 1
            return fallback(requests)
        t = self.torch
        plans, jobs = [], []
        try:
            # Match the stock unique/inverse mapping and read-job geometry.
            # Validate every table before dispatch: never launch half a batch.
            for table, rel, owned in requests:
                if (rel.device.type != 'cpu' or owned.device.type != 'cpu'
                        or rel.dtype != t.int64 or owned.dtype != t.bool
                        or rel.ndim != 1 or owned.shape != rel.shape
                        or table.dim != 256 or table.sb != 8 or rel.numel() > 16384):
                    self.fallback_calls += 1
                    return fallback(requests)
                uniq, inverse = t.unique(rel, return_inverse=True)
                if uniq.numel() and (int(uniq[0]) < 0 or int(uniq[-1]) >= table.num_rows):
                    raise ValueError('rank-local row outside table')
                w = t.empty((uniq.numel(), table.dim), dtype=t.uint8, device='cpu')
                s = t.empty((uniq.numel(), table.sb), dtype=t.uint8, device='cpu')
                # Stock's numpy memoryview cannot cast a zero-row 2D array.
                # There is no I/O or conversion for an empty table request.
                if uniq.numel():
                    jobs += table.read_jobs(uniq.tolist(), w, s)
                plans.append((table, w, s, inverse.contiguous(), owned.to(t.uint8).contiguous()))
            # One existing shared pool, configured to 32 workers. A short batch
            # need not contain enough jobs to occupy every worker simultaneously.
            self.parallel_read(jobs)
            outs = []
            for table, w, s, inverse, mask in plans:
                out = t.empty((inverse.numel(), table.dim), dtype=t.bfloat16, device='cpu')
                if inverse.numel():
                    rc = self.convert(w.shape[0], table.dim, table.sb, inverse.numel(),
                                      w.data_ptr(), s.data_ptr(), inverse.data_ptr(),
                                      mask.data_ptr(), self.lut.data_ptr(), out.data_ptr())
                    if rc:
                        raise OSError(-rc, 'native Engram conversion failed')
                outs.append(out)
            nonempty = [i for i, (_, rel, _) in enumerate(requests) if rel.numel()]
            if self.verify_left and nonempty:
                reference = fallback([requests[i] for i in nonempty])
                if len(reference) != len(nonempty) or any(
                        not t.equal(a.view(t.uint16), outs[i].view(t.uint16))
                        for a, i in zip(reference, nonempty)):
                    raise ValueError('A1/stock raw BF16 bits differ')
                self.verify_left -= 1
                if self.verify_left == 0:
                    print('DGX_V089_NATIVE_QUALIFIED A1 io_threads=32', flush=True)
            self.calls += 1
            self.unique_rows += sum(w.shape[0] for _, w, _, _, _ in plans)
            return outs
        except Exception as exc:
            # Original reader joins its submitted futures before returning on
            # success. On read failure its job buffers remain owned by futures;
            # fallback allocates independent buffers and returns the whole batch.
            self._disarm(exc)
            self.fallback_calls += 1
            return fallback(requests)
