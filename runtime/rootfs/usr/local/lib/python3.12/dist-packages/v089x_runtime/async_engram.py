"""Overlap A1's *existing 32-reader* gather/dequant with the target GPU forward.

Pinned CPU results feed a gated H2D copy on a side stream. Per-layer ready
flags are captured through an opaque custom op. Only the main thread queues
CUDA work. No B12X Engram/kernel dependency and no retained table-row cache.
GPU qualification of stream-memory-op capture is mandatory before a trial.
"""
import ctypes
import os
import threading
import torch
from .async_ticket import ReadTicket

_FLAGS = {}
_REGISTERED = False

def _check(result):
    from cuda.bindings import driver
    status, *values = result
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError('Engram CUDA stream-memory operation failed: '+str(status))
    return values[0] if values else None

class MappedWord:
    def __init__(self, initial):
        from cuda.bindings import driver
        self.driver = driver
        self.host_pointer = _check(driver.cuMemHostAlloc(4, driver.CU_MEMHOSTALLOC_DEVICEMAP))
        self.device_pointer = _check(driver.cuMemHostGetDevicePointer(self.host_pointer, 0))
        self.word = ctypes.c_uint32.from_address(int(self.host_pointer))
        self.word.value = initial
        # Grace is ARM64. A plain ctypes store does not establish a release
        # ordering between the completed pinned-buffer writes and the flag.
        atomic = ctypes.CDLL('libatomic.so.1')
        self.store_release = getattr(atomic, '__atomic_store_4')
        self.store_release.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int]
        self.store_release.restype = None

    def gpu_write(self, stream, value):
        _check(self.driver.cuStreamWriteValue32(stream.cuda_stream, self.device_pointer, value, 0))

    def gpu_wait(self, stream, value):
        _check(self.driver.cuStreamWaitValue32(stream.cuda_stream, self.device_pointer, value,
            int(self.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ)))

    def release(self, value):
        self.store_release(int(self.host_pointer), value, 3)  # __ATOMIC_RELEASE

def _wait(rows: torch.Tensor, key: int) -> None:
    _FLAGS[key].gpu_wait(torch.cuda.current_stream(rows.device), 1)

def _fake(rows: torch.Tensor, key: int) -> None:
    return None

def initialize_layer(layer):
    global _REGISTERED
    if os.environ.get('DGX_V089X_ENGRAM_ASYNC') != '1' or not layer.prestage:
        return
    if not _REGISTERED:
        from vllm.utils.torch_utils import direct_register_custom_op
        direct_register_custom_op(op_name='dgx_v089x_engram_wait', op_func=_wait,
                                  mutates_args=['rows'], fake_impl=_fake)
        _REGISTERED = True
    layer._v089x_ready = MappedWord(1)
    layer._v089x_ready_key = id(layer)
    _FLAGS[layer._v089x_ready_key] = layer._v089x_ready

def wait_layer(layer):
    if hasattr(layer, '_v089x_ready'):
        torch.ops.vllm.dgx_v089x_engram_wait(layer.staged_rows, layer._v089x_ready_key)

class AsyncEngram:
    def __init__(self, stager):
        if os.environ.get('DSV41_ENGRAM_DISK_THREADS') != '32':
            raise RuntimeError('0.8.9X keeps exactly the qualified 32-reader pool')
        if not all(hasattr(x, '_v089x_ready') for x in stager.engrams):
            raise RuntimeError('all target Engram layers must have readiness gates')
        self.stager = stager
        self.owner = threading.get_ident()
        self.side = torch.cuda.Stream(device=stager.engrams[0].staged_rows.device)
        self.read_gate = MappedWord(0)
        self.reset_done = torch.cuda.Event()
        self.copy_done = torch.cuda.Event()
        self.copy_used = False
        self.last_sequence = 0
        self.ticket = ReadTicket(float(os.environ.get('DGX_V089X_ENGRAM_TIMEOUT_S','20')))

    def submit(self, native, requests, n, stock):
        if threading.get_ident() != self.owner or not 0 < n <= 48:
            raise RuntimeError('async staging requires the main thread and <=48 rows')
        self.finish()
        if self.copy_used:
            self.copy_done.synchronize()  # Ownership fence before overwriting pinned buffers.
        main = torch.cuda.current_stream(self.stager.engrams[0].staged_rows.device)
        for layer in self.stager.engrams:
            layer._v089x_ready.gpu_write(main, 0)
        self.reset_done.record(main)
        self.side.wait_event(self.reset_done)
        self.last_sequence += 1
        if self.last_sequence >= 0x7fffffff:
            raise RuntimeError('Engram read sequence exhausted; restart required')
        sequence = self.last_sequence
        self.read_gate.gpu_wait(self.side, sequence)
        # Queue copies before CPU completion; HostGate protects source buffers.
        with torch.cuda.stream(self.side):
            for layer, buf in zip(self.stager.engrams, self.stager.rows_host):
                layer.staged_rows[:n].copy_(buf[:n], non_blocking=True)
                layer._v089x_ready.gpu_write(self.side, 1)
            self.copy_done.record(self.side)
        self.copy_used = True
        def read():
            with torch.inference_mode():
                rows = native(requests, num_tokens=n, stock=stock)
                if len(rows) != len(self.stager.rows_host):
                    raise RuntimeError('incomplete Engram batch')
                for result, buf in zip(rows, self.stager.rows_host):
                    buf[:n].copy_(result.view(n, self.stager.local_heads, self.stager.dim))
        self.ticket.start(read, lambda: self.read_gate.release(sequence))
        from .telemetry import observe
        observe('engram_async', rows=n, io_threads=32)

    def finish(self):
        self.ticket.finish(unblock_timeout=lambda:self.read_gate.release(self.last_sequence))

def stage_if_enabled(stager, native, requests, n, stock):
    if os.environ.get('DGX_V089X_ENGRAM_ASYNC') != '1' or n > 48:
        return False
    async_stage = getattr(stager, '_v089x_async', None)
    if async_stage is None:
        async_stage = stager._v089x_async = AsyncEngram(stager)
    async_stage.submit(native, requests, n, stock)
    return True

def finish_model_state(model_state):
    stager = getattr(model_state, 'engram_stager', None)
    async_stage = getattr(stager, '_v089x_async', None)
    if async_stage is not None:
        async_stage.finish()  # Errors propagate before ExecuteModelState/sampling.
