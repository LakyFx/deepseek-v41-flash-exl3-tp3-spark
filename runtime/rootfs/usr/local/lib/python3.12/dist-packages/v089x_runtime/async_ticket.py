"""Single owner CPU read ticket. Failure poisons the controller, never falls back
to an earlier generation. Device work is supplied exclusively by the caller.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading

class ReadTicket:
    def __init__(self, timeout=20):
        if not 0 < timeout <= 300:
            raise ValueError('invalid reader deadline')
        self.timeout = timeout
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='v089x-stage')
        self.pending = None
        self.poisoned = False
        self.owner = threading.get_ident()
        self.generation = 0

    def _owned(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('only owner may schedule/join device staging')
        if self.poisoned:
            raise RuntimeError('Engram reader poisoned; engine restart required')

    def start(self, read_and_copy, release):
        self._owned()
        if self.pending is not None:
            raise RuntimeError('previous Engram generation has not been joined')
        self.generation += 1
        def run():
            try:
                return read_and_copy()
            finally:
                release()  # Unblock queued device work even when read fails.
        try:
            self.pending = self.pool.submit(run)
        except BaseException:
            self.poisoned = True
            release()
            raise

    def finish(self, *, unblock_timeout):
        self._owned()
        if self.pending is None:
            return
        try:
            self.pending.result(timeout=self.timeout)
        except BaseException:
            self.poisoned = True
            # A stuck read cannot keep the CUDA graph blocked indefinitely.
            # Its result is discarded and this arena must never be reused.
            unblock_timeout()
            raise
        else:
            self.pending = None

    def close(self):
        self._owned()
        if self.pending is not None:
            raise RuntimeError('cannot close an outstanding reader')
        self.pool.shutdown(wait=True)
