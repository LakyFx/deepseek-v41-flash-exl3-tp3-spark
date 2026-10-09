"""Serialize transient CPU media preparation, never autoregressive generation.
Candidate only: requires integration/real C8 qualification before activation.
"""
import asyncio
import functools
import os
import time
from pathlib import Path

def available():
    return int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:")))*1024


def has_media(value):
    if isinstance(value, dict):
        if value.get("type") in {"image_url", "input_image", "video_url", "input_audio", "image", "video"}:
            return True
        return any(has_media(v) for v in value.values() if isinstance(v, (dict, list, tuple)))
    if isinstance(value, (list, tuple)):
        return any(has_media(v) for v in value)
    return False


def serialized_render(fn):
    @functools.wraps(fn)
    async def wrapped(self, *args, **kwargs):
        if os.environ.get("DGX_SERIAL_RENDER") != "1" or not has_media(args[0] if args else kwargs.get("conversations", [])):
            return await fn(self, *args, **kwargs)
        lock = getattr(self, "_dgx_render_gate", None)
        if lock is None:
            self._dgx_render_gate = lock = asyncio.Lock()
        queued=time.monotonic()
        async with lock:
            from vllm.logger import init_logger
            log=init_logger(__name__)
            before=available();started=time.monotonic()
            # Shield CPU executor work from disconnect cancellation. Releasing
            # admission while its thread is still running would break the bound.
            task = asyncio.create_task(fn(self, *args, **kwargs))
            cancelled = False
            while True:
                try:
                    result = await asyncio.shield(task)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    if task.done():
                        raise
                except BaseException:
                    if cancelled:
                        raise asyncio.CancelledError() from None
                    raise
            log.info("DGX render admission wait=%.3fs render=%.3fs available_before=%d available_after=%d", started-queued,time.monotonic()-started,before,available())
            if cancelled:
                raise asyncio.CancelledError()
            return result
    return wrapped
