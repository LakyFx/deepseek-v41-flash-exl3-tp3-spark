"""One image-bearing request at a time, including preparation and streaming.

Original image payloads are preserved. Text requests bypass this admission.
The existing route heartbeat runs while admission waits.
"""
import asyncio
import contextvars
import functools
import logging
import os
import time
try:
    from .media_types import IMAGE_BLOCK_TYPES
except ImportError:  # Standalone regression test runner.
    from media_types import IMAGE_BLOCK_TYPES

_owner = contextvars.ContextVar('media_admission_owner', default=None)
log = logging.getLogger(__name__)


def has_images(value):
    if isinstance(value, dict):
        # Traversal also visits JSON Schema properties: a property named
        # "type" may itself contain a dict or a union list, not a block tag.
        kind = value.get('type')
        if isinstance(kind, str) and kind in IMAGE_BLOCK_TYPES:
            return True
        # Protocol content only: tools schemas/arguments can contain examples
        # that resemble image blocks but are not model image input.
        return any(has_images(value[key]) for key in ('input', 'messages', 'content', 'output')
                   if key in value)
    if isinstance(value, list):
        return any(has_images(child) for child in value)
    return False


def media_request(*, stream=False, close_method=None):
    def decorate(fn):
        @functools.wraps(fn)
        async def wrapped(self, request_body, *args, **kwargs):
            if (os.getenv('DGX_MEDIA_SERIAL_REQUESTS') != '1'
                    or not has_images(request_body)
                    or _owner.get() is self):
                return await fn(self, request_body, *args, **kwargs)
            gate = getattr(self, '_media_request_gate', None)
            if gate is None:
                self._media_request_gate = gate = asyncio.Semaphore(1)
            queued = time.monotonic()
            await gate.acquire()
            released = False
            transferred = False

            def release():
                nonlocal released
                if not released:
                    released = True
                    gate.release()

            token = _owner.set(self)
            cancelled = False
            try:
                log.info('Image request admitted after %.3fs', time.monotonic()-queued)
                # A disconnected client must not release admission while its
                # preparation or nonstream inference is still running.
                task = asyncio.create_task(fn(self, request_body, *args, **kwargs))
                while True:
                    try:
                        result = await asyncio.shield(task)
                        break
                    except asyncio.CancelledError:
                        cancelled = True
                        if task.done():
                            if task.cancelled():
                                raise
                            result = task.result()
                            break
                if stream:
                    result.media_release = release
                    transferred = True
                    if cancelled:
                        # Existing close uses shielded upstream abort/cleanup;
                        # release is owned by that cleanup, not this task.
                        await getattr(self, close_method)(result, outcome='cancelled')
                if cancelled:
                    raise asyncio.CancelledError()
                return result
            finally:
                _owner.reset(token)
                if not transferred:
                    release()
        return wrapped
    return decorate


def media_close(fn):
    @functools.wraps(fn)
    async def wrapped(self, handle, *args, **kwargs):
        try:
            return await fn(self, handle, *args, **kwargs)
        finally:
            release = getattr(handle, 'media_release', None)
            if release is not None:
                handle.media_release = None
                release()
    return wrapped
