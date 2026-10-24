"""Opt-in, bounded private evidence for explicitly allowlisted vision jobs.

Never changes acceptance, retries, budgets or native cancellation. No prompt,
image, auth header or raw SSE is recorded. Trigger text may contain private
reasoning and must stay in the private directory, never in an API response.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat
import time

log = logging.getLogger(__name__)
MAX_JOBS = 16
MAX_FILE_BYTES = 1024 * 1024
MAX_FINAL_BYTES = 128 * 1024
TAIL_CHARS = 16000
IDENTITY = re.compile(r'vision_(?:[a-f0-9]{32}|[a-f0-9]{64})\Z')


def _private(path, directory=False):
    info = path.lstat()
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError('Diagnostic path is not a regular private resource')
    if os.name == 'posix' and (info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077):
        raise ValueError('Diagnostic resource must belong to service uid with private permissions')
    return info


def _pattern(text):
    # Deliberately mirrors the existing detector; it does not decide acceptance.
    words = re.findall(r'[^\W_]+(?:[.,]\d+)?', text[-TAIL_CHARS:].lower())[-384:]
    for period in range(1, min(97, len(words)//5 + 1)):
        copies = min(12, len(words)//period)
        if period * copies >= 60 and words[-period*copies:] == words[-period:]*copies:
            return dict(period_words=period, copies=copies, matched_words=period*copies,
                        pattern_sha256=hashlib.sha256(json.dumps(words[-period:], ensure_ascii=False).encode()).hexdigest())
    return None


class Diagnostics:
    @classmethod
    def open(cls, identity, native_id, dispatch=None):
        root = os.getenv('DGX_VISION_DIAGNOSTICS_DIR')
        if not root:
            return None
        try:
            if not IDENTITY.fullmatch(identity):
                raise ValueError('Invalid diagnostic job identity')
            directory = Path(root)
            if not directory.is_absolute():
                raise ValueError('Absolute diagnostic directory required')
            # Resolve parents too: an allowed path must not redirect through links.
            if directory.resolve() != directory:
                raise ValueError('Diagnostic symlink path refused')
            _private(directory, directory=True)
            allowlist = directory/'ALLOWLIST.json'
            if _private(allowlist).st_size > 8192:
                raise ValueError('Diagnostic allowlist too large')
            config = json.loads(allowlist.read_bytes())
            ids = config['job_ids']
            expiry = float(config['expires_epoch'])
            if not isinstance(ids, list) or len(ids) > MAX_JOBS or len(set(ids)) != len(ids):
                raise ValueError('Bounded unique diagnostic allowlist required')
            if not all(isinstance(i, str) and IDENTITY.fullmatch(i) for i in ids):
                raise ValueError('Invalid allowlisted identity')
            now = time.time()
            if not now < expiry <= now + 86400:
                return None
            if identity not in ids:
                return None
            return cls(directory, identity, native_id, dispatch)
        except Exception as exc:
            log.warning('Vision diagnostics unavailable (%s)', type(exc).__name__)
            return None

    def __init__(self, directory, identity, native_id, dispatch):
        self.path = directory/(identity+'.json')
        # Existing evidence is never overwritten by another generation.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        self.started = time.monotonic()
        self.final = bytearray()
        self.final_seen = 0
        self.events = 0
        self.write_failed = False
        self.record = dict(schema='tensorfold.vision-diagnostics.v1', job_id=identity,
            native_request_id=native_id, gateway_pid=os.getpid(),
            gateway_release=str(Path(__file__).resolve().parents[1]),
            started_epoch=time.time(), status='observing', trigger=None)
        self.record['native_dispatch'] = {k:dispatch[k] for k in ('schema','instance','request_id') if dispatch and k in dispatch}
        self._save()

    def _save(self):
        try:
            self.record['events_observed'] = self.events
            data = json.dumps(self.record, ensure_ascii=False).encode('utf-8')
            if len(data) > MAX_FILE_BYTES:
                raise ValueError('Diagnostic evidence exceeded its bounded size')
            _private(self.path)
            temporary = self.path.with_suffix('.json.tmp')
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except Exception as exc:
            self.write_failed = True
            log.warning('Vision diagnostic evidence write failed (%s)', type(exc).__name__)

    def observe(self, delta):
        self.events += 1
        text = delta.get('content') or ''
        if text:
            data = text.encode('utf-8')
            self.final_seen += len(data)
            remaining = MAX_FINAL_BYTES - len(self.final)
            if remaining > 0:
                self.final.extend(data[:remaining])

    def trigger(self, content, reasoning):
        if self.record['trigger'] is not None:
            return
        channels = []
        for name, text in [('content', content), ('reasoning', reasoning)]:
            match = _pattern(text)
            if match:
                channels.append(dict(channel=name, characters_at_trigger=len(text),
                    private_tail=text[-TAIL_CHARS:], **match))
        self.record['trigger'] = dict(elapsed_seconds=time.monotonic()-self.started,
            event=self.events, channels=channels)
        self.record['status'] = 'triggered_draining_same_generation'
        self._save()

    def finish(self, *, done, finish_reason, repetitive, usage, result_status, outcome):
        # Private continuation survives the collector clearing its public buffers.
        # This is evidence, never an alternative accepted response.
        self.record.update(status='stream_ended', elapsed_seconds=time.monotonic()-self.started,
            stream_done=done, finish_reason=finish_reason, rejected_repetition=repetitive,
            usage=usage, result_status=result_status, outcome=outcome,
            private_final=self.final.decode('utf-8', errors='replace'),
            final_bytes_seen=self.final_seen, final_bytes_retained=len(self.final),
            final_capture_truncated=self.final_seen > len(self.final),
            prior_evidence_write_failed=self.write_failed)
        self._save()


def observe(diagnostics, method, *args, **kwargs):
    """Diagnostic I/O must never alter inference or its cleanup path."""
    if diagnostics is not None:
        try:
            getattr(diagnostics, method)(*args, **kwargs)
        except Exception as exc:
            log.warning('Vision diagnostic observer failed (%s)', type(exc).__name__)
