"""Explicitly armed bounded CPU shape histogram, without tokens or prompts.

The future baseline C1..C8 supplement enables this through the owned worker
RPC. It stays disabled throughout the primary timed Hermes/Owen suites.
"""
from collections import Counter
import re

LIMIT = 4096
_COUNTS = Counter()
_SEEN = 0
_ACTIVE = False
_NONCE = None
_HEAD = False


def begin(rank, nonce):
    global _SEEN, _ACTIVE, _NONCE, _HEAD
    if rank not in (0, 1, 2) or not re.fullmatch('[a-f0-9]{32}', nonce):
        raise ValueError('trace requires rank and exact request nonce')
    if _ACTIVE:
        raise RuntimeError('previous trace still active')
    _COUNTS.clear()
    _SEEN, _NONCE, _HEAD = 0, nonce, rank == 0
    _ACTIVE = _HEAD
    return {'complete': True, 'trace_enabled': _ACTIVE, 'trace_limit': LIMIT}


def observe(batch, real, padded, dummy):
    global _SEEN, _ACTIVE
    if not _ACTIVE or dummy or batch is None or batch.has_prefill:
        return
    if not 1 <= real <= padded <= 48:
        return
    _COUNTS[(int(real), int(padded))] += 1
    _SEEN += 1
    if _SEEN >= LIMIT:
        _ACTIVE = False


def end():
    global _ACTIVE
    if _NONCE is None:
        raise RuntimeError('trace was never armed on this generation')
    _ACTIVE = False
    return {'complete': True, 'trace': {
        'schema':'dgx.next.shape-trace.v2', 'begin_nonce':_NONCE,
        'head':_HEAD, 'ended':True, 'limit':LIMIT, 'steps':_SEEN,
        'truncated':_SEEN == LIMIT,
        'histogram':[{'phase':'decode', 'real_rows':real, 'padded_rows':padded, 'steps':steps}
                     for (real,padded),steps in sorted(_COUNTS.items())]}}
