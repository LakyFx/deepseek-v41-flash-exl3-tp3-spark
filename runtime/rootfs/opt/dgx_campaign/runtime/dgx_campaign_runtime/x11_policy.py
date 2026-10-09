"""CPU-only, deterministic policies shared by preparation and the runtime port."""
from dataclasses import dataclass

BASE_GRAPHS = (5, 6, 10, 12, 15, 18, 20, 24, 25, 30, 35, 36, 40, 42, 48)
SMALL_GRAPHS = tuple(sorted(set(BASE_GRAPHS) | {1, 2, 3, 4, 8, 16, 32}))
MODULES = ('R', 'K', 'D', 'I1', 'I2', 'H', 'M', 'P', 'A')
VARIANTS = {
    '0.8.9X11a': {'kind': 'profile', 'profile_context': 65536},
    '0.8.9X11b': {'kind': 'graphs', 'graphs': list(SMALL_GRAPHS)},
    '0.8.9X11c': {'kind': 'prefetch', 'min_rows': 12, 'max_rows': 48,
                 'ffn_mib': 8, 'attention_mib': 12, 'ctas': 1},
    '0.8.9X11d': {'kind': 'draft', 'max_draft': 4,
                 'graphs': sorted(set(BASE_GRAPHS) | {n*k for n in range(1, 9) for k in (4, 5)})},
    '0.8.9X11e': {'kind': 'dynamic', 'max_draft': 5,
                 'graphs': sorted(set(BASE_GRAPHS) | {n*k for n in range(1, 9) for k in (3, 4, 5, 6)}),
                 'depths': [3, 4, 5], 'downshift_steps': 2},
}


def requested_depth(num_reqs):
    if isinstance(num_reqs, bool) or not isinstance(num_reqs, int) or not 1 <= num_reqs <= 8:
        raise ValueError('exact C1..C8 request count required')
    return 5 if num_reqs <= 3 else 4 if num_reqs <= 5 else 3


@dataclass
class DepthPolicy:
    """Same CPU input on all TP ranks; no timer, randomness or graph-hidden state."""
    depth: int = 5
    pending: int = 5
    streak: int = 0

    def choose(self, num_reqs, *, dummy=False):
        desired = requested_depth(num_reqs)
        if dummy:
            return 5  # Memory/cost profiling remains the maximum-depth reference.
        if desired >= self.depth:
            self.depth, self.pending, self.streak = desired, desired, 0
        else:
            self.streak = self.streak + 1 if self.pending == desired else 1
            self.pending = desired
            if self.streak >= 2:
                self.depth, self.streak = desired, 0
        return self.depth


def prefetch_parameters(rows, environ):
    def integer(name, default):
        raw = environ.get(name, str(default))
        if not raw.isdecimal():
            raise ValueError('invalid prefetch policy: ' + name)
        return int(raw)
    lo = integer('DGX_X11_PREFETCH_MIN_ROWS', 1)
    hi = integer('DGX_X11_PREFETCH_MAX_ROWS', 48)
    ff = integer('DGX_X11_PREFETCH_FFN_MIB', 16)
    attn = integer('DGX_X11_PREFETCH_ATTN_MIB', 20)
    ctas = integer('DGX_X11_PREFETCH_CTAS', 2)
    if not (1 <= lo <= hi <= 48 and 1 <= ff <= 16 and 1 <= attn <= 20 and 1 <= ctas <= 2):
        raise ValueError('prefetch policy exceeds qualified X11 geometry')
    return lo <= rows <= hi, ff*1024**2, attn*1024**2, ctas
