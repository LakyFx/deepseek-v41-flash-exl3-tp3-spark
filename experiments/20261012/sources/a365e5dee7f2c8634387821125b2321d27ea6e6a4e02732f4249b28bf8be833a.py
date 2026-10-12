"""Prefill-only bounded NVMe hints; the original read/dequant path stays intact."""
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Policy:
    min_tokens: int = 512
    max_hint_bytes: int = 16 * 1024 * 1024
    max_spans: int = 1024
    page_bytes: int = 4096
    min_available_bytes: int = 4 * 1024**3

def available_memory():
    try:
        with open('/proc/meminfo',encoding='ascii') as stream:
            for line in stream:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0  # unknown headroom: no hints, original reads still run

def merged_spans(rel, base, row_bytes, page_bytes=4096):
    if row_bytes <= 0 or base < 0 or page_bytes <= 0:
        raise ValueError('invalid disk geometry')
    spans = []
    for row in sorted(set(int(item) for item in rel)):
        if row < 0:
            raise ValueError('negative rank-local row')
        low = (base + row * row_bytes) // page_bytes * page_bytes
        high = (base + (row + 1) * row_bytes + page_bytes-1) // page_bytes * page_bytes
        if spans and low <= spans[-1][1]:
            spans[-1] = (spans[-1][0],max(high,spans[-1][1]))
        else:
            spans.append((low,high))
    return spans

def hint_jobs(jobs, num_tokens, *, policy=Policy(), available=None, advise=None, file_size=None):
    result = {'eligible':False,'hinted_bytes':0,'spans':0,'errors':0}
    if num_tokens is None or num_tokens < policy.min_tokens:
        return result
    if (available if available is not None else available_memory()) < policy.min_available_bytes:
        return result
    if advise is None:
        if not hasattr(os,'posix_fadvise'):
            return result
        advise = lambda fd,lo,size:os.posix_fadvise(fd,lo,size,os.POSIX_FADV_WILLNEED)
    size_of = file_size or (lambda fd:os.fstat(fd).st_size)
    result['eligible'] = True
    # Budgets are per stage call, not independently per shard/table.
    seen = set()
    for fd, base, rel, row_bytes, _buffer in jobs:
        try:
            size = size_of(fd)
            for low, high in merged_spans(rel,base,row_bytes,policy.page_bytes):
                high = min(high,size)
                key = (fd,low,high)
                if key in seen or high <= low:
                    continue
                seen.add(key)
                remaining = policy.max_hint_bytes-result['hinted_bytes']
                if result['spans'] >= policy.max_spans or remaining <= 0:
                    return result
                count = min(high-low,remaining)
                try:
                    advise(fd,low,count)
                    result['hinted_bytes'] += count
                    result['spans'] += 1
                except OSError:
                    result['errors'] += 1
        except (OSError,ValueError):
            result['errors'] += 1
    return result

