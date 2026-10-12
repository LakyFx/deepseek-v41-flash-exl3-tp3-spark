"""CPU policy shared by preparation and hash-pinned runtime overlays."""
from __future__ import annotations


def full_c1_budget(*, request_count, scheduled_drafts, non_draft_tokens,
                   computed_prefill, prefill_length, max_budget, max_steps):
    # Request count and prefill state come from the scheduler, never padded rows.
    # Clipped/EOS drafts remain clipped; do not manufacture a missing proposal.
    if (request_count == 1 and max_steps == 5 and non_draft_tokens == 1
            and computed_prefill >= prefill_length
            and 0 < scheduled_drafts <= max_steps and max_budget >= scheduled_drafts):
        return scheduled_drafts
    return None


def small_decode_mhc(rows, metadata):
    if not 1 <= rows <= 6 or not isinstance(metadata, dict):
        return False
    # The model has multiple attention metadata types. SWA identifies real
    # prefill/decode phase, while an indexer metadata object need not expose it.
    phases = [(m.num_decodes, m.num_prefills) for m in metadata.values()
              if hasattr(m, 'num_decodes') and hasattr(m, 'num_prefills')]
    return bool(phases) and all(d > 0 and p == 0 for d, p in phases)


def small_prefill_eager(batch, rows, dummy):
    # A tiny prefill must never replay a small pure-decode mHC graph. FULL graph
    # capture has decode metadata; prefill/mixed <=6 gets the original eager M.
    return bool(not dummy and batch is not None and 1 <= rows <= 6
                and batch.has_prefill)


def choose_graph_additions(histogram, existing, max_additions=3):
    """Offline graph selection from real/padded CPU step observations.

    Saved histograms must come from the future bounded GPU qualification, not
    invented timings. Only pure decode buckets up to the locked C8*K6 ceiling.
    """
    if not 0 <= max_additions <= 3:
        raise ValueError('at most three additional graph buckets')
    if not existing or max(existing) != 48:
        raise ValueError('baseline C8 graph ceiling changed')
    savings = {}
    for row in histogram:
        n, padded, count = row['real_rows'], row['padded_rows'], row['steps']
        if any(type(v) is not int for v in (n, padded, count)) or count < 0:
            raise ValueError('invalid histogram')
        if not 1 <= n <= padded <= 48:
            raise ValueError('decode shape outside locked bounds')
        if row['phase'] != 'decode' or n in existing:
            continue
        # Rank opportunities, NOT claimed milliseconds or a speed estimate.
        savings[n] = savings.get(n, 0) + count * (padded - n)
    return sorted(sorted((n for n in savings if savings[n] > 0),
                         key=lambda n: (-savings[n], n))[:max_additions])
