"""A activates already-implemented device-length flattening on locked SM121.

Do NOT enable the SM100 indices-based varlen branch: the captured builder
expands block-table rows, so passing original request indices can misaddress
that expanded table. Plain one-query rows need no indices indirection.
"""
import os


def enabled() -> bool:
    value = os.environ.get("DGX_CAMPAIGN_A", "0")
    if value not in ("0", "1"):
        raise ValueError("DGX_CAMPAIGN_A must be 0 or 1")
    return value == "1"


def supports_flattened_device_lengths(platform) -> bool:
    return enabled() and platform.is_cuda() and platform.is_device_capability_family(120)
