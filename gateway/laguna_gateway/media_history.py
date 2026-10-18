"""Compact completed older turns under image pressure; pin the whole live turn."""
from copy import deepcopy
import logging


async def compact_image_history(body, limit, count_images, latest_user_index, summarize, summary_item):
    items = body.get("input")
    if not isinstance(items, list) or count_images(body)["image"] <= max(1, limit * 3 // 4):
        return body, 0, 0
    boundary = latest_user_index(items)
    if boundary is None or boundary <= 0:
        return body, 0, 0
    older, live = items[:boundary], items[boundary:]
    # A tool result must never lose its call across an interrupted turn.
    older_calls = {i.get("call_id") for i in older if i.get("type") in {"function_call", "custom_tool_call"}}
    if any(i.get("call_id") in older_calls for i in live if i.get("type") in {"function_call_output", "custom_tool_call_output"}):
        return body, 0, 0
    image_positions = [i for i, item in enumerate(older) if count_images({"input": [item]})["image"]]
    if not image_positions:
        return body, 0, 0
    # Evidence that a later model step consumed the old image, not an unseen upload.
    if not any(i.get("role") == "assistant" or i.get("type") in {"function_call", "custom_tool_call"}
               for i in older[image_positions[-1] + 1:]):
        return body, 0, 0
    pinned = [i for i in older if i.get("role") in {"system", "developer"}]
    source = [i for i in older if i.get("role") not in {"system", "developer"}]
    if not count_images({"input": source})["image"]:
        return body, 0, 0
    try:
        summary, chunks = await summarize(source)
        if not summary.strip():
            return body, 0, 0
    except Exception:
        logging.getLogger(__name__).warning("Image-history compaction failed; original input retained", exc_info=True)
        return body, 0, 0
    result = deepcopy(body)
    result["input"] = deepcopy(pinned) + [summary_item(summary)] + deepcopy(live)
    result.pop("previous_response_id", None)
    return result, len(source), chunks
