"""Per-request native cancellation receipts, on a separate HTTP connection pool.

A timeout, a 404, or a successful socket close is never a termination receipt.
The native endpoint seals cancelled IDs against late dispatch and acknowledges
termination only after its generation has drained (or its instance changed).
"""
import asyncio
import os
import math
import uuid
import httpx

SCHEMA = 'tensorfold.vision-control.v1'
PATH = '/internal/vision/control'


def confirms_termination(receipt, dispatch):
    if not isinstance(receipt,dict):
        return False
    instance = receipt.get('instance')
    same = instance == dispatch['instance']
    return (receipt.get('schema') == SCHEMA
            and receipt.get('request_id') == dispatch['request_id']
            and receipt.get('requested_instance') == dispatch['instance']
            and isinstance(instance,str) and bool(instance)
            and receipt.get('termination_confirmed') is True
            and ((same and receipt.get('state') == 'terminated')
                 or (not same and receipt.get('state') == 'instance_replaced')))


def seconds(name, default, maximum=3600):
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f'{name} must be positive and at most {maximum} seconds')
    return value


class NativeControl:
    def __init__(self, client):
        self.client = client
        self.grace = seconds('DGX_VISION_CANCEL_GRACE_SECONDS', 60, 120)

    async def dispatch(self, row, ledger):
        response = await self.client.get(PATH)
        response.raise_for_status()
        info = response.json()
        if (not isinstance(info,dict) or info.get('schema') != SCHEMA
                or not isinstance(info.get('instance'), str) or not info['instance']):
            raise ValueError('Native vision cancellation receipts are unavailable')
        request_id = 'dgx-gateway-' + uuid.uuid4().hex
        dispatch = {'instance': info['instance'], 'request_id': request_id,
                    'expires': math.ceil(row['deadline'])}
        # Persist BEFORE the HTTP POST: cancellation can seal an ID even when
        # the gateway dies between the POST and the first response headers.
        ledger.dispatch(row['id'], dispatch)
        return dispatch

    async def cancel(self, dispatch):
        response = await self.client.post(PATH, json={**dispatch, 'action': 'cancel'})
        response.raise_for_status()
        receipt = response.json()
        if (not isinstance(receipt,dict) or receipt.get('schema') != SCHEMA
                or receipt.get('request_id') != dispatch['request_id']
                or receipt.get('requested_instance') != dispatch['instance']):
            raise ValueError('Native vision receipt does not match this dispatch')
        if confirms_termination(receipt,dispatch):
            return receipt
        return None

    async def terminate(self, dispatch):
        end = asyncio.get_running_loop().time() + self.grace
        while True:
            remaining = end - asyncio.get_running_loop().time()
            if remaining <= 0:
                return None
            try:
                async with asyncio.timeout(remaining):
                    receipt = await self.cancel(dispatch)
                if receipt:
                    return receipt
            except (TimeoutError, ValueError, httpx.HTTPError):
                pass
            await asyncio.sleep(min(1, max(0, end-asyncio.get_running_loop().time())))
