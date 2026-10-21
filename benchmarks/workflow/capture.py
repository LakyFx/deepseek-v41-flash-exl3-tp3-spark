"""Frozen native SSE timing capture; no token rate inferred from buffered chunks."""
import json
from .streaming import parse_sse as frozen_parse

def parse_sse(lines):
    extra={};frame=[]
    def captured():
        for when,line in lines:
            text=line.rstrip(b'\r\n')
            if text.startswith(b'data:'):frame.append(text[5:].lstrip(b' '))
            elif not text and frame:
                raw=b'\n'.join(frame);frame.clear()
                if raw!=b'[DONE]':
                    value=json.loads(raw)
                    for key in ('tensorfold','speculative'):
                        if key in value:extra[key]=value[key]
            yield when,line
    reply=frozen_parse(captured());reply.native_extra=extra;return reply

def timing(extra,usage):
    r=extra.get('tensorfold') or {}
    p=r.get('prefill_seconds');ttft=r.get('time_to_first_token')
    rate=r.get('tokens_per_second');total=r.get('seconds')
    decoded_seconds=(usage['completion_tokens']-1)/rate if rate and usage['completion_tokens']>1 else None
    # Native source: total=finished-submitted, decoded=finished-prefill_event,
    # TTFT=first-received. U therefore equals first-prefill_event PLUS
    # submitted-received. Unlike TTFT-prefill it excludes scheduler queueing.
    # The binary does not export received->submitted; NEVER call U exact.
    upper=ttft-total+decoded_seconds if all(x is not None for x in (ttft,total,decoded_seconds)) else None
    if upper is not None and upper < -1e-4:raise ValueError('inconsistent native timing')
    return {'prefill_seconds':p,'computed_prefill_tokens':usage['computed_prompt_tokens'],
        'computed_prefill_tps':usage['computed_prompt_tokens']/p if p else None,
        'native_ttft_seconds':ttft,'native_decode_tps':rate,'native_decode_seconds':decoded_seconds,
        'post_prefill_first_token_upper_bound_seconds':max(0,upper) if upper is not None else None,
        'post_prefill_first_token_exact_seconds':None,
        'post_prefill_scope':'upper bound = exact post-prefill event latency + server preparation before submit; scheduler queue and prefill excluded; frozen binaries do not export preparation duration'}
