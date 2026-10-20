"""Buffer one image inference; reasoning/partial loops never escape as an answer."""
import json
import re
import time

from .vision_jobs import error, repeated


def final_text(text):
    text = re.sub(r'<(think|thinking|reasoning|analysis)>.*?</\1>', '', text, flags=re.I|re.S)
    return re.sub(r'<(?:think|thinking|reasoning|analysis)>.*$', '', text, flags=re.I|re.S).strip()


class Collector:
    def __init__(self):
        self.pending = b''
        self.content = self.reasoning = ''
        self.finish = None
        self.done = False
        self.identity = None
        self.model = None
        self.usage = None
        self.repetitive = False

    def feed(self, chunk):
        self.pending = (self.pending+chunk).replace(b'\r\n',b'\n')
        if len(self.pending) + len(self.content.encode()) + len(self.reasoning.encode()) > 4*1024*1024:
            raise ValueError('Vision response exceeded retained result limit')
        while b'\n\n' in self.pending:
            block,self.pending = self.pending.split(b'\n\n',1)
            data = b'\n'.join(line[5:].lstrip() for line in block.splitlines() if line.startswith(b'data:'))
            if not data:
                continue
            if data == b'[DONE]':
                self.done = True
                continue
            payload = json.loads(data)
            if payload.get('error'):
                raise ValueError('Provider returned a streaming error')
            self.identity = payload.get('id') or self.identity
            self.model = payload.get('model') or self.model
            self.usage = payload.get('usage') or self.usage
            for choice in payload.get('choices') or []:
                delta = choice.get('delta') or {}
                if not self.repetitive:
                    self.content += delta.get('content') or ''
                    self.reasoning += delta.get('reasoning_content') or delta.get('reasoning') or ''
                self.finish = choice.get('finish_reason') or self.finish
                if delta.get('tool_calls'):
                    raise ValueError('Standalone vision returned unexpected tool calls')
            if not self.repetitive and (repeated(self.content) or repeated(self.reasoning)):
                # Reject the answer, but drain this SAME bounded generation.
                # Raising here closes its stream before a terminal receipt;
                # a backend without abort support then leaves every image job
                # blocked as UNKNOWN. Transport errors/deadlines still fail
                # closed through the existing engine and job worker.
                self.repetitive = True
                self.content = self.reasoning = ''

    def result(self, identity):
        if self.repetitive:
            if (not self.done or self.pending.strip()
                    or self.finish not in {'stop','length','content_filter','end_turn','completed'}):
                raise ValueError('Repetitive vision stream ended without confirmed termination')
            return {'status_code':422,'data':error('vision_output_repetitive',
                'Vision generation finished with repetitive output. Output was discarded; do not automatically regenerate.',
                identity),'headers':{}}
        content = final_text(self.content)
        if not self.done or self.pending.strip() or self.finish not in {'stop','end_turn','completed'} or not content:
            code = 'vision_output_truncated' if self.finish == 'length' else 'vision_output_incomplete'
            return {'status_code':422,'data':error(code,
                'Vision returned no complete final answer. Partial/reasoning content was discarded; do not automatically regenerate.',identity),'headers':{}}
        data = {'id':self.identity or identity,'object':'chat.completion','created':int(time.time()),
                'model':self.model,'choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'stop'}]}
        if self.usage:
            data['usage'] = self.usage
        return {'status_code':200,'data':data,'headers':{}}


async def collect(engine, row, on_backend):
    from .engine import GatewayError
    handle = None
    iterator = None
    outcome = 'vision_job_failed'
    try:
        body = json.loads(row['body'])
        body['stream'] = True
        body['stream_options'] = {'include_usage':True}
        handle = await engine.begin_stream(body)
        on_backend(handle.upstream_request_id)
        if handle.response.status_code >= 400:
            await handle.response.aread()
            # A rejected request is never an excuse for a second inference.
            return {'status_code':422,'data':error('vision_provider_rejected',
                'Provider rejected this vision job. Do not submit another generation automatically.',row['id']),'headers':{}}
        result = Collector()
        iterator = engine.iter_stream(handle)
        async for chunk in iterator:
            result.feed(chunk)
        response = result.result(row['id'])
        outcome = 'success'
        response['headers'] = engine._response_headers(handle.prepared,queue_wait_seconds=handle.queue_wait_seconds)
        return response
    except GatewayError as exc:
        if handle is None and exc.status_code in {400,401,403,404,413,422,429}:
            return {'status_code':422,'data':error('vision_input_rejected',str(exc),row['id']),'headers':{}}
        raise
    finally:
        if iterator is not None:
            await iterator.aclose()
        if handle is not None:
            await engine.close_stream(handle,outcome=outcome)
