"""Explicitly armed loopback SSE client; natural completion, owned disconnects."""
import copy
import hashlib
import json
import socket
import threading
import time
import uuid
from .network import Network, metrics
from .capture import parse_sse


class BaseClient:
    def __init__(self, port, *, armed=False, journal=None, stall_seconds=900):
        if not 1024 <= port <= 65535: raise ValueError('explicit loopback port required')
        self.network=Network(port)
        self.armed=armed
        self.journal=journal
        self.stall_seconds=stall_seconds
        self.lock=threading.Lock()
        self.active={}
        self.pending_aborts=set()
        self.cancelled=False

    def checkpoint(self):
        if self.journal:
            self.journal({'epoch':time.time(),'active':sorted(self.active),
                          'unconfirmed_disconnects':sorted(self.pending_aborts)})

    def complete(self, body):
        if not self.armed: raise RuntimeError('client unarmed')
        with self.lock:
            if self.cancelled:raise RuntimeError('campaign cancelled; no further requests')
        ident='dgx-gateway-'+uuid.uuid4().hex
        server_id='chatcmpl-'+ident
        payload=copy.deepcopy(body)
        payload.update(request_id=ident,stream=True,stream_options={'include_usage':True})
        self.network.validate_body(payload)
        raw=json.dumps(payload,ensure_ascii=False).encode()
        started=time.monotonic()
        connection=self.network.connection(self.network.BASE,timeout=5)
        with self.lock:
            if self.cancelled:raise RuntimeError('campaign cancelled; no submission')
            self.active[server_id]=None;self.checkpoint()
        try:
            connection.connect()
            stream_socket=connection.sock
            stream_socket.settimeout(self.stall_seconds)
            with self.lock:
                self.active[server_id]=stream_socket;self.checkpoint()
                if self.cancelled or server_id in self.pending_aborts: raise OSError('owned request cancelled')
            connection.request('POST','/v1/chat/completions',body=raw,
                headers={'Content-Type':'application/json','Connection':'close'})
            with connection.getresponse() as response:
                if response.status!=200: raise ValueError('inference HTTP '+str(response.status))
                def lines():
                    while True:
                        line=response.readline(2*1024**2+1)
                        if not line:return
                        yield time.monotonic(),line
                reply=parse_sse(lines())
            reply.usage_counts()
            return {'reply':reply,'request_id':ident,'body_sha256':hashlib.sha256(raw).hexdigest(),
                'started_monotonic':started,'ended_monotonic':time.monotonic(),
                'first_any_delta_monotonic':reply.first_any_delta,
                'first_visible_delta_monotonic':reply.first_visible_delta,
                'stream_gap_seconds':sorted(b-a for a,b in zip(reply.delta_times,reply.delta_times[1:]))}
        except BaseException:
            with self.lock:
                self.pending_aborts.add(server_id);self.checkpoint()
            try:
                if connection.sock:connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            raise
        finally:
            connection.close()
            with self.lock:
                self.active.pop(server_id,None);self.checkpoint()

    def abort_all_owned(self):
        with self.lock:
            self.cancelled=True
            self.pending_aborts.update(self.active)
            for stream in self.active.values():
                if stream:
                    try:stream.shutdown(socket.SHUT_RDWR)
                    except OSError:pass
            self.checkpoint()

    def require_idle(self):
        value=metrics(self.network)
        with self.lock:
            if self.active or self.pending_aborts:
                raise RuntimeError('owned cancellation debt requires reconciliation; stop campaign')
        if value['requests_running'] or value['requests_waiting']:
            raise RuntimeError('backend not idle; do not start next wave')
        return value
