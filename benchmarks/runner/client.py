"""Explicitly armed, owned SSE requests over the campaign's bound transport.

The future controller supplies the verified network transport only after it
has closed production ingress. This module has no service-management actions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import socket
import threading
import time
import uuid

from .streaming import parse_sse


class Client:
    def __init__(self, network, *, armed: bool = False, journal=None):
        self.network = network
        self.armed = armed
        self.journal = journal
        self.lock = threading.Lock()
        self.active = {}
        self.pending_aborts = set()

    def _checkpoint(self) -> None:
        if self.journal is not None:
            self.journal({"active_ids": sorted(self.active),
                          "abort_pending_ids": sorted(self.pending_aborts), "epoch": time.time()})

    def complete(self, body: dict, *, deadline: float, on_first_delta=None) -> dict:
        if not self.armed:
            raise RuntimeError("inference client is unarmed")
        request_id = "dgx-gateway-" + uuid.uuid4().hex
        server_id = "chatcmpl-" + request_id
        payload = copy.deepcopy(body)
        payload.update(request_id=request_id, stream=True, stream_options={"include_usage": True})
        raw = json.dumps(payload, ensure_ascii=False).encode()
        started = time.monotonic()
        if started >= deadline:
            raise TimeoutError("request deadline before connection")
        connection = self.network.connection(self.network.BASE, timeout=min(5.0, deadline - started))
        with self.lock:
            self.active[server_id] = None
            self._checkpoint()
        try:
            connection.connect()
            request_socket = connection.sock
            request_socket.settimeout(max(0.001, deadline - time.monotonic()))
            with self.lock:
                self.active[server_id] = connection.sock
                self._checkpoint()
            connection.request("POST", "/v1/chat/completions", body=raw,
                               headers={"Content-Type": "application/json", "Connection": "close"})
            with connection.getresponse() as response:
                if response.status != 200:
                    raise ValueError("inference HTTP status: " + str(response.status))
                def lines():
                    while True:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("request stream deadline")
                        if response.isclosed():
                            return
                        # HTTPConnection detaches sock for Connection: close;
                        # the response still owns the underlying file object.
                        request_socket.settimeout(max(0.001, deadline - time.monotonic()))
                        line = response.readline(2 * 1024**2 + 1)
                        if not line:
                            return
                        yield time.monotonic(), line
                reply = parse_sse(lines(), on_first_delta=on_first_delta)
            reply.usage_counts()
            return {"reply": reply, "request_id": request_id,
                    "body_sha256": hashlib.sha256(raw).hexdigest(),
                    "started_monotonic": started, "ended_monotonic": time.monotonic()}
        except BaseException:
            # Abort ONLY this owned request, also when usage/DONE was truncated.
            # Retain cancellation debt until the controller proves the engine
            # idle, even when the abort HTTP call itself fails or is only acked.
            with self.lock:
                self.pending_aborts.add(server_id)
                self._checkpoint()
            try:
                self.network.request(self.network.BASE, "POST", "/abort_requests",
                                     {"request_ids": [server_id]})
            except Exception:
                # Preserve the original stream error; controller retains the
                # owned-ID journal and can abort/recover the whole failed arm.
                pass
            finally:
                try:
                    if connection.sock is not None:
                        connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            raise
        finally:
            connection.close()
            with self.lock:
                self.active.pop(server_id, None)
                self._checkpoint()

    def abort_all_owned(self) -> None:
        if not self.armed:
            return
        with self.lock:
            rows = list(self.active.items())
            ids = sorted(set(self.active) | self.pending_aborts)
        if not ids:
            return
        try:
            self.network.request(self.network.BASE, "POST", "/abort_requests",
                                 {"request_ids": ids})
        finally:
            for _, connection in rows:
                if connection is not None:
                    try:
                        connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

    def acknowledge_owned_idle(self, proof):
        """Clear cancellation debt only after a bound closed-window idle proof."""
        if not self.armed:
            raise RuntimeError("inference client is unarmed")
        with self.lock:
            if self.active:
                raise RuntimeError("client threads still own active requests")
            if (not isinstance(proof,dict) or proof.get("exclusive") is not True
                or not proof.get("engine_generation") or not proof.get("campaign_id")
                or proof.get("running") != 0 or proof.get("waiting") != 0
                or proof.get("owned_ids") != sorted(self.pending_aborts)):
                raise ValueError("exact engine-idle/cancellation ownership proof required")
            self.pending_aborts.clear()
            self._checkpoint()
