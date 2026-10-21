import http.client
import json
import math
import socket
import statistics
import time

class Network:
    BASE="loopback-backend"
    def __init__(self,port):self.port=port
    def connection(self,base,timeout=5):
        if base!=self.BASE:raise ValueError("loopback only")
        return http.client.HTTPConnection("127.0.0.1",self.port,timeout=timeout)
    def validate_body(self,body):
        if "cache_salt" in body:raise ValueError("Use physical namespace, not unproven salt")
    def cancel_owned(self,ids,sockets):
        for s in sockets:
            if s is not None:
                try:s.shutdown(socket.SHUT_RDWR)
                except OSError:pass
        return {"signal":"disconnect","acknowledged":False}
    def get(self,path):
        conn=self.connection(self.BASE)
        try:
            conn.request("GET",path);r=conn.getresponse();data=r.read(8*1024**2+1)
            if r.status!=200 or len(data)>8*1024**2:raise ValueError(f"GET {path}: {r.status}")
            return data.decode() if path=="/metrics" else json.loads(data)
        finally:conn.close()

def metrics(network):
    result={}
    for line in network.get("/metrics").splitlines():
        if line.startswith("tensorfold:") and "{" not in line:
            key,value=line.split()[:2];n=float(value)
            if not math.isfinite(n) or n<0:raise ValueError("invalid native metric")
            result[key.removeprefix("tensorfold:")]=n
    for k in ["requests_running","requests_waiting","generation_tokens_total","request_latency_seconds_count"]:
        if k not in result:raise ValueError("required metric absent: "+k)
    return result

def comparison(before,after,expected,output_tokens):
    keys=["generation_tokens_total","request_latency_seconds_count","prefill_computed_tokens_total",
          "prefill_duration_seconds_total","request_decode_seconds_sum","prefix_cache_queries_total",
          "prefix_cache_hits_total","mtp_drafted_total","mtp_accepted_total","decode_rounds_total"]
    d={k:after[k]-before[k] if k in before and k in after else None for k in keys}
    if any(v is not None and v<0 for v in d.values()):raise ValueError("native counters reset")
    coverage=(d["request_latency_seconds_count"]==expected and d["generation_tokens_total"]==output_tokens)
    def ratio(a,b):return d[a]/d[b] if coverage and d[a] is not None and d[b] else None
    return dict(coverage_verified=coverage,expected_requests=expected,deltas=d,
        computed_prefill_tps=ratio("prefill_computed_tokens_total","prefill_duration_seconds_total"),
        weighted_request_decode_tps=ratio("generation_tokens_total","request_decode_seconds_sum"),
        prefix_token_hit_fraction=ratio("prefix_cache_hits_total","prefix_cache_queries_total"),
        draft_acceptance_fraction=ratio("mtp_accepted_total","mtp_drafted_total"),
        output_tokens_per_request_decode_round=ratio("generation_tokens_total","decode_rounds_total"),
        decode_round_scope="sum of finished-request rounds; NOT unique shared GPU rounds",
        decode_rate_scope="summed request decode time, NOT aggregate GPU throughput")

def idle(network,client,deadline):
    while time.monotonic()<deadline:
        m=metrics(network)
        with client.lock:busy=bool(client.active)
        if not busy and m["requests_running"]==0 and m["requests_waiting"]==0:
            with client.lock:client.pending_aborts.clear()
            return m
        time.sleep(.2)
    raise TimeoutError("engine not idle; next cell must not start")

def summarize_latencies(result):
    seconds=sorted(t["elapsed_seconds"] for t in result["tasks"] if "elapsed_seconds" in t)
    first=[r["first_any_delta_monotonic"]-r["started_monotonic"] for t in result["tasks"] for r in t["requests"] if r["first_any_delta_monotonic"] is not None]
    if seconds:result.update(task_latency_median_s=statistics.median(seconds),task_latency_p95_s=seconds[math.ceil(.95*len(seconds))-1])
    if first:result["TTFT_median_s"]=statistics.median(first)
