"""Candidate inventory. Never equate source preparation with GPU qualification."""
from __future__ import annotations

BASELINE = '0.8.9X11c-host64k'
IMAGE = 'sha256:24b2387d00c974f9aba0c38135457b045e24025b0eec525e8a5e04e9d2c7775c'
BASELINE_FINGERPRINT = 'ac100567958566a48d41fcb9f19a140dda54cb70fa389b120896aa0537cf2f81'
GRAPHS = [5,6,10,12,15,18,20,24,25,30,35,36,40,42,48]
HERMES_WORKLOAD_SHA = 'df3f710fd0cb901c262449c5c23490cf8503d087cb939a92e4f6f2a6dfd67d10'
OWEN_FIXTURE_SHA = '40427145003f73d3db195de5e98810fcc6ed294d68872df8028f9a3181fcb367'
OWEN_SOURCE_SHA = 'b130f172ec4e039532e9fb0b7d4d6256ddc5f16918c88d0689841146631ba46a'

CANDIDATES = {
    'N1': {'description':'Full valid K5 verification for one pure decode request', 'parts':['N1']},
    'N2': {'description':'Stock lagged mHC for small pure decode; mixed/prefill stay fused', 'parts':['N2']},
    'N3': {'description':'Same probabilistic draft, reusable logits scratch and batched position keys', 'parts':['N3']},
    'N4': {'description':'EXL3 prefill tile/warp specialization, retaining existing decode-once arithmetic', 'parts':['N4']},
    'N5': {'description':'Exact compatible indexer optimization', 'parts':['N5']},
    'N6': {'description':'Bounded decoder replay with cache and custom SP integration', 'parts':['N6']},
    'N7a': {'description':'Pure prefill SP threshold 512', 'parts':['N7a']},
    'N7b': {'description':'Pure prefill SP threshold 256', 'parts':['N7b']},
    'N8a': {'description':'B12X MXFP8 A16 rows1..6 splitK2; larger rows unchanged', 'parts':['N8a']},
    'N8b': {'description':'B12X MXFP8 A16 rows1..6 splitK1; larger rows unchanged', 'parts':['N8b']},
    'N9': {'description':'Up to three trace-selected additional graph buckets', 'parts':['N9']},
    'C1': {'description':'Single-stream specialization', 'parts':['N1','N2']},
    'C2': {'description':'Single-stream specialization and cheaper same-distribution draft', 'parts':['N1','N2','N3']},
    'C3': {'description':'Balanced decode plus EXL3 prefill', 'parts':['N1','N2','N3','N4']},
    'C4': {'description':'Balanced decode/prefill plus bounded replay', 'parts':['N1','N2','N3','N4','N6']},
    'C5-indexer': {'description':'Balanced candidate with indexer', 'parts':['N1','N2','N3','N4','N5']},
    'C5-sp512': {'description':'Balanced candidate with SP512', 'parts':['N1','N2','N3','N4','N7a']},
    'C5-sp256': {'description':'Balanced candidate with SP256', 'parts':['N1','N2','N3','N4','N7b']},
}


def benchmark_plan():
    return {
        'schema':'dgx.x11c-next.benchmark-plan.v1',
        'execution_authorized':False,
        'primary':'hermes-coverage20',
        'per_loaded_arm':[
            {'suite':'hermes-coverage20','workload_sha256':HERMES_WORKLOAD_SHA,
             'hard_cap_s':1200,'reasoning':'MAX','status':'frozen standard suite unchanged'},
            {'suite':'owen-common-context','fixture_sha256':OWEN_FIXTURE_SHA,
             'source_sha256':OWEN_SOURCE_SHA,'hard_cap_s':1200,
             'reasoning':'same server MAX; author effort unknown',
             'scope':'code/prose C1,C2,C4,C8; discarded warmup + three samples; cold32K/256K twice'},
        ],
        'owen_every_measured_arm':True,
        'reload_between_suites':False,
        'reset_between_suites':'owned-request drain plus quarantined prefix cache reset',
        'results':'separate suite identifiers and metric denominators; never merge rates',
        'targets':{'hermes_C1_request_decode_tps':70,'hermes_C8_wall_output_tps':180,
                   'prefill':'no demonstrated regression against fresh X11c baseline'},
        'timing':'Each suite has its own 20-minute cap. Two suites are not promised within 20 minutes combined. Loading/qualification is additional.',
        'failure':'Stop on RPC error or foreign traffic. Never retry stale rank queues. Keep ingress closed until qualified final arm or exact X11c rollback.',
        'supplement':'Bounded separate C1..C8 curve; never substitute it for standard Hermes metrics',
        'production':'No deployment or benchmark started during preparation',
    }
