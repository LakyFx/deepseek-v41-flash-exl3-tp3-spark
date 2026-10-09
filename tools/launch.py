"""Print or run one X11c rank, with operator supplied topology and local paths."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def build(config, rank, image_aot_sha):
    p = json.loads((ROOT / 'config/x11c.json').read_text())
    nodes = config['nodes']
    if len(nodes) != 3 or [n['rank'] for n in nodes] != [0, 1, 2]:
        raise ValueError('exactly ranks 0, 1, 2 are required')
    node = nodes[rank]
    for n in nodes:
        if n['control_address'].startswith('REPLACE_'):
            raise ValueError('replace all example topology fields')
    if config['image'].endswith(':latest'):
        raise ValueError('pin a recipe image tag or digest, never latest')
    env = dict(p['environment'])
    env.update({
        'DGX_DIAG_RANK': str(rank), 'VLLM_HOST_IP': node['control_address'],
        'DGX_CAMPAIGN_R_AOT_SHA256': image_aot_sha,
        'DGX_V089_FINGERPRINT': hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest(),
        'NCCL_IB_ROCE_VERSION_NUM': '2', 'NCCL_CUMEM_ENABLE': '0',
        'NCCL_IGNORE_CPU_AFFINITY': '1', 'NCCL_DEBUG': 'INFO', 'NCCL_IB_MERGE_NICS': '0',
        'NCCL_IB_ADDR_FAMILY': 'AF_INET', 'NCCL_IB_DISABLE': '0', 'NCCL_NET': 'IB',
        'NCCL_NVLS_ENABLE': '0', 'NCCL_NET_PLUGIN': 'none', 'NCCL_IB_SUBNET_AWARE_ROUTING': '1',
        'NCCL_IB_ADDR_RANGE': config['roce_cidr'], 'NCCL_IB_HCA': ','.join(node['hcas']),
        'NCCL_SOCKET_IFNAME': '=' + node['control_interface'],
        'GLOO_SOCKET_IFNAME': node['control_interface'],
        'SPARKNET_ROCE_TOPOLOGY': 'direct', 'SPARKNET_ROCE_KERNELS': 'cute',
        'SPARKNET_ROCE_GID_INDEX': str(node['gid_index']),
        'SPARKNET_ROCE_CACHE_DIR': '/opt/dgx_campaign/native/roce',
        'SPARKNET_ROCE_PEER_HCAS': json.dumps(node['peer_hcas']),
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,garbage_collection_threshold:0.6',
    })
    if set(node['peer_hcas']) != {str(i) for i in range(3) if i != rank}:
        raise ValueError('peer_hcas must cover the two other TP ranks')
    if any(len(h) != 2 or any(x not in node['hcas'] for x in h) for h in node['peer_hcas'].values()):
        raise ValueError('the measured TP3 topology uses two HCA lanes per peer')
    benchmark = config.get('benchmark_control', {})
    if benchmark.get('enabled'):
        env['DGX_CAMPAIGN_CONTROL_PEER_ADDRESS'] = benchmark['peer_address']
    argv = ['docker', 'run', '--name', f'dsv41-x11c-rank{rank}', '--network', 'host',
            '--gpus', 'all', '--ipc', 'host', '--cap-add', 'IPC_LOCK', '--ulimit', 'memlock=-1',
            '--device', '/dev/infiniband', '--entrypoint', 'vllm']
    for key, value in sorted(env.items()):
        argv += ['--env', key + '=' + value]
    for field, target, readonly in [
        ('body_path', '/models/dsv41', True), ('engram_path', '/engram-local', True),
        ('cache_path', '/cache', False), ('state_path', '/dgx-v089-profile', False),
        ('state_path', '/dgx-v089x-profile', False), ('diag_path', '/dgx-diag', False),
        ('chunk_path', '/dgx-chunk', False), ('snapshot_path', '/dgx-kv-snapshots', False),
    ]:
        source = node[field]
        if not source.startswith('/') or ',' in source or '\n' in source:
            raise ValueError('absolute Linux paths without commas required')
        argv += ['--mount', f'type=bind,src={source},dst={target}' + (',readonly' if readonly else '')]
    if benchmark.get('enabled'):
        argv += ['--mount', f"type=bind,src={benchmark['lease_path']},dst=/opt/dgx_campaign/lease,readonly"]
    argv += [config['image'], 'serve', '/models/dsv41', '--served-model-name', 'deepseek-v4.1-flash',
             '--host', config.get('backend_bind', '127.0.0.1'), '--port', str(config.get('backend_port', 8003)),
             '--tensor-parallel-size', '3', '--gpu-memory-utilization', '0.80',
             '--kv-cache-memory-bytes', str(p['kv_cache_memory_bytes_per_rank']),
             '--max-model-len', str(p['max_model_len']), '--max-num-seqs', '8',
             '--max-num-batched-tokens', '4096', '--kv-cache-dtype', 'fp8', '--block-size', '128',
             '--prefix-cache-retention-interval', '128', '--engram-config', '{"cpu_offload":false}',
             '--default-chat-template-kwargs', '{"thinking":true,"reasoning_effort":"max"}',
             '--tool-call-parser', 'deepseek_v41', '--enable-auto-tool-choice',
             '--reasoning-parser', 'deepseek_v41', '--limit-mm-per-prompt', '{"image":32}',
             '--mm-processor-cache-gb', '1', '--mm-encoder-tp-mode', 'data',
             '--distributed-executor-backend', 'mp', '--nnodes', '3', '--node-rank', str(rank),
             '--master-addr', nodes[0]['control_address'], '--master-port', str(config.get('master_port', 29541)),
             '--enable-prompt-tokens-details', '--middleware', 'dsv41_gateway_middleware.OwnedAbortMiddleware',
             '--speculative-config', json.dumps({'method': 'dspark', 'num_speculative_tokens': 5,
                    'draft_sample_method': 'probabilistic', 'rejection_sample_method': 'block',
                    'enable_adaptive_verification': True}),
             '--compilation-config', json.dumps({'cudagraph_mode': 'FULL_AND_PIECEWISE',
                    'cudagraph_capture_sizes': p['cuda_graph_capture_sizes']})]
    if benchmark.get('enabled'):
        argv += ['--middleware', 'dgx_campaign_runtime.control.CampaignControl']
    if rank != 0:
        argv += ['--headless']
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/cluster.local.json')
    parser.add_argument('--rank', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--run', action='store_true', help='execute locally on this rank host')
    parser.add_argument('--aot-sha256', help='offline plan only; run always reads the image manifest')
    a = parser.parse_args()
    config = json.loads(a.config.read_text())
    digest = a.aot_sha256
    if a.run or not digest:
        # No GPU access, serving or writable mounts in this image metadata read.
        manifest = subprocess.check_output(['docker', 'run', '--rm', '--entrypoint', 'cat',
                     config['image'], '/opt/dgx_campaign/native/roce-aot/AOT-MANIFEST.json'])
        digest = hashlib.sha256(manifest).hexdigest()
    if len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest):
        parser.error('a lowercase SHA256 AOT manifest digest is required')
    command = build(config, a.rank, digest)
    if not a.run:
        print(shlex.join(command))
        return
    node = config['nodes'][a.rank]
    for field in ['body_path', 'engram_path']:
        if not Path(node[field]).is_dir():
            raise FileNotFoundError(node[field])
    local = json.loads((Path(node['engram_path']) / 'engram-local.json').read_text())
    from model import ranges
    expected = ranges(json.loads((ROOT / 'config/model-config.reference.json').read_text()), a.rank)
    if local['layers'] != expected:
        raise ValueError('local Engram row ranges do not match this TP3 rank')
    if json.loads((Path(node['body_path']) / 'config.json').read_text()) != json.loads(
            (ROOT / 'config/model-config.reference.json').read_text()):
        raise ValueError('body must have the selected TP3 virtual-head config')
    for field in ['cache_path', 'state_path', 'diag_path', 'chunk_path', 'snapshot_path']:
        Path(node[field]).mkdir(parents=True, exist_ok=True)
    # Foreground docker run deliberately fails on an existing same-name rank.
    # The launcher never stops or replaces an unknown running model.
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
