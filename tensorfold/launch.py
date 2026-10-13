"""Print or run one explicitly configured TP3 rank. Does not replace existing services."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent


def command(config, variant, rank, *, benchmark=False, validate_files=False):
    if len(config['ranks']) != 3 or config['ranks'][rank].get('rank') != rank:
        raise ValueError('Exactly three explicitly ordered TP3 ranks required')
    profile = json.loads((HERE / 'profiles' / (variant + '.json')).read_text())
    node = config['ranks'][rank]
    root = Path(node['root'])
    kit = root / 'kit'
    model = root / 'weights/models' / profile['arm']
    cache = root / 'weights' / profile['arm']
    pool = profile['pool_tokens']
    devices = ','.join(node['rdma_devices'])
    if len(node['rdma_devices']) != 4 or len(node['peer_masks']) != 3 or node['peer_masks'][rank] != 0:
        raise ValueError('Four ordered RDMA devices and a three-entry rank-specific mask required')
    for k in ['master_address', 'nccl_interface', 'nccl_hcas', 'nccl_address_range']:
        if not config.get(k):
            raise ValueError('Operator setting required: ' + k)
    env = {'TF_DS_EXPERIMENTAL_TP3': '1', 'TF_DS_START_TIMEOUT': '1800', 'TF_TP_WORLD': '3',
           'TF_TP_MASTER': config['master_address'], 'TF_TP_PORT': str(config['master_port']),
           'TF_DS_KIT': str(kit), 'TENSORFOLD_CUDA_KERNELS': str(kit / 'aot'),
           'TF_DS_RANK_CACHE': str(cache), 'TF_DS_ENGRAM': str(root / 'engram'),
           'TF_DS_TOKEN_MAP': str(kit / 'engram-token-map.json'), 'TF_DS_POOL_TOKENS': str(pool),
           'TF_DS_STREAMS': '8', 'TF_DS_ROUND_ROWS': '16', 'TF_DS_GRAPHS': '1',
           'TF_DS_HC_SIDE': '1', 'TF_DS_L2_PREFETCH': '1', 'TF_DS_ENGRAM_AIO': '1',
           'TF_DS_CACHE_SHORT_CONTINUATION': '1', 'TF_DS_TOOL_SEPARATOR_ROUNDTRIP': '1',
           'TF_RDMA_DEVICES': devices, 'TF_RDMA_GID_INDEX': str(node['gid_index']),
           'TF_RDMA_PEER_MASKS': ','.join(map(str, node['peer_masks'])), 'TF_RDMA_TRACE': '0',
           'NCCL_IB_SUBNET_AWARE_ROUTING': '1', 'NCCL_NVLS_ENABLE': '0',
           'NCCL_IB_ADDR_RANGE': config['nccl_address_range'], 'NCCL_IB_MERGE_NICS': '0',
           'NCCL_IB_ROCE_VERSION_NUM': '2', 'NCCL_IB_DISABLE': '0', 'NCCL_DEBUG': 'INFO',
           'NCCL_IB_ADDR_FAMILY': 'AF_INET', 'NCCL_IGNORE_CPU_AFFINITY': '1',
           'NCCL_IB_HCA': ','.join(config['nccl_hcas']), 'NCCL_NET_PLUGIN': 'none',
           'NCCL_CUMEM_ENABLE': '0', 'NCCL_NET': 'IB', 'NCCL_SOCKET_IFNAME': '=' + config['nccl_interface'],
           'TENSORFOLD_NO_UPDATE_CHECK': '1', 'HF_HUB_OFFLINE': '1'}
    if config.get('library_path'):
        env['LD_LIBRARY_PATH'] = config['library_path']
    if rank == 0:
        argv = [str(root / 'bin/tensorfold-native'), 'serve', str(model), '--backend', 'cuda',
                '--host', '127.0.0.1' if benchmark else config['api_host'], '--port', str(config['api_port']),
                '--name', 'deepseek-v4.1-flash', '--context', '401408', '--parallel', '8',
                '--reasoning-effort', 'max', '--prompt-cache-gib', '1', '--no-update-check']
        if benchmark:
            argv.append('--benchmark-control')
    else:
        argv = [str(root / 'bin/tf-dsv41-lanes'), str(model), str(cache), str(rank), '3',
                config['master_address'], str(config['master_port']), str(kit), '--pool', str(pool),
                '--round-rows', '16', '--streams', '8', '--drafts', '1', '--graphs', '1',
                '--side', '1', '--prefetch', '1', '--aio', '1', '--engram', str(root / 'engram'),
                '--token-map', str(kit / 'engram-token-map.json'), '--rdma', devices]
    if validate_files:
        operator_values = [config['master_address'], config['nccl_interface'], config['nccl_address_range'],
                           *config['nccl_hcas'], *node['rdma_devices']]
        if any('REPLACE_' in value for value in operator_values):
            raise ValueError('Replace every topology placeholder before starting a rank')
        address = ipaddress.ip_address(config['master_address'])
        if any(address in ipaddress.ip_network(cidr) for cidr in ['192.0.2.0/24', '198.51.100.0/24', '203.0.113.0/24']):
            raise ValueError('Replace the documentation example address with your actual head address')
        if not profile['qualified_pool']:
            raise ValueError('This profile has no recorded pool qualification')
        for path in [Path(argv[0]), model / 'config.json', model / 'vision.safetensors', cache,
                     kit / 'aot/aot.json', kit / 'engram-token-map.json', root / 'engram/model.safetensors.index.json']:
            if not path.exists():
                raise FileNotFoundError(path)
        rope = json.loads((kit / 'rope.json').read_text())
        if rope['rows'] < pool + 49:
            raise ValueError('RoPE tables too short for this pool')
        for name in ['rope-plain-cos.f32', 'rope-plain-sin.f32', 'rope-compressed-cos.f32', 'rope-compressed-sin.f32']:
            if (kit / name).stat().st_size != rope['rows'] * rope['half'] * 4:
                raise ValueError('Incomplete RoPE table: ' + name)
        cfg = json.loads((model / 'config.json').read_text())
        if cfg['tensorfold_tp3']['arm'] != profile['arm'] or cfg['tensorfold_tp3']['world_size'] != 3:
            raise ValueError('Wrong model metadata for rank cache/profile')
        if not list(cache.glob('rank' + str(rank) + '*.bin')):
            raise FileNotFoundError('No exported binary for rank ' + str(rank) + ' in ' + str(cache))
    return argv, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--variant', choices=['TF351', 'TF351-3.3M', 'TF29'], required=True)
    parser.add_argument('--rank', choices=range(3), type=int, required=True)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    argv, env = command(json.loads(args.config.read_text()), args.variant, args.rank,
                        benchmark=args.benchmark, validate_files=args.run)
    if args.run:
        os.execvpe(argv[0], argv, {**os.environ, **env})
    print(json.dumps({'variant': args.variant, 'rank': args.rank, 'argv': argv, 'environment': env,
                      'shell_command': 'env ' + ' '.join(shlex.quote(k + '=' + v) for k, v in env.items())
                      + ' ' + shlex.join(argv), 'started': False}, indent=2))


if __name__ == '__main__':
    main()
