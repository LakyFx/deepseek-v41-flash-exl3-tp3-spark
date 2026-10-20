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
    binary_root = root / profile.get('binary_subdir', 'bin')
    kit = root / 'kit'
    model = root / 'weights/models' / profile['arm']
    cache = root / profile.get('rank_cache_subdir', 'weights/' + profile['arm'])
    engram = root / profile.get('engram_subdir', 'engram')
    pool = profile['pool_tokens']
    streams = str(profile['parallel'])
    rows = str(profile['round_rows'])
    devices = ','.join(node['rdma_devices'])
    if len(node['rdma_devices']) != 4 or len(node['peer_masks']) != 3 or node['peer_masks'][rank] != 0:
        raise ValueError('Four ordered RDMA devices and a three-entry rank-specific mask required')
    for k in ['master_address', 'nccl_interface', 'nccl_hcas', 'nccl_address_range']:
        if not config.get(k):
            raise ValueError('Operator setting required: ' + k)
    env = {'TF_DS_EXPERIMENTAL_TP3': '1', 'TF_DS_START_TIMEOUT': '1800', 'TF_TP_WORLD': '3',
           'TF_TP_MASTER': config['master_address'], 'TF_TP_PORT': str(config['master_port']),
           'TF_DS_KIT': str(kit), 'TENSORFOLD_CUDA_KERNELS': str(kit / 'aot'),
           'TF_DS_RANK_CACHE': str(cache), 'TF_DS_ENGRAM': str(engram),
           'TF_DS_TOKEN_MAP': str(kit / 'engram-token-map.json'), 'TF_DS_POOL_TOKENS': str(pool),
           'TF_DS_STREAMS': streams, 'TF_DS_ROUND_ROWS': rows, 'TF_DS_GRAPHS': '1',
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
    if profile.get('round_ms'):
        env['TF_DS_ROUND_MS'] = profile['round_ms']
    if profile.get('adaptive_learning'):
        env['TF_DS_ADAPTIVE_LEARNING'] = '1'
    if config.get('library_path'):
        env['LD_LIBRARY_PATH'] = config['library_path']
    if rank == 0:
        argv = [str(binary_root / 'tensorfold-native'), 'serve', str(model), '--backend', 'cuda',
                '--host', '127.0.0.1' if benchmark else config['api_host'], '--port', str(config['api_port']),
                '--name', 'deepseek-v4.1-flash', '--context', str(profile['context_tokens']), '--parallel', streams,
                '--reasoning-effort', profile['reasoning'], '--prompt-cache-gib', str(profile['prompt_snapshot_gib_head']), '--no-update-check']
        if benchmark:
            argv.append('--benchmark-control')
    else:
        argv = [str(binary_root / 'tf-dsv41-lanes'), str(model), str(cache), str(rank), '3',
                config['master_address'], str(config['master_port']), str(kit), '--pool', str(pool),
                '--round-rows', rows, '--streams', streams, '--drafts', '1', '--graphs', '1',
                '--side', '1', '--prefetch', '1', '--aio', '1', '--engram', str(engram),
                '--token-map', str(kit / 'engram-token-map.json'), '--rdma', devices]
    if validate_files:
        if profile.get('adaptive_learning'):
            import hashlib
            receipt = json.loads((binary_root / 'BUILD.json').read_text(encoding='utf-8'))
            manifest = json.loads((HERE/'adaptive/manifest.json').read_text(encoding='utf-8'))
            if (receipt.get('version') != '1.2.3' or receipt.get('adaptive_learning') is not True
                    or receipt.get('source_hashes') != manifest['patched_sha256']
                    or receipt.get('overlay_sha256') != profile['adaptive_overlay_sha256']
                    or receipt.get('source_archive_sha256') != manifest['source_archive_sha256']):
                raise ValueError('Require the verified adaptive 1.2.3 build receipt')
            for binary in ['tensorfold-native','tf-dsv41-lanes']:
                if hashlib.sha256((binary_root/binary).read_bytes()).hexdigest() != receipt['binaries'][binary]['sha256']:
                    raise ValueError('Adaptive binary differs from its build receipt: '+binary)
        operator_values = [config['master_address'], config['nccl_interface'], config['nccl_address_range'],
                           *config['nccl_hcas'], *node['rdma_devices']]
        if any('REPLACE_' in value for value in operator_values):
            raise ValueError('Replace every topology placeholder before starting a rank')
        address = ipaddress.ip_address(config['master_address'])
        if any(address in ipaddress.ip_network(cidr) for cidr in ['192.0.2.0/24', '198.51.100.0/24', '203.0.113.0/24']):
            raise ValueError('Replace the documentation example address with your actual head address')
        if not profile['qualified_pool'] and not profile.get('serving_smoke_qualified'):
            raise ValueError('This profile has neither recorded pool nor serving smoke qualification')
        for path in [Path(argv[0]), model / 'config.json', model / 'vision.safetensors', cache,
                     kit / 'aot/aot.json', kit / 'engram-token-map.json',
                     engram / profile.get('engram_marker', 'model.safetensors.index.json')]:
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
        if profile.get('source_revision'):
            if cfg['tensorfold_tp3']['original_revision'] != profile['source_revision']:
                raise ValueError('Wrong source model revision')
            template = model / 'chat_template.jinja'
            import hashlib
            if hashlib.sha256(template.read_bytes()).hexdigest() != profile['chat_template_sha256']:
                raise ValueError('Runtime chat template differs from the qualified 1017 template')
            source = json.loads((engram / 'SOURCE.json').read_text())
            if (source['rank'] != rank or source['world'] != 3
                    or source['original_revision'] != profile['source_revision']):
                raise ValueError('Wrong Engram rank or source revision')
            for name, metadata in source['files'].items():
                path = engram / name
                if path.parent != engram or path.stat().st_size != metadata['size']:
                    raise ValueError('Wrong Engram file or size: ' + name)
            original = cache / f'rank{rank}of3.bin'
            discovered = [p for p in cache.glob(f'rank{rank}of3-*.bin') if p.is_file() and not p.is_symlink()]
            if (original.is_symlink() or len(discovered) != 1
                    or not original.samefile(discovered[0])):
                raise ValueError('Require one regular rankNof3-*.bin hardlink to the published rank cache')
        elif not list(cache.glob('rank' + str(rank) + '*.bin')):
            raise FileNotFoundError('No exported binary for rank ' + str(rank) + ' in ' + str(cache))
    return argv, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--variant', choices=['TF351', 'TF351-3.3M', 'TF29', '1.2.0', '1.2.3'], required=True)
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
