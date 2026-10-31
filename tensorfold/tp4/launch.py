"""Print or run ONE explicitly configured TP4 rank; never replaces a service."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import shlex

from tensorfold.tp4 import assets
from tensorfold.variants import patch

HERE = Path(__file__).resolve().parent
INCOMPATIBLE_ENV = ('TF_DS_EXPERIMENTAL_TP3', 'TF_RDMA_DEVICES', 'TF_RDMA_PEER_MASKS',
                    'TF_RDMA_GID_INDEX', 'TF_DS_RDMA_PROXY_CPUS', 'NCCL_MIN_NCHANNELS',
                    'NCCL_MAX_NCHANNELS')


def process_environment(env):
    return {**{k: v for k, v in os.environ.items() if k not in INCOMPATIBLE_ENV}, **env}


def command(config, rank, *, benchmark=False, validate_files=False):
    profile = json.loads((HERE/'profile.json').read_bytes())
    if len(config['ranks']) != 4 or [r['rank'] for r in config['ranks']] != list(range(4)):
        raise ValueError('Exactly four explicitly ordered TP4 ranks are required')
    if rank not in range(4):
        raise ValueError('Rank must be 0..3')
    node = config['ranks'][rank]
    root = Path(node['root'])
    kit, model, cache = root/'kit', root/'model', root/'rank-cache'
    engram = root/f'engram-tp2-r{rank % 2}'
    first, second = node['neighbor_first'], node['neighbor_second']
    hcas = node['nccl_hcas']
    if (len(first) != 2 or len(second) != 2 or len(hcas) != 4
            or len(set(first + second)) != 4 or set(first + second) != set(hcas)):
        raise ValueError('Two disjoint two-device neighbor rails and four matching NCCL HCAs required')
    required = [config['master_address'], node['nccl_interface'], config['nccl_address_range'],
                node['library_path'], node['root'], *hcas]
    if not all(isinstance(v, str) and v for v in required):
        raise ValueError('Operator-owned topology values are required')
    env = dict(profile['runtime_environment'])
    env.update(TF_TP_MASTER=config['master_address'],
        TF_TP_PORT=str(config['master_port']), TF_DS_KIT=str(kit),
        TF_DS_RANK_CACHE=str(cache), TF_DS_ENGRAM=str(engram),
        TF_DS_TOKEN_MAP=str(kit/'engram-token-map.json'),
        TENSORFOLD_CUDA_KERNELS=str(root/'prefill/aot'),
        TF_DS_PREFILL_RAGGED_CUBIN=str(root/'prefill/auxiliary/tp4_mma.cubin'),
        TF_DS_NEIGHBOR_FIRST=','.join(first), TF_DS_NEIGHBOR_SECOND=','.join(second),
        TF_DS_NEIGHBOR_GID_INDEX=str(node['gid_index']),
        LD_LIBRARY_PATH=node['library_path'], NCCL_IB_ADDR_RANGE=config['nccl_address_range'],
        NCCL_IB_HCA=','.join(hcas), NCCL_SOCKET_IFNAME='='+node['nccl_interface'])
    if rank == 0:
        argv = [str(root/'native/tensorfold-native'), 'serve', str(model), '--backend', 'cuda',
            '--host', '127.0.0.1' if benchmark else config['api_host'], '--port', str(config['api_port']),
            '--name', 'deepseek-v4.1-flash', '--context', '401408', '--parallel', '16',
            '--reasoning-effort', 'max', '--prompt-cache-gib', '3.5', '--no-update-check', '--loop-guard']
        if benchmark:
            argv.append('--benchmark-control')
    else:
        argv = [str(root/'native/tf-dsv41-lanes'), str(model), str(cache), str(rank), '4',
            config['master_address'], str(config['master_port']), str(kit), '--pool', '9000000',
            '--prompt-context', '401408', '--streams', '16', '--round-rows', '48',
            '--round-ms', profile['round_ms'], '--engram', str(engram),
            '--token-map', str(kit/'engram-token-map.json'), '--aio', '1', '--prefetch', '1',
            '--side', '1', '--graphs', '1']
    if validate_files:
        if any('REPLACE_' in value for value in required):
            raise ValueError('Replace every topology placeholder before launching')
        address = ipaddress.ip_address(config['master_address'])
        if any(address in ipaddress.ip_network(n) for n in
               ['192.0.2.0/24', '198.51.100.0/24', '203.0.113.0/24']):
            raise ValueError('Documentation address cannot be used for a running cluster')
        ipaddress.ip_network(config['nccl_address_range'])
        spec, identity = patch.recipe('v400')
        receipt = json.loads((root/'BUILD.json').read_bytes())
        if (receipt.get('schema') != 'ds41f.tp4-build.v1' or receipt.get('version') != '4.0.0'
                or receipt.get('recipe_manifest_sha256') != identity
                or receipt.get('source_tree_sha256') != spec['source_tree_sha256']
                or receipt.get('prefill_assets_sha256') != profile['prefill_assets_sha256']
                or set(receipt.get('binaries', {})) != set(profile['historical_native_binaries'])):
            raise ValueError('Exact TP4 build receipt required')
        for name, record in receipt['binaries'].items():
            if record['path'] != 'native/'+name or patch.digest((root/'native'/name).read_bytes()) != record['sha256']:
                raise ValueError('Native executable differs from build receipt')
        assets.check(root/'prefill', profile['prefill_assets_sha256'])
        for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                     'model.safetensors.index.json', 'vision.safetensors']:
            if not (model/name).is_file():
                raise FileNotFoundError(model/name)
        cfg = json.loads((model/'config.json').read_bytes())
        marker = cfg['tensorfold_tp4']
        if (marker['world_size'] != 4 or marker['weights_requantized'] is not False
                or marker['parent_tp3_provenance']['original_revision'] != profile['source_revision']
                or cfg['text_config']['num_attention_heads'] != 64 or cfg['text_config']['o_groups'] != 8
                or cfg['text_config']['vocab_size'] != 129280):
            raise ValueError('Require native four-rank 1017 metadata, without TP3 padding')
        for name in ['config.json', 'tokenizer_config.json', 'chat_template.jinja']:
            if patch.digest((model/name).read_bytes()) != patch.digest((HERE/'model'/name).read_bytes()):
                raise ValueError('Qualified TP4 model metadata/template differs: '+name)
        rank_file = cache/f'rank{rank}of4.bin'
        record = json.loads(rank_file.with_suffix('.receipt.json').read_bytes())
        keys = json.loads((HERE/'plan/KEYS.json').read_bytes())[str(rank)]
        if (record.get('status') != 'TP3_TO_TP4_BYTES_REPARTITIONED_FULL_READBACK_NOT_GPU_QUALIFIED'
                or record.get('rank') != rank or record.get('requantized') is not False
                or record.get('keys_sha256') != keys['sha256']
                or record.get('artifact_bytes') != rank_file.stat().st_size
                or record.get('plan_sha256') != marker['parent_tp3_provenance']['weight_plan_sha256']):
            raise ValueError('Require the verified native TP4 rank cache and receipt')
        native_files = list(cache.glob(f'rank{rank}of4-*.bin'))
        expected_name = f'rank{rank}of4-{record["artifact_sha256"][:20]}.bin'
        if (len(native_files) != 1 or native_files[0].name != expected_name
                or native_files[0].is_symlink() or rank_file.is_symlink()
                or not rank_file.samefile(native_files[0])):
            raise ValueError('Require the unique native-discovery rank cache hardlink')
        # Full cache hashes are verified once during preparation, not on every restart.
        em = json.loads((engram/'SOURCE.json').read_bytes())
        if (em['world'] != 2 or em['rank'] != rank % 2 or em['requantized'] is not False
                or em['status'] != 'TP2_OWNED_RANGES_FULL_READBACK_VERIFIED'):
            raise ValueError('Wrong two-way Engram ownership')
        for name, rec in em['files'].items():
            patch.relative(name)
            if (engram/name).stat().st_size != rec['size']:
                raise ValueError('Incomplete Engram table: '+name)
        rope = json.loads((kit/'rope.json').read_bytes())
        if rope['rows'] < 9000000 + 49:
            raise ValueError('9M pool requires the existing verified 10M RoPE tables')
        for kind in ['plain', 'compressed']:
            for part in ['cos', 'sin']:
                name = f'rope-{kind}-{part}.f32'
                if (kit/name).stat().st_size != rope['rows'] * rope['half'] * 4:
                    raise ValueError('Incomplete RoPE table: '+name)
        if not (kit/'engram-token-map.json').is_file():
            raise FileNotFoundError('Missing pinned Engram token map')
        nr = json.loads((Path(node['library_path']).parent/'BUILD.json').read_bytes())
        lib = Path(node['library_path'])/'libnccl.so.2.30.7'
        if (nr.get('source_revision') != '73cf112295c33aee2b895f329f592f2a9b4b0f97'
                or nr.get('cpu_compatibility') != 'PASS'
                or nr.get('patches') != {name: patch.digest((HERE/'nccl-ring'/name).read_bytes())
                       for name in ['dual-pci-domain.patch', 'cts-acquire-fence.patch']}
                or patch.digest(lib.read_bytes()) != nr.get('library_sha256')):
            raise ValueError('Require the separately built, verified switchless NCCL library')
    return argv, env


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--rank', type=int, choices=range(4), required=True)
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    argv, env = command(json.loads(a.config.read_bytes()), a.rank,
                        benchmark=a.benchmark, validate_files=a.run)
    if a.run:
        os.execvpe(argv[0], argv, process_environment(env))
    print(json.dumps(dict(version='4.0.0', rank=a.rank, argv=argv, environment=env,
        shell_command='env '+' '.join('-u '+k for k in INCOMPATIBLE_ENV)+' '
            +' '.join(shlex.quote(k+'='+v) for k, v in env.items())+' '+shlex.join(argv),
        started=False), indent=2))
