"""Generate qualified RoPE arithmetic for 3.3M or 10M logical pool positions, CPU only."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

OPS_SHA = 'a1716efb08b687baa28c932a406e412b1d6809707969858291513588a8edfe4b'
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--kit', required=True, type=Path)
    parser.add_argument('--pool', choices=[3300000, 10000000], type=int, required=True)
    args = parser.parse_args()
    source = args.reference / 'src/tensorfold/families/deepseek_v41/ops.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != OPS_SHA:
        raise ValueError('RoPE arithmetic source differs from qualified reference')
    targets = [args.kit / f'rope-{kind}-{part}.f32' for kind in ('plain', 'compressed') for part in ('cos', 'sin')]
    if any(p.exists() for p in targets):
        raise FileExistsError('Use a new kit, or a kit without RoPE binaries')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(args.reference / 'src'))
    import torch
    from tensorfold.families.deepseek_v41.ops import freqs_cis, rope_block, ROPE_COMPAT, ROPE_BLOCK
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cfg = json.loads((args.kit / 'models/TF351/config.json').read_text())['text_config']
    yarn = cfg['rope_scaling']
    half = cfg['qk_rope_head_dim'] // 2
    rows = ((args.pool + 49 + 4095) // 4096) * 4096
    expected = json.loads((ROOT / 'rope-hashes.json').read_text())[str(args.pool)]
    result = dict(rows=rows, half=half)
    for kind in ('plain', 'compressed'):
        params = (yarn['original_max_position_embeddings'] if kind == 'compressed' else 0,
                  cfg['compress_rope_theta'] if kind == 'compressed' else cfg['rope_theta'],
                  yarn['factor'], yarn['beta_fast'], yarn['beta_slow'])
        names = [f'rope-{kind}-{part}.f32' for part in ('cos', 'sin')]
        hashes = [hashlib.sha256(), hashlib.sha256()]
        with (args.kit / names[0]).open('xb') as cos, (args.kit / names[1]).open('xb') as sin:
            for start in [0] + list(range(ROPE_COMPAT, rows, ROPE_BLOCK)):
                count = ROPE_COMPAT if start == 0 else ROPE_BLOCK
                f = (freqs_cis.__wrapped__(2 * half, count, *params, device='cpu') if start == 0
                     else rope_block(2 * half, start, count, *params))
                keep = min(count, rows - start)
                for i, (part, output) in enumerate(zip((f.real, f.imag), (cos, sin))):
                    data = part[:keep].contiguous().numpy().tobytes()
                    output.write(data)
                    hashes[i].update(data)
                del f
            for stream in (cos, sin):
                stream.flush()
                os.fsync(stream.fileno())
        for i, name in enumerate(names):
            result[name] = hashes[i].hexdigest()
            if result[name] != expected[name]:
                raise ValueError('Generated RoPE differs from qualified bytes: ' + name)
    if result != expected:
        raise ValueError('RoPE metadata differs')
    (args.kit / 'rope.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'CPU_BYTES_MATCH_QUALIFIED_ROPE', 'pool': args.pool, 'rows': rows,
                      'torch_version': torch.__version__}, indent=2))


if __name__ == '__main__':
    main()
