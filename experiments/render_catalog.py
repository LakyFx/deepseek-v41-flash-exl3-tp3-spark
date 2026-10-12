"""Render public variant tables from measured results; --check rejects stale docs."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- variant-catalog:start -->'
END = '<!-- variant-catalog:end -->'


def render():
    rows = json.loads((ROOT / 'results/20261012-host64k-campaign/summary.json').read_text(encoding='utf-8'))
    arms = {r['arm']: r for r in json.loads((ROOT / 'experiments/20261012/variants.json').read_text(encoding='utf-8'))['arms']}
    notes = {}
    for line in (ROOT / 'docs/CAMPAIGN-20261012.md').read_text(encoding='utf-8').splitlines():
        cells = [c.strip().replace('**', '') for c in line.split('|')[1:-1]]
        if len(cells) == 3:
            arm = cells[0].split(' / ')[0]
            if arm in arms:
                notes[arm] = cells[1:]

    def table(headers, keys):
        out = ['| Variant | ' + ' | '.join(headers) + ' |', '|---|' + '---:|' * len(keys)]
        maxima = {k: max(r[k] for r in rows if r[k] is not None) for k in keys}
        for r in rows:
            name = r['arm']
            label = name + (' / 1.0.0' if name == 'C4' else '')
            values = []
            for key in keys:
                v = r[key]
                value = 'n/a' if v is None else f'{v:,.1f}'
                if v is not None and v == maxima[key]:
                    value = '**' + value + '**'
                if key == 'h_128k_prefill' and v is not None and not r['h_128k_complete']:
                    value += '*'
                values.append(value)
            out.append(f'| [{label}](docs/VARIANTS.md#{name.lower()}) | ' + ' | '.join(values) + ' |')
        return '\n'.join(out)

    overview = '''## Choose your variant

**18 measured recipes. Same EXL3 weights. Different workload tradeoffs.**

| Your priority | Start by comparing | Why |
|---|---|---|
| Long-context agent work | **C4 / DS41F 1.0.0**, N6 | The only two arms to finish the full bounded 128K-C3 cold/hot phase |
| Hermes single-request decode | C3, C5-indexer | Highest and second-highest measured Hermes C1 |
| Hermes eight-request throughput | C5-indexer, N8b, B0 | Highest measured Hermes C8 wall throughput |
| Code generation at eight requests | N3, N1, N4 | Highest measured Owen code C8 decode-window throughput |
| Reference comparison | B0 | Fresh X11c baseline on the same upgraded hosts |

These are candidates to evaluate, not universal winners. **C4 is our production choice; the others are experimental alternatives.**
Click any variant below for its changes, tradeoffs, build commands and verification scope.
[All variant recipes](docs/VARIANTS.md) · [Reproduction guide](docs/EXPERIMENT-REPLAY.md) · [Verification status](docs/VARIANT-VALIDATION.md)

### Hermes: our agent workload

All values are **tok/s**. C1/C3 are weighted **per-request engine decode**; C8 is **aggregate output / cohort wall time**, including first-token latency. Prefill counts newly computed tokens over native prefill time. This is not one concurrency scaling curve.

'''
    overview += table(['C1 decode', 'C3 decode', 'C8 wall', '64K prefill', '128K prefill'],
                      ['h_c1_decode', 'h_c3_decode', 'h_c8_wall', 'h_64k_prefill', 'h_128k_prefill'])
    overview += '''

`*` Cold prime measured, but the full 128K-C3 phase did not finish. `n/a` is missing, not zero. Only N6 and C4 completed that full phase.

### Christopher Owen benchmark: code

Adapted from [Christopher Owen's spark-ds41f](https://github.com/christopherowen/spark-ds41f), with [frozen source, inputs and methodology](docs/OWEN-BENCHMARK.md). Values are aggregate **decode-window tok/s**, averaged over three measured repeats after a discarded warmup.

'''
    overview += table(['C1', 'C2', 'C4', 'C8'], ['o_code_c1', 'o_code_c2', 'o_code_c4', 'o_code_c8'])
    overview += '\n\n### Christopher Owen benchmark: prose and cold prefill\n\nProse uses the same decode-window metric. Prefill is **input tokens / TTFT**, including first-token overhead; it differs from the Hermes native computed-prefill metric.\n\n'
    overview += table(['Prose C1', 'Prose C2', 'Prose C4', 'Prose C8', '32K prefill', '256K prefill'],
                      ['o_prose_c1', 'o_prose_c2', 'o_prose_c4', 'o_prose_c8', 'o_32k_prefill', 'o_256k_prefill'])
    overview += '''

**Reading these tables:** bold marks the largest observed value in a column, not a statistically established winner. Hermes has one campaign pass per arm; Owen decode has three repeats. MAX reasoning was used for chat requests. Candidate names C1/C3/C4 are not concurrency levels. Results depend on prompt, acceptance, context and cache state. See [complete campaign limitations](docs/CAMPAIGN-20261012.md#completeness-correctness-and-interpretation).

### Reproduce a choice

Clone the current repository with tags, select an arm and build on Linux ARM64. For example:

```bash
git fetch origin --tags
python3 experiments/materialize.py --arm C5-indexer --output /srv/dsv41-build-c5-indexer
cd /srv/dsv41-build-c5-indexer
RECIPE_IMAGE=ds41f:experiment-c5-indexer bash build/build.sh
cp config/cluster.example.json config/cluster.local.json
```

Fill in your topology and paths, then follow [installation](docs/INSTALL.md#3-download-and-prepare-the-checkpoint) and [acceptance checks](docs/VERIFICATION.md#operator-acceptance). The commands above prepare/build only; they do not deploy. Source reconstruction is checked separately from public image builds and GPU qualification. Historical measurements do not certify a new image.
'''
    details = '''# Variant recipes

[Choose by measured workload](../README.md#choose-your-variant) · [Complete results](CAMPAIGN-20261012.md) · [Build verification](VARIANT-VALIDATION.md)

All 18 measured arms use the same pinned EXL3 checkpoint, TP3 and cache allocation. N9 was a no-op and is not an installable measured arm. C4 is DS41F 1.0.0. Each command below starts in the repository root with tags fetched and creates a new directory outside it. Use a fresh output path if one exists; no automatic cleanup occurs.

After building, copy `config/cluster.example.json` to `config/cluster.local.json`, configure your own topology, weights and independent writable caches, and follow [installation from checkpoint preparation](INSTALL.md#3-download-and-prepare-the-checkpoint) and [operator acceptance](VERIFICATION.md#operator-acceptance). Build once, use the identical image on all three ranks. No recipe here deploys automatically.

Verification labels for every arm: **historical GPU campaign recorded**; **source reconstruction separately checked**; **fresh public image build/ABI and GPU requalification are not implied**. Consult the [status and receipts](VARIANT-VALIDATION.md) for actual evidence.

'''
    baseline = next(r for r in rows if r['arm'] == 'B0')
    for r in rows:
        arm = r['arm']
        change, note = notes[arm]
        details += f'## {arm}\n\n**Change:** {change}.\n\n**Reason to compare / limitation:** {note}.\n\n'
        details += '**Observed versus B0:** ' + '; '.join(
            f'{label} {r[k]:.1f} tok/s ({(r[k]/baseline[k]-1)*100:+.1f}%)'
            for label, k in [('Hermes C1','h_c1_decode'), ('Hermes C8','h_c8_wall'), ('64K prefill','h_64k_prefill'), ('Owen code C8','o_code_c8')]) + '.\n\n'
        details += '**Full bounded 128K-C3 phase:** ' + ('completed.' if r['h_128k_complete'] else 'incomplete; do not interpret a cold-prime value as a completed phase.') + '\n\n'
        details += f'''```bash
git fetch origin --tags
python3 experiments/materialize.py --arm {arm} --output /srv/dsv41-build-{arm.lower()}
cd /srv/dsv41-build-{arm.lower()}
RECIPE_IMAGE=ds41f:experiment-{arm.lower()} bash build/build.sh
```

'''
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    block = START + '\n' + overview + '\n' + END
    if START in readme:
        readme = readme[:readme.index(START)] + block + readme[readme.index(END) + len(END):]
    else:
        readme = readme.replace('## Selected settings', block + '\n\n## Selected settings')
    return {'README.md': readme, 'docs/VARIANTS.md': details.rstrip() + '\n'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for name, content in render().items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_text(encoding='utf-8') != content:
                raise SystemExit('Generated catalog differs: ' + name)
        else:
            path.write_text(content, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
