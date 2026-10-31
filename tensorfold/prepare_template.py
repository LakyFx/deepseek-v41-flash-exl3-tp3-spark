"""Create a published head metadata overlay; original files stay intact."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def prepare(runtime, variant='1.2.5'):
    if variant not in {p.stem for p in (HERE/'profiles').glob('*.json')
                       if json.loads(p.read_bytes()).get('head_chat_template_sha256')}:
        raise ValueError('Select a published direct-output template profile')
    profile = json.loads((HERE / 'profiles' / (variant+'.json')).read_bytes())
    original = runtime / 'weights/models' / profile['arm']
    destination = runtime / profile['head_model_subdir']
    candidate = HERE / 'templates/1017-direct-output.jinja'
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != profile['head_chat_template_sha256']:
        raise ValueError('Published direct-output template changed')
    if hashlib.sha256((original / 'chat_template.jinja').read_bytes()).hexdigest() != profile['chat_template_sha256']:
        raise ValueError('Original model template differs from the pinned release')
    entries = list(original.iterdir())
    if not (original / 'config.json').is_file():
        raise ValueError('Prepare the pinned 1017 model first')
    destination.mkdir(parents=True, exist_ok=False)
    # Native reads chat_template.jinja directly. Config/tokenizer bytes are reused.
    for item in entries:
        target = candidate if item.name == 'chat_template.jinja' else item
        (destination / item.name).symlink_to(target.resolve(), target_is_directory=target.is_dir())
    return {'head_model': str(destination), 'original_changed': False,
            'template_sha256': profile['head_chat_template_sha256'], 'model_requests': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--variant', default='1.2.8', choices=sorted(
        p.stem for p in (HERE/'profiles').glob('*.json')
        if json.loads(p.read_bytes()).get('head_chat_template_sha256')))
    args=parser.parse_args()
    print(json.dumps(prepare(args.runtime.resolve(),args.variant), indent=2))
