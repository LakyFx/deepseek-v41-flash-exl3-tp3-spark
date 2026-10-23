"""Create the measured 1.2.5 head metadata overlay; original files stay intact."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def prepare(runtime):
    profile = json.loads((HERE / 'profiles/1.2.5.json').read_bytes())
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
    print(json.dumps(prepare(parser.parse_args().runtime.resolve()), indent=2))
