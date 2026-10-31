"""Verify and stage the isolated TP4/4.0.0 AOT set, preserving the base kit."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent / 'assets'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(name):
    from tensorfold.variants.patch import relative as checked
    return checked(name)


def manifest(expected):
    path = HERE / 'MANIFEST.json'
    if digest(path) != expected:
        raise ValueError('Wrong 4.0.0 assets manifest')
    spec = json.loads(path.read_bytes())
    for name in [*spec['base_files'], *spec['payload_files']]:
        relative(name)
    return spec


def inputs(kit, expected):
    """Validate the complete source set before creating any destination."""
    spec = manifest(expected)
    if digest(kit / 'aot/aot.json') != spec['base_aot_sha256']:
        raise ValueError('Wrong base AOT kit for 4.0.0')
    for name, wanted in spec['base_files'].items():
        path = kit.joinpath(*relative(name).parts)
        if path.is_symlink() or digest(path) != wanted:
            raise ValueError('Base AOT file differs: ' + name)
    archive = HERE.joinpath(*relative(spec['archive']).parts)
    if digest(archive) != spec['archive_sha256']:
        raise ValueError('Wrong 4.0.0 kernel payload')
    payload = {}
    with tarfile.open(archive) as stream:
        for item in stream:
            relative(item.name)
            if not item.isfile() or item.name in payload:
                raise ValueError('Unsafe or duplicate kernel payload member')
            raw = stream.extractfile(item).read()
            if hashlib.sha256(raw).hexdigest() != spec['payload_files'].get(item.name):
                raise ValueError('Wrong kernel payload member: ' + item.name)
            payload[item.name] = raw
    if set(payload) != set(spec['payload_files']):
        raise ValueError('Incomplete kernel payload')
    return spec, payload


def prepare(kit, destination, expected):
    spec, payload = inputs(kit, expected)
    destination.mkdir(parents=True, exist_ok=False)
    for name in spec['base_files']:
        target = destination.joinpath(*relative(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(kit / name, target)
    for name, raw in payload.items():
        target = destination.joinpath(*relative(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    check(destination, expected)


def check(directory, expected):
    spec = manifest(expected)
    files = dict(spec['base_files'], **spec['payload_files'])
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual != set(files) or any(p.is_symlink() for p in directory.rglob('*')):
        raise ValueError('Incomplete or unexpected installed 4.0.0 kernels')
    for name, wanted in files.items():
        if digest(directory.joinpath(*relative(name).parts)) != wanted:
            raise ValueError('Installed kernel differs: ' + name)
