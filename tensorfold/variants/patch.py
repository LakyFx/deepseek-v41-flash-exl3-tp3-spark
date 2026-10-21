"""Materialize one exact measured native source recipe from the frozen archive."""
import hashlib
import json
from pathlib import Path, PurePosixPath

HERE=Path(__file__).resolve().parent


def digest(data):return hashlib.sha256(data).hexdigest()
def tree_digest(files):return digest(json.dumps(files,sort_keys=True,separators=(',',':')).encode())


def relative(name):
    value=PurePosixPath(name)
    if value.is_absolute() or '..' in value.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe source path')
    return value


def recipe(name):
    if name not in {p.stem for p in (HERE/'recipes').glob('*.json')}:
        raise ValueError('Unknown native recipe: '+name)
    path=HERE/'recipes'/f'{name}.json'
    value=json.loads(path.read_bytes())
    if value['recipe']!=name or tree_digest(value['source_files'])!=value['source_tree_sha256']:
        raise ValueError('Invalid recipe manifest')
    return value,digest(path.read_bytes())


def verify(source,files):
    source=Path(source)
    actual=set()
    # Prune compiler output rather than traversing potentially huge caches.
    import os
    for directory,dirs,names in os.walk(source,followlinks=False):
        if any((Path(directory)/d).is_symlink() for d in dirs):
            raise ValueError('Symlink directory in source tree')
        dirs[:]=[d for d in dirs if d not in ('.zig-cache','zig-out')]
        for name in names:
            p=Path(directory)/name
            rel=p.relative_to(source).as_posix();actual.add(rel)
            if p.is_symlink() or not p.is_file() or rel not in files or digest(p.read_bytes())!=files[rel]:
                raise ValueError('Source differs from pinned manifest: '+rel)
    if actual!=set(files):raise ValueError('Missing source files')


def apply(source,name):
    source=Path(source)
    spec,identity=recipe(name)
    base=json.loads((HERE/'base.json').read_bytes())
    archive=HERE.parent/'artifacts/tensorfold-source.tar.gz'
    if digest(archive.read_bytes())!=base['archive_sha256'] or spec['base_archive_sha256']!=base['archive_sha256']:
        raise ValueError('Wrong source archive')
    verify(source,base['files'])
    # Validate every object and target before mutating any source file.
    writes=[]
    for name,record in spec['overlays'].items():
        relative(name);obj=HERE.joinpath(*relative(record['object']).parts)
        data=obj.read_bytes()
        if obj.is_symlink() or digest(data)!=record['sha256'] or spec['source_files'][name]!=record['sha256']:
            raise ValueError('Source object hash mismatch: '+name)
        writes.append((source/name,data))
    removals=[source.joinpath(*relative(name).parts) for name in spec['remove']]
    for target,data in writes:
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    for target in removals:target.unlink()
    verify(source,spec['source_files'])
    return spec,identity
