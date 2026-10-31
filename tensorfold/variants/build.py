"""Build any published workflow recipe; no inference, SSH or service changes."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

try:
    from . import patch
except ImportError:
    import patch
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build(build_root,out,name,zig='zig',prepare_only=False):
    source=build_root/'source'
    spec,identity=patch.apply(source,name)
    assets_sha = None
    if name == 'v128':
        from tensorfold import prefill_assets
        profile = json.loads((Path(__file__).parents[1]/'profiles/1.2.8.json').read_bytes())
        assets_sha = profile['prefill_assets_sha256']
        prefill_assets.inputs(build_root/'kit', assets_sha)
    if prepare_only:return dict(status='EXACT_SOURCE_VERIFIED',recipe=name,source_tree_sha256=spec['source_tree_sha256'],model_requests=0)
    out.mkdir(parents=True,exist_ok=False)
    commands=[[zig,'build','test','-j2','--summary','all']]
    modules=[('core/lanes/adaptive.zig',False),('core/lanes/refresh_budget.zig',False),('families/dsv41/sampling.zig',True),('families/dsv41/penalty_pool.zig',True)]
    if name=='v126':
        modules += [('families/dsv41/kv_pressure.zig',False),('families/dsv41/kv_move_budget.zig',False)]
    if spec.get('release_version') in ('1.2.7', '1.2.8') or name in ('x1','x2','x3','x4'):
        modules += [(module, True) for module in (
            'core/checkpoint_plan.zig', 'core/checkpoint_policy.zig',
            'core/prompt_cache.zig', 'core/adaptive_prefix.zig',
            'families/dsv41/kv_pressure.zig', 'families/dsv41/kv_move_budget.zig')]
    if name == 'v128':
        modules.append(('families/dsv41/row_state.zig', True))
    for module,libc in modules:
        path='zig/src/'+module
        if (source/path).exists():commands.append([zig,'test',path,'-O','ReleaseSafe']+(['-lc'] if libc else []))
    commands.append([zig,'build','native','dsv41-tools','-j2','-Dtarget=aarch64-linux-gnu','-Dfatbins='+str((build_root/'kit/build-fatbins').resolve()),'--prefix',str(out.resolve()),'--summary','all'])
    logs=[]
    for i,command in enumerate(commands,1):
        path=out/f'build-{i}.log'
        with path.open('wb') as log:subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
        logs.append(dict(step=i,sha256=patch.digest(path.read_bytes()),exit_code=0))
    patch.verify(source,spec['source_files'])
    if assets_sha:
        prefill_assets.prepare(build_root/'kit', out/'prefill-1.2.8', assets_sha)
    binaries={}
    for name in ('tensorfold-native','tf-dsv41-lanes'):
        paths=[p for p in out.rglob(name) if p.is_file()]
        if len(paths)!=1:raise ValueError('Expected one built executable: '+name)
        binaries[name]=dict(path=paths[0].relative_to(out).as_posix(),sha256=patch.digest(paths[0].read_bytes()))
    receipt=dict(schema='ds41f.native-build.v1',recipe=spec['recipe'],version=spec.get('release_version'),adaptive_learning=spec['adaptive_learning'],recipe_manifest_sha256=identity,source_archive_sha256=spec['base_archive_sha256'],source_tree_sha256=spec['source_tree_sha256'],binaries=binaries,cpu_checks=logs,model_requests=0,gpu_qualified_on_this_build_host=False)
    if assets_sha:
        receipt['prefill_assets_sha256'] = assets_sha
    (out/'BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


def install(built,runtime,selector):
    if selector not in {p.stem for p in (Path(__file__).parents[1]/'profiles').glob('*.json')}:
        raise ValueError('Unknown launcher profile')
    profile=json.loads((Path(__file__).parents[1]/'profiles'/f'{selector}.json').read_bytes())
    receipt=json.loads((built/'BUILD.json').read_bytes())
    spec,identity=patch.recipe(profile['build_variant'])
    if (receipt['recipe_manifest_sha256']!=identity or receipt['recipe']!=spec['recipe']
            or receipt['source_tree_sha256']!=spec['source_tree_sha256']
            or receipt['source_archive_sha256']!=spec['base_archive_sha256']
            or receipt['adaptive_learning']!=spec['adaptive_learning']
            or set(receipt['binaries'])!={'tensorfold-native','tf-dsv41-lanes'}):
        raise ValueError('Build receipt does not match selected recipe')
    files=[]
    for name,record in receipt['binaries'].items():
        path=built.joinpath(*patch.relative(record['path']).parts)
        if patch.digest(path.read_bytes())!=record['sha256']:raise ValueError('Built binary changed')
        files.append((path,name))
    dest=runtime/profile['binary_subdir']
    if dest.exists(): raise ValueError('Binary installation already exists')
    if profile.get('prefill_assets_sha256'):
        from tensorfold import prefill_assets
        expected = profile['prefill_assets_sha256']
        if receipt.get('prefill_assets_sha256') != expected:
            raise ValueError('Build receipt does not include matching 1.2.8 kernels')
        prefill_assets.check(built/'prefill-1.2.8', expected)
        shutil.copytree(built/'prefill-1.2.8', runtime/'prefill-1.2.8')
    dest.mkdir(parents=True,exist_ok=False)
    for path,name in files:shutil.copy2(path,dest/name)
    shutil.copy2(built/'BUILD.json',dest/'BUILD.json')
    return dict(installed=str(dest),recipe=receipt['recipe'],services_touched=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    b=sub.add_parser('build');b.add_argument('--recipe',required=True);b.add_argument('--build-root',type=Path,required=True);b.add_argument('--out',type=Path,required=True);b.add_argument('--zig',default='zig');b.add_argument('--prepare-only',action='store_true')
    i=sub.add_parser('install');i.add_argument('--built',type=Path,required=True);i.add_argument('--runtime',type=Path,required=True);i.add_argument('--variant',required=True)
    a=p.parse_args()
    value=build(a.build_root.resolve(),a.out.resolve(),a.recipe,a.zig,a.prepare_only) if a.action=='build' else install(a.built.resolve(),a.runtime.resolve(),a.variant)
    print(json.dumps(value,indent=2))
