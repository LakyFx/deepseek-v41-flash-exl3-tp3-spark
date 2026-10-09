import pathlib,shutil
root=pathlib.Path('/opt/dsv41-patch');site=pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm')
for line in (root/'mounts.txt').read_text().splitlines():
 if not line.strip() or line.lstrip().startswith('#'):continue
 name,dst=line.split();target=pathlib.Path(dst) if dst.startswith('/') else site/dst
 assert (root/name).is_file(),name
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
