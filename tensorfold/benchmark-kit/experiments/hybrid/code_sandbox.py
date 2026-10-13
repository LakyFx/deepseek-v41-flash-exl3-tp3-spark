"""Bounded code-task evaluation in an isolated, CPU-only Docker container.

No generated code executes in the host process. Named containers and receipts
are retained for explicit cleanup. Use an already prepared immutable image ID.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import uuid

HARNESS = r'''
import ast,builtins,contextlib,io,json,sys
import collections,datetime,decimal,functools,itertools,math,re,statistics,unicodedata
payload=json.load(sys.stdin)
source=payload['source']
allowed={'collections','datetime','decimal','functools','itertools','math','re','statistics','unicodedata','json'}
tree=ast.parse(source)
for node in ast.walk(tree):
 if isinstance(node,ast.Attribute) and node.attr.startswith('__'):raise ValueError('dunder access')
 if isinstance(node,ast.Name) and node.id.startswith('__'):raise ValueError('dunder name')
 if isinstance(node,ast.Import) and any(x.name not in allowed for x in node.names):raise ValueError('import denied')
 if isinstance(node,ast.ImportFrom) and (node.level or node.module not in allowed):raise ValueError('import denied')
def importing(name,*args,**kwargs):
 if name not in allowed:raise ValueError('import denied')
 return builtins.__import__(name,*args,**kwargs)
safe={name:getattr(builtins,name) for name in ('abs','all','any','bool','bytes','chr','dict','divmod','enumerate','filter','float','format','frozenset','int','isinstance','issubclass','iter','len','list','map','max','min','next','ord','pow','range','repr','reversed','round','set','slice','sorted','str','sum','tuple','type','zip','Exception','ValueError','TypeError','KeyError','IndexError','ZeroDivisionError')}
safe['__import__']=importing
scope={'__builtins__':safe}
exec(compile(tree,'candidate.py','exec'),scope)
fn=scope.get('solve')
if not callable(fn):raise ValueError('missing solve')
def equal(a,b):
 if isinstance(b,(float,int)) and not isinstance(b,bool):return isinstance(a,(float,int)) and not isinstance(a,bool) and a==b
 if type(a) is not type(b):return False
 if isinstance(b,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 if isinstance(b,dict):return set(a)==set(b) and all(equal(a[k],v) for k,v in b.items())
 return a==b
rows=[]
for i,(argument,expected) in enumerate(payload['checks']):
 try:
  actual=fn(argument)
  ok=equal(actual,expected)
  rows.append({'index':i,'pass':ok})
 except Exception as e:rows.append({'index':i,'pass':False,'error':type(e).__name__})
print(json.dumps({'passed':all(x['pass'] for x in rows),'checks':rows}))
'''


def evaluate(case, response, image, output):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('immutable, locally available Docker image ID required')
    source = response.get('message', {}).get('content') or ''
    name = 'ds41f-quality-' + uuid.uuid4().hex
    payload = json.dumps({'source': source, 'checks': case['code_checks']}).encode('utf-8')
    command = ['docker', 'run', '--name', name, '--runtime', 'runc', '--network', 'none',
        '--read-only', '--cap-drop=ALL', '--security-opt', 'no-new-privileges',
        '--user', '65534:65534', '--pids-limit', '32', '--memory', '256m', '--memory-swap', '256m',
        '--cpus', '1', '--env', 'NVIDIA_VISIBLE_DEVICES=void', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--workdir', '/tmp', '--entrypoint', 'python3', '-i', image, '-I', '-c', HARNESS]
    receipt = {'case_id': case['id'], 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
               'checks_sha256': hashlib.sha256(json.dumps(case['code_checks'], sort_keys=True).encode()).hexdigest(),
               'harness_sha256': hashlib.sha256(HARNESS.encode()).hexdigest(),
               'image': image, 'container': name, 'gpu': False, 'network': False, 'passed': False}
    try:
        if response.get('finish_reason') != 'stop' or response.get('error'):
            raise ValueError('incomplete model output')
        run = subprocess.run(command, input=payload, capture_output=True, timeout=15)
        receipt['exit_code'] = run.returncode
        if run.returncode == 0:
            measured = json.loads(run.stdout)
            if len(measured['checks']) != len(case['code_checks']):
                raise ValueError('sandbox check count differs')
            receipt['passed'] = measured['passed'] is True and all(x['pass'] is True for x in measured['checks'])
            receipt['checks'] = measured['checks']
        else:
            receipt['error'] = run.stderr.decode('utf-8', errors='replace')[-2000:]
    except subprocess.TimeoutExpired:
        subprocess.run(['docker', 'kill', name], capture_output=True, timeout=10)
        receipt['error'] = 'sandbox_timeout'
    except (OSError, ValueError, KeyError) as exc:
        receipt['error'] = str(exc)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--responses', type=Path, required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    fixtures = json.loads(args.fixtures.read_text(encoding='utf-8'))
    responses = {x['case_id']: x for x in json.loads(args.responses.read_text(encoding='utf-8'))}
    if not args.execute:
        print(json.dumps({'action': 'PLAN_ONLY', 'code_cases': sum('code_checks' in c for c in fixtures['cases'])}))
    else:
        for case in fixtures['cases']:
            if 'code_checks' in case:
                evaluate(case, responses[case['id']], args.image, args.output / (case['id'] + '.json'))
