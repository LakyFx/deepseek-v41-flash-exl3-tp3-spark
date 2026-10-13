"""CPU Docker evaluation of six frozen EvalPlus-derived tasks. Retains containers."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,uuid
HARNESS=r'''
import ast,builtins,contextlib,io,json,sys
payload=json.load(sys.stdin);source=payload['source'];tree=ast.parse(source)
allowed={'typing','collections','math','itertools','functools','statistics','re','decimal','json'}
for node in ast.walk(tree):
 if isinstance(node,ast.Attribute) and node.attr.startswith('__'):raise ValueError('dunder access')
 if isinstance(node,ast.Name) and node.id.startswith('__'):raise ValueError('dunder name')
 if isinstance(node,ast.Import) and any(x.name not in allowed for x in node.names):raise ValueError('import denied')
 if isinstance(node,ast.ImportFrom) and (node.level or node.module not in allowed):raise ValueError('import denied')
def importing(name,*args,**kwargs):
 if name not in allowed:raise ValueError('import denied')
 return builtins.__import__(name,*args,**kwargs)
safe={name:getattr(builtins,name) for name in ('abs','all','any','bool','bytes','chr','dict','divmod','enumerate','filter','float','format','frozenset','int','isinstance','issubclass','iter','len','list','map','max','min','next','ord','pow','range','repr','reversed','round','set','slice','sorted','str','sum','tuple','type','zip','Exception','ValueError','TypeError','KeyError','IndexError','ZeroDivisionError')}
safe['__import__']=importing;scope={'__builtins__':safe}
with contextlib.redirect_stdout(io.StringIO()):exec(compile(tree,'candidate.py','exec'),scope)
fn=scope.get(payload['entry_point']);assert callable(fn),'missing entry point'
def equal(a,b):
 if type(a) is not type(b):return False
 if isinstance(b,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
checks=[]
for i,row in enumerate(payload['checks']):
 try:
  with contextlib.redirect_stdout(io.StringIO()):actual=fn(*row['args'])
  checks.append(dict(index=i,passed=equal(actual,row['expected']),group=row['group']))
 except Exception as e:checks.append(dict(index=i,passed=False,error=type(e).__name__,group=row['group']))
print(json.dumps(dict(passed=all(x['passed'] for x in checks),checks=checks)))
'''
def extract(text):
 blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,flags=re.S|re.I)
 return blocks[0].strip() if len(blocks)==1 else text.strip()
def evaluate(case,response,image,output):
 assert re.fullmatch(r'sha256:[0-9a-f]{64}',image)
 text=response.get('message',{}).get('content') or '';source=extract(text)
 name='ds41f-q20-'+uuid.uuid4().hex
 receipt=dict(attempt_id=response['attempt_id'],container=name,image=image,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
  harness_sha256=hashlib.sha256(HARNESS.encode()).hexdigest(),gpu=False,network=False,passed=False)
 if response.get('error') or response.get('finish_reason')!='stop' or not source:receipt['error']='incomplete code'
 else:
  args=['docker','run','--name',name,'--runtime','runc','--network','none','--read-only','--cap-drop=ALL','--security-opt','no-new-privileges',
   '--user','65534:65534','--pids-limit','32','--memory','256m','--memory-swap','256m','--cpus','1','--env','NVIDIA_VISIBLE_DEVICES=void',
   '--env','PYTHONDONTWRITEBYTECODE=1','--workdir','/tmp','--entrypoint','python3','-i',image,'-I','-c',HARNESS]
  payload=json.dumps(dict(source=source,entry_point=case['grade']['entry_point'],checks=case['grade']['checks'])).encode()
  try:
   run=subprocess.run(args,input=payload,capture_output=True,timeout=12)
   receipt['exit_code']=run.returncode
   if run.returncode==0:
    value=json.loads(run.stdout);assert len(value['checks'])==len(case['grade']['checks']);receipt.update(value)
   else:receipt['error']=run.stderr.decode(errors='replace')[-2000:]
  except subprocess.TimeoutExpired:
   subprocess.run(['docker','kill',name],capture_output=True,timeout=10);receipt['error']='sandbox_timeout'
 with output.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
 return receipt
def main():
 p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--responses',type=Path,required=True)
 p.add_argument('--image',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:print(json.dumps(dict(status='PLAN_ONLY',gpu=False)));return
 a.output.mkdir(parents=True,exist_ok=False);cases={c['id']:c for c in json.loads(a.fixtures.read_bytes())['cases']};rows=[]
 for response in json.loads(a.responses.read_bytes()):
  case=cases[response['case_id']]
  if case['grade']['kind']=='code':rows.append(evaluate(case,response,a.image,a.output/(response['attempt_id']+'.json')))
 (a.output/'RESULT.json').write_text(json.dumps(dict(passed=sum(x['passed'] for x in rows),total=len(rows),receipts=rows),indent=2)+'\n')
if __name__=='__main__':main()
