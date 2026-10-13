"""Six-minute, loopback-only MAX reasoning quality capture. No service changes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,hashlib,json,sys,time,uuid
from q20_grade import grade

def main():
 p=argparse.ArgumentParser();p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--benchmark-kit',type=Path,required=True)
 p.add_argument('--config',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--token-reference',type=Path);p.add_argument('--exclusive',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
 fixture=json.loads(a.fixtures.read_bytes());config=json.loads(a.config.read_bytes());control=config['benchmark_control']
 if not a.execute:print(json.dumps(dict(status='PLAN_ONLY',attempts=len(fixture['attempts']),seconds=360)));return
 assert a.exclusive and config['backend_bind']=='127.0.0.1' and control['enabled'] and control['peer_address']=='127.0.0.1'
 sys.path.insert(0,str(a.benchmark_kit/'experiments/hybrid'))
 from quality_run import run_case
 from run import Network,ProfileClient,Window,dump
 class MatchedClient(ProfileClient):
  def complete(self,body,**kwargs):
   return super().complete({**body,'top_p':1,'top_k':0,'min_p':0},**kwargs)
 a.output.mkdir(parents=True,exist_ok=False)
 lease=Path(control['lease_path'])/'lease.json';lease.parent.mkdir(parents=True,exist_ok=True)
 owner=dict(campaign_id='q20-'+uuid.uuid4().hex,window_closed=True,plan_sha256=hashlib.sha256(a.fixtures.read_bytes()).hexdigest())
 dump(lease,owner)
 net=Network(config.get('backend_port',8003),engine=config.get('benchmark_engine','vllm'));client=MatchedClient(net,profile='max',armed=True)
 window=Window(net,client,lease,owner,lambda k,v:dump(a.output/(k+'-'+uuid.uuid4().hex+'.json'),v))
 results=[];started=time.monotonic();deadline=started+360
 try:
  window.reset();net.validate_body({'cache_salt':'q20-capability-check'})
  cases={c['id']:c for c in fixture['cases']}
  from runner.profiles import body_for_profile
  tokenization=[]
  for c in fixture['cases']:
   body=body_for_profile(dict(model='deepseek-v4.1-flash',messages=c['messages'],**({'tools':c['tools']} if c.get('tools') else {})),'max')
   count=net.count(body);ids=count.get('token_ids')
   assert isinstance(ids,list) and len(ids)==count['count'],'exact rendered token IDs required'
   tokenization.append(dict(case_id=c['id'],**count))
  dump(a.output/'TOKENIZATION.json',tokenization)
  if a.arm.startswith('TF'):
   assert a.token_reference is not None,'new C4 Q20 token reference required'
   reference=json.loads(a.token_reference.read_bytes())
   assert [(r['case_id'],r['token_ids']) for r in reference]==[(r['case_id'],r['token_ids']) for r in tokenization],'quality rendered inputs differ'
  def execute(attempt):
   c=cases[attempt['case_id']]
   result=run_case(c,client,seed=attempt['seed'],temperature=attempt['temperature'],deadline=deadline,
    campaign='q20-fixed-'+attempt['attempt_id'],cache_policy='salted-case')
   result.update(attempt_id=attempt['attempt_id'],grade=grade(c,result))
   result['content_chars']=len(result.get('message',{}).get('content') or '')
   result['reasoning_chars']=sum(len(r.get('message',{}).get('reasoning_content') or '') for r in result.get('requests',[]))
   dump(a.output/(attempt['attempt_id']+'.json'),result);return result
  short=[x for x in fixture['attempts'] if not x['case_id'].startswith('long-')]
  long=[x for x in fixture['attempts'] if x['case_id'].startswith('long-')]
  with ThreadPoolExecutor(max_workers=4) as pool:
   for task in as_completed([pool.submit(execute,x) for x in short]):
    results.append(task.result());window.guard()
  for attempt in long:results.append(execute(attempt))
  window.drain(deadline=time.monotonic()+15)
  results.sort(key=lambda x:x['attempt_id'])
  dump(a.output/'RESPONSES.json',results)
  dump(a.output/'RESULT.json',dict(status='CAPTURED_PENDING_CODE_AND_EDITORIAL_REVIEW',arm=a.arm,
   fixture_sha256=owner['plan_sha256'],elapsed_seconds=time.monotonic()-started,
   hard_inference_budget_seconds=360,attempts=len(results),complete=sum(r['grade']['complete'] for r in results),
   noncode_pass=sum(r['grade']['passed'] for r in results),all_max=True,sampling=fixture['sampling'],
   errors=[dict(id=r['attempt_id'],error=r['error']) for r in results if r.get('error')]))
 finally:
  client.abort_all_owned()
  lease.write_text(json.dumps({**owner,'window_closed':False,'finished':True}))

if __name__=='__main__':main()
