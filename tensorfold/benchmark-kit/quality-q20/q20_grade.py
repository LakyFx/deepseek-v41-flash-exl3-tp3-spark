"""Frozen automatic graders; no generated code runs in this process."""
import json,re,unicodedata
import ifeval_selected

def typed_equal(a,b):
 if type(a) is not type(b):return False
 if isinstance(b,dict):return a.keys()==b.keys() and all(typed_equal(a[k],b[k]) for k in b)
 if isinstance(b,list):return len(a)==len(b) and all(typed_equal(x,y) for x,y in zip(a,b))
 return a==b

def json_object(text):
 stripped=text.strip();unfenced=re.sub(r'^```(?:json)?\s*\n?(.*?)\n?```$',r'\1',stripped,flags=re.S|re.I)
 try:return json.loads(unfenced),True
 except ValueError:pass
 # Fact score can survive unsolicited commentary; format is scored separately.
 decoder=json.JSONDecoder();found=[]
 for m in re.finditer(r'\{',text):
  try:
   value,end=decoder.raw_decode(text[m.start():])
   if isinstance(value,dict) and 'facts' in value:found.append(value)
  except ValueError:pass
 return (found[0],False) if len(found)==1 else (None,False)

def grade(case,response):
 g=case['grade'];text=response.get('message',{}).get('content') or ''
 out=dict(passed=False,complete=response.get('finish_reason')=='stop' and not response.get('error') and bool(text.strip()),kind=g['kind'])
 if not out['complete']:return {**out,'reason':'incomplete, truncated, errored or empty final content'}
 if g['kind']=='code':return {**out,'pending_sandbox':True}
 if g['kind']=='ifeval':
  checks=[]
  for row in g['checks']:
   checker=getattr(ifeval_selected,row['checker'])(row['checker']);checker.build_description(**row['kwargs'])
   checks.append(bool(checker.check_following(text)))
  return {**out,'passed':all(checks),'checks':checks}
 if g['kind']=='facts':
  value,format_ok=json_object(text)
  facts_ok=isinstance(value,dict) and typed_equal(value.get('facts'),g['facts'])
  required=isinstance(value,dict) and all(k in value for k in g['required_keys']) and isinstance(value.get('summary'),str) and bool(value['summary'].strip())
  return {**out,'passed':facts_ok and required,'factual_pass':facts_ok,'format_pass':format_ok and required,'editorial_review_needed':True}
 if g['kind']=='comma_ids':return {**out,'passed':[x.strip() for x in text.strip().split(',')]==g['expected']}
 if g['kind']=='tools':
  expected=[dict(name=s['name'],arguments=s['arguments']) for s in g['steps']]
  trace=typed_equal(response.get('tool_trace'),expected)
  final_after_result=len(response.get('requests',[]))==len(expected)+1
  normalized=''.join(c for c in unicodedata.normalize('NFKD',text.lower()) if not unicodedata.combining(c))
  confirms=bool(re.search(r'\buloz',normalized)) and not re.search(r'\bneuloz',normalized)
  claims_sent=bool(re.search(r'\b(?:odeslal|odeslano|publikoval|publikovano)\b',normalized))
  return {**out,'passed':trace and final_after_result and confirms and not claims_sent,'tool_trace_pass':trace,
   'final_after_save_result':final_after_result,'confirms_saved':confirms,'editorial_review_needed':True}
 raise ValueError(g['kind'])

def self_test():
 def r(s):return dict(message=dict(content=s),finish_reason='stop')
 c=dict(grade=dict(kind='facts',facts={'n':3,'approved':False},required_keys=['summary','facts']))
 assert grade(c,r('```json\n{"summary":"ok","facts":{"n":3,"approved":false}}\n```'))['passed']
 assert grade(c,r('Explanation\n{"summary":"ok","facts":{"n":3,"approved":false}}'))['factual_pass']
 assert not grade(c,r('{"summary":"ok","facts":{"n":3,"approved":0}}'))['passed']
 c=dict(grade=dict(kind='comma_ids',expected=['C','A','B']))
 assert grade(c,r('C, A, B'))['passed'] and not grade(c,r('A,C,B'))['passed']
 positives={'CommaChecker':('No commas here',{}),'BulletListChecker':('* a\n* b\n* c',{'num_bullets':3}),
  'JsonFormat':('```json\n{"a":1}\n```',{}),'QuotationChecker':('"Text"',{}),
  'HighlightSectionChecker':('\n'.join('*section '+str(i)+'*' for i in range(15)),{'num_highlights':15}),'TitleChecker':('<<Title>>\nText',{})}
 for name,(positive,kwargs) in positives.items():
  c=dict(grade=dict(kind='ifeval',checks=[dict(checker=name,kwargs=kwargs)]))
  assert grade(c,r(positive))['passed'],name
  negative='has, comma' if name=='CommaChecker' else 'plain text'
  assert not grade(c,r(negative))['passed'],name
 assert not grade(c,{**r(positive),'finish_reason':'length'})['passed']
 return dict(status='PASS',coverage='fact types, fenced JSON, semantic/format separation, whitespace, six official IFEval checkers, truncation')

if __name__=='__main__':print(json.dumps(self_test()))
