"""One realistic repair/accounting/report task with actual safe local tools."""
import copy
import hashlib
import json
import re
from pathlib import Path
import sys
import time

from . import fixtures
from . import evaluate as fixture_eval

INITIAL = '''def merge_events(events):
    latest = {}
    for event in events:
        latest[event["request_id"]] = event
    return list(latest.values())
'''
INSTRUCTION = '''Oprav malý problém z agentního workflow: events.py obsahuje merge_events(events).
Pro každý request_id musí vrátit nejnovější záznam podle timestamp; při stejné
timestamp vyhrává pozdější záznam ve vstupním seznamu. Výsledek seřaď podle
request_id. Vstupní seznam ani záznamy neměň a vracej kopie záznamů.
Přečti soubor, oprav jej přes nástroj a skutečně spusť testy. Načti obě stránky
usage z query_records; mezi stránkami mohou být duplicitní id. Pro každé id
ponech nejnovější time (při shodě pozdější výskyt), pak započítej pouze
backend=ds41 a status=done. Spočítej requests, prompt_tokens a completion_tokens.
Nakonec odevzdej finish_task s těmito počty a krátkým českým vysvětlením
pro uživatele (4 až 6 vět): co bylo špatně, proč oprava funguje, co ověřily
testy a jak dopadlo vyúčtování. Samotné tvrzení, že testy prošly, nestačí.
Používej jen poskytnuté nástroje. Úkol dokonči, nepřidávej jiné změny.'''

def function(name,description,properties,required):
    return {'type':'function','function':{'name':name,'description':description,
        'parameters':{'type':'object','properties':properties,'required':required,'additionalProperties':False}}}

TOOLS = [
    function('read_file','Přečti lokální soubor úlohy.',{'path':{'type':'string'}},['path']),
    function('replace_file','Nahraď celý events.py opraveným čistým Pythonem.',{'path':{'type':'string'},'content':{'type':'string'}},['path','content']),
    function('run_tests','Spusť skutečné izolované testy aktuálního events.py.',{},[]),
    function('query_records','Načti jednu stránku účtovacích záznamů.',{'collection':{'type':'string','enum':['usage']},'page':{'type':'integer','enum':[0,1]}},['collection','page']),
    function('finish_task','Odevzdej dokončený výsledek a vysvětlení pro uživatele.',{
        'summary_cs':{'type':'string'},'requests':{'type':'integer'},
        'prompt_tokens':{'type':'integer'},'completion_tokens':{'type':'integer'}},
        ['summary_cs','requests','prompt_tokens','completion_tokens']),
]

def expected_totals():
    records=[r for page in fixtures.USAGE_PAGES for r in page]
    answer=[]
    for ident in sorted({r['id'] for r in records}):
        row=max(((r['time'],i,r) for i,r in enumerate(records) if r['id']==ident),key=lambda item:item[:2])[2]
        if row['backend']=='ds41' and row['status']=='done':answer.append(row)
    return {'requests':len(answer),'prompt_tokens':sum(r['prompt'] for r in answer),
            'completion_tokens':sum(r['completion'] for r in answer)}

class Task:
    def __init__(self):
        self.code=INITIAL
        self.calls=[]
        self.pages=set()
        self.read=False
        self.patch=False
        self.tested_sha=None
        self.tests=None
        self.finished=False
        self.result=None

    @property
    def code_sha(self):return hashlib.sha256(self.code.encode()).hexdigest()

    def invoke(self,name,args):
        if self.finished:raise ValueError('tool after completed task')
        if not isinstance(args,dict):raise ValueError('object arguments required')
        self.calls.append({'name':name,'arguments':copy.deepcopy(args)})
        if name=='read_file' and args=={'path':'events.py'}:
            self.read=True;return {'path':'events.py','content':self.code}
        if name=='replace_file' and set(args)=={'path','content'} and args['path']=='events.py':
            if not self.read:raise ValueError('read events.py first')
            fixture_eval.validate(args['content'])
            self.code=args['content'];self.patch=True;self.tested_sha=None
            return {'written':True,'sha256':self.code_sha}
        if name=='run_tests' and args=={}:
            self.tests=fixture_eval.evaluate(self.code)
            self.tested_sha=self.code_sha if self.tests.get('passed') else None
            return {**self.tests,'source_sha256':self.code_sha}
        if name=='query_records' and set(args)=={'collection','page'} and args['collection']=='usage' and type(args['page']) is int and args['page'] in [0,1]:
            self.pages.add(args['page']);return {'page':args['page'],'records':copy.deepcopy(fixtures.USAGE_PAGES[args['page']]),'pages':2}
        if name=='finish_task' and set(args)=={'summary_cs','requests','prompt_tokens','completion_tokens'}:
            if not self.patch or self.tested_sha!=self.code_sha or self.pages!={0,1}:
                raise ValueError('complete code repair, actual tests and both usage pages first')
            for key,value in expected_totals().items():
                if type(args[key]) is not int or args[key]!=value:raise ValueError('incorrect accounting: '+key)
            summary=args['summary_cs']
            if not isinstance(summary,str) or not 30<=len(summary.split())<=180:
                raise ValueError('provide a concise Czech explanation, 30 to180 words')
            sentences=[s for s in re.split(r'[.!?]+(?=\s|$)',summary) if s.strip()]
            if not 4<=len(sentences)<=6:raise ValueError('provide four to six explanatory sentences')
            self.result=copy.deepcopy(args);self.finished=True
            return {'accepted':True,'source_sha256':self.code_sha,'tests':self.tests}
        raise ValueError('unknown tool or invalid arguments')

def body_for(campaign,concurrency,replica):
    marker=hashlib.sha256(f'{campaign}:{concurrency}:{replica}'.encode()).hexdigest()
    return {'model':'deepseek-v4.1-flash','reasoning_effort':'max',
        'chat_template_kwargs':{'thinking':True,'enable_thinking':True,'reasoning_effort':'max'},
        'temperature':0.6,'seed':42,'max_tokens':65536,
        'tools':copy.deepcopy(TOOLS),'tool_choice':'auto',
        'messages':[{'role':'system','content':f'Workflow instance {marker}.\nJsi vývojářský agent. Použij nástroje, ověř výsledek a dokonči zadání přes finish_task.'},
                    {'role':'user','content':INSTRUCTION}]}

def run_task(client,campaign,concurrency,replica):
    task=Task();body=body_for(campaign,concurrency,replica)
    started=time.monotonic();requests=[];errors=[];failure=None
    # No two-minute cutoff: it is a design target, not a deadline or score trick.
    # Protocol/loop guard counts failures explicitly; it cannot manufacture a pass.
    try:
        for turn in range(16):
            call=client.complete(copy.deepcopy(body))
            reply=call.pop('reply');message=reply.message();usage=reply.usage_counts()
            requests.append({**call,'assistant':message,'usage':usage,'finish_reason':reply.finish_reason})
            if reply.finish_reason=='length':raise ValueError('capacity_limit_hit_not_completed')
            body['messages'].append(message)
            calls=message.get('tool_calls',[])
            if not calls:raise ValueError('missing_required_task_submission')
            for i,tool in enumerate(calls):
                try:
                    args=json.loads(tool['function']['arguments'])
                    outcome=task.invoke(tool['function']['name'],args)
                except (ValueError,TypeError,KeyError,SyntaxError) as exc:
                    outcome={'error':str(exc)[:300]};errors.append(outcome)
                body['messages'].append({'role':'tool','tool_call_id':tool['id'],'content':json.dumps(outcome,ensure_ascii=False)})
                if task.finished and i+1!=len(calls):raise ValueError('extra_tool_after_submission')
            if task.finished:break
        else:raise ValueError('protocol_loop_guard_not_completed')
    except Exception as exc:failure=type(exc).__name__+': '+str(exc)[:300]
    return {'concurrency':concurrency,'replica':replica,'correct':task.finished and failure is None,
        'finished':task.finished,'elapsed_seconds':time.monotonic()-started,'failure':failure,
        'tool_errors':errors,'tools':task.calls,'result':task.result,'final_code':task.code,
        'requests':requests,'prose_semantics':'retained for human review; deterministic checks validate length/accounting/code, not full prose quality'}
