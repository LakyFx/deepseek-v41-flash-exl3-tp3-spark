"""Durable standalone-image jobs behind the existing Chat Completions API.

Enabled only by DGX_VISION_JOBS_DB. No agent/tool/image-history requests are
rerouted. Accepted work outlives an HTTP client; interrupted dispatch never
replays. An owned UNKNOWN reserves its slot until a native termination receipt;
legacy dispatches without that contract retain operator reconciliation.
"""
import asyncio
from contextlib import contextmanager
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path

from .media_admission import has_images, media_concurrency
from .vision_control import seconds, confirms_termination

log = logging.getLogger(__name__)
TERMINAL = {'succeeded', 'failed', 'unknown'}


def repeated(text):
    # Numeric values/row identifiers carry evidence in tables and charts.
    # Removing them turns distinct measurements into identical word cycles.
    words = re.findall(r'[^\W_]+(?:[.,]\d+)?', text[-16000:].lower())[-384:]
    for period in range(1, min(97, len(words)//5 + 1)):
        copies = min(12, len(words)//period)
        if period*copies >= 60 and words[-period*copies:] == words[-period:]*copies:
            return True
    return False


def contaminated(body):
    for message in body.get('messages') or []:
        if isinstance(message, dict) and message.get('role') == 'assistant':
            for field in ('reasoning', 'reasoning_content'):
                value = message.get(field)
                if isinstance(value, str) and repeated(value):
                    return True
    return False


def standalone(body):
    messages = body.get('messages')
    return (not body.get('tools') and not body.get('tool_choice')
            and body.get('n', 1) == 1 and isinstance(messages, list)
            and all(isinstance(m, dict) and m.get('role') in {'system','developer','user'} for m in messages)
            and sum(m.get('role') == 'user' for m in messages) == 1 and has_images(body))


def error(code, message, identity=None):
    return {'error': {'type': 'vision_job_error', 'code': code, 'message': message,
                      'retryable': False, 'job_id': identity}}


class AdmissionError(ValueError):
    pass


class Ledger:
    def __init__(self, path, clock=time.time, controlled=False):
        self.path, self.clock = Path(path), clock
        self.controlled = controlled
        self.max_running = media_concurrency()
        self.queue_timeout = seconds('DGX_VISION_QUEUE_TIMEOUT_SECONDS', 1800)
        self.job_timeout = seconds('DGX_VISION_JOB_TIMEOUT_SECONDS', 1800)
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, fingerprint TEXT NOT NULL,
                state TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                deadline REAL NOT NULL, body TEXT NOT NULL, result TEXT,
                backend_id TEXT)''')
            db.execute('CREATE INDEX IF NOT EXISTS jobs_lookup ON jobs(owner,fingerprint,created)')
            db.execute('''CREATE TABLE IF NOT EXISTS vision_dispatch (
                job_id TEXT PRIMARY KEY, body TEXT NOT NULL, receipt TEXT)''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def accept(self, owner, body, key=None, *, generation_body=None):
        canonical = {k:v for k,v in body.items() if k not in {'stream','stream_options'}}
        raw = json.dumps(canonical, sort_keys=True, separators=(',',':'), ensure_ascii=False)
        fingerprint = hashlib.sha256((os.getenv('DGX_VISION_JOBS_REVISION','v1')+'\0'+raw).encode()).hexdigest()
        # Deduplication describes the caller's input, not a later operator policy.
        # Keep accepted jobs stable across policy changes; never replay them.
        if generation_body is not None:
            canonical = {k:v for k,v in generation_body.items() if k not in {'stream','stream_options'}}
            raw = json.dumps(canonical, sort_keys=True, separators=(',',':'), ensure_ascii=False)
        now = self.clock()
        identity = 'vision_' + (hashlib.sha256((owner+'\0'+key).encode()).hexdigest() if key else uuid.uuid4().hex)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = (db.execute('SELECT * FROM jobs WHERE id=? AND owner=?',(identity,owner)).fetchone()
                   if key else db.execute('SELECT * FROM jobs WHERE owner=? AND fingerprint=? AND deadline>? ORDER BY created DESC LIMIT 1',
                                          (owner,fingerprint,now-900)).fetchone())
            if old:
                if old['fingerprint'] != fingerprint:
                    raise AdmissionError('Idempotency key belongs to different input')
                return old['id'], False
            if self.legacy_unknown(db):
                raise AdmissionError('Backend termination awaits confirmation; do not resubmit accepted jobs')
            if db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0] >= 64:
                raise AdmissionError('Vision queue is full; no new job accepted')
            count, size = db.execute('SELECT count(*), coalesce(sum(length(body)+coalesce(length(result),0)),0) FROM jobs').fetchone()
            if count >= 4096 or size + len(raw.encode()) > 512*1024*1024:
                raise AdmissionError('Retained vision ledger reached its storage limit; operator review required')
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,NULL,NULL)',
                       (identity,owner,fingerprint,'queued',now,now,now+self.queue_timeout,raw))
        return identity, True

    def read(self, identity, owner=None):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (identity,)).fetchone()
        if row is None or (owner is not None and row['owner'] != owner):
            return None
        return dict(row)

    def recover(self):
        with self.connect() as db:
            db.execute("UPDATE jobs SET state='unknown',updated=?,result=? WHERE state='running'",
                       (self.clock(),json.dumps(error('termination_unconfirmed','Gateway interrupted; do not replay this job'))))

    def claim(self):
        now = self.clock()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("UPDATE jobs SET state='failed',updated=?,result=? WHERE state='queued' AND deadline<=?",
                       (now,json.dumps(error('queue_deadline','Job expired before dispatch')),now))
            if self.legacy_unknown(db):
                return None
            # Each unconfirmed owned dispatch retains ONE physical image slot,
            # so an ambiguous job cannot freeze another healthy vision lane.
            if db.execute("SELECT count(*) FROM jobs WHERE state IN ('running','unknown')").fetchone()[0] >= self.max_running:
                return None
            row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created,rowid LIMIT 1").fetchone()
            if row:
                deadline = now+self.job_timeout
                if self.controlled:
                    # Distinguish a crash BEFORE dispatch from legacy jobs.
                    # A complete dispatch record must replace this marker
                    # atomically before the engine can receive a POST.
                    db.execute('INSERT INTO vision_dispatch(job_id,body) VALUES (?,?)',
                               (row['id'],json.dumps({'pending':True})))
                db.execute("UPDATE jobs SET state='running',updated=?,deadline=? WHERE id=?",(now,deadline,row['id']))
                return {**dict(row), 'state':'running', 'deadline':deadline}

    def finish(self, identity, state, result):
        if state not in TERMINAL:
            raise ValueError('Invalid terminal state')
        with self.connect() as db:
            db.execute('UPDATE jobs SET state=?,updated=?,result=? WHERE id=?',
                       (state,self.clock(),json.dumps(result,ensure_ascii=False),identity))

    def backend(self, identity, backend_id):
        with self.connect() as db:
            db.execute('UPDATE jobs SET backend_id=?,updated=? WHERE id=?',(backend_id,self.clock(),identity))

    def dispatch(self, identity, dispatch):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT state FROM jobs WHERE id=?',(identity,)).fetchone()
            previous=db.execute('SELECT body FROM vision_dispatch WHERE job_id=?',(identity,)).fetchone()
            if (not row or row['state']!='running'
                    or (previous and not json.loads(previous['body']).get('pending'))):
                raise ValueError('Vision dispatch is no longer owned and running')
            db.execute('''INSERT INTO vision_dispatch(job_id,body) VALUES (?,?)
                ON CONFLICT(job_id) DO UPDATE SET body=excluded.body''',(identity,json.dumps(dispatch)))
            db.execute('UPDATE jobs SET backend_id=?,updated=? WHERE id=?',
                       (dispatch['request_id'],self.clock(),identity))

    def dispatch_for(self, identity):
        with self.connect() as db:
            row = db.execute('SELECT body FROM vision_dispatch WHERE job_id=?',(identity,)).fetchone()
        return json.loads(row[0]) if row else None

    def unknown(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM jobs WHERE state='unknown'")]

    def legacy_unknown(self, db):
        if not self.controlled:
            return db.execute("SELECT count(*) FROM jobs WHERE state='unknown'").fetchone()[0]
        return db.execute("""SELECT count(*) FROM jobs j LEFT JOIN vision_dispatch d ON d.job_id=j.id
            WHERE j.state='unknown' AND d.job_id IS NULL""").fetchone()[0]

    def blocked(self):
        with self.connect() as db:
            return bool(self.legacy_unknown(db) or db.execute(
                "SELECT count(*) FROM jobs WHERE state='unknown'").fetchone()[0] >= self.max_running)

    def not_dispatched(self, identity, code):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('''SELECT j.state,j.backend_id,d.body,d.receipt FROM jobs j
                JOIN vision_dispatch d ON d.job_id=j.id WHERE j.id=?''',(identity,)).fetchone()
            if row and row['state']=='failed' and row['receipt']:
                previous=json.loads(row['receipt'])
                if previous.get('schema')=='dgx.vision-not-dispatched.v1' and previous.get('job_id')==identity:
                    return  # Worker and background reconciliation can race.
            if (not row or row['state']!='unknown' or row['backend_id'] is not None
                    or json.loads(row['body'])!={'pending':True}):
                raise ValueError('A persisted pre-dispatch marker is required')
            receipt={'schema':'dgx.vision-not-dispatched.v1','termination_confirmed':True,
                     'state':'not_dispatched','job_id':identity}
            result={'status_code':504 if code=='vision_deadline' else 422,'headers':{},
                    'data':error(code,'Vision stopped before backend dispatch. No automatic replay.',identity)}
            db.execute('UPDATE vision_dispatch SET receipt=? WHERE job_id=?',(json.dumps(receipt),identity))
            db.execute("UPDATE jobs SET state='failed',updated=?,result=? WHERE id=?",
                       (self.clock(),json.dumps(result),identity))

    def confirmed(self, identity, receipt, code='vision_interrupted'):
        result = {'status_code':504 if code=='vision_deadline' else 422,
                  'data':error(code,'Vision stopped; backend termination confirmed. No automatic replay.',identity),
                  'headers':{}}
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            dispatch = db.execute('SELECT body FROM vision_dispatch WHERE job_id=?',(identity,)).fetchone()
            if not dispatch:
                raise ValueError('No owned dispatch for termination receipt')
            owned = json.loads(dispatch[0])
            if not confirms_termination(receipt,owned):
                raise ValueError('Termination receipt does not match owned dispatch')
            db.execute('UPDATE vision_dispatch SET receipt=? WHERE job_id=?',(json.dumps(receipt),identity))
            db.execute("UPDATE jobs SET state='failed',updated=?,result=? WHERE id=? AND state='unknown'",
                       (self.clock(),json.dumps(result),identity))

    def status(self, identity, owner):
        row = self.read(identity,owner)
        if row is None:
            return None
        if row['state']=='queued' and row['deadline']<=self.clock():
            # A stopped dispatcher must not leave a caller polling forever.
            # The conditional update cannot overwrite a concurrent claim.
            with self.connect() as db:
                db.execute("UPDATE jobs SET state='failed',updated=?,result=? WHERE id=? AND state='queued' AND deadline<=?",
                           (self.clock(),json.dumps(error('queue_deadline','Job expired before dispatch')),
                            identity,self.clock()))
            row=self.read(identity,owner)
        return {'job_id':identity,'status':row['state'],'accepted':True,'retryable':False,
                'elapsed_seconds':round(max(0,self.clock()-row['created']),3),
                'deadline':row['deadline'],'backend_request_id':row['backend_id'],
                'status_url':'/v1/vision/jobs/'+identity,
                'result':json.loads(row['result']) if row['result'] else None}

    def counts(self):
        with self.connect() as db:
            return dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state').fetchall())


class Jobs:
    def __init__(self, path, engine, sender=None, control=None):
        self.ledger, self.engine = Ledger(path,controlled=control is not None), engine
        self.control = control
        self.final_answer_tokens = int(os.getenv('DGX_VISION_FINAL_ANSWER_TOKENS', '0'))
        if not 0 <= self.final_answer_tokens <= 65536:
            raise ValueError('DGX_VISION_FINAL_ANSWER_TOKENS must be between 0 and 65536')
        # An operator-selected standalone-image profile. Without this opt-in,
        # retain the previous medium-only headroom policy for existing installs.
        effort = os.getenv('DGX_VISION_REASONING_EFFORT', '')
        if effort not in {'', 'medium', 'max'}:
            raise ValueError('DGX_VISION_REASONING_EFFORT must be medium or max')
        self.force_reasoning = bool(effort)
        self.reasoning_effort = effort or 'medium'
        self.thinking_budget = int(os.getenv('DGX_VISION_THINKING_BUDGET', '3072'))
        if not 1 <= self.thinking_budget <= 65536:
            raise ValueError('DGX_VISION_THINKING_BUDGET must be between 1 and 65536')
        self.sender = sender or self.collect
        self.wake = asyncio.Event()
        self.runner = None
        self.active = {}

    async def start(self):
        self.lease = self.ledger.path.with_suffix('.lock').open('a')
        if os.name == 'posix':
            import fcntl
            fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.ledger.recover()
        self.runner = asyncio.create_task(self.run())

    async def close(self):
        self.runner.cancel()
        await asyncio.gather(self.runner, return_exceptions=True)
        self.lease.close()

    def output_budget(self):
        if not self.final_answer_tokens:
            return None
        return {'reasoning_effort':self.reasoning_effort, 'thinking_budget':self.thinking_budget,
                'final_answer_tokens':self.final_answer_tokens,
                'max_tokens':self.thinking_budget+self.final_answer_tokens}

    def generation_body(self, body):
        # Jobs admits standalone image questions only. Main-agent tool/history
        # requests never reach this method; their text reasoning is independent.
        if not self.final_answer_tokens or (not self.force_reasoning and body.get('reasoning_effort') != 'medium'):
            return body
        result = dict(body)
        result.pop('max_completion_tokens', None)
        result['max_tokens'] = self.thinking_budget + self.final_answer_tokens
        result['thinking_budget'] = self.thinking_budget
        if self.force_reasoning:
            result['reasoning_effort'] = self.reasoning_effort
        return result

    def accept(self, owner, body, key=None):
        if self.runner.done():
            raise AdmissionError('Vision worker stopped; no job accepted')
        identity, created = self.ledger.accept(owner,body,key, generation_body=self.generation_body(body))
        self.wake.set()
        log.info('Vision job %s %s',identity,'accepted; waiting' if created else 'existing; no resubmission')
        return identity, created

    async def run(self):
        workers = [asyncio.create_task(self.work()) for _ in range(self.ledger.max_running)]
        if self.control is not None:
            workers.append(asyncio.create_task(self.reconcile_unknown()))
        try:
            await asyncio.gather(*workers)
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    async def work(self):
        while True:
            self.wake.clear()
            row = self.ledger.claim()
            if not row:
                try:
                    # Register the waiter in this task before yielding; a peer
                    # clearing the shared event must not lose a wakeup here.
                    async with asyncio.timeout(5):
                        await self.wake.wait()
                except TimeoutError:
                    pass
                continue
            identity = row['id']
            task = self.active[identity] = asyncio.create_task(self.sender(row))
            try:
                done,_ = await asyncio.wait({task},timeout=max(0,row['deadline']-self.ledger.clock()))
                if not done:
                    raise TimeoutError()
                result = task.result()
                self.ledger.finish(identity,'succeeded' if result['status_code']==200 else 'failed',result)
            except (asyncio.CancelledError,TimeoutError) as exc:
                self.ledger.finish(identity,'unknown',{'status_code':409,'data':error('termination_unconfirmed',
                    'Vision dispatch interrupted; backend termination is unconfirmed. Keep this job ID; do not resubmit.',identity),'headers':{}})
                task.cancel()
                # The existing engine owns shielded abort/admission cleanup. Do
                # not wait forever for cancellation or dispatch another image.
                task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
                if asyncio.current_task().cancelling():
                    raise
                await self.terminate(row, 'vision_deadline' if isinstance(exc,TimeoutError) else 'vision_interrupted')
            except Exception:
                log.exception('Vision job %s has an unconfirmed outcome',identity)
                self.ledger.finish(identity,'unknown',{'status_code':409,'data':error('backend_outcome_unknown',
                    'Backend outcome unknown. Keep the job ID; do not resubmit.',identity),'headers':{}})
                await self.terminate(row, 'vision_transport_failed')
            finally:
                self.active.pop(identity, None)
                self.wake.set()

    async def terminate(self, row, code):
        dispatch = self.ledger.dispatch_for(row['id'])
        if self.control is not None and dispatch:
            if dispatch.get('pending'):
                self.ledger.not_dispatched(row['id'],code)
                self.wake.set()
                return
            receipt = await self.control.terminate(dispatch)
            if receipt:
                self.ledger.confirmed(row['id'],receipt,code)
                self.wake.set()

    async def reconcile_unknown(self):
        while True:
            for row in self.ledger.unknown():
                dispatch = self.ledger.dispatch_for(row['id'])
                if not dispatch:
                    continue  # Legacy UNKNOWN needs its existing operator receipt.
                try:
                    if dispatch.get('pending'):
                        self.ledger.not_dispatched(row['id'],'vision_interrupted')
                        self.wake.set()
                        continue
                    receipt = await self.control.cancel(dispatch)
                    if receipt:
                        self.ledger.confirmed(row['id'],receipt)
                        self.wake.set()
                except Exception:
                    log.warning('Vision cancellation receipt unavailable for %s',row['id'])
            await asyncio.sleep(5)

    async def wait(self, identity, owner):
        while True:
            row = self.ledger.status(identity,owner)
            if row['status'] in TERMINAL:
                result = row['result']
                # Recovery/queue expiry errors also use the OpenAI error envelope.
                if 'status_code' not in result:
                    result = {'status_code':409,'data':result,'headers':{}}
                return result
            await asyncio.sleep(0.1)

    async def collect(self, row):
        from .vision_result import collect
        dispatch = None
        if self.control is not None:
            try:
                dispatch = await self.control.dispatch(row,self.ledger)
            except Exception:
                # Capability check failed before any POST. Never fall back to an
                # untracked generation on a backend lacking cancellation proof.
                return {'status_code':503,'data':error('vision_control_unavailable',
                    'Vision cancellation control unavailable; image was not dispatched.',row['id']),'headers':{}}
        return await collect(self.engine,row,lambda backend:self.ledger.backend(row['id'],backend),dispatch)


def owner(request):
    return hashlib.sha256(request.headers.get('authorization','').encode()).hexdigest()


def job_headers(identity):
    return {'X-DGX-Vision-Job-ID':identity,'Location':'/v1/vision/jobs/'+identity,
            'X-Should-Retry':'false','Cache-Control':'no-store'}


async def stream_result(jobs, identity, principal, heartbeat):
    """SSE comments are transport progress, never model/assistant messages."""
    task = asyncio.create_task(jobs.wait(identity,principal))
    try:
        while True:
            status = jobs.ledger.status(identity,principal)
            yield f": vision-job {identity} {status['status']}; accepted, wait for same job\n\n".encode()
            done,_ = await asyncio.wait({task},timeout=heartbeat)
            if done:
                break
        result = task.result()
        if result['status_code'] != 200:
            yield ('data: '+json.dumps(result['data'])+'\n\n').encode()
        else:
            data = result['data']
            chunk = {k:data[k] for k in ('id','created','model')}
            chunk.update(object='chat.completion.chunk', choices=[{'index':0,
                'delta':data['choices'][0]['message'],'finish_reason':None}])
            yield ('data: '+json.dumps(chunk,ensure_ascii=False)+'\n\n').encode()
            chunk['choices'] = [{'index':0,'delta':{},'finish_reason':'stop'}]
            yield ('data: '+json.dumps(chunk)+'\n\n').encode()
            if data.get('usage'):
                chunk.update(choices=[],usage=data['usage'])
                yield ('data: '+json.dumps(chunk)+'\n\n').encode()
        yield b'data: [DONE]\n\n'
    finally:
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)
