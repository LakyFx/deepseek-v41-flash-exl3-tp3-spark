"""Durable standalone-image jobs behind the existing Chat Completions API.

Enabled only by DGX_VISION_JOBS_DB. No agent/tool/image-history requests are
rerouted. Accepted work outlives an HTTP client; interrupted dispatch never
replays. Unknown dispatch blocks this worker until an operator reconciles it.
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

log = logging.getLogger(__name__)
TERMINAL = {'succeeded', 'failed', 'unknown'}


def repeated(text):
    words = re.findall(r'[^\W\d_]+', text[-16000:].lower())[-384:]
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
    def __init__(self, path, clock=time.time):
        self.path, self.clock = Path(path), clock
        self.max_running = media_concurrency()
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, fingerprint TEXT NOT NULL,
                state TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                deadline REAL NOT NULL, body TEXT NOT NULL, result TEXT,
                backend_id TEXT)''')
            db.execute('CREATE INDEX IF NOT EXISTS jobs_lookup ON jobs(owner,fingerprint,created)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def accept(self, owner, body, key=None):
        canonical = {k:v for k,v in body.items() if k not in {'stream','stream_options'}}
        raw = json.dumps(canonical, sort_keys=True, separators=(',',':'), ensure_ascii=False)
        fingerprint = hashlib.sha256((os.getenv('DGX_VISION_JOBS_REVISION','v1')+'\0'+raw).encode()).hexdigest()
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
            if db.execute("SELECT count(*) FROM jobs WHERE state='unknown'").fetchone()[0]:
                raise AdmissionError('Unconfirmed backend dispatch requires operator reconciliation; do not resubmit')
            if db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0] >= 64:
                raise AdmissionError('Vision queue is full; no new job accepted')
            count, size = db.execute('SELECT count(*), coalesce(sum(length(body)+coalesce(length(result),0)),0) FROM jobs').fetchone()
            if count >= 4096 or size + len(raw.encode()) > 512*1024*1024:
                raise AdmissionError('Retained vision ledger reached its storage limit; operator review required')
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,NULL,NULL)',
                       (identity,owner,fingerprint,'queued',now,now,now+900,raw))
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
            if db.execute("SELECT count(*) FROM jobs WHERE state='unknown'").fetchone()[0]:
                return None
            if db.execute("SELECT count(*) FROM jobs WHERE state='running'").fetchone()[0] >= self.max_running:
                return None
            row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created,rowid LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE jobs SET state='running',updated=? WHERE id=?",(now,row['id']))
                return dict(row)

    def finish(self, identity, state, result):
        if state not in TERMINAL:
            raise ValueError('Invalid terminal state')
        with self.connect() as db:
            db.execute('UPDATE jobs SET state=?,updated=?,result=? WHERE id=?',
                       (state,self.clock(),json.dumps(result,ensure_ascii=False),identity))

    def backend(self, identity, backend_id):
        with self.connect() as db:
            db.execute('UPDATE jobs SET backend_id=?,updated=? WHERE id=?',(backend_id,self.clock(),identity))

    def status(self, identity, owner):
        row = self.read(identity,owner)
        if row is None:
            return None
        return {'job_id':identity,'status':row['state'],'accepted':True,'retryable':False,
                'elapsed_seconds':round(max(0,self.clock()-row['created']),3),
                'deadline':row['deadline'],'backend_request_id':row['backend_id'],
                'status_url':'/v1/vision/jobs/'+identity,
                'result':json.loads(row['result']) if row['result'] else None}

    def counts(self):
        with self.connect() as db:
            return dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state').fetchall())


class Jobs:
    def __init__(self, path, engine, sender=None):
        self.ledger, self.engine = Ledger(path), engine
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

    def accept(self, owner, body, key=None):
        if self.runner.done():
            raise AdmissionError('Vision worker stopped; no job accepted')
        identity, created = self.ledger.accept(owner,body,key)
        self.wake.set()
        log.info('Vision job %s %s',identity,'accepted; waiting' if created else 'existing; no resubmission')
        return identity, created

    async def run(self):
        workers = [asyncio.create_task(self.work()) for _ in range(self.ledger.max_running)]
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
            except (asyncio.CancelledError,TimeoutError):
                self.ledger.finish(identity,'unknown',{'status_code':409,'data':error('termination_unconfirmed',
                    'Vision dispatch interrupted; backend termination is unconfirmed. Keep this job ID; do not resubmit.',identity),'headers':{}})
                task.cancel()
                # The existing engine owns shielded abort/admission cleanup. Do
                # not wait forever for cancellation or dispatch another image.
                task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
                if asyncio.current_task().cancelling():
                    raise
            except Exception:
                log.exception('Vision job %s has an unconfirmed outcome',identity)
                self.ledger.finish(identity,'unknown',{'status_code':409,'data':error('backend_outcome_unknown',
                    'Backend outcome unknown. Keep the job ID; do not resubmit.',identity),'headers':{}})
            finally:
                self.active.pop(identity, None)
                self.wake.set()

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
        return await collect(self.engine,row,lambda backend:self.ledger.backend(row['id'],backend))


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
