"""Bounded parallel semantic workflows with a first-request start barrier."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
import time

from .scoring import workflow_summary
from .workflow import run_task


class FirstRequestBarrier:
    def __init__(self, client, barrier):
        self.client, self.barrier = client, barrier
        self.first = True

    def complete(self, body, *, deadline):
        if self.first:
            self.first = False
            try:
                self.barrier.wait(timeout=max(0.001,deadline-time.monotonic()))
            except threading.BrokenBarrierError as exc:
                raise TimeoutError("cohort first-request barrier") from exc
        return self.client.complete(body,deadline=deadline)


def run_workflow_cohort(tasks, client, *, campaign: str, context: str,
                        deadline: float, save_completed=None, output_tokens=1536):
    if getattr(client,"armed",False) is not True:
        raise RuntimeError("cohort requires an explicitly armed client")
    if not 1 <= len(tasks) <= 8 or len({task.name for task in tasks}) != len(tasks):
        raise ValueError("distinct bounded cohort tasks required")
    started=time.monotonic()
    if started>=deadline:
        raise TimeoutError("cohort deadline before dispatch")
    barrier=threading.Barrier(len(tasks))
    pool=ThreadPoolExecutor(max_workers=len(tasks))
    jobs=[]
    rows_by_job={}
    def collect(done):
        for job in done:
            row=job.result()
            rows_by_job[job]=row
            if save_completed is not None:
                save_completed(row)
    try:
        for slot,task in enumerate(tasks):
            jobs.append(pool.submit(run_task,task,FirstRequestBarrier(client,barrier),
                campaign=campaign,concurrency=len(tasks),slot=slot,deadline=deadline,
                context=context[task.name] if isinstance(context,dict) else context,
                output_tokens=output_tokens))
        pending=set(jobs)
        while pending and time.monotonic()<deadline:
            done,pending=wait(pending,timeout=max(0,deadline-time.monotonic()),return_when=FIRST_COMPLETED)
            collect(done)
        if pending:
            client.abort_all_owned()
            done,pending=wait(pending,timeout=5)
            collect(done)
        if pending:
            raise TimeoutError("owned workflow threads did not stop after abort")
        rows=[rows_by_job[job] for job in jobs]
        ended=time.monotonic()
        return {"regime":"C"+str(len(tasks)),"tasks":rows,
            "started_monotonic":started,"ended_monotonic":ended,
            "cohort_wall_seconds":ended-started,
            "scores":workflow_summary(rows,wall_seconds=ended-started),
            "usage_complete":all(row["usage_complete"] for row in rows),
            "concurrency_scope":"HTTP/workflow overlap, not proof of exact GPU batch size"}
    except BaseException:
        barrier.abort()
        client.abort_all_owned()
        raise
    finally:
        pool.shutdown(wait=False,cancel_futures=True)
