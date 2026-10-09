"""One bounded sequential workload suite, with durable per-request evidence.

Preparing a suite never sends requests. Fleet/controller owns ingress, native
generation/idle proofs and cache resets. Same regime allocations on every arm
retain a specialised gain instead of testing a candidate only at short C1.
"""
from __future__ import annotations

import hashlib
import json
import time

from .cohort import run_workflow_cohort
from .fixtures import make_tasks, initial_body
from .legacy import run_legacy
from .load_regimes import require_armed, run_cold_hot, run_mixed
from .monitor import workflow_attempts


PRIMARY = (("workflow-C1",80),("workflow-C3",130),("legacy-screen6",300),
           ("mixed",140),("64K-C1",95),("128K-C3",325),("8K-C8",80))
FULL_LEGACY = (("legacy-full16",720),)


def allocation(profile="coverage20"):
    phases = PRIMARY if profile == "coverage20" else FULL_LEGACY if profile == "legacy-full16" else None
    if phases is None:
        raise ValueError("unknown frozen workload profile")
    value={"schema":"dgx.frozen-workload-allocation.v1","profile":profile,
           "phases":[{"id":name,"seconds":seconds} for name,seconds in phases],
           "hard_seconds":1200,"phase_budget_seconds":sum(seconds for _,seconds in phases),
           "drain_reserve_seconds":50,"primary_same_on_all_arms":profile=="coverage20",
           "coverage":"all seven regimes; original full16 remains a separate prepared option"
              if profile=="coverage20" else "exact original sixteen cold/hot C1..C4 cells",
           "missing_cells":"UNPAIRED_NOT_ZERO_GAIN","inference_now":False}
    if value["phase_budget_seconds"]+value["drain_reserve_seconds"]>value["hard_seconds"]:
        raise ValueError("frozen workload allocation exceeds hard budget")
    value["allocation_sha256"]=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return value


class Suite:
    def __init__(self, client, legacy, workloads, *, campaign, version, monitor,
                 save, drain, cold_reset, profile="coverage20", clock=time.monotonic):
        self.client,self.legacy,self.workloads=client,legacy,workloads
        self.campaign,self.version=campaign,version
        self.monitor,self.save,self.drain,self.cold_reset=monitor,save,drain,cold_reset
        self.allocation=allocation(profile)
        self.clock=clock
        # Each workflow's exact reference context/tools/main-max input is the
        # measured frozen first body. Do not replace it with one shared dossier.
        for task in make_tasks():
            entry=workloads[task.name]
            if initial_body(task,context=entry["context"])!=entry["body"]:
                raise ValueError("workflow fixture differs from frozen tokenized input")

    def run(self, *, deadline):
        require_armed(self.client)
        started=self.clock()
        if not started<deadline<=started+self.allocation["hard_seconds"]+0.001:
            raise ValueError("nonexpired at-most20minute workload budget required")
        result={"schema":"dgx.actual-expanded-variant-workloads.v1","version":self.version,
                "campaign_id":self.campaign,"allocation":self.allocation,"phases":[],
                "complete":False,"model_test":True}
        for phase in self.allocation["phases"]:
            name=phase["id"]
            phase_start=self.clock()
            if phase_start>=deadline-20:
                result["phases"].append({"id":name,"status":"MISSING_DEADLINE_NOT_ZERO_GAIN"})
                continue
            # Leave room for owned socket/abort completion and engine-idle
            # proof. No new request is admitted in the last20seconds.
            phase_deadline=min(deadline-20,phase_start+phase["seconds"])
            record={"id":name,"started_monotonic":phase_start,"complete":False,"result":None}
            try:
                value=self.phase(name,phase_deadline)
                record.update(result=value,complete=self.complete(name,value),status="OBSERVED")
            except (TimeoutError,OSError) as exc:
                record.update(status="PARTIAL_ROWS_RETAINED",error=type(exc).__name__+":"+str(exc)[:300])
            finally:
                record["wall_seconds"]=self.clock()-phase_start
                self.save("regime",record)
            # Never start the next cell while an earlier owned request keeps
            # computing. Failure to prove idle is fatal and triggers rollback.
            proof=self.drain(deadline=min(deadline,self.clock()+7))
            self.save("drain",proof)
            result["phases"].append(record)
        result["wall_seconds"]=self.clock()-started
        result["complete"]=all(row.get("complete") is True for row in result["phases"])
        result["comparison_policy"]="Report each regime and quality; one pass has no confidence interval or universal20percent claim"
        return result

    def phase(self,name,deadline):
        save_request=lambda row:self.save("request",row)
        if name.startswith("workflow-"):
            tasks=make_tasks()
            if name=="workflow-C1":tasks=tasks[:1]
            contexts={task.name:self.workloads[task.name]["context"] for task in tasks}
            run=lambda:run_workflow_cohort(tasks,self.client,campaign=self.campaign,context=contexts,
                deadline=deadline,save_completed=lambda row:self.save("workflow",row))
            return self.monitor(name,workflow_attempts,deadline,run)
        # Reset is an operation, not a truthy function masquerading as a
        # receipt. Persist only the actual serializable success evidence.
        reset=self.cold_reset() if callable(self.cold_reset) else self.cold_reset
        if not isinstance(reset,dict) or not reset:
            raise ValueError('actual cold-reset receipt required before load regime')
        if name.startswith("legacy-"):
            return run_legacy(self.client,self.legacy,campaign=self.campaign,deadline=deadline,
                full=name=="legacy-full16",cold_reset_receipt=reset,
                save_completed=save_request,measure=self.monitor)
        if name=="mixed":
            run=lambda:run_mixed(self.client,[self.workloads["short-"+str(i)] for i in range(3)],
                self.workloads["mixed-cold-32k"],campaign=self.campaign,deadline=deadline,
                cold_reset_receipt=reset,save_completed=save_request)
            # Three primes, three leaders and (if sent) one actual cold arrival.
            count=lambda value:3+len(value.get("leaders",{}).get("rows",[]))+int(value.get("injected") is not None)
            return self.monitor(name,count,deadline,run)
        ids=(["long-64k"] if name=="64K-C1" else
             ["long-128k-"+str(i) for i in range(3)] if name=="128K-C3" else
             ["short-"+str(i) for i in range(8)] if name=="8K-C8" else None)
        if ids is None:raise ValueError("unknown workload regime")
        return run_cold_hot(self.client,[self.workloads[key] for key in ids],campaign=self.campaign,
            group=name,deadline=deadline,output_tokens=256 if name=="8K-C8" else 128,
            cold_reset_receipt=reset,save_completed=save_request,measure=self.monitor)

    @staticmethod
    def complete(name,value):
        if name.startswith("workflow-"):
            # Execution completeness and correctness are distinct. Wrong but
            # fully measured tasks remain in the quality denominator.
            return value["usage_complete"]
        if name.startswith("legacy-"):return value["complete"]
        if name=="mixed":
            return bool(value.get("injected") and value["injected"]["complete"]
                and value["leaders"]["usage_complete"] and value.get("cache_regime_verified") is True)
        return bool(value.get("cold_cache_verified") and value.get("hot_cache_verified"))
