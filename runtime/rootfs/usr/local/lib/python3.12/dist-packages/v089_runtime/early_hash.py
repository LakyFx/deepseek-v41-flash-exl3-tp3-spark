"""One-use Engram hash tickets: current batch only, no next-step prediction."""
from dataclasses import dataclass

def tensor_stamp(tensor):
    return (tensor.data_ptr(),tuple(tensor.shape),tuple(tensor.stride()),str(tensor.dtype))

@dataclass
class Ticket:
    stamp: tuple
    host: object
    ready: object
    generation: int

class EarlyHash:
    def __init__(self,verify_steps=8,max_tokens=64):
        if verify_steps<1 or not 1<=max_tokens<=64:
            raise ValueError('invalid early-hash policy')
        self.verify_left=verify_steps;self.max_tokens=max_tokens
        self.enabled=True;self.ticket=None;self.generation=0;self.reason=None;self.batch=None

    @staticmethod
    def stamp(ids,positions,query_start,lookback,n):
        return (n,tensor_stamp(ids),tensor_stamp(positions),tensor_stamp(query_start),tensor_stamp(lookback))

    def publish(self,stamp,host,ready):
        self.generation+=1
        self.ticket=Ticket(stamp,host,ready,self.generation)

    def invalidate(self):
        self.ticket=None

    def consume(self,stamp,*,recompute=None,equal=None):
        ticket,self.ticket=self.ticket,None  # consume once even on mismatch
        if not self.enabled or ticket is None or stamp!=ticket.stamp:
            return None
        ticket.ready.synchronize()
        if self.verify_left:
            if recompute is None or equal is None:
                print('DGX_V089_EARLY_HASH_DISARMED callback',flush=True)
                self.enabled=False;self.reason='verification callback missing';return None
            # late_hash may reuse hash_host. Snapshot BEFORE recomputing or a
            # corrupted early result could compare equal to itself after refill.
            expected=ticket.host.clone()
            reference=recompute()
            if not equal(expected,reference):
                print('DGX_V089_EARLY_HASH_DISARMED mismatch',flush=True)
                self.enabled=False;self.reason='early/late hashes differ';return None
            self.verify_left-=1
            if self.verify_left==0:
                print('DGX_V089_EARLY_HASH_QUALIFIED',flush=True)
        return ticket.host

def launch_for_model_state(state,batch,req_states,lookback_kernel,triton,image_mask):
    """Called at the start of prepare_attn after runner.prepare_inputs is done."""
    stager=state.engram_stager
    window=state.lookback_token_ids
    if stager is None or window is None or batch.input_ids is None:
        return
    helper=getattr(stager,'_dgx_early',None)
    if helper is None:
        helper=stager._dgx_early=EarlyHash()
    helper.invalidate()
    helper.batch=batch
    n=min(int(batch.num_tokens),stager.max_tokens)
    if not helper.enabled or not 0<n<=helper.max_tokens or not stager.hash_state.ensure_cache():
        return
    all_ids=req_states.all_token_ids.gpu
    depth=window.shape[1]
    lookback_kernel[(window.shape[0],)](window,batch.idx_mapping,
        req_states.num_computed_tokens.gpu,all_ids,all_ids.stride(0),batch.idx_mapping.shape[0],
        DEPTH=depth,BLOCK_DEPTH=triton.next_power_of_2(depth))
    ids=batch.input_ids[:n];positions=batch.positions[:n]
    qsl=batch.query_start_loc[:batch.num_reqs+1]
    hashes=stager.hash_state(ids,positions,qsl,image_mask(ids),window,image_mask(window),None,None)
    host=stager.hash_host[:n]
    host.copy_(hashes[:,:,stager.head_start:stager.head_end],non_blocking=True)
    stager.hashes_ready.record()
    helper.publish(helper.stamp(ids,positions,qsl,window,n),host,stager.hashes_ready)
