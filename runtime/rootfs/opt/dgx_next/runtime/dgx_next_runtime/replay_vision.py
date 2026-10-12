"""CPU multimodal boundaries for N6's fixed-window replay.

Image tokens attend bidirectionally inside a feature. A replay beginning in
that feature cannot reconstruct it from the trailing causal window alone.
Keep the preceding text prefix, and recompute the complete image feature.
"""


def feature_spans(features):
    result=[]
    for feature in features:
        start,length=feature.mm_position.offset,feature.mm_position.length
        if type(start) is not int or type(length) is not int or start<0 or length<=0:
            raise ValueError('invalid multimodal token interval')
        result.append((start,start+length))
    return tuple(sorted(result))


def safe_hit(hit,spans,window):
    if type(hit) is not int or hit<0 or type(window) is not int or window<=0:
        raise ValueError('invalid prefix replay boundary')
    bound=hit
    # Moving before one image may put the rewind inside an earlier image.
    # Repeat until BOTH the hit and rewind edges avoid every feature interior.
    while True:
        rewind=max(0,bound-window)
        touched=[start for start,end in spans if start<bound<end or start<rewind<end]
        if not touched:return bound
        reduced=min(touched)
        if reduced>=bound:raise AssertionError('nondecreasing image replay boundary')
        bound=reduced


def intersects(start,end,spans):
    if not 0<=start<=end:raise ValueError('invalid scheduled row interval')
    return any(start<right and left<end for left,right in spans)


def atomic_rows(start,count,spans):
    """Stop before an image if the remaining step budget cannot fit it.

    Apply before and after encoder-cache/budget handling: a cached encoder
    embedding is not permission to split the model's bidirectional rows.
    """
    if start<0 or count<0:raise ValueError('negative scheduled image interval')
    if count==0:return 0
    if any(left<start<right for left,right in spans):
        raise ValueError('N6 cannot resume a prefill inside an image feature')
    end=start+count
    while True:
        overlaps=[left for left,right in spans if left<end<right]
        if not overlaps:return end-start
        end=min(overlaps)
        if end<start:raise ValueError('overlapping image interval before scheduled start')


def restrict_hit(manager,request,computed,hit,uncached):
    """Re-query the real coordinator, so block lists/statistics match the cap.

    No request token mutation and no speculative truncation of block groups.
    Rechecking the returned hit matters: coordinator block alignment can move
    the boundary back into an earlier image.
    """
    if not request.mm_features:return computed,hit,uncached
    windows={group.kv_cache_spec.prefix_replay_tokens
             for group in manager.kv_cache_config.kv_cache_groups
             if group.kv_cache_spec.prefix_replay_tokens>0}
    if not windows:return computed,hit,uncached
    if windows!={128}:raise ValueError('unreviewed multimodal replay window')
    spans=feature_spans(request.mm_features)
    while True:
        cap=safe_hit(hit,spans,128)
        if cap==hit:return computed,hit,uncached
        computed,actual,uncached=manager.coordinator.find_longest_cache_hit(request.block_hashes,cap)
        if not 0<=actual<=cap or actual>=hit:raise AssertionError('cache coordinator exceeded safe multimodal cap')
        hit=actual
