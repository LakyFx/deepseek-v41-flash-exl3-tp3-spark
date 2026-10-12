"""Future N5 exact packer check on synthetic, segregated FP8 cache pages.

No access to serving KV contents and no allocations at import. The worker RPC
owns execution and its authorization, pinned-source checks and memory reserve.
"""
CASES = ((1,1,97,16,False), (6,6,1024,128,False),
         (18,6,4096,512,False), (48,6,16384,2048,False),
         (7,1,1024,128,True))


def inputs(rows, next_n, width, k, varlen, device):
    import torch
    pages = (width+63)//64
    requests = rows if varlen else (rows+next_n-1)//next_n
    cache = torch.empty((requests*pages,64,1,132),dtype=torch.uint8,device=device)
    flat = cache.reshape(requests*pages,64*132)
    # Actual cache layout is all 128-byte values followed by float32 scales,
    # despite the nominal [...,132] shape. Finite deterministic FP8 bytes.
    flat[:,:64*128].copy_((torch.arange(64*128,device=device)%112).to(torch.uint8))
    flat[:,64*128:].view(torch.float32).fill_(0.125)
    table = torch.arange(requests*pages,device=device,dtype=torch.int32).view(requests,pages)
    candidates = torch.arange(k,device=device,dtype=torch.int32).expand(rows,-1).clone()
    candidates[:,::11] = -1
    candidates[:,1] = 2  # duplicate: only first owner contributes
    candidates[:,-1] = (width+7)//8+3  # stock out-of-range edge sentinel
    lengths = torch.tensor([width if i%3 == 0 else 0 if i%3 == 1 else width-3
                            for i in range(rows)],device=device,dtype=torch.int32)
    request_ids = torch.arange(rows,device=device,dtype=torch.int32) if varlen else None
    return cache,table,request_ids,lengths,candidates


def compare(actual, expected):
    import torch
    reports = {}
    for name,a,b in zip(('positions','packed_bytes','table','lengths'),actual,expected,strict=True):
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a,b):
            raise AssertionError('N5 changed '+name+' relative to exact baseline packer')
        reports[name] = {'bit_equal':True,'shape':list(a.shape)}
    return reports


def run(worker):
    import torch
    from dgx_campaign_runtime.candidate_kernels import pack_candidates
    from .indexer_reference import pack_candidates as baseline
    from .indexer import page_table
    device = worker.model_runner.device
    # Initialize the persistent owner outside graph capture. All future shapes
    # are views of this same allocation, so another case cannot evict it.
    owner = page_table(device,48,256)
    pointer = owner.data_ptr()
    reports = []
    for rows,next_n,width,k,varlen in CASES:
        args = inputs(rows,next_n,width,k,varlen,device)
        def call():
            return pack_candidates(*args,next_n=next_n,width=width)
        def reference():
            return baseline(*args,next_n=next_n,width=width)
        eager = compare(call(),reference())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream): call(); call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): captured = call()
        if captured[2].data_ptr() != pointer:
            raise AssertionError('N5 table does not retain the permanent owner')
        replays = []
        for turn in range(3):
            # Changed cache bytes, visibility and candidate ownership, all in
            # gate-owned buffers. Exercise partial and full visible lengths.
            args[0].reshape(-1,64*132)[:,:64*128].fill_(16+turn)
            args[3].fill_(width-turn)
            args[4][:,0] = turn
            expected = reference()
            graph.replay()
            torch.cuda.synchronize()
            replays.append(compare(captured,expected))
        reports.append({'rows':rows,'next_n':next_n,'width':width,'candidates':k,
                        'varlen':varlen,'eager':eager,'graphs':replays})
        del graph, captured, expected, args
    return {'complete':True,'cases':reports,'permanent_table_bytes':48*256*4,
            'scope':'exact baseline vs patched packing, private synthetic FP8 cache, changed-input CUDA graph replays; scoring/top-k untouched'}
