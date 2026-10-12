"""Future actual graph inventory check; does not replay live request buffers."""
from .registry import GRAPHS


def inventory(runner):
    expected=list(runner.compilation_config.cudagraph_capture_sizes)
    additions=sorted(set(expected)-set(GRAPHS))
    if (expected!=sorted(set(expected)) or not set(GRAPHS)<=set(expected)
            or not 1<=len(additions)<=3 or any(not 1<=n<=48 for n in additions)):
        raise AssertionError('N9 graph delta exceeds the trace-selected contract')
    manager=runner.cudagraph_manager
    if manager is None or manager._graphs_captured is not True:
        raise AssertionError('N9 graph capture incomplete')
    actual=manager.captured_token_counts()
    if not set(expected)<=set(actual) or max(actual)>48:
        raise AssertionError('configured N9 buckets not actually captured')
    # This method enumerates manager.graphs keys, not configured intentions.
    if any(graph is None for graph in manager.graphs.values()):
        raise AssertionError('missing actual CUDA graph object')
    return {'complete':True,'configured':expected,'actual_captured':actual,'additions':additions,
            'scope':'actual captured graph objects and protected geometry; request-level correctness covered by common quality gate'}
