"""Small CUDA-driver check: RB4-6 versus the established RB8 arithmetic."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import struct
import sys


def main(root):
    root = Path(root)
    manifest = json.loads((root / 'aot/aot.json').read_bytes())
    driver = C.CDLL('libcuda.so.1')
    def call(name, *args):
        result = getattr(driver, name)(*args)
        if result:
            raise RuntimeError(f'{name}: CUDA {result}')
    call('cuInit', 0)
    device = C.c_int()
    call('cuDeviceGet', C.byref(device), 0)
    context = C.c_void_p()
    call('cuDevicePrimaryCtxRetain', C.byref(context), device)
    call('cuCtxSetCurrent', context)
    allocations, modules = [], []
    def alloc(raw):
        pointer = C.c_uint64()
        call('cuMemAlloc_v2', C.byref(pointer), C.c_size_t(len(raw)))
        allocations.append(pointer)
        buf = C.create_string_buffer(raw)
        call('cuMemcpyHtoD_v2', pointer, buf, C.c_size_t(len(raw)))
        return pointer
    def read(pointer, size):
        out = C.create_string_buffer(size)
        call('cuMemcpyDtoH_v2', out, pointer, C.c_size_t(size))
        return out.raw
    def floats(n, scale):
        return b''.join(struct.pack('<f', ((i * 17) % 101 - 50) * scale) for i in range(n))
    def bf16(n):
        return b''.join(struct.pack('<f', ((i * 13) % 67 - 33) / 33)[2:] for i in range(n))
    def spec(rb, posted):
        return next(k for k in manifest['kernels'] if k['fn'] == '_hc_mix_pf'
                    and k['consts']['RB']['int'] == rb
                    and k['consts']['WORLD']['int'] == (3 if posted else 1)
                    and k['consts']['POSTED']['int'] == int(posted))
    results = []
    try:
        for posted in (False, True):
            for rows in (4, 5, 6):
                d, wide, nb = 5120, 20480, 40
                x = alloc(bf16(rows * wide))
                g = alloc(floats(3 * rows * d, .01)) if posted else x
                post = alloc(floats(rows * 4, .005))
                comb = alloc(floats(rows * 16, .003))
                fn = alloc(floats(24 * wide, .0002))
                outputs = []
                for rb in (8, rows):
                    k = spec(rb, posted)
                    assert not k['global_scratch'] and not k['profile_scratch']
                    module = C.c_void_p()
                    cubin = root / 'aot/cubins' / (k['hash'] + '.cubin')
                    assert hashlib.sha256(cubin.read_bytes()).hexdigest() == k['hash']
                    call('cuModuleLoad', C.byref(module), str(cubin).encode())
                    modules.append(module)
                    function = C.c_void_p()
                    call('cuModuleGetFunction', C.byref(function), module, k['name'].encode())
                    xo = alloc(b'\xa5\xa5' * (rows * wide))
                    part = alloc(struct.pack('<f', -123) * (rows * nb * 32))
                    values = [x, xo, g, C.c_int(rows * d if posted else 0), post,
                              comb, fn, part, C.c_int(rows), C.c_void_p(), C.c_void_p()]
                    args = (C.c_void_p * len(values))(*(C.cast(C.byref(v), C.c_void_p) for v in values))
                    call('cuLaunchKernel', function, C.c_uint(nb), C.c_uint(1), C.c_uint(1),
                         C.c_uint(k['num_warps'] * 32), C.c_uint(1), C.c_uint(1),
                         C.c_uint(k['shared']), C.c_void_p(), args, C.c_void_p())
                    call('cuCtxSynchronize')
                    outputs.append((read(xo, rows * wide * 2), read(part, rows * nb * 32 * 4)))
                assert outputs[0] == outputs[1], (rows, posted, 'bitwise mismatch')
                results.append({'rows': rows, 'posted': posted, 'bitwise_equal_to_RB8': True})
                for pointer in allocations:
                    call('cuMemFree_v2', pointer)
                allocations.clear()
                for module in modules:
                    call('cuModuleUnload', module)
                modules.clear()
    finally:
        for pointer in allocations:
            call('cuMemFree_v2', pointer)
        for module in modules:
            call('cuModuleUnload', module)
        call('cuDevicePrimaryCtxRelease_v2', device)
    print(json.dumps({'status': 'GPU_BITWISE_PASS', 'cases': results}))


if __name__ == '__main__':
    main(sys.argv[1])
