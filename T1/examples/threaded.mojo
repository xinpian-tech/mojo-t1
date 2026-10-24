"""Checks independent harts, reusable block barriers and shared buffer access."""

from max.gpu import thread_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.sys import argv


def exchange(
    shared: Pointer[Int32, MutAnyOrigin],
    results: Pointer[Int32, MutAnyOrigin],
):
    var thread = thread_idx.x
    var threads = block_dim.x
    var total = Int32(0)
    for generation in range(32):
        shared.unsafe_store(thread, Int32(generation * 10 + thread))
        barrier()
        total += shared.unsafe_load((thread + 1) % threads)
        barrier()
    results.unsafe_store(thread, total)


def main() raises:
    var arguments = argv()
    var threads = Int(String(arguments[1])) if len(arguments) > 1 else 4
    if threads < 1 or threads > 4:
        raise Error("thread count must be between 1 and 4")
    var context = DeviceContext(api="t1")
    var shared = context.enqueue_create_buffer[DType.int32](threads)
    var results = context.enqueue_create_buffer[DType.int32](threads)
    context.enqueue_function[exchange](shared, results, grid_dim=1, block_dim=threads)
    context.synchronize()
    with results.map_to_host() as values:
        for thread in range(threads):
            var expected = Int32(4960 + 32 * ((thread + 1) % threads))
            if values[thread] != expected:
                raise Error("thread exchange mismatch on hart " + String(thread))
    print("Thread exchange OK:", threads, "harts, 64 barriers")
