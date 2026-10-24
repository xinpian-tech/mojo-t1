"""Checks the full-width normalization used by TinyLlama on four RTL harts."""

from max.gpu.host import DeviceContext
from std.math import sqrt
from std.sys import argv
from t1llama import _rmsnorm


def normalize(
    outputs: Pointer[Float32, MutAnyOrigin],
    inputs: Pointer[Float32, MutAnyOrigin],
    weights: Pointer[Float32, MutAnyOrigin],
):
    _rmsnorm(outputs, inputs, weights, 2048)


def main() raises:
    var arguments = argv()
    var threads = Int(arguments[1]) if len(arguments) > 1 else 4
    if threads < 1 or threads > 4:
        raise Error("RMSNorm requires one to four harts")
    var context = DeviceContext(api="t1")
    var outputs = context.enqueue_create_buffer[DType.float32](2048)
    var inputs = context.enqueue_create_buffer[DType.float32](2048)
    var weights = context.enqueue_create_buffer[DType.float32](2048)
    var squares = Float32(0)
    with inputs.map_to_host() as host:
        for index in range(2048):
            var value = Float32(index % 31 - 15) / 16
            host[index] = value
            squares += value * value
    with weights.map_to_host() as host:
        for index in range(2048):
            host[index] = Float32(index % 7 + 1) / 8
    context.enqueue_function[normalize](outputs, inputs, weights, grid_dim=1, block_dim=threads)
    context.synchronize()
    var inverse = Float32(1) / sqrt(squares / 2048 + Float32(0.00001))
    with outputs.map_to_host() as host:
        for index in range(2048):
            var expected = Float32(index % 7 + 1) / 8 * (Float32(index % 31 - 15) / 16 * inverse)
            if not abs(host[index] - expected) <= Float32(0.00001):
                print("RMSNorm mismatch:", index, host[index], expected)
                raise Error("RMSNorm mismatch")
    print("RMSNorm OK:", threads, "harts, 2048 elements")
