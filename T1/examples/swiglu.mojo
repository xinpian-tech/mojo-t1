"""Isolates vector SwiGLU arithmetic from matrix operations and attention."""

from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from std.math import exp
from std.sys.info import simd_width_of
from std.sys import argv
from t1llama import _read_f32


def swiglu(
    output: Pointer[Float32, MutAnyOrigin],
    gate: Pointer[Float32, MutAnyOrigin],
    up: Pointer[Float32, MutAnyOrigin],
):
    comptime WIDTH = simd_width_of[DType.float32]()
    var index = thread_idx.x * 32
    var last = index + 32
    while index + WIDTH <= last:
        var value = gate.unsafe_load[width=WIDTH](index)
        var multiplier = up.unsafe_load[width=WIDTH](index)
        var sigmoid = value / (SIMD[DType.float32, WIDTH](1) + exp(-value))
        output.unsafe_store(index, sigmoid * multiplier)
        index += WIDTH
    while index < last:
        var value = gate[index]
        output[index] = value / (Float32(1) + exp(-value)) * up[index]
        index += 1


def main() raises:
    var context = DeviceContext(api="t1")
    var output = context.enqueue_create_buffer[DType.float32](128)
    var gate = context.enqueue_create_buffer[DType.float32](128)
    var up = context.enqueue_create_buffer[DType.float32](128)
    var gate_values = List[Float32](length=128, fill=0)
    var up_values = List[Float32](length=128, fill=0)
    var arguments = argv()
    for index in range(128):
        gate_values[index] = Float32(index % 65 - 32) / 8
        up_values[index] = Float32(index % 17 - 8) / 4
    if len(arguments) > 1:
        var raw = open(String(arguments[1]), "r").read_bytes()
        if len(raw) != 1024:
            raise Error("Expected 128 gate floats followed by 128 up floats")
        for index in range(128):
            gate_values[index] = _read_f32(raw.unsafe_ptr(), index * 4)
            up_values[index] = _read_f32(raw.unsafe_ptr(), (128 + index) * 4)
    with gate.map_to_host() as host:
        for index in range(128):
            host[index] = gate_values[index]
    with up.map_to_host() as host:
        for index in range(128):
            host[index] = up_values[index]
    context.enqueue_function[swiglu](output, gate, up, grid_dim=1, block_dim=4)
    context.synchronize()
    with output.map_to_host() as host:
        for index in range(128):
            var value = gate_values[index]
            var expected = value / (Float32(1) + exp(-value)) * up_values[index]
            if not abs(host[index] - expected) <= Float32(1e-5) * max(Float32(1), abs(expected)):
                print("SwiGLU mismatch:", index, host[index], expected)
                raise Error("SwiGLU arithmetic mismatch")
    print("SwiGLU OK: four harts, negative/zero/positive gates")
