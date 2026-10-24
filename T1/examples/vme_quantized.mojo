"""Checks the inference matvec, group scales, zero points and partial tiles."""

from max.gpu.host import DeviceContext
from t1llama import _matvec


def matrix(
    outputs: Pointer[Float32, MutAnyOrigin],
    weights: Pointer[Float32, MutAnyOrigin],
    inputs: Pointer[Float32, MutAnyOrigin],
):
    _matvec(outputs, weights, inputs, 140, 64)


def main() raises:
    var context = DeviceContext(api="t1")
    var outputs = context.enqueue_create_buffer[DType.float32](140)
    var weights = context.enqueue_create_buffer[DType.float32](140 * 64)
    var inputs = context.enqueue_create_buffer[DType.float32](64)
    with weights.map_to_host() as host:
        var packed = host.unsafe_ptr().bitcast[Int8]()
        var scales = host.unsafe_ptr().unsafe_offset(140 * 64 // 4)
        var sums = scales.unsafe_offset(280).bitcast[Int32]()
        for group in range(2):
            for column in range(140):
                var total = Int32(0)
                for inner in range(32):
                    var value = Int8((column + inner * 3 + group) % 255 - 127)
                    packed[(group * 32 + inner) * 140 + column] = value
                    total += Int32(value)
                scales[group * 140 + column] = Float32(group + 1) / 16
                sums[group * 140 + column] = total
    with inputs.map_to_host() as host:
        for group in range(2):
            for inner in range(32):
                host[group * 32 + inner] = Float32(127 if inner == 0 else inner * 7 - 127) / Float32(4 << group)
    context.enqueue_function[matrix](outputs, weights, inputs, grid_dim=1, block_dim=4)
    context.synchronize()
    with outputs.map_to_host() as host:
        for column in range(140):
            var expected = Float32(0)
            for group in range(2):
                for inner in range(32):
                    var activation = Float32(127 if inner == 0 else inner * 7 - 127) / Float32(4 << group)
                    var weight = Float32((column + inner * 3 + group) % 255 - 127) * Float32(group + 1) / 16
                    expected += activation * weight
            if host[column] != expected:
                print("Quantized mismatch:", column, host[column], expected)
                raise Error("Quantized VME matrix mismatch")
    print("Quantized VME OK: 4 harts, two groups, partial tiles")
