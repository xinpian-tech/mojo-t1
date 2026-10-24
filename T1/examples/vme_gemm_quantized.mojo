"""Checks quantized token batches, zero rows, row tails and column tails."""

from max.gpu.host import DeviceContext
from t1llama import _matmul, _matmul_prepared, _prepare_activations, _pack_segmented_weights


def matrix(
    outputs: Pointer[Float32, MutAnyOrigin],
    weights: Pointer[Float32, MutAnyOrigin],
    inputs: Pointer[Float32, MutAnyOrigin],
    prepared_outputs: Pointer[Float32, MutAnyOrigin],
    workspace: Pointer[Float32, MutAnyOrigin],
):
    _matmul(outputs, weights, inputs, 5, 140, 64, 67, 143)
    for first in range(0, 5, 4):
        var rows = min(4, 5 - first)
        var source = inputs.unsafe_offset(first * 67)
        _prepare_activations(source, workspace, rows, 64, 67)
        _matmul_prepared(prepared_outputs.unsafe_offset(first * 143), weights, source, rows, 140, 64, 67, 143, workspace)


def activation(row: Int, group: Int, inner: Int) -> Float32:
    if row == 4:
        return 0
    if inner == 2:
        return Float32(0.5 if row % 2 == 0 else -0.5) / Float32(4 << (group + row))
    if inner == 3:
        return Float32(1.5 if row % 2 == 0 else -1.5) / Float32(4 << (group + row))
    if inner == 4:
        return Float32(0.499 if row % 2 == 0 else -0.499) / Float32(4 << (group + row))
    return Float32(127 if inner == 0 else inner * 7 - 127) / Float32(4 << (group + row))


def main() raises:
    var context = DeviceContext(api="t1")
    var outputs = context.enqueue_create_buffer[DType.float32](5 * 143)
    var prepared_outputs = context.enqueue_create_buffer[DType.float32](5 * 143)
    var workspace = context.enqueue_create_buffer[DType.float32](64 * 9 // 8)
    var weights = context.enqueue_create_buffer[DType.float32](140 * 64)
    var inputs = context.enqueue_create_buffer[DType.float32](5 * 67)
    with outputs.map_to_host() as host:
        for element in range(5 * 143):
            host[element] = -999
    with prepared_outputs.map_to_host() as host:
        for element in range(5 * 143):
            host[element] = -999
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
        _pack_segmented_weights(host.unsafe_ptr(), 140, 64)
    with inputs.map_to_host() as host:
        for row in range(5):
            for inner in range(67):
                host[row * 67 + inner] = activation(row, inner // 32, inner % 32) if inner < 64 else Float32(999)
    context.enqueue_function[matrix](outputs, weights, inputs, prepared_outputs, workspace, grid_dim=1, block_dim=4)
    context.synchronize()
    with outputs.map_to_host() as host:
        for row in range(5):
            for column in range(143):
                var expected = Float32(-999)
                if column < 140:
                    expected = 0
                    for group in range(2):
                        for inner in range(32):
                            var weight = Float32((column + inner * 3 + group) % 255 - 127) * Float32(group + 1) / 16
                            var scale = Float32(1) / Float32(4 << (group + row))
                            var value = activation(row, group, inner) / scale
                            var rounded = Int(value + (Float32(0.5) if value >= 0 else Float32(-0.5)))
                            expected += Float32(rounded) * scale * weight
                if host[row * 143 + column] != expected:
                    print("Quantized GEMM mismatch:", row, column, host[row * 143 + column], expected)
                    raise Error("Quantized VME GEMM mismatch")
    with outputs.map_to_host() as reference:
        with prepared_outputs.map_to_host() as actual:
            for element in range(5 * 143):
                if reference[element] != actual[element]:
                    raise Error("Shared quantization differs from scalar reference")
    print("Quantized GEMM OK: shared RVV/scalar packs, distinct scales, zero row, row/column tails, padded strides")
