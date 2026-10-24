"""Checks quantized token batches, zero rows, row tails and column tails."""

from max.gpu.host import DeviceContext
from t1llama import _matmul


def matrix(
    outputs: Pointer[Float32, MutAnyOrigin],
    weights: Pointer[Float32, MutAnyOrigin],
    inputs: Pointer[Float32, MutAnyOrigin],
):
    _matmul(outputs, weights, inputs, 5, 140, 64, 67, 143)


def activation(row: Int, group: Int, inner: Int) -> Float32:
    if row == 4:
        return 0
    return Float32(127 if inner == 0 else inner * 7 - 127) / Float32(4 << (group + row))


def main() raises:
    var context = DeviceContext(api="t1")
    var outputs = context.enqueue_create_buffer[DType.float32](5 * 143)
    var weights = context.enqueue_create_buffer[DType.float32](140 * 64)
    var inputs = context.enqueue_create_buffer[DType.float32](5 * 67)
    with outputs.map_to_host() as host:
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
    with inputs.map_to_host() as host:
        for row in range(5):
            for inner in range(67):
                host[row * 67 + inner] = activation(row, inner // 32, inner % 32) if inner < 64 else Float32(999)
    context.enqueue_function[matrix](outputs, weights, inputs, grid_dim=1, block_dim=4)
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
                            expected += activation(row, group, inner) * weight
                if host[row * 143 + column] != expected:
                    print("Quantized GEMM mismatch:", row, column, host[row * 143 + column], expected)
                    raise Error("Quantized VME GEMM mismatch")
    print("Quantized GEMM OK: distinct scales, zero row, row/column tails, padded strides")
