"""Checks real multi-row VME multiplication against scalar integer arithmetic."""

from max._plugin.t1_vme import matrix_u8_i8
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from std.sys.defines import get_defined_int

comptime BATCH = get_defined_int["T1_GEMM_ROWS", 4]()


def matrix(
    outputs: Pointer[Int32, MutAnyOrigin],
    weights: Pointer[Int8, MutAnyOrigin],
    inputs: Pointer[UInt8, MutAnyOrigin],
):
    var first = thread_idx.x * 35
    var column = first
    while column < first + 35:
        var count = min(32, first + 35 - column)
        for row in range(0, 4, BATCH):
            matrix_u8_i8(
                outputs.unsafe_offset(row * 140 + column), weights.unsafe_offset(column),
                inputs.unsafe_offset(row), 8, min(BATCH, 4 - row), count, 140, 4, 140,
            )
        column += count


def main() raises:
    var context = DeviceContext(api="t1")
    var outputs = context.enqueue_create_buffer[DType.int32](4 * 140)
    var weights = context.enqueue_create_buffer[DType.int8](8 * 140)
    var inputs = context.enqueue_create_buffer[DType.uint8](8 * 4)
    with weights.map_to_host() as host:
        for reduction in range(8):
            for column in range(140):
                host[reduction * 140 + column] = Int8((column + reduction * 3) % 31 - 15)
    with inputs.map_to_host() as host:
        for reduction in range(8):
            for row in range(4):
                host[reduction * 4 + row] = UInt8(row * 40 + reduction * 7)
    context.enqueue_function[matrix](outputs, weights, inputs, grid_dim=1, block_dim=4)
    context.synchronize()
    with outputs.map_to_host() as host:
        for row in range(4):
            for column in range(140):
                var expected = Int32(0)
                for reduction in range(8):
                    expected += Int32((row * 40 + reduction * 7) * ((column + reduction * 3) % 31 - 15))
                if host[row * 140 + column] != expected:
                    print("GEMM mismatch:", row, column, host[row * 140 + column], expected)
                    raise Error("VME GEMM mismatch")
    print("VME GEMM OK: 4 distinct rows, 4 harts, partial column tiles; batch", BATCH)
