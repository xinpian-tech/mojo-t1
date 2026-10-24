"""Checks the actual VME matrix instruction on independent hardware harts."""

from max._plugin.t1_vme import matvec_u8_i8
from max.gpu import thread_idx
from max.gpu.host import DeviceContext


def matrix(
    outputs: Pointer[Int32, MutAnyOrigin],
    weights: Pointer[Int8, MutAnyOrigin],
    inputs: Pointer[UInt8, MutAnyOrigin],
):
    var thread = thread_idx.x
    matvec_u8_i8(
        outputs.unsafe_offset(thread * 32), weights.unsafe_offset(thread * 32),
        inputs.unsafe_offset(thread * 8), 8, 32, 128,
    )


def main() raises:
    var context = DeviceContext(api="t1")
    var outputs = context.enqueue_create_buffer[DType.int32](128)
    var weights = context.enqueue_create_buffer[DType.int8](8 * 128)
    var inputs = context.enqueue_create_buffer[DType.uint8](4 * 8)
    with weights.map_to_host() as host:
        for reduction in range(8):
            for column in range(128):
                host[reduction * 128 + column] = Int8((column + reduction * 3) % 31 - 15)
    with inputs.map_to_host() as host:
        for thread in range(4):
            for reduction in range(8):
                host[thread * 8 + reduction] = UInt8(thread * 40 + reduction * 7)
    context.enqueue_function[matrix](outputs, weights, inputs, grid_dim=1, block_dim=4)
    context.synchronize()
    with outputs.map_to_host() as host:
        for column in range(128):
            var expected = Int32(0)
            for reduction in range(8):
                expected += Int32((column // 32 * 40 + reduction * 7) * ((column + reduction * 3) % 31 - 15))
            if host[column] != expected:
                print("VME mismatch:", column, host[column], expected)
                raise Error("VME matrix mismatch")
    print("VME matrix OK: 4 harts, 128 outputs, signed weights")
