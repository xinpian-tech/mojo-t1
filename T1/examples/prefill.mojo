"""Checks one batched Llama layer against sequential causal token execution."""

from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.math import cos, exp, log, sin
from std.sys import argv
from t1llama import t1_layer, t1_prefill_layer, F32Ptr, I32Ptr

comptime ROWS = 4
comptime DIM = 64
comptime HIDDEN = 128
comptime HEAD_SIZE = 16
comptime KV_DIM = 32
comptime SEQ_CAP = 6
comptime LAYER_WORDS = 2 * DIM + 2 * DIM * DIM + 2 * KV_DIM * DIM + 3 * DIM * HIDDEN


def sequential[stop_before_activation: Bool = False](hdr: I32Ptr, rope: F32Ptr, x: F32Ptr, weights: F32Ptr, kv: F32Ptr, scratch: F32Ptr, token_count: Int32):
    var start = hdr[0]
    for row in range(Int(token_count)):
        if thread_idx.x == 0:
            hdr[0] = start + Int32(row)
        barrier()
        t1_layer[stop_before_activation](hdr, rope.unsafe_offset(row * HEAD_SIZE), x.unsafe_offset(row * DIM), weights, kv, scratch)
    if thread_idx.x == 0:
        hdr[0] = start
    barrier()


def batched(hdr: I32Ptr, rope: F32Ptr, x: F32Ptr, weights: F32Ptr, kv: F32Ptr, scratch: F32Ptr, token_count: Int32):
    t1_prefill_layer(hdr, rope, x, weights, kv, scratch, token_count)


def pack_matrix[origin: MutOrigin, //](destination: Pointer[Float32, origin], outputs: Int, reduction: Int, salt: Int):
    var packed = destination.bitcast[Int8]()
    var scales = destination.unsafe_offset(outputs * reduction // 4)
    var sums = scales.unsafe_offset(outputs * reduction // 32).bitcast[Int32]()
    for group in range(reduction // 32):
        for column in range(outputs):
            var total = Int32(0)
            for inner in range(32):
                var value = Int8((column * 13 + inner * 7 + group * 11 + salt) % 31 - 15)
                packed[(group * 32 + inner) * outputs + column] = value
                total += Int32(value)
            scales[group * outputs + column] = Float32(1 + (column + group) % 3) / 128
            sums[group * outputs + column] = total


def main() raises:
    var arguments = argv()
    var reference_only = len(arguments) > 1 and String(arguments[1]) == "--reference-only"
    var activation_input_only = len(arguments) > 1 and String(arguments[1]) == "--activation-input-only"
    var select_tokens = reference_only or (len(arguments) > 1 and String(arguments[1]) == "--tokens")
    var reference_tokens = Int(String(arguments[2])) if select_tokens and len(arguments) > 2 else ROWS
    if reference_tokens < 1 or reference_tokens > ROWS:
        raise Error("Reference token count must be in 1..4")
    var context = DeviceContext(api="t1")
    var hdr = context.enqueue_create_buffer[DType.int32](8)
    var rope = context.enqueue_create_buffer[DType.float32](ROWS * HEAD_SIZE)
    var x = context.enqueue_create_buffer[DType.float32](ROWS * DIM)
    var weights = context.enqueue_create_buffer[DType.float32](LAYER_WORDS)
    var kv = context.enqueue_create_buffer[DType.float32](2 * SEQ_CAP * KV_DIM)
    var scratch = context.enqueue_create_buffer[DType.float32](2 * ROWS * (DIM + HIDDEN) + 4 * SEQ_CAP)
    var initial_x = List[Float32](length=ROWS * DIM, fill=0)
    var initial_kv = List[Float32](length=2 * SEQ_CAP * KV_DIM, fill=0)
    for row in range(ROWS):
        for element in range(DIM):
            initial_x[row * DIM + element] = Float32((row * 17 + element * 3) % 43 - 21) / Float32(8 << row)
    for element in range(2 * SEQ_CAP * KV_DIM):
        initial_kv[element] = Float32(element % 23 - 11) / 32
    with hdr.map_to_host() as host:
        host[0] = 1
        host[1] = DIM
        host[2] = HIDDEN
        host[3] = 4
        host[4] = 2
        host[5] = HEAD_SIZE
        host[6] = SEQ_CAP
        host[7] = 0
    with rope.map_to_host() as host:
        for row in range(ROWS):
            for element in range(0, HEAD_SIZE, 2):
                var angle = Float32(row + 1) * exp(-log(Float32(10000)) * Float32(element) / HEAD_SIZE)
                host[row * HEAD_SIZE + element] = cos(angle)
                host[row * HEAD_SIZE + element + 1] = sin(angle)
    with weights.map_to_host() as host:
        var pointer = host.unsafe_ptr()
        for element in range(DIM):
            pointer[element] = Float32(1) + Float32(element % 9) / 64
        var offset = DIM
        pack_matrix(pointer.unsafe_offset(offset), DIM, DIM, 1)
        offset += DIM * DIM
        pack_matrix(pointer.unsafe_offset(offset), KV_DIM, DIM, 2)
        offset += KV_DIM * DIM
        pack_matrix(pointer.unsafe_offset(offset), KV_DIM, DIM, 3)
        offset += KV_DIM * DIM
        pack_matrix(pointer.unsafe_offset(offset), DIM, DIM, 4)
        offset += DIM * DIM
        for element in range(DIM):
            pointer[offset + element] = Float32(1) + Float32(element % 7) / 32
        offset += DIM
        pack_matrix(pointer.unsafe_offset(offset), HIDDEN, DIM, 5)
        offset += HIDDEN * DIM
        pack_matrix(pointer.unsafe_offset(offset), DIM, HIDDEN, 6)
        offset += DIM * HIDDEN
        pack_matrix(pointer.unsafe_offset(offset), HIDDEN, DIM, 7)
    x.enqueue_copy_from(initial_x.unsafe_ptr())
    kv.enqueue_copy_from(initial_kv.unsafe_ptr())
    if activation_input_only:
        context.enqueue_function[sequential[True]](hdr, rope, x, weights, kv, scratch, Int32(1), grid_dim=1, block_dim=4)
    else:
        context.enqueue_function[sequential[False]](hdr, rope, x, weights, kv, scratch, Int32(reference_tokens), grid_dim=1, block_dim=4)
    context.synchronize()
    if reference_only or activation_input_only:
        print("Sequential reference layer complete")
        return
    var expected_x = List[Float32](length=ROWS * DIM, fill=0)
    var expected_kv = List[Float32](length=2 * SEQ_CAP * KV_DIM, fill=0)
    with x.map_to_host() as host:
        for element in range(ROWS * DIM):
            expected_x[element] = host[element]
    with kv.map_to_host() as host:
        for element in range(2 * SEQ_CAP * KV_DIM):
            expected_kv[element] = host[element]
    x.enqueue_copy_from(initial_x.unsafe_ptr())
    kv.enqueue_copy_from(initial_kv.unsafe_ptr())
    context.enqueue_function[batched](hdr, rope, x, weights, kv, scratch, Int32(reference_tokens), grid_dim=1, block_dim=4)
    context.synchronize()
    with x.map_to_host() as host:
        for element in range(ROWS * DIM):
            if host[element] != expected_x[element]:
                print("Prefill residual mismatch:", element, host[element], expected_x[element])
                raise Error("Prefill differs from sequential tokens")
    with kv.map_to_host() as host:
        for element in range(2 * SEQ_CAP * KV_DIM):
            if host[element] != expected_kv[element]:
                print("Prefill KV mismatch:", element, host[element], expected_kv[element])
                raise Error("Prefill KV cache differs from sequential tokens")
    print("Prefill layer OK:", reference_tokens, "tokens, nonzero position, GQA, causal attention, residual and full KV match")
