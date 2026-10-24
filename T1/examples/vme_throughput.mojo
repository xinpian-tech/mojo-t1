"""Separates resident-operand VME throughput from reload and register-reuse costs."""

from max._plugin.t1_vme import matrix_submit, matrix_finish
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from std.sys import argv, inlined_assembly


def _reload_asm[activation_base: Int, weight_base: Int, compute: Bool]() -> String:
    var assembly = "mv t1, $0\nmv t2, $1\nli t0, 4\n.word 0x8802f057\n"
    for inner in range(4):
        assembly += "vle8.v v" + String(activation_base + inner * 2) + ", (t2)\naddi t2, t2, 4\n"
    assembly += "li t0, 32\n.word 0x8802f057\n"
    for inner in range(4):
        assembly += "vle8.v v" + String(weight_base + inner * 2) + ", (t1)\naddi t1, t1, 32\n"
    if compute:
        assembly += ".word " + String(0xF2000477 | (activation_base << 20) | (weight_base << 15)) + "\n"
    return assembly


@always_inline
def reload[activation_base: Int, weight_base: Int, compute: Bool = True](
    weights: Pointer[Int8, MutAnyOrigin], inputs: Pointer[UInt8, MutAnyOrigin],
):
    comptime assembly = StaticString(materialize[_reload_asm[activation_base, weight_base, compute]()]())
    inlined_assembly[
        assembly, NoneType,
        constraints="r,r,~{t0},~{t1},~{t2},"
        "~{v0},~{v2},~{v4},~{v6},~{v8},~{v10},~{v12},~{v14},"
        "~{v16},~{v18},~{v20},~{v22},~{v24},~{v26},~{v28},~{v30},~{memory}",
    ](weights, inputs)


def throughput(
    output: Pointer[Int32, MutAnyOrigin], weights: Pointer[Int8, MutAnyOrigin],
    inputs: Pointer[UInt8, MutAnyOrigin], iterations: Int32, mode: Int32,
):
    matrix_submit(weights, inputs, 4, 4, 32, 32, 4)
    if mode == 0:
        for iteration in range(1, Int(iterations)):
            inlined_assembly[".word 0xf2880477", NoneType, constraints="~{memory}"]()
    elif mode == 1:
        for iteration in range(1, Int(iterations)):
            reload[8, 16](weights, inputs)
    elif mode == 2:
        for iteration in range(1, Int(iterations)):
            if iteration % 2 == 0:
                reload[0, 8](weights, inputs)
            else:
                reload[16, 24](weights, inputs)
    elif mode == 3:
        for iteration in range(1, Int(iterations)):
            inlined_assembly[
                "li t0, 4\n.word 0x8802f057\nli t0, 32\n.word 0x8802f057\n.word 0xf2880477",
                NoneType, constraints="~{t0},~{memory}",
            ]()
    elif mode == 4:
        for iteration in range(1, Int(iterations)):
            reload[8, 16, False](weights, inputs)
    else:
        for iteration in range(1, Int(iterations)):
            inlined_assembly[
                "li t0, 4\n"
                "vsetvli zero, t0, e8, m2, ta, ma\n"
                "vlseg4e8.v v0, ($1)\n"
                "li t0, 32\n"
                "vsetvli zero, t0, e8, m2, ta, ma\n"
                "vlseg4e8.v v8, ($0)\n"
                "li t0, 4227\n"
                "li t1, 256\n"
                ".word 0x8462f057\n"
                "li t0, 32\n"
                ".word 0x8802f057\n"
                ".word 0xf2040477\n",
                NoneType,
                constraints="r,r,~{t0},~{t1},~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},"
                "~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{memory}",
            ](weights.unsafe_offset(128), inputs.unsafe_offset(16))
    matrix_finish(output.unsafe_offset(thread_idx.x * 128), 4, 32, 32)


def main() raises:
    var arguments = argv()
    if len(arguments) != 4:
        raise Error("usage: t1-vme-throughput MODE(0=resident,1=reload,2=pingpong,3=config,4=loads,5=segmented) HARTS(1..4) ITERATIONS(1..4096)")
    var mode = Int(arguments[1])
    var harts = Int(arguments[2])
    var iterations = Int(arguments[3])
    if mode < 0 or mode > 5 or harts < 1 or harts > 4 or iterations < 1 or iterations > 4096:
        raise Error("invalid throughput configuration")
    var context = DeviceContext(api="t1")
    var output = context.enqueue_create_buffer[DType.int32](4 * 128)
    var weights = context.enqueue_create_buffer[DType.int8](2 * 4 * 32)
    var inputs = context.enqueue_create_buffer[DType.uint8](2 * 4 * 4)
    with output.map_to_host() as host:
        for element in range(4 * 128):
            host[element] = -12345
    with weights.map_to_host() as host:
        for inner in range(4):
            for column in range(32):
                host[inner * 32 + column] = Int8((column + inner) % 7 - 3)
                host[128 + column * 4 + inner] = Int8((column + inner) % 7 - 3)
    with inputs.map_to_host() as host:
        for inner in range(4):
            for row in range(4):
                host[inner * 4 + row] = UInt8(row * 2 + inner + 1)
                host[16 + row * 4 + inner] = UInt8(row * 2 + inner + 1)
    context.enqueue_function[throughput](output, weights, inputs, Int32(iterations), Int32(mode), grid_dim=1, block_dim=harts)
    context.synchronize()
    with output.map_to_host() as host:
        for hart in range(4):
            for row in range(4):
                for column in range(32):
                    var expected = Int32(-12345)
                    if hart < harts:
                        expected = 0
                        for inner in range(4):
                            expected += Int32((row * 2 + inner + 1) * ((column + inner) % 7 - 3) * (1 if mode == 4 else iterations))
                    if host[hart * 128 + row * 32 + column] != expected:
                        raise Error("VME throughput readback mismatch")
    print("VME throughput OK: mode", mode, "harts", harts, "iterations", iterations, "MACs", harts * (1 if mode == 4 else iterations) * 512)
