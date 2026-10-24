"""T1 Zvt integer matrix operations for group-quantized inference."""

from std.sys import inlined_assembly


@always_inline
def matvec_u8_i8[
    output_origin: MutOrigin, weight_origin: MutOrigin, input_origin: MutOrigin, //
](
    output: Pointer[Int32, output_origin],
    weights: Pointer[Int8, weight_origin],
    inputs: Pointer[UInt8, input_origin],
    reduction: Int,
    outputs: Int,
    weight_stride: Int,
):
    """Computes up to 32 outputs using VME u8 × i8 → i32 accumulation.

    Weights are reduction-major with `weight_stride` bytes per column.
    The reduction length must be a positive multiple of four; the caller
    supplies one to 32 outputs. Each hart owns its tile-4 accumulator.
    """
    matvec_submit(weights, inputs, reduction, outputs, weight_stride)
    matvec_finish(output, outputs)


@always_inline
def matvec_submit[
    weight_origin: MutOrigin, input_origin: MutOrigin, //
](
    weights: Pointer[Int8, weight_origin],
    inputs: Pointer[UInt8, input_origin],
    reduction: Int,
    outputs: Int,
    weight_stride: Int,
):
    """Issues tile-4 computation without waiting for its result in memory.

    Call matvec_finish before reusing tile 4. Independent vector work may
    run between submission and finish; it must not overwrite the operands
    in v8/v10/v12/v14 or v16/v18/v20/v22 before their reads complete.
    """
    inlined_assembly[
        "mv t2, $0\n"
        "mv t3, $1\n"
        "mv t4, $2\n"
        "mv t5, $4\n"
        "vsetvli zero, $3, e8, m1, ta, ma\n"
        "li t0, 1155\n"
        "li t1, 256\n"
        ".word 0x8462f057\n"
        "mv t0, $3\n"
        ".word 0x8802f057\n"
        ".word 0x43e06457\n"
        "1:\n"
        "lbu t0, 0(t3)\n"
        "vmv.v.x v8, t0\n"
        "lbu t0, 1(t3)\n"
        "vmv.v.x v10, t0\n"
        "lbu t0, 2(t3)\n"
        "vmv.v.x v12, t0\n"
        "lbu t0, 3(t3)\n"
        "vmv.v.x v14, t0\n"
        "vle8.v v16, (t2)\n"
        "add t2, t2, t5\n"
        "vle8.v v18, (t2)\n"
        "add t2, t2, t5\n"
        "vle8.v v20, (t2)\n"
        "add t2, t2, t5\n"
        "vle8.v v22, (t2)\n"
        "add t2, t2, t5\n"
        ".word 0xf2880477\n"
        "addi t3, t3, 4\n"
        "addi t4, t4, -4\n"
        "bnez t4, 1b\n",
        NoneType,
        constraints="r,r,r,r,r,~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},"
        "~{v8},~{v10},~{v12},~{v14},~{v16},~{v18},~{v20},~{v22},~{memory}",
    ](weights, inputs, reduction, outputs, weight_stride)


@always_inline
def matvec_finish[output_origin: MutOrigin, //](
    output: Pointer[Int32, output_origin], outputs: Int,
):
    """Stores the submitted tile-4 result and orders its memory visibility."""
    inlined_assembly[
        "vsetvli zero, $1, e8, m1, ta, ma\n"
        "li t0, 1155\n"
        "li t1, 256\n"
        ".word 0x8462f057\n"
        "mv t0, $1\n"
        ".word 0x8802f057\n"
        "mv a0, $0\n"
        "li a1, 0x20000000\n"
        ".word 0x52b57027\n"
        "fence iorw, iorw\n",
        NoneType,
        constraints="r,r,~{t0},~{t1},~{a0},~{a1},~{memory}",
    ](output, outputs)


@always_inline
def accumulate_scaled[
    output_origin: MutOrigin, accumulator_origin: MutOrigin,
    sum_origin: MutOrigin, scale_origin: MutOrigin, //
](
    output: Pointer[Float32, output_origin],
    accumulator: Pointer[Int32, accumulator_origin],
    sums: Pointer[Int32, sum_origin],
    scales: Pointer[Float32, scale_origin],
    activation_scale: Float32, count: Int,
):
    """Applies unsigned zero-point correction and scales on independent RVV registers."""
    inlined_assembly[
        "vsetvli zero, $5, e32, m1, ta, ma\n"
        "vle32.v v24, ($1)\n"
        "vle32.v v25, ($2)\n"
        "vsll.vi v25, v25, 7\n"
        "vsub.vv v24, v24, v25\n"
        "vfcvt.f.x.v v24, v24\n"
        "vfmul.vf v24, v24, $4\n"
        "vle32.v v26, ($3)\n"
        "vfmul.vv v24, v24, v26\n"
        "vle32.v v27, ($0)\n"
        "vfadd.vv v24, v24, v27\n"
        "vse32.v v24, ($0)\n",
        NoneType,
        constraints="r,r,r,r,f,r,~{v24},~{v25},~{v26},~{v27},~{memory}",
    ](output, accumulator, sums, scales, activation_scale, count)
