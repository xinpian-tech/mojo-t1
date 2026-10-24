"""T1 Zvt integer matrix operations for group-quantized inference."""

from std.sys import inlined_assembly


@always_inline
def quantize_group32[input_origin: MutOrigin, output_origin: MutOrigin, //](
    source: Pointer[Float32, input_origin], destination: Pointer[UInt8, output_origin],
    stride: Int,
) -> Float32:
    """Packs 32 finite FP32 values with symmetric scaling and ties away from zero.

    Requires VLEN >= 1024. Destination has room for 32 strided bytes.
    Division and RTZ conversion preserve the scalar group-32 quantizer.
    """
    var magnitude = inlined_assembly[
        "li t0, 32\n"
        "vsetvli zero, t0, e32, m1, ta, ma\n"
        "vle32.v v8, ($1)\n"
        "vfabs.v v9, v8\n"
        "vmv.s.x v10, zero\n"
        "vfredmax.vs v10, v9, v10\n"
        "vfmv.f.s $0, v10\n",
        Float32,
        constraints="=f,r,~{t0},~{v8},~{v9},~{v10},~{memory}",
    ](source)
    var scale = magnitude / 127 if magnitude > 0 else Float32(1)
    inlined_assembly[
        "li t0, 32\n"
        "vsetvli zero, t0, e32, m1, ta, ma\n"
        "vle32.v v8, ($0)\n"
        "vfdiv.vf v8, v8, $3\n"
        "vfmv.v.f v10, $4\n"
        "vfsgnj.vv v10, v10, v8\n"
        "vfadd.vv v8, v8, v10\n"
        "vfcvt.rtz.x.f.v v12, v8\n"
        "li t1, -127\n"
        "vmax.vx v12, v12, t1\n"
        "li t1, 127\n"
        "vmin.vx v12, v12, t1\n"
        "li t1, 128\n"
        "vadd.vx v12, v12, t1\n"
        "vsetvli zero, t0, e16, m1, ta, ma\n"
        "vnsrl.wi v10, v12, 0\n"
        "vsetvli zero, t0, e8, m1, ta, ma\n"
        "vnsrl.wi v8, v10, 0\n"
        "vsse8.v v8, ($1), $2\n",
        NoneType,
        constraints="r,r,r,f,f,~{t0},~{t1},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{memory}",
    ](source, destination, stride, scale, Float32(0.5))
    return scale


@always_inline
def matrix_u8_i8[
    output_origin: MutOrigin, weight_origin: MutOrigin, input_origin: MutOrigin, //
](
    output: Pointer[Int32, output_origin],
    weights: Pointer[Int8, weight_origin],
    inputs: Pointer[UInt8, input_origin],
    reduction: Int,
    rows: Int,
    columns: Int,
    weight_stride: Int,
    input_stride: Int,
    output_stride: Int,
):
    """Computes C[M,N] = A[K,M].T @ B[K,N] in tile 4.

    Rows and columns must be in 1..32, and reduction a positive multiple
    of four. Input and weight strides are bytes between successive K rows;
    output_stride counts Int32 elements between output rows. Output must
    be naturally aligned. Only the requested rectangle is accessed.
    """
    matrix_submit(weights, inputs, reduction, rows, columns, weight_stride, input_stride)
    matrix_finish(output, rows, columns, output_stride)


@always_inline
def matrix_submit[weight_origin: MutOrigin, input_origin: MutOrigin, //](
    weights: Pointer[Int8, weight_origin], inputs: Pointer[UInt8, input_origin],
    reduction: Int, rows: Int, columns: Int, weight_stride: Int, input_stride: Int,
):
    """Submits tile 4; finish it before another tile-4 computation.

    Uses the same shape/layout contract as matrix_u8_i8. Independent RVV
    work may follow, but must not overwrite v8/v10/v12/v14/v16/v18/v20/v22.
    """
    inlined_assembly[
        "mv t2, $0\n"
        "mv t3, $1\n"
        "mv t4, $2\n"
        "mv t5, $5\n"
        "mv t6, $6\n"
        "slli t0, $3, 10\n"
        "ori t0, t0, 131\n"
        "li t1, 256\n"
        ".word 0x8462f057\n"
        "mv t0, $4\n"
        ".word 0x8802f057\n"
        ".word 0x43e06457\n"
        "1:\n"
        "mv t0, $3\n"
        ".word 0x8802f057\n"
        "vle8.v v8, (t3)\n"
        "add t3, t3, t6\n"
        "vle8.v v10, (t3)\n"
        "add t3, t3, t6\n"
        "vle8.v v12, (t3)\n"
        "add t3, t3, t6\n"
        "vle8.v v14, (t3)\n"
        "add t3, t3, t6\n"
        "mv t0, $4\n"
        ".word 0x8802f057\n"
        "vle8.v v16, (t2)\n"
        "add t2, t2, t5\n"
        "vle8.v v18, (t2)\n"
        "add t2, t2, t5\n"
        "vle8.v v20, (t2)\n"
        "add t2, t2, t5\n"
        "vle8.v v22, (t2)\n"
        "add t2, t2, t5\n"
        ".word 0xf2880477\n"
        "addi t4, t4, -4\n"
        "bnez t4, 1b\n",
        NoneType,
        constraints="r,r,r,r,r,r,r,~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},"
        "~{v8},~{v10},~{v12},~{v14},~{v16},~{v18},~{v20},~{v22},~{memory}",
    ](weights, inputs, reduction, rows, columns, weight_stride, input_stride)


@always_inline
def matrix_finish[output_origin: MutOrigin, //](
    output: Pointer[Int32, output_origin], rows: Int, columns: Int, output_stride: Int,
):
    """Restores matrix configuration and stores each submitted output row.

    The final fence orders all result stores before subsequent memory work.
    Rows, columns and element stride follow matrix_u8_i8's contract.
    """
    for row in range(rows):
        inlined_assembly[
            "li t0, 1155\n"
            "li t1, 256\n"
            ".word 0x8462f057\n"
            "mv t0, $1\n"
            ".word 0x8802f057\n"
            "mv a0, $0\n"
            "li a1, 0x20000000\n"
            "or a1, a1, $2\n"
            ".word 0x52b57027\n",
            NoneType,
            constraints="r,r,r,~{t0},~{t1},~{a0},~{a1},~{memory}",
        ](output.unsafe_offset(row * output_stride), columns, row)
    inlined_assembly["fence iorw, iorw", NoneType, constraints="~{memory}"]()


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
    Inputs must be four-byte aligned.
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
        "lw t0, 0(t3)\n"
        "vle8.v v16, (t2)\n"
        "add t2, t2, t5\n"
        "vle8.v v18, (t2)\n"
        "add t2, t2, t5\n"
        "vle8.v v20, (t2)\n"
        "add t2, t2, t5\n"
        "vle8.v v22, (t2)\n"
        "add t2, t2, t5\n"
        "vmv.v.x v8, t0\n"
        "srli t0, t0, 8\n"
        "vmv.v.x v10, t0\n"
        "srli t0, t0, 8\n"
        "vmv.v.x v12, t0\n"
        "srli t0, t0, 8\n"
        "vmv.v.x v14, t0\n"
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
def accumulate_scaled_rows[
    output_origin: MutOrigin, accumulator_origin: MutOrigin,
    sum_origin: MutOrigin, scale_origin: MutOrigin, activation_origin: MutOrigin, //
](
    output: Pointer[Float32, output_origin],
    accumulator: Pointer[Int32, accumulator_origin],
    sums: Pointer[Int32, sum_origin],
    scales: Pointer[Float32, scale_origin],
    activation_scales: Pointer[Float32, activation_origin],
    count: Int, rows: Int, output_stride: Int, accumulator_stride: Int,
):
    """Scales tile rows while loading their shared weight metadata only once.

    Count is in 1..32 and rows is positive. Strides count 32-bit elements.
    Uses only v24-v27, independently of
    matrix operand registers, and preserves the single-row arithmetic order.
    """
    inlined_assembly[
        "mv t0, $0\n"
        "mv t1, $1\n"
        "mv t2, $4\n"
        "mv t3, $6\n"
        "slli t4, $7, 2\n"
        "slli t5, $8, 2\n"
        "vsetvli zero, $5, e32, m1, ta, ma\n"
        "vle32.v v25, ($2)\n"
        "vsll.vi v25, v25, 7\n"
        "vle32.v v26, ($3)\n"
        "1:\n"
        "flw ft0, 0(t2)\n"
        "vle32.v v24, (t1)\n"
        "vsub.vv v24, v24, v25\n"
        "vfcvt.f.x.v v24, v24\n"
        "vfmul.vf v24, v24, ft0\n"
        "vfmul.vv v24, v24, v26\n"
        "vle32.v v27, (t0)\n"
        "vfadd.vv v24, v24, v27\n"
        "vse32.v v24, (t0)\n"
        "addi t2, t2, 4\n"
        "add t0, t0, t4\n"
        "add t1, t1, t5\n"
        "addi t3, t3, -1\n"
        "bnez t3, 1b\n",
        NoneType,
        constraints="r,r,r,r,r,r,r,r,r,~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{ft0},"
        "~{v24},~{v25},~{v26},~{v27},~{memory}",
    ](output, accumulator, sums, scales, activation_scales, count, rows, output_stride, accumulator_stride)


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
