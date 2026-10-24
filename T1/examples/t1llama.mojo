# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Xinpian Tech. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# Llama-architecture inference (llama2.c .bin format) written entirely in
# Mojo, with every transformer computation executed on the T1 RVV
# accelerator through the standard DeviceContext API.
#
# Works for any llama2.c legacy-format model; validated against llama2.c
# run.c (greedy) with:
#   - karpathy/tinyllamas stories15M
#   - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (converted with a numpy exporter)
#
# Weights are streamed from disk one layer per launch (the device window
# cannot hold 1.1B fp32 parameters at once); the residual stream, the KV
# cache, and the logits live in device buffers across launches.
#
#   mojo build --target-accelerator=t1 t1llama.mojo -o t1llama
#   ./t1llama model.bin tokenizer.bin "prompt" 12
# ===----------------------------------------------------------------------=== #

from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu import thread_idx, block_dim
from max.gpu.sync import barrier
from max._plugin.t1_vme import matvec_submit, matvec_finish, accumulate_scaled, matrix_submit, matrix_finish, quantize_group32
from std.memory import unsafe_stack_allocation
from std.collections import Dict
from std.io.file import FileHandle
from std.math import cos, exp, log, sin, sqrt
from std.sys import argv, inlined_assembly
from std.sys.defines import get_defined_int
from std.sys.info import simd_width_of
from std.time import perf_counter_ns

comptime F32Ptr = Pointer[Float32, MutAnyOrigin]
comptime I32Ptr = Pointer[Int32, MutAnyOrigin]


# ===----------------------------------------------------------------------=== #
# Device kernels
# ===----------------------------------------------------------------------=== #


# ===----------------------------------------------------------------------=== #
# Templated RVV kernels.
#
# Every hand-written vector helper below is generated at compile time from
# SEW / LMUL parameters (`vl` follows as VLMAX of the chosen shape, with any
# remainder strip-mined by `vsetvli`), so shapes can be swept and evaluated
# without touching the assembly:
#
#   comptime MV_LMUL = 4   # matvec chunk register-group width
#   comptime EW_LMUL = 8   # elementwise (rmsnorm/residual) strip width
#
# The matvec generator derives its whole register allocation from LMUL:
# w buffer v28, x buffer v[L], double-buffered accumulator pairs A/B at
# v[2L]..v[5L], reduce stubs at v[6L].. — so L in {1,2,4}.  No kernel ever
# writes v0: it is the RVV mask register, and on this machine a v0 write
# serializes every dependent consumer (T1 issue #180) — dependent
# accumulation measures II 35 on v8/v24 vs 89-140 on v0.
# ===----------------------------------------------------------------------=== #

comptime SEW = 32
# Must match the hardware VLEN: the generated `vsetvli`s request exact
# VLMAX chunk lengths and assume they are granted in full.
comptime VLEN = get_defined_int["T1_VLEN", 1024]()
comptime THREADS = get_defined_int["T1_THREADS", 4]()
comptime USE_VME = get_defined_int["T1_VME", 1]() != 0
comptime PREFILL_GEMM = get_defined_int["T1_PREFILL_GEMM", 1]() != 0
comptime PREFILL_INTERLEAVE = get_defined_int["T1_PREFILL_INTERLEAVE", 0]() != 0
comptime PREFILL_BATCH = get_defined_int["T1_PREFILL_BATCH", 4]()
comptime MV_LMUL = get_defined_int["T1_MV_LMUL", 2]()
comptime MV_TRANSPOSED = get_defined_int["T1_MV_T", 1]() != 0
# Columns per scalar-prefetch batch in the transposed matvec (<= 8).
comptime MV_BATCH = get_defined_int["T1_MV_BATCH", 8]()
comptime EW_LMUL = get_defined_int["T1_EW_LMUL", 8]()

comptime _VCLOB2 = (
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},"
    "~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},"
    "~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},"
    "~{v31},~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},~{a4},~{a5},~{s2},"
    "~{s3},~{s4},~{s5},~{s6},~{s7},~{ft0},~{ft1},~{ft2},~{ft3},~{ft4},"
    "~{ft5},~{ft6},~{ft7},~{memory}"
)

comptime _VCLOB = (
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},"
    "~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},"
    "~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},"
    "~{v31},~{t0},~{t1},~{t2},~{t3},~{t4},~{t5},~{t6},~{ft0},~{memory}"
)


def _vlmax(sew: Int, lmul: Int) -> Int:
    return VLEN * lmul // sew


def _vle[sew: Int]() -> String:
    return "vle" + String(sew) + ".v"


def _vse[sew: Int]() -> String:
    return "vse" + String(sew) + ".v"


def _cfg(sew: Int, lmul: Int, vl: Int) -> String:
    """`vsetvli` to a fixed vl of shape e{sew}/m{lmul} (through t5)."""
    return (
        "li t5, " + String(vl) + "\n"
        + "vsetvli zero, t5, e" + String(sew) + ", m" + String(lmul)
        + ", ta, ma\n"
    )


def _mv_compute_pair[
    sew: Int, lmul: Int
](acc0: Int, acc1: Int, label: Int) -> String:
    """One 2-row pair of the matvec: zero-init accumulators, then a single
    vsetvli-strip-mined `tu` loop — correct for any n (including n smaller
    than one chunk, which matters when VLEN is raised)."""
    comptime chunk = _vlmax(sew, lmul)
    var w = "v28"
    var x = "v" + String(lmul)
    var a0 = "v" + String(acc0)
    var a1 = "v" + String(acc1)
    return (
        "mv t2, s4\n"
        + "mv t3, s3\n"
        + "add t4, t3, s6\n"
        + "add s3, t4, s6\n"
        + "mv t1, $2\n"
        + _cfg(sew, lmul, chunk)
        + "vmv.v.i " + a0 + ", 0\n"
        + "vmv.v.i " + a1 + ", 0\n"
        + String(label) + ":\n"
        + "vsetvli t0, t1, e" + String(sew) + ", m" + String(lmul)
        + ", tu, ma\n"
        + _vle[sew]() + " " + x + ", (t2)\n"
        + _vle[sew]() + " " + w + ", (t3)\n"
        + "vfmacc.vv " + a0 + ", " + x + ", " + w + "\n"
        + _vle[sew]() + " " + w + ", (t4)\n"
        + "vfmacc.vv " + a1 + ", " + x + ", " + w + "\n"
        + "slli t0, t0, " + String(2 if sew == 32 else 1) + "\n"
        + "add t2, t2, t0\n"
        + "add t3, t3, t0\n"
        + "add t4, t4, t0\n"
        + "srli t0, t0, " + String(2 if sew == 32 else 1) + "\n"
        + "sub t1, t1, t0\n"
        + "bnez t1, " + String(label) + "b\n"
    )


def _mv_fold_pair[sew: Int, lmul: Int](acc0: Int, acc1: Int) -> String:
    """log2(LMUL) lane-parallel vfadd stages folding both accumulators of a
    pair to one m1 register each."""
    var body = String("")
    var l = lmul // 2
    while l >= 1:
        body += _cfg(sew, l, _vlmax(sew, l))
        body += (
            "vfadd.vv v" + String(acc0) + ", v" + String(acc0)
            + ", v" + String(acc0 + l) + "\n"
        )
        body += (
            "vfadd.vv v" + String(acc1) + ", v" + String(acc1)
            + ", v" + String(acc1 + l) + "\n"
        )
        l //= 2
    return body


def _mv_reduce_pair[
    sew: Int, lmul: Int
](acc0: Int, acc1: Int, dest: Int) -> String:
    """Both unordered reduces of a folded pair, issued back-to-back so they
    run on the reduce unit while the next pair's loads stream."""
    var body = _cfg(sew, 1, _vlmax(sew, 1))
    body += (
        "vfredusum.vs v" + String(dest) + ", v" + String(acc0)
        + ", v" + String(dest + 2) + "\n"
    )
    body += (
        "vfredusum.vs v" + String(dest + 1) + ", v" + String(acc1)
        + ", v" + String(dest + 2) + "\n"
    )
    return body


def _mv_store_pair[sew: Int](dest: Int) -> String:
    return (
        "vsetivli zero, 2, e" + String(sew) + ", m1, tu, ma\n"
        + "vslideup.vi v" + String(dest) + ", v" + String(dest + 1) + ", 1\n"
        + _vse[sew]() + " v" + String(dest) + ", (s5)\n"
        + "addi s5, s5, " + String(2 * sew // 8) + "\n"
    )


def _mv_asm[sew: Int, lmul: Int]() -> String:
    """The software-pipelined matvec: a pair's fold and both vfredusum are
    issued immediately before the next pair's load stream, so the (slow,
    non-pipelined) reduce unit crunches while the LSU streams.  Register
    plan derives from LMUL: w v28 (v0 is the mask register — writing it
    serializes dependent consumers, issue #180), x v[L], pair A v[2L]/v[3L], pair B
    v[4L]/v[5L], reduce stubs v[6L]..v[6L+2] — 6L+3 regs, LMUL in {1,2,4}."""
    comptime a0 = 2 * lmul
    comptime a1 = 3 * lmul
    comptime b0 = 4 * lmul
    comptime b1 = 5 * lmul
    comptime dst = 6 * lmul
    var body = (
        "mv s2, $3\n"
        + "mv s3, $0\n"
        + "mv s4, $1\n"
        + "mv s5, $4\n"
        + "slli s6, $2, " + String(2 if sew == 32 else 1) + "\n"
        + _cfg(sew, 1, 1)
        + "vmv.s.x v" + String(dst + 2) + ", zero\n"
    )
    body += _mv_compute_pair[sew, lmul](a0, a1, 1)
    body += "addi s2, s2, -2\n"
    body += "4:\n"
    body += _mv_fold_pair[sew, lmul](a0, a1)
    body += _mv_reduce_pair[sew, lmul](a0, a1, dst)
    body += _mv_compute_pair[sew, lmul](b0, b1, 5)
    body += _mv_store_pair[sew](dst)
    body += "addi s2, s2, -2\n"
    body += "beqz s2, 11f\n"
    body += _mv_fold_pair[sew, lmul](b0, b1)
    body += _mv_reduce_pair[sew, lmul](b0, b1, dst)
    body += _mv_compute_pair[sew, lmul](a0, a1, 7)
    body += _mv_store_pair[sew](dst)
    body += "addi s2, s2, -2\n"
    body += "bnez s2, 4b\n"
    body += _mv_fold_pair[sew, lmul](a0, a1)
    body += _mv_reduce_pair[sew, lmul](a0, a1, dst)
    body += _mv_store_pair[sew](dst)
    body += "j 99f\n"
    body += "11:\n"
    body += _mv_fold_pair[sew, lmul](b0, b1)
    body += _mv_reduce_pair[sew, lmul](b0, b1, dst)
    body += _mv_store_pair[sew](dst)
    body += "99:\n"
    return body


def _mv_t_asm[sew: Int, lmul: Int, batch: Int]() -> String:
    """Reduction-free matvec over an offline-TRANSPOSED weight matrix.

    Register plan: single accumulator chain v8, double-buffered column
    loads v16/v24.  v0 is never written — it is the RVV mask register,
    and writing it pins the instruction to the one shared sequencer slot
    (T1 issue #180).  Dependent arithmetic on ordinary registers chains
    element-wise (II 35, same as independent streams), so one
    accumulator suffices.

    The scalars are the interesting part.  Each column needs x[j] in an
    f-register for `vfmacc.vf`, and Rocket issues in order: a `flw`
    immediately before its consumer costs ~45 cycles of stalled issue
    even when the address is already resident (measured: 61.4 cy/column
    with no scalar load, 104-107 with one per column, at vl=288/m8).
    Hoisting `batch` independent `flw`s to the top of the batch drops
    that to 71.4 — the loads overlap each other and the vector stream
    instead of serializing against it.  It also subsumes the #178
    stale-`fs1` workaround: every scalar is >= `batch` instructions away
    from its use.

    Contract: n divisible by `batch` (2 <= batch <= 8); the caller's
    dimensions (288, 768, 2048, 5632, ...) all are.
    $0=wt, $1=x, $2=n (columns), $3=m (outputs), $4=out."""
    comptime shift = String(2 if sew == 32 else 1)
    comptime esz = sew // 8
    var loads = String("")
    var macs = String("")
    for i in range(batch):
        loads += "flw ft" + String(i) + ", " + String(i * esz) + "(t2)\n"
        var buf = "v16" if i % 2 == 0 else "v24"
        macs += (
            _vle[sew]() + " " + buf + ", (t3)\n"
            + "add t3, t3, s6\n"
            + "vfmacc.vf v8, ft" + String(i) + ", " + buf + "\n"
        )
    return (
        "mv s2, $3\n"
        + "mv s5, $4\n"
        + "mv s3, $0\n"
        + "slli s6, $5, " + shift + "\n"
        + "1:\n"
        + "vsetvli t0, s2, e" + String(sew) + ", m" + String(lmul)
        + ", ta, ma\n"
        + "vmv.v.i v8, 0\n"
        + "mv t1, $2\n"
        + "mv t2, $1\n"
        + "mv t3, s3\n"
        + "2:\n"
        + loads
        + "addi t2, t2, " + String(batch * esz) + "\n"
        + macs
        + "addi t1, t1, -" + String(batch) + "\n"
        + "bnez t1, 2b\n"
        + _vse[sew]() + " v8, (s5)\n"
        + "slli t0, t0, " + shift + "\n"
        + "add s5, s5, t0\n"
        + "add s3, s3, t0\n"
        + "srli t0, t0, " + shift + "\n"
        + "sub s2, s2, t0\n"
        + "bnez s2, 1b\n"
    )


@always_inline
def _mv_transposed(wt: F32Ptr, x: F32Ptr, n: Int, m: Int, out_ptr: F32Ptr, stride: Int):
    comptime asm = StaticString(materialize[_mv_t_asm[SEW, 8, MV_BATCH]()]())
    inlined_assembly[
        asm,
        NoneType,
        constraints="r,r,r,r,r,r," + _VCLOB2,
    ](wt, x, n, m, out_ptr, stride)


@always_inline
def _mv(w: F32Ptr, x: F32Ptr, n: Int, rows: Int, out_ptr: F32Ptr):
    """Software-pipelined matvec at the file-level MV_LMUL/SEW shape.
    Contract: n >= VLMAX(MV_LMUL), rows % 4 == 0, rows >= 4."""
    comptime assert MV_LMUL in (1, 2, 4), "matvec register budget is 6L+3"
    comptime asm = StaticString(materialize[_mv_asm[SEW, MV_LMUL]()]())
    inlined_assembly[
        asm,
        NoneType,
        constraints="r,r,r,r,r," + _VCLOB2,
    ](w, x, n, rows, out_ptr)


def _mv1_asm[sew: Int, lmul: Int]() -> String:
    comptime chunk = _vlmax(sew, lmul)
    comptime cb = String(chunk * sew // 8)
    var body = (
        "mv t1, $2\n"
        + "mv t2, $1\n"
        + "mv t3, $0\n"
        + "li t0, " + String(chunk) + "\n"
        + "vsetvli zero, t0, e" + String(sew) + ", m" + String(lmul)
        + ", ta, ma\n"
        + "vmv.s.x v24, zero\n"
        + _vle[sew]() + " v" + String(lmul) + ", (t2)\n"
        + _vle[sew]() + " v" + String(2 * lmul) + ", (t3)\n"
        + "vfmul.vv v" + String(3 * lmul) + ", v" + String(lmul) + ", v"
        + String(2 * lmul) + "\n"
        + "addi t1, t1, -" + String(chunk) + "\n"
        + "addi t2, t2, " + cb + "\n"
        + "addi t3, t3, " + cb + "\n"
        + "bltu t1, t0, 2f\n"
        + "1:\n"
        + _vle[sew]() + " v" + String(lmul) + ", (t2)\n"
        + _vle[sew]() + " v" + String(2 * lmul) + ", (t3)\n"
        + "vfmacc.vv v" + String(3 * lmul) + ", v" + String(lmul) + ", v"
        + String(2 * lmul) + "\n"
        + "addi t1, t1, -" + String(chunk) + "\n"
        + "addi t2, t2, " + cb + "\n"
        + "addi t3, t3, " + cb + "\n"
        + "bgeu t1, t0, 1b\n"
        + "2:\n"
        + "beqz t1, 3f\n"
        + "vsetvli t0, t1, e" + String(sew) + ", m" + String(lmul)
        + ", tu, ma\n"
        + _vle[sew]() + " v" + String(lmul) + ", (t2)\n"
        + _vle[sew]() + " v" + String(2 * lmul) + ", (t3)\n"
        + "vfmacc.vv v" + String(3 * lmul) + ", v" + String(lmul) + ", v"
        + String(2 * lmul) + "\n"
        + "3:\n"
    )
    var l = lmul // 2
    var acc = 3 * lmul
    while l >= 1:
        body += _cfg(sew, l, _vlmax(sew, l))
        body += (
            "vfadd.vv v" + String(acc) + ", v" + String(acc) + ", v"
            + String(acc + l) + "\n"
        )
        l //= 2
    body += _cfg(sew, 1, _vlmax(sew, 1))
    body += "vfredusum.vs v25, v" + String(acc) + ", v24\n"
    body += "vfmv.f.s ft0, v25\n"
    body += "fsw ft0, 0($3)\n"
    return body


@always_inline
def _mv1(w: F32Ptr, x: F32Ptr, n: Int, out_ptr: F32Ptr):
    """Single-row matvec remainder; same n >= VLMAX contract as `_mv`."""
    comptime asm = StaticString(materialize[_mv1_asm[SEW, MV_LMUL]()]())
    inlined_assembly[
        asm,
        NoneType,
        constraints="r,r,r,r," + _VCLOB,
    ](w, x, n, out_ptr)


@always_inline
def _matvec_vme(out_ptr: F32Ptr, weights: F32Ptr, inputs: F32Ptr, outputs: Int, reduction: Int):
    """Runs group-32 quantized matrix multiplication on each hart's VME tile."""
    var quantized = unsafe_stack_allocation[8, DType.uint32]().bitcast[UInt8]()
    var accumulator = unsafe_stack_allocation[64, DType.int32]()
    var packed = weights.bitcast[Int8]()
    var scales = weights.unsafe_offset(outputs * reduction // 4)
    var sums = scales.unsafe_offset(outputs * reduction // 32).bitcast[Int32]()
    var first = outputs * thread_idx.x // block_dim.x
    var last = outputs * (thread_idx.x + 1) // block_dim.x
    for column in range(first, last):
        out_ptr[column] = 0
    for group in range(reduction // 32):
        var magnitude = Float32(0)
        for inner in range(32):
            magnitude = max(magnitude, abs(inputs[group * 32 + inner]))
        var scale = magnitude / 127 if magnitude > 0 else Float32(1)
        for inner in range(32):
            var value = inputs[group * 32 + inner] / scale
            var rounded = Int(value + (Float32(0.5) if value >= 0 else Float32(-0.5)))
            quantized[inner] = UInt8(max(-127, min(127, rounded)) + 128)
        var column = first
        var previous = first
        var previous_count = 0
        var slot = 0
        while column < last:
            var count = min(32, last - column)
            matvec_submit(
                packed.unsafe_offset(group * 32 * outputs + column),
                quantized, 32, count, outputs,
            )
            if previous_count:
                var metadata = group * outputs + previous
                accumulate_scaled(
                    out_ptr.unsafe_offset(previous), accumulator.unsafe_offset((1 - slot) * 32),
                    sums.unsafe_offset(metadata), scales.unsafe_offset(metadata), scale, previous_count,
                )
            matvec_finish(accumulator.unsafe_offset(slot * 32), count)
            previous = column
            previous_count = count
            slot = 1 - slot
            column += count
        if previous_count:
            var metadata = group * outputs + previous
            accumulate_scaled(
                out_ptr.unsafe_offset(previous), accumulator.unsafe_offset((1 - slot) * 32),
                sums.unsafe_offset(metadata), scales.unsafe_offset(metadata), scale, previous_count,
            )


@always_inline
def _matmul(
    out_ptr: F32Ptr, weights: F32Ptr, inputs: F32Ptr,
    rows: Int, outputs: Int, reduction: Int, input_stride: Int, output_stride: Int,
):
    """Projects independent token rows, sharing group-32 weights within each tile."""
    _matmul_prepared[False](out_ptr, weights, inputs, rows, outputs, reduction, input_stride, output_stride, inputs)


@always_inline
def _prepare_activations(inputs: F32Ptr, workspace: F32Ptr, rows: Int, reduction: Int, input_stride: Int):
    """Cooperatively quantizes up to four rows into shared K-major group tiles."""
    comptime if USE_VME and PREFILL_GEMM:
        var groups = reduction // 32
        var packed = workspace.bitcast[UInt8]()
        var scales = workspace.unsafe_offset(reduction)
        for task in range(thread_idx.x, groups * rows, block_dim.x):
            var group = task // rows
            var row = task % rows
            scales[group * 4 + row] = quantize_group32(
                inputs.unsafe_offset(row * input_stride + group * 32),
                packed.unsafe_offset(group * 128 + row), 4,
            )
        barrier()


@always_inline
def _matmul_prepared[prepared: Bool = True](
    out_ptr: F32Ptr, weights: F32Ptr, inputs: F32Ptr,
    rows: Int, outputs: Int, reduction: Int, input_stride: Int, output_stride: Int,
    workspace: F32Ptr,
):
    """Projects rows using shared prepared activations, or the scalar reference packer.

    The prepared path accepts one to four rows and requires a completed
    _prepare_activations call for these inputs. All harts participate.
    """
    comptime if not USE_VME or not PREFILL_GEMM:
        for row in range(rows):
            _matvec(
                out_ptr.unsafe_offset(row * output_stride), weights,
                inputs.unsafe_offset(row * input_stride), outputs, reduction,
            )
    else:
        var quantized = unsafe_stack_allocation[32, DType.uint32]().bitcast[UInt8]().as_unsafe_any_origin()
        var activation_scales = unsafe_stack_allocation[4, DType.float32]().as_unsafe_any_origin()
        var accumulator = unsafe_stack_allocation[256, DType.int32]()
        var packed = weights.bitcast[Int8]()
        var scales = weights.unsafe_offset(outputs * reduction // 4)
        var sums = scales.unsafe_offset(outputs * reduction // 32).bitcast[Int32]()
        var first = outputs * thread_idx.x // block_dim.x
        var last = outputs * (thread_idx.x + 1) // block_dim.x
        for row_base in range(0, rows, 4):
            var row_count = min(4, rows - row_base)
            for row in range(row_count):
                for column in range(first, last):
                    out_ptr[(row_base + row) * output_stride + column] = 0
            for group in range(reduction // 32):
                comptime if prepared:
                    quantized = workspace.bitcast[UInt8]().unsafe_offset(group * 128)
                    activation_scales = workspace.unsafe_offset(reduction + group * 4)
                else:
                    for row in range(row_count):
                        var source = inputs.unsafe_offset((row_base + row) * input_stride + group * 32)
                        var magnitude = Float32(0)
                        for inner in range(32):
                            magnitude = max(magnitude, abs(source[inner]))
                        var scale = magnitude / 127 if magnitude > 0 else Float32(1)
                        activation_scales[row] = scale
                        for inner in range(32):
                            var value = source[inner] / scale
                            var rounded = Int(value + (Float32(0.5) if value >= 0 else Float32(-0.5)))
                            quantized[inner * 4 + row] = UInt8(max(-127, min(127, rounded)) + 128)
                var previous = first
                var previous_count = 0
                var slot = 0
                for column in range(first, last, 32):
                    var count = min(32, last - column)
                    matrix_submit(
                        packed.unsafe_offset(group * 32 * outputs + column),
                        quantized, 32, row_count, count, outputs, 4,
                    )
                    comptime if not PREFILL_INTERLEAVE:
                        matrix_finish(accumulator, row_count, count, 32)
                    var result_column = previous if PREFILL_INTERLEAVE else column
                    var result_count = previous_count if PREFILL_INTERLEAVE else count
                    var result_slot = (1 - slot) * 128 if PREFILL_INTERLEAVE else 0
                    var metadata = group * outputs + result_column
                    if result_count:
                        for row in range(row_count):
                            accumulate_scaled(
                                out_ptr.unsafe_offset((row_base + row) * output_stride + result_column),
                                accumulator.unsafe_offset(result_slot + row * 32), sums.unsafe_offset(metadata),
                                scales.unsafe_offset(metadata), activation_scales[row], result_count,
                            )
                    comptime if PREFILL_INTERLEAVE:
                        matrix_finish(accumulator.unsafe_offset(slot * 128), row_count, count, 32)
                        previous = column
                        previous_count = count
                        slot = 1 - slot
                comptime if PREFILL_INTERLEAVE:
                    var metadata = group * outputs + previous
                    for row in range(row_count):
                        if previous_count:
                            accumulate_scaled(
                                out_ptr.unsafe_offset((row_base + row) * output_stride + previous),
                                accumulator.unsafe_offset((1 - slot) * 128 + row * 32), sums.unsafe_offset(metadata),
                                scales.unsafe_offset(metadata), activation_scales[row], previous_count,
                            )
        barrier()


@always_inline
def _matvec(out_ptr: F32Ptr, w: F32Ptr, x: F32Ptr, m: Int, n: Int):
    """out[i] = dot(w[i, :], x).  With MV_TRANSPOSED the weight buffer
    holds the offline-transposed matrix (see `_stage`) and the
    reduction-free column-accumulation kernel runs instead."""
    var first = m * thread_idx.x // block_dim.x
    var last = m * (thread_idx.x + 1) // block_dim.x
    comptime if USE_VME:
        _matvec_vme(out_ptr, w, x, m, n)
    elif MV_TRANSPOSED:
        if first < last:
            _mv_transposed(w.unsafe_offset(first), x, n, last - first, out_ptr.unsafe_offset(first), m)
    else:
        var rows = last - first
        var groups = rows - rows % 4
        if groups > 0:
            _mv(w.unsafe_offset(first * n), x, n, groups, out_ptr.unsafe_offset(first))
        var i = first + groups
        while i < last:
            _mv1(w.unsafe_offset(i * n), x, n, out_ptr.unsafe_offset(i))
            i += 1
    barrier()


def _dot_asm[sew: Int, lmul: Int]() -> String:
    return (
        "vsetvli t0, $3, e" + String(sew) + ", m" + String(lmul)
        + ", ta, ma\n"
        + _vle[sew]() + " v8, ($1)\n"
        + _vle[sew]() + " v" + String(8 + lmul) + ", ($2)\n"
        + "vfmul.vv v8, v8, v" + String(8 + lmul) + "\n"
        + "vmv.s.x v" + String(8 + 2 * lmul) + ", zero\n"
        + "vfredusum.vs v" + String(8 + 2 * lmul) + ", v8, v"
        + String(8 + 2 * lmul) + "\n"
        + "vfmv.f.s $0, v" + String(8 + 2 * lmul) + "\n"
    )


# dot(a[0..n), b[0..n)) for n <= VLMAX(lmul).
@always_inline
def _dot(a: F32Ptr, b: F32Ptr, n: Int) -> Float32:
    comptime asm = StaticString(materialize[_dot_asm[SEW, 2]()]())
    var result = Float32(0)
    var offset = 0
    while offset < n:
        var count = min(n - offset, _vlmax(SEW, 2))
        result += inlined_assembly[
            asm,
            Float32,
            constraints="=f,r,r,r," + _VCLOB,
        ](a.unsafe_offset(offset), b.unsafe_offset(offset), count)
        offset += count
    return result


def _axpy_asm[sew: Int, lmul: Int](first: Bool) -> String:
    var body = (
        "vsetvli t0, $3, e" + String(sew) + ", m" + String(lmul)
        + ", ta, ma\n"
        + _vle[sew]() + " v8, ($1)\n"
    )
    if first:
        body += "vfmul.vf v8, v8, $2\n"
        body += _vse[sew]() + " v8, ($0)\n"
    else:
        body += _vle[sew]() + " v" + String(8 + lmul) + ", ($0)\n"
        body += "vfmacc.vf v" + String(8 + lmul) + ", $2, v8\n"
        body += _vse[sew]() + " v" + String(8 + lmul) + ", ($0)\n"
    return body


# acc[0..n) += a * v[0..n) for n <= VLMAX(lmul).
@always_inline
def _axpy(acc: F32Ptr, v: F32Ptr, a: Float32, n: Int):
    comptime asm = StaticString(materialize[_axpy_asm[SEW, 1](False)]())
    var offset = 0
    while offset < n:
        var count = min(n - offset, _vlmax(SEW, 1))
        inlined_assembly[
            asm,
            NoneType,
            constraints="r,r,f,r," + _VCLOB,
        ](acc.unsafe_offset(offset), v.unsafe_offset(offset), a, count)
        offset += count


# acc[0..n) = a * v[0..n) (first position: no read, no zero-fill).
@always_inline
def _axpy_first(acc: F32Ptr, v: F32Ptr, a: Float32, n: Int):
    comptime asm = StaticString(materialize[_axpy_asm[SEW, 1](True)]())
    var offset = 0
    while offset < n:
        var count = min(n - offset, _vlmax(SEW, 1))
        inlined_assembly[
            asm,
            NoneType,
            constraints="r,r,f,r," + _VCLOB,
        ](acc.unsafe_offset(offset), v.unsafe_offset(offset), a, count)
        offset += count


def _ssq_asm[sew: Int, lmul: Int]() -> String:
    comptime chunk = _vlmax(sew, lmul)
    var body = (
        "li t0, " + String(chunk) + "\n"
        + "vsetvli zero, t0, e" + String(sew) + ", m" + String(lmul)
        + ", ta, ma\n"
        + "vmv.v.i v8, 0\n"
        + "mv t1, $2\n"
        + "mv t2, $1\n"
        + "1:\n"
        + "vsetvli t0, t1, e" + String(sew) + ", m" + String(lmul)
        + ", tu, ma\n"
        + _vle[sew]() + " v" + String(8 + lmul) + ", (t2)\n"
        + "vfmacc.vv v8, v" + String(8 + lmul) + ", v" + String(8 + lmul)
        + "\n"
        + "slli t0, t0, 2\n"
        + "add t2, t2, t0\n"
        + "srli t0, t0, 2\n"
        + "sub t1, t1, t0\n"
        + "bnez t1, 1b\n"
    )
    var l = lmul // 2
    while l >= 1:
        body += _cfg(sew, l, _vlmax(sew, l))
        body += "vfadd.vv v8, v8, v" + String(8 + l) + "\n"
        l //= 2
    body += _cfg(sew, 1, _vlmax(sew, 1))
    body += "vmv.s.x v24, zero\n"
    body += "vfredusum.vs v25, v8, v24\n"
    body += "vfmv.f.s $0, v25\n"
    return body


# Strip-mined sum of squares (unordered reduce) at EW_LMUL.
@always_inline
def _ssq(x: F32Ptr, n: Int) -> Float32:
    comptime asm = StaticString(materialize[_ssq_asm[SEW, EW_LMUL]()]())
    return inlined_assembly[
        asm,
        Float32,
        constraints="=f,r,r," + _VCLOB,
    ](x, n)


def _scale_asm[sew: Int, lmul: Int]() -> String:
    return (
        "mv t1, $4\n"
        + "mv t2, $1\n"
        + "mv t3, $2\n"
        + "mv t4, $0\n"
        + "1:\n"
        + "vsetvli t0, t1, e" + String(sew) + ", m" + String(lmul)
        + ", ta, ma\n"
        + _vle[sew]() + " v8, (t2)\n"
        + _vle[sew]() + " v" + String(8 + lmul) + ", (t3)\n"
        + "vfmul.vf v8, v8, $3\n"
        + "vfmul.vv v8, v8, v" + String(8 + lmul) + "\n"
        + _vse[sew]() + " v8, (t4)\n"
        + "slli t0, t0, 2\n"
        + "add t2, t2, t0\n"
        + "add t3, t3, t0\n"
        + "add t4, t4, t0\n"
        + "srli t0, t0, 2\n"
        + "sub t1, t1, t0\n"
        + "bnez t1, 1b\n"
    )


# out[j] = g[j] * (x[j] * inv), strip-mined at EW_LMUL.
@always_inline
def _scale(out_ptr: F32Ptr, x: F32Ptr, g: F32Ptr, inv: Float32, n: Int):
    comptime asm = StaticString(materialize[_scale_asm[SEW, EW_LMUL]()]())
    inlined_assembly[
        asm,
        NoneType,
        constraints="r,r,r,f,r," + _VCLOB,
    ](out_ptr, x, g, inv, n)


def _vadd_asm[sew: Int, lmul: Int]() -> String:
    return (
        "mv t1, $2\n"
        + "mv t2, $0\n"
        + "mv t3, $1\n"
        + "1:\n"
        + "vsetvli t0, t1, e" + String(sew) + ", m" + String(lmul)
        + ", ta, ma\n"
        + _vle[sew]() + " v8, (t2)\n"
        + _vle[sew]() + " v" + String(8 + lmul) + ", (t3)\n"
        + "vfadd.vv v8, v8, v" + String(8 + lmul) + "\n"
        + _vse[sew]() + " v8, (t2)\n"
        + "slli t0, t0, 2\n"
        + "add t2, t2, t0\n"
        + "add t3, t3, t0\n"
        + "srli t0, t0, 2\n"
        + "sub t1, t1, t0\n"
        + "bnez t1, 1b\n"
    )


# x[i] += r[i], strip-mined at EW_LMUL.
@always_inline
def _vadd(x: F32Ptr, r: F32Ptr, n: Int):
    comptime asm = StaticString(materialize[_vadd_asm[SEW, EW_LMUL]()]())
    var first = n * thread_idx.x // block_dim.x
    var count = n * (thread_idx.x + 1) // block_dim.x - first
    if count > 0:
        inlined_assembly[
            asm,
            NoneType,
            constraints="r,r,r," + _VCLOB,
        ](x.unsafe_offset(first), r.unsafe_offset(first), count)
    barrier()


@always_inline
def _rmsnorm(out_ptr: F32Ptr, x: F32Ptr, g: F32Ptr, n: Int):
    var ss = _ssq(x, n)
    var inv = Float32(1.0) / sqrt(ss / Float32(n) + Float32(1e-5))
    var first = n * thread_idx.x // block_dim.x
    var count = n * (thread_idx.x + 1) // block_dim.x - first
    if count > 0:
        _scale(out_ptr.unsafe_offset(first), x.unsafe_offset(first), g.unsafe_offset(first), inv, count)
    barrier()


def _rope_asm[sew: Int]() -> String:
    return (
        "srli t1, $2, 1\n"
        + "vsetvli t0, t1, e" + String(sew) + ", m1, ta, ma\n"
        + "vlseg2e" + String(sew) + ".v v8, ($1)\n"
        + "vlseg2e" + String(sew) + ".v v10, ($0)\n"
        + "vfmul.vv v12, v10, v8\n"
        + "vfmul.vv v13, v11, v9\n"
        + "vfsub.vv v12, v12, v13\n"
        + "vfmul.vv v14, v10, v9\n"
        + "vfmul.vv v15, v11, v8\n"
        + "vfadd.vv v13, v14, v15\n"
        + "vmv.v.v v10, v12\n"
        + "vmv.v.v v11, v13\n"
        + "vsseg2e" + String(sew) + ".v v10, ($0)\n"
    )


# Rotate one head's (even, odd) pairs by the interleaved (cos, sin) table.
@always_inline
def _rope_head(q: F32Ptr, rope: F32Ptr, hs: Int):
    comptime asm = StaticString(materialize[_rope_asm[SEW]()]())
    var offset = 0
    while offset < hs:
        var count = min(hs - offset, 2 * _vlmax(SEW, 1))
        inlined_assembly[
            asm,
            NoneType,
            constraints="r,r,r," + _VCLOB,
        ](q.unsafe_offset(offset), rope.unsafe_offset(offset), count)
        offset += count


def t1_layer[stop_before_activation: Bool = False](hdr: I32Ptr, rope: F32Ptr, x: F32Ptr, w: F32Ptr, kv: F32Ptr, sc: F32Ptr):
    """One transformer layer for one token position.

    hdr:  [pos, dim, hidden, n_heads, n_kv_heads, head_size, seq_cap, vocab]
    rope: [head_size] floats: cos/sin interleaved for the current position
    x:    [dim] residual stream (device resident across launches)
    w:    this layer's weights: rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3
    kv:   this layer's cache: k[seq_cap][kv_dim] then v[seq_cap][kv_dim]
    sc:   scratch: xb[dim], q[dim], hb[hidden], hb2[hidden], att[seq_cap]
    """
    comptime W = simd_width_of[DType.float32]()
    var pos = Int(hdr.unsafe_load(0))
    var dim = Int(hdr.unsafe_load(1))
    var hidden = Int(hdr.unsafe_load(2))
    var n_heads = Int(hdr.unsafe_load(3))
    var n_kv = Int(hdr.unsafe_load(4))
    var hs = Int(hdr.unsafe_load(5))
    var seq_cap = Int(hdr.unsafe_load(6))
    var kv_dim = n_kv * hs
    var kv_mul = n_heads // n_kv

    var o_wq = dim
    var o_wk = o_wq + dim * dim
    var o_wv = o_wk + kv_dim * dim
    var o_wo = o_wv + kv_dim * dim
    var o_rms_ffn = o_wo + dim * dim
    var o_w1 = o_rms_ffn + dim
    var o_w2 = o_w1 + hidden * dim
    var o_w3 = o_w2 + dim * hidden

    var xb = sc
    var q = sc.unsafe_offset(dim)
    var hb = sc.unsafe_offset(2 * dim)
    var hb2 = sc.unsafe_offset(2 * dim + hidden)
    var att = sc.unsafe_offset(2 * dim + 2 * hidden + thread_idx.x * seq_cap)

    var krow = kv.unsafe_offset(pos * kv_dim)
    var vbase = kv.unsafe_offset(seq_cap * kv_dim)

    # Attention block.
    _rmsnorm(xb, x, w, dim)
    _matvec(q, w.unsafe_offset(o_wq), xb, dim, dim)
    _matvec(krow, w.unsafe_offset(o_wk), xb, kv_dim, dim)
    _matvec(vbase.unsafe_offset(pos * kv_dim), w.unsafe_offset(o_wv), xb, kv_dim, dim)

    # RoPE: rotate adjacent pairs (llama2.c convention), one head at a time
    # via segment loads (the rope buffer is interleaved cos/sin per pair).
    var h0 = thread_idx.x
    while h0 < n_heads:
        _rope_head(q.unsafe_offset(h0 * hs), rope, hs)
        h0 += block_dim.x
    h0 = thread_idx.x
    while h0 < n_kv:
        _rope_head(krow.unsafe_offset(h0 * hs), rope, hs)
        h0 += block_dim.x
    barrier()

    var scale = Float32(1.0) / sqrt(Float32(hs))
    var h = thread_idx.x
    while h < n_heads:
        var qo = h * hs
        var kvo = (h // kv_mul) * hs
        # Scores (vector dot per cached position).
        var t = 0
        while t <= pos:
            var s = _dot(
                q.unsafe_offset(qo), kv.unsafe_offset(t * kv_dim + kvo), hs
            )
            att.unsafe_store(t, s * scale)
            t += 1
        # Softmax over att[0..=pos].
        var mx = att.unsafe_load(0)
        t = 1
        while t <= pos:
            var a = att.unsafe_load(t)
            if a > mx:
                mx = a
            t += 1
        var sum = Float32(0)
        t = 0
        while t <= pos:
            var e = exp(att.unsafe_load(t) - mx)
            att.unsafe_store(t, e)
            sum += e
            t += 1
        var isum = Float32(1.0) / sum
        # Weighted sum of values into xb[head]: the first position writes
        # (no zero-fill pass), the rest accumulate.
        _axpy_first(
            xb.unsafe_offset(qo), vbase.unsafe_offset(kvo),
            att.unsafe_load(0) * isum, hs,
        )
        t = 1
        while t <= pos:
            var a = att.unsafe_load(t) * isum
            _axpy(
                xb.unsafe_offset(qo),
                vbase.unsafe_offset(t * kv_dim + kvo),
                a, hs,
            )
            t += 1
        h += block_dim.x
    barrier()

    # Output projection + residual (q reused as temporary).
    _matvec(q, w.unsafe_offset(o_wo), xb, dim, dim)
    _vadd(x, q, dim)

    # FFN: SwiGLU.
    _rmsnorm(xb, x, w.unsafe_offset(o_rms_ffn), dim)
    _matvec(hb, w.unsafe_offset(o_w1), xb, hidden, dim)
    _matvec(hb2, w.unsafe_offset(o_w3), xb, hidden, dim)
    if stop_before_activation:
        return
    var j = hidden * thread_idx.x // block_dim.x
    var end = hidden * (thread_idx.x + 1) // block_dim.x
    while j + W <= end:
        var g = hb.unsafe_load[width=W](j)
        var u = hb2.unsafe_load[width=W](j)
        var s = g / (SIMD[DType.float32, W](1) + exp(-g))
        hb.unsafe_store(j, s * u)
        j += W
    while j < end:
        var g = hb.unsafe_load(j)
        hb.unsafe_store(j, g / (Float32(1) + exp(-g)) * hb2.unsafe_load(j))
        j += 1
    barrier()
    _matvec(xb, w.unsafe_offset(o_w2), hb, dim, hidden)
    _vadd(x, xb, dim)


def t1_prefill_layer(
    hdr: I32Ptr, rope: F32Ptr, x: F32Ptr, w: F32Ptr, kv: F32Ptr, sc: F32Ptr, token_count: Int32,
):
    """Processes known consecutive tokens with causal attention and shared weights.

    The header position is the first token position. Residual and RoPE rows
    are contiguous. Scratch holds xb[rows,dim], q[rows,dim], hb[rows,hidden],
    hb2[rows,hidden], att[THREADS,seq_cap], and shared packed activations
    (max(dim,hidden) * 9 // 8 Float32 words). KV uses the decode layout.
    """
    comptime W = simd_width_of[DType.float32]()
    var rows = Int(token_count)
    var start = Int(hdr[0])
    var dim = Int(hdr[1])
    var hidden = Int(hdr[2])
    var n_heads = Int(hdr[3])
    var n_kv = Int(hdr[4])
    var hs = Int(hdr[5])
    var seq_cap = Int(hdr[6])
    var kv_dim = n_kv * hs
    var kv_mul = n_heads // n_kv
    var o_wq = dim
    var o_wk = o_wq + dim * dim
    var o_wv = o_wk + kv_dim * dim
    var o_wo = o_wv + kv_dim * dim
    var o_rms_ffn = o_wo + dim * dim
    var o_w1 = o_rms_ffn + dim
    var o_w2 = o_w1 + hidden * dim
    var o_w3 = o_w2 + dim * hidden
    var xb = sc
    var query = sc.unsafe_offset(rows * dim)
    var gate = sc.unsafe_offset(2 * rows * dim)
    var up = sc.unsafe_offset(2 * rows * dim + rows * hidden)
    var att = sc.unsafe_offset(2 * rows * (dim + hidden) + thread_idx.x * seq_cap)
    var workspace = sc.unsafe_offset(2 * rows * (dim + hidden) + block_dim.x * seq_cap)
    var value_cache = kv.unsafe_offset(seq_cap * kv_dim)
    for row in range(rows):
        _rmsnorm(xb.unsafe_offset(row * dim), x.unsafe_offset(row * dim), w, dim)
    _prepare_activations(xb, workspace, rows, dim, dim)
    _matmul_prepared(query, w.unsafe_offset(o_wq), xb, rows, dim, dim, dim, dim, workspace)
    _matmul_prepared(kv.unsafe_offset(start * kv_dim), w.unsafe_offset(o_wk), xb, rows, kv_dim, dim, dim, kv_dim, workspace)
    _matmul_prepared(value_cache.unsafe_offset(start * kv_dim), w.unsafe_offset(o_wv), xb, rows, kv_dim, dim, dim, kv_dim, workspace)
    for row in range(rows):
        for head in range(thread_idx.x, n_heads, block_dim.x):
            _rope_head(query.unsafe_offset(row * dim + head * hs), rope.unsafe_offset(row * hs), hs)
        for head in range(thread_idx.x, n_kv, block_dim.x):
            _rope_head(kv.unsafe_offset((start + row) * kv_dim + head * hs), rope.unsafe_offset(row * hs), hs)
    barrier()
    var scale = Float32(1) / sqrt(Float32(hs))
    for row in range(rows):
        var position = start + row
        for head in range(thread_idx.x, n_heads, block_dim.x):
            var query_offset = row * dim + head * hs
            var kv_offset = (head // kv_mul) * hs
            for previous in range(position + 1):
                att[previous] = _dot(
                    query.unsafe_offset(query_offset), kv.unsafe_offset(previous * kv_dim + kv_offset), hs,
                ) * scale
            var maximum = att[0]
            for previous in range(1, position + 1):
                if att[previous] > maximum:
                    maximum = att[previous]
            var total = Float32(0)
            for previous in range(position + 1):
                var exponential = exp(att[previous] - maximum)
                att[previous] = exponential
                total += exponential
            var inverse = Float32(1) / total
            _axpy_first(xb.unsafe_offset(query_offset), value_cache.unsafe_offset(kv_offset), att[0] * inverse, hs)
            for previous in range(1, position + 1):
                _axpy(
                    xb.unsafe_offset(query_offset), value_cache.unsafe_offset(previous * kv_dim + kv_offset),
                    att[previous] * inverse, hs,
                )
    barrier()
    _prepare_activations(xb, workspace, rows, dim, dim)
    _matmul_prepared(query, w.unsafe_offset(o_wo), xb, rows, dim, dim, dim, dim, workspace)
    for row in range(rows):
        _vadd(x.unsafe_offset(row * dim), query.unsafe_offset(row * dim), dim)
        _rmsnorm(xb.unsafe_offset(row * dim), x.unsafe_offset(row * dim), w.unsafe_offset(o_rms_ffn), dim)
    _prepare_activations(xb, workspace, rows, dim, dim)
    _matmul_prepared(gate, w.unsafe_offset(o_w1), xb, rows, hidden, dim, dim, hidden, workspace)
    _matmul_prepared(up, w.unsafe_offset(o_w3), xb, rows, hidden, dim, dim, hidden, workspace)
    for row in range(rows):
        var element = hidden * thread_idx.x // block_dim.x
        var last = hidden * (thread_idx.x + 1) // block_dim.x
        while element + W <= last:
            var offset = row * hidden + element
            var value = gate.unsafe_load[width=W](offset)
            var multiplier = up.unsafe_load[width=W](offset)
            var sigmoid = value / (SIMD[DType.float32, W](1) + exp(-value))
            gate.unsafe_store(offset, sigmoid * multiplier)
            element += W
        while element < last:
            var offset = row * hidden + element
            var value = gate[offset]
            gate[offset] = value / (Float32(1) + exp(-value)) * up[offset]
            element += 1
    barrier()
    _prepare_activations(gate, workspace, rows, hidden, hidden)
    _matmul_prepared(xb, w.unsafe_offset(o_w2), gate, rows, dim, hidden, hidden, dim, workspace)
    for row in range(rows):
        _vadd(x.unsafe_offset(row * dim), xb.unsafe_offset(row * dim), dim)


def t1_head(hdr: I32Ptr, x: F32Ptr, w: F32Ptr, logits: F32Ptr, sc: F32Ptr):
    """Final RMSNorm + classifier matvec. w = rms_final[dim], wcls[vocab*dim]."""
    var dim = Int(hdr.unsafe_load(1))
    var vocab = Int(hdr.unsafe_load(7))
    _rmsnorm(sc, x, w, dim)
    _matvec(logits, w.unsafe_offset(dim), sc, vocab, dim)


def t1_prefill_head(hdr: I32Ptr, x: F32Ptr, w: F32Ptr, logits: F32Ptr, sc: F32Ptr, rows: Int32):
    """Classifies only the final known token in a prefill batch."""
    t1_head(hdr, x.unsafe_offset((Int(rows) - 1) * Int(hdr[1])), w, logits, sc)


# ===----------------------------------------------------------------------=== #
# Host side
# ===----------------------------------------------------------------------=== #


def _read_i32(b: Pointer[UInt8, _], off: Int) -> Int32:
    var u = (
        UInt32(Int(b[off]))
        | (UInt32(Int(b[off + 1])) << 8)
        | (UInt32(Int(b[off + 2])) << 16)
        | (UInt32(Int(b[off + 3])) << 24)
    )
    return Int32(from_bits=u)


def _read_f32(b: Pointer[UInt8, _], off: Int) -> Float32:
    var u = (
        UInt32(Int(b[off]))
        | (UInt32(Int(b[off + 1])) << 8)
        | (UInt32(Int(b[off + 2])) << 16)
        | (UInt32(Int(b[off + 3])) << 24)
    )
    return Float32(from_bits=u)


def _read_full(f: FileHandle, ptr: Pointer[mut=True, Float32, _], n: Int) raises:
    """Reads exactly n Float32 elements at the current file position."""
    var got = 0
    while got < n:
        var bytes_read = f.read(
            Span(unsafe_ptr=ptr.unsafe_offset(got), length=n - got)
        )
        if bytes_read == 0:
            raise Error("t1llama: unexpected EOF in model file")
        got += bytes_read // 4


def _read_mat[
    o1: MutOrigin, //
](f: FileHandle, dst: Pointer[Float32, o1], m: Int, n: Int,
  mut tmp: List[Float32]) raises:
    """Reads an [m, n] row-major matrix; with MV_TRANSPOSED it is stored
    transposed ([n, m]) so the reduction-free matvec can accumulate
    columns with vfmacc.vf."""
    comptime if USE_VME:
        if n % 32 != 0:
            raise Error("VME matrix reduction must be a multiple of 32")
        _read_full(f, tmp.unsafe_ptr(), m * n)
        var packed = dst.bitcast[Int8]()
        var scales = dst.unsafe_offset(m * n // 4)
        var sums = scales.unsafe_offset(m * n // 32).bitcast[Int32]()
        for row in range(m):
            for group in range(n // 32):
                var magnitude = Float32(0)
                for inner in range(32):
                    magnitude = max(magnitude, abs(tmp[row * n + group * 32 + inner]))
                var scale = magnitude / 127 if magnitude > 0 else Float32(1)
                var total = Int32(0)
                for inner in range(32):
                    var value = tmp[row * n + group * 32 + inner] / scale
                    var rounded = Int(value + (Float32(0.5) if value >= 0 else Float32(-0.5)))
                    var quantized = Int8(max(-127, min(127, rounded)))
                    packed[(group * 32 + inner) * m + row] = quantized
                    total += Int32(quantized)
                scales[group * m + row] = scale
                sums[group * m + row] = total
    elif MV_TRANSPOSED:
        _read_full(f, tmp.unsafe_ptr(), m * n)
        var tp = tmp.unsafe_ptr()
        var i = 0
        while i < m:
            var j = 0
            while j < n:
                dst.unsafe_store(j * m + i, tp.unsafe_load(i * n + j))
                j += 1
            i += 1
    else:
        _read_full(f, dst, m * n)


@fieldwise_init
struct Config(Copyable, Movable):
    var dim: Int
    var hidden: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv: Int
    var vocab: Int
    var seq_len: Int
    var shared_classifier: Bool
    var head_size: Int
    var kv_dim: Int


struct Tokenizer:
    var pieces: List[String]
    var scores: List[Float32]
    var lookup: Dict[String, Int]

    def __init__(out self, path: String) raises:
        self.pieces = List[String]()
        self.scores = List[Float32]()
        self.lookup = Dict[String, Int]()
        var raw = open(path, "r").read_bytes()
        var b = raw.unsafe_ptr()
        var off = 4  # skip max_token_length
        var idx = 0
        while off < len(raw):
            var score = _read_f32(b, off)
            var n = Int(_read_i32(b, off + 4))
            off += 8
            var piece = String(
                StringSlice(
                    unsafe_from_utf8=Span(unsafe_ptr=b.unsafe_offset(off), length=n)
                )
            )
            off += n
            self.scores.append(score)
            self.pieces.append(piece)
            self.lookup[piece] = idx
            idx += 1

    def find(self, s: String) -> Int:
        try:
            return self.lookup[s]
        except:
            return -1

    def encode(self, text: String) -> List[Int]:
        """BOS + sentencepiece-style BPE, ported from llama2.c encode()."""
        var toks = List[Int]()
        toks.append(1)  # BOS
        if text.byte_length() == 0:
            return toks^
        var dummy = self.find(" ")
        if dummy >= 0:
            toks.append(dummy)
        var tb = text.as_bytes()
        var i = 0
        while i < len(tb):
            # Group one UTF-8 codepoint.
            var j = i + 1
            while j < len(tb) and (Int(tb[j]) & 0xC0) == 0x80 and j - i < 4:
                j += 1
            var cp = String(
                StringSlice(
                    unsafe_from_utf8=Span(
                        unsafe_ptr=tb.unsafe_ptr().unsafe_offset(i), length=j - i
                    )
                )
            )
            var id = self.find(cp)
            if id >= 0:
                toks.append(id)
            else:
                var k = i
                while k < j:
                    toks.append(Int(tb[k]) + 3)  # byte-fallback tokens
                    k += 1
            i = j
        # Merge loop: repeatedly fuse the best-scoring adjacent pair.
        while True:
            var best_score = Float32(-1e10)
            var best_id = -1
            var best_idx = -1
            var t = 0
            while t + 1 < len(toks):
                var merged = self.pieces[toks[t]] + self.pieces[toks[t + 1]]
                var id = self.find(merged)
                if id >= 0 and self.scores[id] > best_score:
                    best_score = self.scores[id]
                    best_id = id
                    best_idx = t
                t += 1
            if best_idx < 0:
                break
            toks[best_idx] = best_id
            _ = toks.pop(best_idx + 1)
        return toks^

    def decode(self, prev: Int, token: Int) -> String:
        var piece = self.pieces[token]
        if prev == 1 and piece.startswith(" "):
            var trimmed = String(piece[byte=1:])
            piece = trimmed^
        # Raw-byte tokens are stored as "<0xXX>".
        if (
            piece.byte_length() == 6
            and piece.startswith("<0x")
            and piece.endswith(">")
        ):
            var hex = String(piece[byte=3:5])
            var v = 0
            for ch in hex.codepoint_slices():
                var c = Int(ord(ch))
                v = v * 16 + (
                    c - 48 if c < 58 else (c | 32) - 97 + 10
                )
            var bytes = List[UInt8]()
            bytes.append(UInt8(v))
            return String(
                StringSlice(unsafe_from_utf8=Span(bytes))
            )
        return piece


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: t1llama <model.bin> <tokenizer.bin> [prompt] [steps] [--first-layer-only]")
        return
    var model_path = String(args[1])
    var tok_path = String(args[2])
    var prompt = String(args[3]) if len(args) > 3 else String("")
    var steps = Int(String(args[4])) if len(args) > 4 else 64
    var first_layer_only = len(args) > 5 and String(args[5]) == "--first-layer-only"
    if PREFILL_BATCH < 1 or PREFILL_BATCH > 4:
        raise Error("T1_PREFILL_BATCH must be in 1..4")

    # --- Model header -----------------------------------------------------
    var f = open(model_path, "r")
    var hdr_raw = f.read_bytes(28)
    var hb = hdr_raw.unsafe_ptr()
    var vocab_raw = Int(_read_i32(hb, 20))
    var cfg = Config(
        dim=Int(_read_i32(hb, 0)),
        hidden=Int(_read_i32(hb, 4)),
        n_layers=Int(_read_i32(hb, 8)),
        n_heads=Int(_read_i32(hb, 12)),
        n_kv=Int(_read_i32(hb, 16)),
        vocab=vocab_raw if vocab_raw > 0 else -vocab_raw,
        seq_len=Int(_read_i32(hb, 24)),
        shared_classifier=vocab_raw > 0,
        head_size=0,
        kv_dim=0,
    )
    cfg.head_size = cfg.dim // cfg.n_heads
    cfg.kv_dim = cfg.n_kv * cfg.head_size
    print(
        "t1llama:", model_path, "dim", cfg.dim, "hidden", cfg.hidden,
        "layers", cfg.n_layers, "heads", cfg.n_heads, "kv", cfg.n_kv,
        "vocab", cfg.vocab, "seq", cfg.seq_len,
    )

    var tok = Tokenizer(tok_path)
    var prompt_toks = tok.encode(prompt)
    if steps > cfg.seq_len:
        steps = cfg.seq_len
    var seq_cap = steps + 1

    # --- File offsets (bytes) of each grouped tensor ----------------------
    var L = cfg.n_layers
    var emb_off = 28
    var rms_att_off = emb_off + 4 * cfg.vocab * cfg.dim
    var wq_off = rms_att_off + 4 * L * cfg.dim
    var wk_off = wq_off + 4 * L * cfg.dim * cfg.dim
    var wv_off = wk_off + 4 * L * cfg.dim * cfg.kv_dim
    var wo_off = wv_off + 4 * L * cfg.dim * cfg.kv_dim
    var rms_ffn_off = wo_off + 4 * L * cfg.dim * cfg.dim
    var w1_off = rms_ffn_off + 4 * L * cfg.dim
    var w2_off = w1_off + 4 * L * cfg.dim * cfg.hidden
    var w3_off = w2_off + 4 * L * cfg.dim * cfg.hidden
    var rms_final_off = w3_off + 4 * L * cfg.dim * cfg.hidden
    var freq_off = rms_final_off + 4 * cfg.dim
    var wcls_off = freq_off + 4 * cfg.seq_len * cfg.head_size
    if cfg.shared_classifier:
        wcls_off = emb_off

    var layer_words = (
        cfg.dim + cfg.dim * cfg.dim + 2 * cfg.kv_dim * cfg.dim
        + cfg.dim * cfg.dim + cfg.dim + 3 * cfg.dim * cfg.hidden
    )
    var head_words = cfg.dim + cfg.vocab * cfg.dim
    var sc_words = PREFILL_BATCH * (2 * cfg.dim + 2 * cfg.hidden) + THREADS * seq_cap + max(cfg.dim, cfg.hidden) * 9 // 8

    # --- Device setup (standard Mojo DeviceContext API) -------------------
    var ctx = DeviceContext(api="t1")
    print("device:", ctx.name(), "api:", ctx.api())
    # The streamed weight buffers are allocated first so that the small,
    # kernel-written buffers form one compact arena range (cheap readback).
    var wl_b = ctx.enqueue_create_buffer[DType.float32](layer_words)
    var wh_b = ctx.enqueue_create_buffer[DType.float32](head_words)
    var hdr_b = ctx.enqueue_create_buffer[DType.int32](8)
    var rope_b = ctx.enqueue_create_buffer[DType.float32](PREFILL_BATCH * cfg.head_size)
    var x_b = ctx.enqueue_create_buffer[DType.float32](PREFILL_BATCH * cfg.dim)
    var sc_b = ctx.enqueue_create_buffer[DType.float32](sc_words)
    var logits_b = ctx.enqueue_create_buffer[DType.float32](cfg.vocab)
    var kv_bufs = List[DeviceBuffer[DType.float32]]()
    for _ in range(L):
        kv_bufs.append(
            ctx.enqueue_create_buffer[DType.float32](
                2 * seq_cap * cfg.kv_dim
            )
        )

    # Head weights never change: load them once.
    var staging = List[Float32](length=head_words, fill=0)
    var sp = staging.unsafe_ptr()
    var tmp = List[Float32](
        length=cfg.vocab * cfg.dim if MV_TRANSPOSED else 1, fill=0
    )
    _ = f.seek(rms_final_off)
    _read_full(f, sp, cfg.dim)
    _ = f.seek(wcls_off)
    _read_mat(f, sp.unsafe_offset(cfg.dim), cfg.vocab, cfg.dim, tmp)
    wh_b.enqueue_copy_from(sp)

    var hdr_host = List[Int32](length=8, fill=0)
    hdr_host[1] = Int32(cfg.dim)
    hdr_host[2] = Int32(cfg.hidden)
    hdr_host[3] = Int32(cfg.n_heads)
    hdr_host[4] = Int32(cfg.n_kv)
    hdr_host[5] = Int32(cfg.head_size)
    hdr_host[6] = Int32(seq_cap)
    hdr_host[7] = Int32(cfg.vocab)
    var rope_host = List[Float32](length=PREFILL_BATCH * cfg.head_size, fill=0)
    var emb_host = List[Float32](length=PREFILL_BATCH * cfg.dim, fill=0)

    # --- Generation loop --------------------------------------------------
    print("prompt tokens:", len(prompt_toks), " steps:", steps)
    print("prefill batch:", PREFILL_BATCH)
    var token = prompt_toks[0]
    var pos = 0
    var out_text = String("")
    var t_start = perf_counter_ns()
    while pos < steps:
        var t0 = perf_counter_ns()
        var rows = min(PREFILL_BATCH, min(steps - pos, max(1, len(prompt_toks) - pos)))

        # Embedding row -> x (host lookup, device write).
        for row in range(rows):
            var input_token = prompt_toks[pos + row] if pos < len(prompt_toks) else token
            _ = f.seek(emb_off + 4 * input_token * cfg.dim)
            _read_full(f, emb_host.unsafe_ptr().unsafe_offset(row * cfg.dim), cfg.dim)
        if pos < len(prompt_toks):
            token = prompt_toks[pos + rows - 1]
        x_b.enqueue_copy_from(emb_host.unsafe_ptr())

        # Position-dependent inputs.
        hdr_host[0] = Int32(pos)
        hdr_b.enqueue_copy_from(hdr_host.unsafe_ptr())
        for row in range(rows):
            var hd = 0
            while hd < cfg.head_size:
                var freq = exp(
                    -log(Float32(10000.0)) * Float32(hd) / Float32(cfg.head_size)
                )
                var val = Float32(pos + row) * freq
                rope_host[row * cfg.head_size + hd] = cos(val)
                rope_host[row * cfg.head_size + hd + 1] = sin(val)
                hd += 2
        rope_b.enqueue_copy_from(rope_host.unsafe_ptr())

        # One launch per layer; weights streamed from disk.
        for l in range(L):
            var o = 0
            _ = f.seek(rms_att_off + 4 * l * cfg.dim)
            _read_full(f, sp.unsafe_offset(o), cfg.dim)
            o += cfg.dim
            _ = f.seek(wq_off + 4 * l * cfg.dim * cfg.dim)
            _read_mat(f, sp.unsafe_offset(o), cfg.dim, cfg.dim, tmp)
            o += cfg.dim * cfg.dim
            _ = f.seek(wk_off + 4 * l * cfg.dim * cfg.kv_dim)
            _read_mat(f, sp.unsafe_offset(o), cfg.kv_dim, cfg.dim, tmp)
            o += cfg.dim * cfg.kv_dim
            _ = f.seek(wv_off + 4 * l * cfg.dim * cfg.kv_dim)
            _read_mat(f, sp.unsafe_offset(o), cfg.kv_dim, cfg.dim, tmp)
            o += cfg.dim * cfg.kv_dim
            _ = f.seek(wo_off + 4 * l * cfg.dim * cfg.dim)
            _read_mat(f, sp.unsafe_offset(o), cfg.dim, cfg.dim, tmp)
            o += cfg.dim * cfg.dim
            _ = f.seek(rms_ffn_off + 4 * l * cfg.dim)
            _read_full(f, sp.unsafe_offset(o), cfg.dim)
            o += cfg.dim
            _ = f.seek(w1_off + 4 * l * cfg.dim * cfg.hidden)
            _read_mat(f, sp.unsafe_offset(o), cfg.hidden, cfg.dim, tmp)
            o += cfg.dim * cfg.hidden
            _ = f.seek(w2_off + 4 * l * cfg.dim * cfg.hidden)
            _read_mat(f, sp.unsafe_offset(o), cfg.dim, cfg.hidden, tmp)
            o += cfg.dim * cfg.hidden
            _ = f.seek(w3_off + 4 * l * cfg.dim * cfg.hidden)
            _read_mat(f, sp.unsafe_offset(o), cfg.hidden, cfg.dim, tmp)
            wl_b.enqueue_copy_from(sp)
            if rows > 1:
                ctx.enqueue_function[t1_prefill_layer](
                    hdr_b, rope_b, x_b, wl_b, kv_bufs[l], sc_b, Int32(rows),
                    grid_dim=1, block_dim=THREADS,
                )
            else:
                ctx.enqueue_function[t1_layer[False]](
                    hdr_b, rope_b, x_b, wl_b, kv_bufs[l], sc_b,
                    grid_dim=1, block_dim=THREADS,
                )
            if first_layer_only:
                ctx.synchronize()
                print("First layer complete: position", pos, "tokens", rows, "layer", l)
                return

        ctx.enqueue_function[t1_prefill_head](
            hdr_b, x_b, wh_b, logits_b, sc_b, Int32(rows), grid_dim=1, block_dim=THREADS
        )
        ctx.synchronize()

        # Next token: forced while consuming the prompt, then greedy argmax.
        var next_tok: Int
        if pos + rows < len(prompt_toks):
            next_tok = prompt_toks[pos + rows]
        else:
            var best = 0
            var best_v = Float32(-1e30)
            with logits_b.map_to_host() as hl:
                for i in range(cfg.vocab):
                    if hl[i] > best_v:
                        best_v = hl[i]
                        best = i
            next_tok = best
        if next_tok == 1:
            break

        var piece = tok.decode(token, next_tok)
        # Only generated tokens go into the transcript; forced prompt
        # tokens are already part of the prompt text.
        if pos + rows >= len(prompt_toks):
            out_text += piece
        var ms = (perf_counter_ns() - t0) // 1_000_000
        print(
            "[pos", pos + rows - 1, "->", next_tok, "|", ms, "ms]", piece,
        )
        token = next_tok
        pos += rows

    var total_s = (perf_counter_ns() - t_start) // 1_000_000_000
    print("")
    print("=== T1 output ===")
    print(prompt + out_text)
    print("=== ", pos, "positions in", total_s, "s ===")
