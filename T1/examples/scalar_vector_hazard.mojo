"""Checks scalar FP producer dependencies consumed by vector instructions."""

from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from std.sys import inlined_assembly


def probe(output: Pointer[Float32, MutAnyOrigin]):
    var destination = output.unsafe_offset(thread_idx.x * 32)
    inlined_assembly[
        "li t0, 32\n"
        "vsetvli zero, t0, e32, m1, ta, ma\n"
        "li t0, 0x40000000\n"
        "fmv.w.x fa5, t0\n"
        "vfmv.v.f v8, fa5\n"
        "li t0, 0x3f000000\n"
        "vmv.v.x v10, t0\n"
        "li t0, 0x40400000\n"
        "fmv.w.x fa5, t0\n"
        "vmv.v.v v11, v8\n"
        "vfmadd.vf v11, fa5, v10\n"
        "vse32.v v11, ($0)\n"
        "fence iorw, iorw\n",
        NoneType,
        constraints="r,~{t0},~{fa5},~{v8},~{v10},~{v11},~{memory}",
    ](destination)


def main() raises:
    var context = DeviceContext(api="t1")
    var output = context.enqueue_create_buffer[DType.float32](128)
    context.enqueue_function[probe](output, grid_dim=1, block_dim=4)
    context.synchronize()
    with output.map_to_host() as host:
        for index in range(128):
            if host[index] != Float32(6.5):
                print("Scalar/vector RAW mismatch:", index, host[index], "expected", Float32(6.5))
                raise Error("Vector consumer observed an incorrect scalar FP operand")
    print("Scalar/vector RAW dependencies OK")
