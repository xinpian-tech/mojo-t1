"""Defines the T1 accelerator and its standard-library hooks."""

from std.collections.string.string_span import _get_kgen_string
from std.sys import CompilationTarget
from std.sys import inlined_assembly
from std.collections import OptionalReg
from std._gpu.primitives.id import _GPUIdPluginHookFnType
from max.gpu.sync.sync import _BarrierPluginHookFnType
from ._trait import MaxPlugin, MaxPluginHooks
from std._plugin import StdPlugin, PluginHooks, TargetDeviceKind
from std._gpu.host.info import (
    GPUInfo,
    TargetAccelerator,
    TargetAcceleratorCollection,
    TargetAcceleratorType,
)


@always_inline
def _t1_gpu_id[name: StaticString, dim: StaticString]() -> Int:
    comptime if name == "thread_idx" and dim == "x":
        return Int(inlined_assembly["csrr $0, mhartid", UInt32, constraints="=r"]())
    elif name == "block_dim" and dim == "x":
        return Int(inlined_assembly[
            "li t0, 0x800f0000; lw $0, 12(t0)", UInt32,
            constraints="=r,~{t0},~{memory}",
        ]())
    elif name == "block_dim" or name == "grid_dim":
        return 1
    else:
        return 0


@always_inline
def _t1_barrier():
    inlined_assembly[
        "fence iorw, iorw\n"
        "csrr t0, mhartid\n"
        "slli t0, t0, 6\n"
        "li t1, 0x800e0000\n"
        "add t0, t0, t1\n"
        "li t2, 1\n"
        "vsetvli zero, t2, e32, m1, ta, ma\n"
        "vle32.v v1, (t0)\n"
        "vmv.x.s t2, v1\n"
        "addi t2, t2, 1\n"
        "vmv.v.x v1, t2\n"
        "vse32.v v1, (t0)\n"
        "fence iorw, iorw\n"
        "li t0, 0x800f0000\n"
        "lw t3, 12(t0)\n"
        "1:\n"
        "vle32.v v1, (t1)\n"
        "vmv.x.s t4, v1\n"
        "bltu t4, t2, 1b\n"
        "addi t1, t1, 64\n"
        "addi t3, t3, -1\n"
        "bnez t3, 1b\n"
        "fence iorw, iorw\n",
        NoneType,
        constraints="~{t0},~{t1},~{t2},~{t3},~{t4},~{v1},~{memory}",
    ]()


struct T1Plugin(StdPlugin, MaxPlugin):
    """Selects bare-metal RV32 behavior for T1 kernels."""

    comptime name = _get_kgen_string["t1"]()
    comptime HookFor[target: CompilationTarget] = T1Hooks[target]


struct T1Hooks[target: CompilationTarget](PluginHooks, MaxPluginHooks):
    """Maps a single thread block to T1's independent Rocket harts."""

    comptime device_kind = TargetDeviceKind.other("t1", True)
    comptime _handles_debug_assert: Bool = True
    comptime supports_float64: Bool = False
    comptime warp_size: Int = 1
    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = _t1_barrier
    comptime gpu_id_fn[name: StaticString, dim: StaticString]: OptionalReg[
        _GPUIdPluginHookFnType
    ] = _t1_gpu_id[name, dim] if (
        name == "thread_idx" or name == "block_idx" or name == "block_dim"
        or name == "grid_dim" or name == "lane_id"
    ) else None


comptime _target = __mlir_attr[
    `#kgen.target<triple = "riscv32-unknown-elf", `,
    `arch = "generic-rv32", `,
    `abi = "ilp32", `,
    `features = "+m,+a,+c,+f,+zve32f,+zvl1024b", `,
    `tune_cpu = "generic-rv32", `,
    `data_layout = "e-m:e-p:32:32-i64:64-n32-S128", `,
    `index_bit_width = 32, simd_bit_width = 1024, `,
    `opaque_plugin = `,
    TypeList.of[T1Plugin]().values,
    `> : !kgen.target`,
]

comptime T1 = TargetAccelerator[
    GPUInfo(
        name="T1",
        api="t1",
        arch_name="t1",
        compute=0,
        version="t1",
        sm_count=1,
        warp_size=1,
        threads_per_multiprocessor=4,
        shared_memory_per_multiprocessor=0,
        max_registers_per_block=0,
        max_thread_block_size=4,
    ),
    CompilationTarget[_mlir_value=_target](),
    ["t1"],
]
"""T1 thread-tilelang: RV32, Zve32f, per-thread VLEN 1024."""


struct T1Targets(TargetAcceleratorCollection):
    """Contributes the T1 target to a MAX overlay build."""

    comptime vendor_name: String = "T1"
    comptime encode_device_types_with_host_layout: Bool = True
    comptime RAW_TARGETS = TypeList.of[Trait=TargetAcceleratorType, T1]().values

    @staticmethod
    def normalize_target_arch(target_arch: StaticString) -> String:
        """Returns the canonical T1 target name."""
        return String(target_arch)
