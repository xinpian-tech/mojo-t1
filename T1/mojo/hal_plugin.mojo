"""Adapts the compatibility launch configuration to a T1 thread block."""

from std.collections.string.string_span import _get_kgen_string
from std.sys.info import CompilationTarget
from ._trait import HalPlugin, HalPluginHooks
from .._machine import DeviceRef
from ..execution_config.gpu_execution_config import GPUExecutionConfiguration


struct T1HalPlugin(HalPlugin):
    """Selects T1 launch hooks for the upstream HAL client."""

    comptime name = _get_kgen_string["t1"]()
    comptime HookFor[target: CompilationTarget] = T1HalHooks[target]


struct T1HalHooks[target: CompilationTarget](HalPluginHooks):
    """Uses the HAL grid configuration, restricted to one block by T1."""

    comptime ExecutionConfigType[device: DeviceRef] = GPUExecutionConfiguration
