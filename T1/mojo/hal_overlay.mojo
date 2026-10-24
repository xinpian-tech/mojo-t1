"""Registers T1's HAL hooks without changing the upstream source tree."""

from std._plugin.selector import PluginSelector
from ._trait import DefaultHalPlugin
from .cuda import CudaHalPlugin
from .hip import HipHalPlugin
from .metal import MetalHalPlugin
from .t1 import T1HalPlugin

comptime HAL_PLUGINS = PluginSelector[
    DefaultHalPlugin, MetalHalPlugin, CudaHalPlugin, HipHalPlugin, T1HalPlugin
]
