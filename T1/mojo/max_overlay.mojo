"""Registers T1 alongside upstream's accelerator collections."""

from std._plugin.selector import PluginSelector
from std._gpu.host.info import TargetAcceleratorCollection
from ._trait import DefaultTargets, MaxPlugin
from .cuda.targets import CudaTargets
from .hip.targets import HipTargets
from .metal.targets import MetalTargets
from .t1 import T1Targets

comptime MAX_PLUGINS = PluginSelector[Trait=MaxPlugin]
comptime TARGETS = TypeList.of[
    Trait=TargetAcceleratorCollection,
    DefaultTargets,
    CudaTargets,
    HipTargets,
    MetalTargets,
    T1Targets,
]()
