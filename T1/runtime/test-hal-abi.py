"""Check the driver against the matching upstream HAL symbol list."""

import ctypes
import pathlib
import re
import sys

library = ctypes.CDLL(str(pathlib.Path(sys.argv[1]).resolve()))
symbols = pathlib.Path(sys.argv[2]).read_text()
for name in re.findall(r"M_DRIVER_HAL_SYMBOL\((\w+)\)", symbols):
    getattr(library, "M_driver_" + name)
try:
    library.AsyncRT_DeviceContext_create
except AttributeError:
    pass
else:
    raise AssertionError("legacy AsyncRT ABI must not be exported")
