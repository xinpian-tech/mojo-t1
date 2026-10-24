"""Regression tests for development SDK installation and response files."""

import importlib.util
import json
import pathlib
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("installer", sys.argv.pop(1))
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class ExportTests(unittest.TestCase):
    def test_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            execroot = root / "execroot"
            output = root / "sdk with spaces"
            header = "external/llvm/include/llvm/Config.h"
            library = "bazel-out/bin/libCompiler.a"
            shared = "bazel-out/bin/libGlobals.so"
            for name in [header, library, shared]:
                source = execroot / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(name)
            manifest = root / "manifest.json"
            data = {
                "files": [header, library, shared],
                "includes": [], "quote_includes": ["."],
                "system_includes": [], "framework_includes": [],
                "external_includes": ["external/llvm/include"],
                "defines": ['VERSION="test version"'],
                "libraries": [
                    {"path": library, "whole": True, "shared": False},
                    {"path": library, "whole": True, "shared": False},
                    {"path": shared, "whole": False, "shared": True},
                ],
                "link_flags": ["-pthread"],
            }
            manifest.write_text(json.dumps(data))
            with patch.object(installer.subprocess, "run") as strip:
                installer.install(execroot, manifest, output, pathlib.Path("llvm-strip"))
            self.assertEqual(strip.call_count, 2)
            self.assertEqual((output / "include" / header).read_text(), header)
            self.assertEqual((output / "lib" / library).read_text(), library)
            metadata = output / "share/mojo-dev"
            compile_flags = shlex.split((metadata / "compile.rsp").read_text())
            link_flags = shlex.split((metadata / "link.rsp").read_text())
            self.assertIn('-DVERSION="test version"', compile_flags)
            self.assertIn(str(output / "include/external/llvm/include"), compile_flags)
            self.assertEqual(link_flags.count(str(output / "lib" / library)), 1)
            self.assertIn("-Wl,--whole-archive", link_flags)
            self.assertIn("-Wl,-rpath," + str((output / "lib" / shared).parent), link_flags)
            data["files"] = ["../escape.h"]
            manifest.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                installer.install(execroot, manifest, root / "invalid", pathlib.Path("llvm-strip"))


if __name__ == "__main__":
    unittest.main()
