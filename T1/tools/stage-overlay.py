"""Stages upstream's HAL compatibility layer and T1 target registration."""

import argparse
import shutil
from pathlib import Path


def copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)
    destination.chmod(destination.stat().st_mode | 0o200)
    for directory in destination.rglob("*"):
        if directory.is_dir():
            directory.chmod(directory.stat().st_mode | 0o200)


def stage(upstream: Path, extension: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    copy_tree(upstream / "max/mojo/max", output / "max")
    driver = upstream / "max/driver/src/max_license"
    host = output / "max/gpu/host"
    copy_tree(driver / "_hal", host / "_hal")
    copy_tree(driver / "machine", host / "_hal/machine")
    compatibility = driver / "_device_context_hal"
    replacements = {
        compatibility / "__init__.mojo": host / "_legacy/device_context.mojo",
        compatibility / "_nvidia_cuda.mojo": host / "_legacy/_nvidia_cuda.mojo",
        compatibility / "_device_graph.mojo": host / "_legacy/device_graph.mojo",
        compatibility / "_machine.mojo": host / "_hal/_machine.mojo",
        extension / "mojo/max_overlay.mojo": output / "max/_plugin/_overlay.mojo",
        extension / "mojo/targets.mojo": output / "max/_plugin/t1.mojo",
        extension / "mojo/vme.mojo": output / "max/_plugin/t1_vme.mojo",
        extension / "mojo/hal_overlay.mojo": host / "_hal/_plugin/_overlay.mojo",
        extension / "mojo/hal_plugin.mojo": host / "_hal/_plugin/t1.mojo",
    }
    for source, destination in replacements.items():
        destination.unlink(missing_ok=True)
        shutil.copyfile(source, destination)
    for name in ("_device_context_extras", "_device_context_metal", "_metal"):
        (host / f"_legacy/{name}.mojo").unlink()
    for build_file in output.rglob("BUILD.bazel"):
        build_file.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    stage(arguments.upstream, arguments.extension, arguments.output)
