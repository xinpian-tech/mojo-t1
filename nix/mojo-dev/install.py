"""Install a relocatable compiler SDK from Bazel's configured CcInfo."""

import json
import pathlib
import shutil
import subprocess
import sys


def install(execroot, manifest, output, strip):
    data = json.loads(manifest.read_text())
    tree = output / "include"
    tree.mkdir(parents=True)
    library_paths = {library["path"] for library in data["libraries"]}

    def exported(path):
        relative = pathlib.PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Non-relocatable Bazel path: {path}")
        return (output / "lib" if path in library_paths else tree) / relative

    for source in data["files"]:
        destination = exported(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(execroot / source, destination)
        if source in library_paths:
            subprocess.run([str(strip), "--strip-debug", str(destination)], check=True)

    compile_flags = [
        "-std=c++20", "-fPIC", "-fno-exceptions", "-fno-rtti",
        "-fvisibility=hidden", "-fvisibility-inlines-hidden", "-DNDEBUG",
        "-U_DEBUG", "-U_GLIBCXX_ASSERTIONS", "-DLLVM_BUILD_STATIC",
        "-DMLIR_USE_FALLBACK_TYPE_IDS=1", "-DGOOGLE_PROTOBUF_NO_RTTI=1",
        "-I" + str(tree),
    ]
    for key, flag in [
        ("includes", "-I"), ("quote_includes", "-iquote"),
        ("system_includes", "-isystem"), ("framework_includes", "-F"),
        ("external_includes", "-isystem"),
    ]:
        for directory in data[key]:
            compile_flags.extend([flag, str(exported(directory))])
    compile_flags.extend("-D" + define for define in data["defines"])

    link_flags = ["-Wl,--start-group"]
    seen = set()
    for library in data["libraries"]:
        path = library["path"]
        if path in seen:
            continue
        seen.add(path)
        if library["whole"]:
            link_flags.append("-Wl,--whole-archive")
        link_flags.append(str(exported(path)))
        if library["whole"]:
            link_flags.append("-Wl,--no-whole-archive")
        if library["shared"]:
            link_flags.append("-Wl,-rpath," + str(exported(path).parent))
    link_flags.append("-Wl,--end-group")
    replacements = sorted(data["files"], key=len, reverse=True)
    for flag in data["link_flags"]:
        for path in replacements:
            if path in flag:
                flag = flag.replace(path, str(exported(path)))
                break
        link_flags.append(flag)

    metadata = output / "share/mojo-dev"
    metadata.mkdir(parents=True)
    for name, flags in [("compile", compile_flags), ("link", link_flags)]:
        (metadata / (name + ".rsp")).write_text(
            "\n".join(json.dumps(flag) for flag in flags) + "\n"
        )
    (metadata / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    install(*(pathlib.Path(argument) for argument in sys.argv[1:]))
