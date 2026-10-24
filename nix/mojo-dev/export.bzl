load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _export_cc_sdk_impl(ctx):
    info = cc_common.merge_cc_infos(cc_infos = [dep[CcInfo] for dep in ctx.attr.deps])
    compilation = info.compilation_context
    files = compilation.headers.to_list()
    libraries = []
    flags = []
    for linker_input in info.linking_context.linker_inputs.to_list():
        flags.extend(linker_input.user_link_flags)
        files.extend(linker_input.additional_inputs)
        for library in linker_input.libraries:
            archive = library.pic_static_library or library.static_library
            dynamic = library.resolved_symlink_dynamic_library or library.dynamic_library
            if archive:
                files.append(archive)
                libraries.append({"path": archive.path, "whole": library.alwayslink, "shared": False})
            elif dynamic:
                files.append(dynamic)
                libraries.append({"path": dynamic.path, "whole": False, "shared": True})
            else:
                objects = library.pic_objects or library.objects
                files.extend(objects)
                libraries.extend([{"path": obj.path, "whole": False, "shared": False} for obj in objects])
    manifest = ctx.actions.declare_file(ctx.label.name + ".json")
    ctx.actions.write(manifest, json.encode({
        "files": [file.path for file in depset(files).to_list()],
        "includes": compilation.includes.to_list(),
        "quote_includes": compilation.quote_includes.to_list(),
        "system_includes": compilation.system_includes.to_list(),
        "external_includes": compilation.external_includes.to_list(),
        "framework_includes": compilation.framework_includes.to_list(),
        "defines": compilation.defines.to_list(),
        "libraries": libraries,
        "link_flags": flags,
    }))
    return [DefaultInfo(files = depset([manifest] + files))]

export_cc_sdk = rule(
    implementation = _export_cc_sdk_impl,
    attrs = {"deps": attr.label_list(providers = [CcInfo])},
)
