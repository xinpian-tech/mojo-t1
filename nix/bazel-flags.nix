{ lib, versions }:
rec {

  build = [
    "--curses=no"
    "--color=no"
    "--show_progress_rate_limit=30"
    "--experimental_convenience_symlinks=ignore"



    "--compilation_mode=opt"
    "--//:modular_config=production"
    "--//:release_type=production"
    "--//:modular_version_sha=${versions.rev}"
    "--nostamp"


    "--remote_cache="
    "--bes_backend="
    "--noremote_upload_local_results"
    "--disk_cache="





    "--repo_contents_cache="


    "--config=disable-mypy"
  ]
  ++ versions.bazelFlags;


  targets = [
    "//Mojo/tools/mojo:mojo-full"
    "//Mojo:CompilerRT"
    "//AsyncRT:RuntimeGlobals"
    "//Support:Globals"
    "//Mojo/stdlib/std"
    "@llvm-project//lld:lld"
    "@crashpad//:modular-crashpad-handler"
    "//Mojo:MojoLLDB"
    "//Mojo:MojoJupyter"
    "@llvm-project//lldb:lldb24.0.0git"
    "//Mojo:mojo-lldb"
    "//Mojo:gdb-server"
    "@llvm-project//lldb:lldb-dap"
    "@llvm-project//lldb:lldb-argdumper"
    "@llvm-project//llvm:llvm-symbolizer"
    "//Mojo/tools/mojo-lsp-server"
    "//Mojo/tools/mojo-repl-entry-point"
  ];
}
