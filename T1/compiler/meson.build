shared_library('T1Compiler',
  ['T1Backend.cpp', 'T1Traits.cpp'],
  cpp_args: ['@' + mojo_dev / 'share/mojo-dev/compile.rsp'],
  link_args: [
    '@' + mojo_dev / 'share/mojo-dev/link.rsp',
    '-fuse-ld=lld', '-Wl,-z,defs', '-Wl,-z,nodelete',
    '-Wl,--disable-new-dtags', '-Wl,--gc-sections',
  ],
  install: true)
