project('mojo-t1', version: '0.1.0', meson_version: '>=1.3.0',
  default_options: ['buildtype=release', 'libdir=lib'])

mojo_dev = get_option('mojo_dev')
if mojo_dev == ''
  error('Set -Dmojo_dev to the matching Mojo development package')
endif

if get_option('compiler')
  add_languages('cpp', native: false)
  subdir('T1/compiler')
endif

if get_option('runtime')
  add_languages('c', native: false)
  subdir('T1/runtime')
endif
