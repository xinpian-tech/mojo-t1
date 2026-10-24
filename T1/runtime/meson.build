hal_args = ['-I' + mojo_dev / 'include', '-Wall', '-fvisibility=hidden']
driver = shared_library('T1Driver', 'driver.c',
  c_args: hal_args,
  override_options: ['c_std=gnu23', 'optimization=2'],
  install: true)

executable('t1-hal-smoke', 'hal_smoke.c',
  c_args: hal_args,
  link_with: driver,
  install_rpath: get_option('prefix') / get_option('libdir'),
  install: true)

test('hal-abi', find_program('python3'),
  args: [files('test-hal-abi.py'), driver.full_path(),
    mojo_dev / 'include/HAL/M_driver_hal_symbols.inc'],
  depends: driver)

rv_clang = find_program(get_option('rv_clang'))
rv_lld = find_program(get_option('rv_lld'))
rv_objcopy = find_program(get_option('rv_objcopy'))
rv_flags = ['--target=riscv32-unknown-elf',
  '-march=rv32imafc_zve32f_zvl2048b', '-mabi=ilp32']

trampoline_object = custom_target('trampoline-object',
  input: 'trampoline.S', output: 'trampoline.o',
  command: [rv_clang, rv_flags, '-c', '@INPUT@', '-o', '@OUTPUT@'])
trampoline_elf = custom_target('trampoline-elf',
  input: trampoline_object, output: 'trampoline.elf',
  command: [rv_lld, '-m', 'elf32lriscv', '-nostdlib', '-static', '-n',
    '-Ttext=0x80000000', '--entry=_start', '@INPUT@', '-o', '@OUTPUT@'])
custom_target('trampoline-bin',
  input: trampoline_elf, output: 'trampoline.bin',
  command: [rv_objcopy, '-O', 'binary', '@INPUT@', '@OUTPUT@'],
  install: true, install_dir: get_option('datadir') / 't1')

saxpy_object = custom_target('saxpy-object',
  input: 'test_saxpy.c', output: 'saxpy.o',
  command: [rv_clang, rv_flags, '-O3', '-ffreestanding', '-c', '@INPUT@', '-o', '@OUTPUT@'])
custom_target('saxpy-elf',
  input: saxpy_object, output: 'test_saxpy.elf',
  command: [rv_lld, '-m', 'elf32lriscv', '-static', '-nostdlib', '-n',
    '--image-base=0x80100000', '--entry=saxpy', '@INPUT@', '-o', '@OUTPUT@'],
  install: true, install_dir: get_option('datadir') / 't1')

custom_target('t1-builtins-object',
  input: 't1_builtins.S', output: 't1_builtins.o',
  command: [rv_clang, rv_flags, '-c', '@INPUT@', '-o', '@OUTPUT@'],
  build_by_default: true)
