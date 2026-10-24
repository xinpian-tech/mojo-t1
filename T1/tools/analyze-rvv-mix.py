#!/usr/bin/env python3
"""Dynamic RVV-vs-scalar analysis from a pokedex pc histogram."""
import subprocess, sys, re, collections

OBJDUMP = '/nix/store/86ry0rw06r1vpvqi49c7mq03pgznsadi-bazel-repo-clang-22.1.8/bin/llvm-objdump'

def is_vector(word):
    if word & 3 != 3:
        return False  # compressed: no RVV
    op = word & 0x7F
    if op == 0x57:
        return True   # OP-V (arith + vsetvli/vsetivli)
    if op in (0x07, 0x27):
        width = (word >> 12) & 7
        return width in (0, 5, 6, 7)  # vector load/store element widths
    return False

def load_symbols(elf):
    """(start, end, name) function ranges + addr->mnemonic from objdump."""
    out = subprocess.run([OBJDUMP, '-d', elf], capture_output=True, text=True).stdout
    funcs, insts = [], {}
    cur = None
    for line in out.splitlines():
        m = re.match(r'^([0-9a-f]+) <(.+)>:$', line)
        if m:
            cur = m.group(2)
            funcs.append([int(m.group(1), 16), None, cur])
            continue
        m = re.match(r'^\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)\s*(.*)$', line)
        if m and cur:
            insts[int(m.group(1), 16)] = (m.group(2), m.group(3), cur)
    for i in range(len(funcs) - 1):
        funcs[i][1] = funcs[i + 1][0]
    if funcs:
        funcs[-1][1] = 1 << 63
    return funcs, insts

def main(hist_path, elfs):
    insts = {}
    for e in elfs:
        _, i = load_symbols(e)
        insts.update(i)
    rows = []
    with open(hist_path) as f:
        next(f)
        for line in f:
            pc, count, word = line.strip().split(',')
            rows.append((int(pc, 16), int(count), int(word, 16)))
    tot = sum(c for _, c, _ in rows)
    vec = sum(c for _, c, w in rows if is_vector(w))
    tramp = sum(c for pc, c, _ in rows if pc < 0x80100000)
    print(f"executed: {tot}  RVV: {vec} ({100*vec/tot:.1f}%)  "
          f"scalar: {tot-vec} ({100*(tot-vec)/tot:.1f}%)  "
          f"[trampoline+runtime: {tramp} ({100*tramp/tot:.2f}%)]")
    # scalar breakdown by (function, mnemonic)
    sc = collections.Counter()
    for pc, c, w in rows:
        if is_vector(w):
            continue
        mn, ops, fn = insts.get(pc, ('?', '', 'trampoline' if pc < 0x80100000 else '?'))
        sc[(fn.split('(')[0][:48], mn)] += c
    print("\ntop scalar (function, mnemonic, count, share):")
    for (fn, mn), c in sc.most_common(24):
        print(f"  {c:>12,}  {100*c/tot:5.2f}%  {mn:<10} {fn}")
    # vector breakdown
    vc = collections.Counter()
    for pc, c, w in rows:
        if is_vector(w):
            mn = insts.get(pc, ('?',))[0]
            vc[mn] += c
    print("\nvector mix:", dict(vc.most_common(10)))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
