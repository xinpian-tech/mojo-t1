#!/usr/bin/env python3
"""Non-invasive microarchitectural analysis from a T1 RTL retirement trace."""
import re, sys, collections

def vclass(inst):
    op = inst & 0x7F
    if op == 0x57:
        f3 = (inst >> 12) & 7
        if f3 == 7:
            return 'vsetvli'
        f6 = (inst >> 26) & 0x3F
        names = {0x24: 'vfmul', 0x2C: 'vfmacc.OPFVV?', 0x17: 'vmv/vmerge',
                 0x01: 'vfredusum', 0x00: 'vfadd', 0x02: 'vfsub',
                 0x0E: 'vslideup', 0x0F: 'vslidedown', 0x10: 'vmv.s/x'}
        # distinguish by funct6 for OPFVV(f3=1)/OPIVI(f3=3)...
        if f3 == 1:  # OPFVV
            return {0x00: 'vfadd', 0x01: 'vfredusum', 0x08: 'vfmul?',
                    0x10: 'vfmv/wr', 0x24: 'vfmul', 0x2C: 'vfmacc'}.get(f6, f'opfvv{f6:#x}')
        if f3 == 5:  # OPFVF
            return {0x24: 'vfmul.vf', 0x2C: 'vfmacc.vf'}.get(f6, f'opfvf{f6:#x}')
        if f3 == 3:  # OPIVI
            return {0x0E: 'vslideup', 0x17: 'vmv.v.i'}.get(f6, f'opivi{f6:#x}')
        if f3 == 6:  # OPMVX
            return {0x10: 'vmv.s.x'}.get(f6, f'opmvx{f6:#x}')
        if f3 == 4:
            return f'opivx{f6:#x}'
        return f'opv_f3{f3}_f6{f6:#x}'
    if op == 0x07:
        return 'vlseg' if ((inst >> 29) & 7) else 'vle32'
    if op == 0x27:
        return 'vsseg' if ((inst >> 29) & 7) else 'vse32'
    return f'op{op:#x}'

ISSUE = re.compile(r'T=\s*(\d+), T1Issue\(inst=0x([0-9a-f]+), vtype=0x([0-9a-f]+), vl=0x([0-9a-f]+), .*t1_tag=(\d+), t1_seq=\s*(\d+)')
RETIRE = re.compile(r'T=\s*(\d+), T1Retire\(t1_tag=(\d+),.*t1_seq=\s*(\d+)')
SCALAR = re.compile(r'T=\s*(\d+), (?:Retire|RetireX)\(pc=0x([0-9a-f]+)')

def main(path, t_lo=0, t_hi=1 << 62):
    issues = {}   # seq -> (t, inst, vtype, vl, tag)
    recs = []     # (t_issue, t_retire, inst, vtype, vl)
    scalar_ts = []
    with open(path) as f:
        for line in f:
            m = ISSUE.match(line)
            if m:
                t, inst, vt, vl, tag, seq = (int(m.group(1)), int(m.group(2), 16),
                    int(m.group(3), 16), int(m.group(4), 16), int(m.group(5)), int(m.group(6)))
                if t_lo <= t <= t_hi:
                    issues[seq] = (t, inst, vt, vl)
                continue
            m = RETIRE.match(line)
            if m:
                t, tag, seq = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if seq in issues:
                    ti, inst, vt, vl = issues.pop(seq)
                    recs.append((ti, t, inst, vt, vl))
                continue
            m = SCALAR.match(line)
            if m:
                t = int(m.group(1))
                if t_lo <= t <= t_hi:
                    scalar_ts.append((t, int(m.group(2), 16)))
    recs.sort()
    print(f"vector instructions: {len(recs):,}   scalar retires: {len(scalar_ts):,}")

    # occupancy and issue-gap by class
    by = collections.defaultdict(list)
    gaps = collections.defaultdict(list)
    prev_t = None
    for i, (ti, tr, inst, vt, vl) in enumerate(recs):
        c = vclass(inst)
        lmul = {0: 'm1', 1: 'm2', 2: 'm4', 3: 'm8'}.get(vt & 7, '?')
        by[(c, lmul, vl)].append(tr - ti)
        if prev_t is not None:
            gaps[c].append(ti - prev_t)
        prev_t = ti
    print("\nper-class occupancy (issue->retire cycles) and issue-to-issue gap:")
    print(f"{'class':<12}{'lmul':<5}{'vl':>5}{'n':>9}{'occ p50':>9}{'occ p90':>9}{'gap p50':>9}")
    import statistics
    for (c, lmul, vl), occ in sorted(by.items(), key=lambda kv: -len(kv[1]))[:14]:
        g = gaps.get(c, [0])
        print(f"{c:<12}{lmul:<5}{vl:>5}{len(occ):>9,}{int(statistics.median(occ)):>9}"
              f"{int(sorted(occ)[int(len(occ)*0.9)]):>9}{int(statistics.median(g)):>9}")

    # concurrency: how many vector instructions in flight, cycle-weighted
    events = []
    for ti, tr, *_ in recs:
        events += [(ti, 1), (tr, -1)]
    events.sort()
    conc = collections.Counter()
    cur, last_t = 0, None
    for t, d in events:
        if last_t is not None and t > last_t:
            conc[cur] += t - last_t
        cur += d
        last_t = t
    total_t = sum(conc.values())
    print("\nvector instructions in flight (cycle-weighted):")
    for k in sorted(conc):
        print(f"  {k}: {100*conc[k]/total_t:5.1f}%")

    # scalar-vector overlap: scalar retires while >=1 vector in flight
    iv = sorted((ti, tr) for ti, tr, *_ in recs)
    import bisect
    starts = [a for a, b in iv]
    ends_max = []
    mx = 0
    for a, b in iv:
        mx = max(mx, b)
        ends_max.append(mx)
    inflight = 0
    for t, pc in scalar_ts:
        i = bisect.bisect_right(starts, t) - 1
        if i >= 0 and ends_max[i] > t:
            inflight += 1
    if scalar_ts:
        print(f"\nscalar retires while vector in flight: {inflight:,}/{len(scalar_ts):,}"
              f" ({100*inflight/len(scalar_ts):.1f}%)")

if __name__ == '__main__':
    main(sys.argv[1], *(int(x) for x in sys.argv[2:]))
