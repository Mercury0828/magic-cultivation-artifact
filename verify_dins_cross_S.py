"""Across DIFFERENT insertion configurations: the ambiguity is {I, Zbar}.

Two facts, and they are not on the same footing.

CONTAINMENT is a rank fact, not something a sample can support. Any Z-type
operator with trivial X-check syndrome lies in ker(HX), and for a code with one
logical qubit dim ker(HX) - rank(HZ) = 1, so ker(HX) has exactly two classes
modulo the Z-stabilizers, represented by I and Zbar. This script prints the ranks
that make that so. No sampling is involved and no counterexample is possible.

NONTRIVIALITY is what the sample is for. The ambiguity group is not {I}: there
are genuinely distinct configurations S != S' carrying terms of equal syndrome
that differ by Zbar. Those are counted below, over a random sample.

Same-configuration pairs are excluded, since those are the subject of
verify_dins_standard_round.py. Exit code is nonzero on any counterexample.
"""
import itertools
import random
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotated_surface_code import build_rotated, rank2, nullspace2, validate

SEED = 7
N_CONFIGS = 120


def in_span(rows, r0, v):
    r, _ = rank2(np.vstack([rows, v[None, :]]))
    return r == r0


def run(d):
    random.seed(SEED)
    HX, HZ = validate(d, verbose=False)
    n = d * d
    rz, _ = rank2(HZ)
    ker = nullspace2(HX)
    kdim = ker.shape[0]

    zbar = None
    for bits in range(1, 1 << kdim):
        v = np.zeros(n, dtype=np.uint8)
        for i in range(kdim):
            if bits >> i & 1:
                v ^= ker[i]
        if not in_span(HZ, rz, v):
            zbar = v
            break
    HZL = np.vstack([HZ, zbar[None, :]])
    rzl, _ = rank2(HZL)

    print("d=%d  dim ker(HX)=%d  rank(HZ)=%d  rank<HZ,Zbar>=%d  (Zbar weight %d)"
          % (d, kdim, rz, rzl, int(zbar.sum())))
    print("      containment is forced: dim ker(HX) - rank(HZ) = %d, so ker(HX) has "
          "exactly two classes modulo the Z-stabilizers. Nothing else is possible."
          % (kdim - rz))

    subsets = []
    for size in range(1, d):
        for seed_q in range(n):
            for _ in range(40):
                blob = {seed_q}
                while len(blob) < size:
                    r, c = divmod(random.choice(sorted(blob)), d)
                    dr, dc = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < d and 0 <= nc < d):
                        break
                    blob.add(nr * d + nc)
                if len(blob) == size:
                    subsets.append(frozenset(blob))
    subsets = list(dict.fromkeys(subsets))
    random.shuffle(subsets)
    subsets = subsets[:N_CONFIGS]

    bysyn = {}
    for S in subsets:
        Sl = sorted(S)
        for k in range(len(Sl) + 1):
            for T in itertools.combinations(Sl, k):
                v = np.zeros(n, dtype=np.uint8)
                for q in T:
                    v[q] ^= 1
                bysyn.setdefault(tuple((HX @ v) % 2), []).append((S, v))

    stab = zb = other = 0
    fibres = 0
    for s, lst in bysyn.items():
        pairs = [(a, b) for a, b in itertools.combinations(lst, 2) if a[0] != b[0]]
        if not pairs:
            continue
        fibres += 1
        for (S0, v0), (S1, v1) in pairs:
            diff = v0 ^ v1
            if in_span(HZ, rz, diff):
                stab += 1
            elif in_span(HZL, rzl, diff):
                zb += 1
            else:
                other += 1
                print("  COUNTEREXAMPLE d=%d: %s vs %s" % (sorted(S0), sorted(S1)))
    print("      sample of %d configurations, %d syndrome fibres containing terms from "
          "two different configurations: %d pairs differ by a Z-stabilizer, %d by Zbar, "
          "%d by anything else" % (len(subsets), fibres, stab, zb, other))
    if zb:
        print("      the Zbar case OCCURS, so the ambiguity group is {I, Zbar} and not "
              "{I}: the following free operation has to absorb Zbar.")
    return other == 0 and zb > 0


if __name__ == "__main__":
    bad = 0
    for d in (3, 5):
        bad += (not run(d))
    print("NOTE: the nontriviality counts come from a random sample with seed %d and "
          "%d configurations per distance; the containment above does not." % (SEED, N_CONFIGS))
    sys.exit(1 if bad else 0)
