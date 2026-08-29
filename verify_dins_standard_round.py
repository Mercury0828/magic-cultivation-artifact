"""Within one insertion configuration: all surviving terms agree on the code space.

This is the step of prop:standard that USES the code distance. For an insertion
configuration whose components each have fewer than d cells and no two of which
are touched by a common check, any two subsets T, T' of the configuration with
the same X-check syndrome satisfy Z_T P = Z_T' P, i.e. T xor T' is a Z-stabilizer.

Separation is imposed here as "no X-check touches two components", which is what
the syndrome-splitting step of the proof needs. The paper's rule, that G_d joins
cells whose qubits lie within range 2*r_0, implies it: a range-r_0 check sits in a
radius-r_0 ball, so any two of its qubits are within 2*r_0 and would be joined.

This is a RANDOM SAMPLE, not an exhaustive check. Exit code is nonzero on any
counterexample.
"""
import itertools
import random
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotated_surface_code import build_rotated, rank2, validate

SEED = 20260829
TRIALS = 300


def in_rowspace(HZ, rz, v):
    r, _ = rank2(np.vstack([HZ, v[None, :]]))
    return r == rz


def checks_touching(HX, qubits):
    return {i for i, row in enumerate(HX) if any(row[q] for q in qubits)}


def run(d, trials=TRIALS):
    random.seed(SEED)
    HX, HZ = validate(d, verbose=False)
    n = d * d
    rz, _ = rank2(HZ)

    def syndrome(T):
        v = np.zeros(n, dtype=np.uint8)
        for q in T:
            v[q] ^= 1
        return tuple((HX @ v) % 2), v

    configs = 0
    fibres = 0
    for _ in range(trials):
        blobs = []
        for _ in range(random.randint(1, 3)):
            size = random.randint(1, d - 1)         # fewer than d cells
            for _attempt in range(200):
                blob = {random.randrange(n)}
                while len(blob) < size:
                    r, c = divmod(random.choice(sorted(blob)), d)
                    dr, dc = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < d and 0 <= nc < d):
                        break
                    blob.add(nr * d + nc)
                if len(blob) != size:
                    continue
                ch = checks_touching(HX, blob)
                if all(not (ch & checks_touching(HX, b)) for b in blobs):
                    blobs.append(blob)
                    break
        S = sorted(set().union(*blobs)) if blobs else []
        if not S or len(S) > 16:
            continue
        groups = {}
        for k in range(len(S) + 1):
            for T in itertools.combinations(S, k):
                s, v = syndrome(T)
                groups.setdefault(s, []).append(v)
        for s, vs in groups.items():
            if len(vs) < 2:
                continue
            fibres += 1
            for a, b in itertools.combinations(range(len(vs)), 2):
                if not in_rowspace(HZ, rz, vs[a] ^ vs[b]):
                    print("COUNTEREXAMPLE d=%d blobs=%s syndrome=%s" % (d, blobs, s))
                    return False, configs, fibres
        configs += 1
    print("d=%d: %d random separated configurations; %d syndrome fibres with more than "
          "one member; every pair in every fibre differs by a Z-stabilizer"
          % (d, configs, fibres))
    return True, configs, fibres


def tightness(d):
    """A component covering a minimum-weight Z-logical breaks the coset form."""
    HX, HZ = validate(d, verbose=False)
    n = d * d
    rz, _ = rank2(HZ)
    for c in range(d):
        col = [r * d + c for r in range(d)]
        v = np.zeros(n, dtype=np.uint8)
        v[col] = 1
        if ((HX @ v) % 2 == 0).all() and not in_rowspace(HZ, rz, v):
            print("d=%d: column %d is a weight-%d Z-logical, so within the single "
                  "configuration S = that column, T = S and T' = empty share syndrome 0 "
                  "and differ by Zbar. The coset form has no single L_E at component "
                  "size %d." % (d, c, d, d))
            return True
    print("d=%d: no axis-aligned Z-logical found" % d)
    return False


if __name__ == "__main__":
    bad = 0
    for d in (3, 5):
        ok, _, _ = run(d)
        bad += (not ok)
        bad += (not tightness(d))
    print("NOTE: random sample with seed %d, %d trials per distance; not exhaustive."
          % (SEED, TRIALS))
    sys.exit(1 if bad else 0)
