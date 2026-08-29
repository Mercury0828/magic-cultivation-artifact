"""Rotated surface code: build it, and ASSERT its parameters.

Self-validation is by assertion, not by printing: checks commute, k = 1, and
d_X = d_Z = d are all asserted, so a wrong lattice fails loudly. Only d = 3 and
d = 5 are validated; the construction is not verified for the general odd-d family.
"""
import numpy as np


def rank2(M):
    M = M.copy() % 2
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r, M


def nullspace2(M):
    """basis of {x : M x = 0} over F2, M is m x n"""
    m, n = M.shape
    A = M.copy() % 2
    pivots = []
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if A[i, c]:
                piv = i
                break
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        pivots.append(c)
        r += 1
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for i, c in enumerate(pivots):
            v[c] = A[i, f]
        basis.append(v)
    return np.array(basis, dtype=np.uint8) if basis else np.zeros((0, n), dtype=np.uint8)


def build_rotated(d):
    """d x d data qubits. Returns (HX, HZ) check matrices over F2."""
    def q(r, c):
        return r * d + c

    n = d * d
    HX, HZ = [], []

    for r in range(d - 1):                       # bulk faces
        for c in range(d - 1):
            sup = [q(r, c), q(r, c + 1), q(r + 1, c), q(r + 1, c + 1)]
            row = np.zeros(n, dtype=np.uint8)
            row[sup] = 1
            (HZ if (r + c) % 2 == 0 else HX).append(row)

    for c in range(d - 1):                       # weight-2 boundary checks
        if c % 2 == 1:
            row = np.zeros(n, dtype=np.uint8)
            row[[q(0, c), q(0, c + 1)]] = 1
            HZ.append(row)
    for c in range(d - 1):
        if c % 2 == 0:
            row = np.zeros(n, dtype=np.uint8)
            row[[q(d - 1, c), q(d - 1, c + 1)]] = 1
            HZ.append(row)
    for r in range(d - 1):
        if r % 2 == 0:
            row = np.zeros(n, dtype=np.uint8)
            row[[q(r, 0), q(r + 1, 0)]] = 1
            HX.append(row)
    for r in range(d - 1):
        if r % 2 == 1:
            row = np.zeros(n, dtype=np.uint8)
            row[[q(r, d - 1), q(r + 1, d - 1)]] = 1
            HX.append(row)

    return np.array(HX, dtype=np.uint8), np.array(HZ, dtype=np.uint8)


def min_weight_logical(Hdetect, Hsame):
    """min weight over ker(Hdetect) minus rowspace(Hsame)"""
    n = Hdetect.shape[1]
    ker = nullspace2(Hdetect)
    r0, _ = rank2(Hsame)
    assert ker.shape[0] <= 20, "nullity too large to enumerate"
    best = None
    for bits in range(1, 1 << ker.shape[0]):
        v = np.zeros(n, dtype=np.uint8)
        for i in range(ker.shape[0]):
            if bits >> i & 1:
                v ^= ker[i]
        w = int(v.sum())
        if best is not None and w >= best:
            continue
        r, _ = rank2(np.vstack([Hsame, v[None, :]]))
        if r > r0:
            best = w
    return best


def validate(d, verbose=True):
    HX, HZ = build_rotated(d)
    n = d * d
    rx, _ = rank2(HX)
    rz, _ = rank2(HZ)
    assert ((HX @ HZ.T) % 2 == 0).all(), "checks do not commute at d=%d" % d
    k = n - rx - rz
    assert k == 1, "k=%d at d=%d" % (k, d)
    dz = min_weight_logical(HX, HZ)      # Z-type logicals, detected by the X checks
    dx = min_weight_logical(HZ, HX)      # X-type logicals, detected by the Z checks
    assert dz == d, "d_Z=%s at d=%d" % (dz, d)
    assert dx == d, "d_X=%s at d=%d" % (dx, d)
    if verbose:
        print("d=%d validated: [[%d,1,%d]] with d_X=%d, d_Z=%d, checks commute"
              % (d, n, d, dx, dz))
    return HX, HZ


if __name__ == "__main__":
    for d in (3, 5):
        validate(d)
