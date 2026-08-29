"""prop:family-magic: what the split-readout family's accepted effect is, and when
it carries magic.

Containing a Z-logical support is NECESSARY for beta != 0 and is not sufficient for
magic. A bare logical string at theta = pi/4 has |alpha| = |beta| and lands back on
the stabilizer boundary; that case is one of the tests below.

Checks, by direct 2^n state-vector simulation on the rotated surface code at d=3:

  (1) Pi_0 U P = (alpha I + beta Zbar) P on the code space, with alpha the sum of
      c_T over T with Z_T a stabilizer and beta the sum over T with Z_T logical;
  (2) F_A = (alpha I + beta Zbar)^dag M^X_+ (alpha I + beta Zbar) matches the
      simulated accepted effect;
  (3) the Bloch form eq:family-bloch;
  (4) magic > 0 when alpha*beta != 0 AND |alpha| != |beta|, and beta = 0 whenever
      the marked set contains no Z-logical support. Containing a logical support is
      necessary for beta != 0, not sufficient for magic: a bare logical string at
      theta = pi/4 has |alpha| = |beta| and lands back on the stabilizer boundary.

Exits nonzero on any mismatch.
"""
import itertools
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotated_surface_code import build_rotated, rank2, nullspace2, validate

I2 = np.eye(2, dtype=complex)
Xm = np.array([[0, 1], [1, 0]], dtype=complex)
Zm = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_op(n, ops):
    """ops: dict qubit -> 2x2; identity elsewhere"""
    M = np.array([[1]], dtype=complex)
    for q in range(n):
        M = np.kron(M, ops.get(q, I2))
    return M


def pauli_z_set(n, T):
    return kron_op(n, {q: Zm for q in T})


def pauli_x_set(n, T):
    return kron_op(n, {q: Xm for q in T})


def code_basis(HX, HZ, d):
    """two orthonormal code states, plus Xbar, Zbar as matrices"""
    n = d * d
    dim = 1 << n
    # projector onto the joint +1 eigenspace, built as a product of check projectors
    P = np.eye(dim, dtype=complex)
    for row in HZ:                       # Z-type checks
        g = pauli_z_set(n, np.flatnonzero(row))
        P = P @ ((np.eye(dim, dtype=complex) + g) / 2)
    for row in HX:                       # X-type checks
        g = pauli_x_set(n, np.flatnonzero(row))
        P = P @ ((np.eye(dim, dtype=complex) + g) / 2)
    # orthonormal basis of the range
    w, v = np.linalg.eigh((P + P.conj().T) / 2)
    cols = v[:, w > 0.5]
    assert cols.shape[1] == 2, "code space dim %d" % cols.shape[1]
    return cols, P


def logical_ops(HX, HZ, d, V):
    """Zbar, Xbar restricted to the code space, from minimum-weight representatives"""
    n = d * d
    rz, _ = rank2(HZ)
    rx, _ = rank2(HX)
    ker_x = nullspace2(HX)               # Z-type normalizer
    ker_z = nullspace2(HZ)               # X-type normalizer

    def pick(ker, Hsame, r0, kind):
        best = None
        for bits in range(1, 1 << ker.shape[0]):
            u = np.zeros(n, dtype=np.uint8)
            for i in range(ker.shape[0]):
                if bits >> i & 1:
                    u ^= ker[i]
            r, _ = rank2(np.vstack([Hsame, u[None, :]]))
            if r > r0 and (best is None or u.sum() < best.sum()):
                best = u
        return best

    zsup = pick(ker_x, HZ, rz, 'Z')
    xsup = pick(ker_z, HX, rx, 'X')
    Zbar = V.conj().T @ pauli_z_set(n, np.flatnonzero(zsup)) @ V
    Xbar = V.conj().T @ pauli_x_set(n, np.flatnonzero(xsup)) @ V
    return Zbar, Xbar, zsup, xsup


def syndrome_of_Z(HX, T, n):
    u = np.zeros(n, dtype=np.uint8)
    for q in T:
        u[q] ^= 1
    return tuple((HX @ u) % 2)


def dstab(F):
    a = np.real(np.trace(F)) / 2
    if a <= 1e-14:
        return 0.0, 0.0
    v = np.array([np.real(np.trace(F @ Q)) / 2 for Q in PAULIS])
    return a, max(0.0, np.abs(v).sum() / a - 1)


d = 3
HX, HZ = validate(d, verbose=False)
n = d * d
V, Pfull = code_basis(HX, HZ, d)
Zbar, Xbar, zsup, xsup = logical_ops(HX, HZ, d, V)
Ybar = 1j * Xbar @ Zbar
PAULIS = (Xbar, Ybar, Zbar)
print("d=3 code space built; Zbar support %s (weight %d), Xbar support %s"
      % (np.flatnonzero(zsup).tolist(), int(zsup.sum()), np.flatnonzero(xsup).tolist()))

rz, _ = rank2(HZ)
FAIL = False
rng = np.random.default_rng(11)

cases = []
cases.append(("logical column (contains a Z-logical)", sorted(np.flatnonzero(zsup).tolist())))
cases.append(("logical column plus one extra qubit",
              sorted(set(np.flatnonzero(zsup).tolist()) | {1})))
cases.append(("two qubits only (no Z-logical inside)", [0, 1]))
cases.append(("all nine qubits", list(range(n))))

EDGE = "bare logical string at pi/4 (|alpha| = |beta|)"
cases.append((EDGE, sorted(np.flatnonzero(zsup).tolist())))

for name, S in cases:
    if name == EDGE:
        thetas = {q: float(np.pi / 4) for q in S}
    else:
        thetas = {q: float(rng.uniform(0.1, 0.5)) for q in S}
    # --- combinatorial alpha, beta ---
    alpha = 0j
    beta = 0j
    for k in range(len(S) + 1):
        for T in itertools.combinations(S, k):
            if syndrome_of_Z(HX, T, n) != tuple([0] * HX.shape[0]):
                continue
            c = 1.0 + 0j
            for q in S:
                c *= (1j * np.sin(thetas[q])) if q in T else np.cos(thetas[q])
            u = np.zeros(n, dtype=np.uint8)
            for q in T:
                u[q] ^= 1
            r, _ = rank2(np.vstack([HZ, u[None, :]]))
            if r == rz:
                alpha += c
            else:
                beta += c

    # --- direct simulation ---
    dim = 1 << n
    U = np.eye(dim, dtype=complex)
    for q in S:
        th = thetas[q]
        U = U @ (np.cos(th) * np.eye(dim, dtype=complex)
                 + 1j * np.sin(th) * pauli_z_set(n, [q]))
    # zero-syndrome projector = the full code projector's Z-check part is already
    # satisfied; Pi_0 for Z errors is the product of X-check projectors
    Pi0 = np.eye(dim, dtype=complex)
    for row in HX:
        g = pauli_x_set(n, np.flatnonzero(row))
        Pi0 = Pi0 @ ((np.eye(dim, dtype=complex) + g) / 2)
    Blog = V.conj().T @ (Pi0 @ U) @ V                     # on the code space
    Bpred = alpha * np.eye(2, dtype=complex) + beta * Zbar
    e1 = np.abs(Blog - Bpred).max()

    Mx = (np.eye(2, dtype=complex) + Xbar) / 2
    FA = Bpred.conj().T @ Mx @ Bpred
    FAsim = Blog.conj().T @ Mx @ Blog
    e2 = np.abs(FA - FAsim).max()

    a, b = np.abs(alpha) ** 2, np.abs(beta) ** 2
    w = np.conj(alpha) * beta
    Fbloch = 0.5 * ((a + b) * np.eye(2, dtype=complex) + (a - b) * Xbar
                    + 2 * np.imag(w) * Ybar + 2 * np.real(w) * Zbar)
    e3 = np.abs(FA - Fbloch).max()

    pacc, Ds = dstab(FA)
    ok = (e1 < 1e-9) and (e2 < 1e-9) and (e3 < 1e-9)
    expect_pos = (abs(alpha) > 1e-12 and abs(beta) > 1e-12
                  and abs(abs(alpha) - abs(beta)) > 1e-9)
    ok_sign = (Ds > 1e-12) if expect_pos else True
    if name == EDGE:
        # the proposition claims nothing here, and the paper says the effect
        # returns to the stabilizer boundary: check that it does
        ok_sign = (abs(abs(alpha) - abs(beta)) < 1e-12) and (Ds < 1e-12)
    if not (ok and ok_sign):
        FAIL = True
    print("  %-38s |alpha|=%.4f |beta|=%.3e  p_acc=%.4f  Dstab=%.3e  "
          "err(Pi0UP,F_A,Bloch)=(%.1e,%.1e,%.1e)  %s"
          % (name, abs(alpha), abs(beta), pacc, Ds, e1, e2, e3,
             "ok" if (ok and ok_sign) else "MISMATCH"))


# ---------------------------------------------------------------- genericity
# The proposition also claims that once the marked set contains a logical support,
# the three failure conditions alpha = 0, beta = 0, |alpha| = |beta| confine the
# angles to a measure-zero set. Sample and count how often magic actually fails.
print()
print("genericity: random angles on a marked set containing a logical support")
S = sorted(np.flatnonzero(zsup).tolist())
rng2 = np.random.default_rng(2026)
TRIALS = 3000
zero = 0
worst = np.inf
for _ in range(TRIALS):
    thetas = {q: float(rng2.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)) for q in S}
    alpha = 0j
    beta = 0j
    for k in range(len(S) + 1):
        for T in itertools.combinations(S, k):
            if syndrome_of_Z(HX, T, n) != tuple([0] * HX.shape[0]):
                continue
            c = 1.0 + 0j
            for q in S:
                c *= (1j * np.sin(thetas[q])) if q in T else np.cos(thetas[q])
            u = np.zeros(n, dtype=np.uint8)
            for q in T:
                u[q] ^= 1
            r, _ = rank2(np.vstack([HZ, u[None, :]]))
            if r == rz:
                alpha += c
            else:
                beta += c
    A = alpha * np.eye(2, dtype=complex) + beta * Zbar
    FA = A.conj().T @ ((np.eye(2, dtype=complex) + Xbar) / 2) @ A
    _, ds = dstab(FA)
    worst = min(worst, ds)
    if ds <= 1e-12:
        zero += 1
print("   %d random angle vectors, %d with zero magic, smallest magic %.3e"
      % (TRIALS, zero, worst))
if zero:
    print("   MISMATCH: the failure set is supposed to have measure zero")
    FAIL = True
else:
    print("   none of the sampled angles fell in the failure set, as the measure-zero")
    print("   claim predicts; the pi/4 case above shows the set is nonempty")

print()
print("NOTE: direct 2^9 state-vector simulation at d=3 only.")
sys.exit(1 if FAIL else 0)
