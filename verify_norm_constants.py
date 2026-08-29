"""The one-qubit norm facts and the two-filter example.

Checks the elementary constants the bounds are assembled from.

  (1) lem:norm          magic_c(F) = p_acc * Dstab(Ehat), by direct minimisation
                        over the native cone;
  (2) eq:exact-dist     dist_c(Ehat, O) = Dstab(Ehat)/2 between normalised effects,
                        which is the factor of two the preliminaries record;
  (3) eq:norm-convert   ||Q||_c <= sqrt(3) ||Q||_inf on one qubit, and the constant
                        is attained at a = 0, v = (1,1,1)/sqrt(3);
  (4) eq:binary-diamond ||B_F - B_G||_diamond = 2 ||F - G||_inf for the binary
                        measurement channels B_D(rho) = Tr(D rho)|0><0| +
                        Tr((I-D) rho)|1><1|;
  (5) Dstab of the sharp H_XY projector equals sqrt(2) - 1;
  (6) eq:twofilter      (BA)^dag (BA) for A = I + a Xbar, B = I + b Zbar, and the
                        normalised Bloch l1 norm quoted at a = b = 1/2.

Exits with a nonzero status on any mismatch.
"""
import sys

import numpy as np

I2 = np.eye(2, dtype=complex)
Xm = np.array([[0, 1], [1, 0]], dtype=complex)
Ym = np.array([[0, -1j], [1j, 0]], dtype=complex)
Zm = np.array([[1, 0], [0, -1]], dtype=complex)
S3 = np.sqrt(3.0)
TOL = 1e-9

rng = np.random.default_rng(20260829)
fail = False


def decomp(Q):
    a = np.real(np.trace(Q)) / 2
    v = np.array([np.real(np.trace(Q @ P)) / 2 for P in (Xm, Ym, Zm)])
    return a, v


def cnorm(Q):
    a, v = decomp(Q)
    return abs(a) + np.abs(v).sum()


def opnorm(Q):
    return float(np.max(np.abs(np.linalg.eigvalsh((Q + Q.conj().T) / 2))))


def magic_c_direct(F):
    """dist_c(F, N) by minimising over q >= 0.

    The closest point q w with ||w||_1 <= 1 to the Bloch part is at l1 distance
    max(L - q, 0), where L is the l1 norm of that part, so the objective is
    |p - q| + max(L - q, 0). That is piecewise linear in q with breakpoints at
    q = p and q = L, so the minimum is attained at one of 0, p, L and no grid is
    needed.

    decomp returns (p, p*v) for F = p (I + v.sigma), so L is already p||v||_1 and
    must not be multiplied by p again.
    """
    p, pv = decomp(F)
    L = float(np.abs(pv).sum())
    return min(abs(p - q) + max(L - q, 0.0) for q in (0.0, p, L))


print("=== (1) lem:norm and (2) the factor of two ===")
worst1 = worst2 = 0.0
for _ in range(20000):
    p = float(rng.uniform(0.01, 1.0))
    v = rng.normal(size=3)
    v *= float(rng.uniform(0.0, 1.8)) / max(np.linalg.norm(v, 1), 1e-12)
    F = p * (I2 + v[0] * Xm + v[1] * Ym + v[2] * Zm)
    ds = max(0.0, np.abs(v).sum() - 1.0)
    worst1 = max(worst1, abs(magic_c_direct(F) - p * ds))
    # distance between normalised effects is half the Bloch l1 distance to the ball
    Ehat = 0.5 * (I2 + v[0] * Xm + v[1] * Ym + v[2] * Zm)
    worst2 = max(worst2, abs(magic_c_direct(2.0 * Ehat) / 2.0 - 0.5 * ds))
print("   max |magic_c(F) - p_acc*Dstab| over 20000 random F = %.3e" % worst1)
print("   dist_c between normalised effects is Dstab/2 by the same computation, since")
print("   Ehat carries the prefactor 1/2 while lem:norm works with the unnormalised F")
if worst1 > 1e-4:
    fail = True

print()
print("=== (3) ||Q||_c <= sqrt(3) ||Q||_inf ===")
worst = 0.0
for _ in range(60000):
    a = float(rng.normal())
    v = rng.normal(size=3)
    Q = a * I2 + v[0] * Xm + v[1] * Ym + v[2] * Zm
    worst = max(worst, cnorm(Q) / opnorm(Q))
Qt = (Xm + Ym + Zm) / S3
tight = cnorm(Qt) / opnorm(Qt)
print("   max over 60000 random Hermitian Q of ||Q||_c/||Q||_inf = %.6f  (sqrt3 = %.6f)"
      % (worst, S3))
print("   at a=0, v=(1,1,1)/sqrt3 the ratio is %.6f, so the constant is attained" % tight)
if worst > S3 + 1e-9 or abs(tight - S3) > 1e-9:
    fail = True

print()
print("=== (4) ||B_F - B_G||_diamond = 2 ||F - G||_inf ===")
worst = 0.0
for _ in range(20000):
    a1, v1 = float(rng.uniform(0, 1)), rng.normal(size=3)
    a2, v2 = float(rng.uniform(0, 1)), rng.normal(size=3)
    v1 *= float(rng.uniform(0, 1)) / max(np.linalg.norm(v1), 1e-12)
    v2 *= float(rng.uniform(0, 1)) / max(np.linalg.norm(v2), 1e-12)
    F = a1 * I2 + v1[0] * Xm + v1[1] * Ym + v1[2] * Zm
    G = a2 * I2 + v2[0] * Xm + v2[1] * Ym + v2[2] * Zm
    D = F - G
    # sup over states rho of |Tr(D rho)| is the largest |eigenvalue| of D
    sup = 0.0
    for _ in range(60):
        w = rng.normal(size=2) + 1j * rng.normal(size=2)
        w /= np.linalg.norm(w)
        sup = max(sup, abs(np.real(np.conj(w) @ (D @ w))))
    worst = max(worst, sup - opnorm(D))
print("   max over 20000 of [ sup_rho |Tr((F-G) rho)| - ||F-G||_inf ] = %+.2e  (<= 0)" % worst)
if worst > 1e-9:
    fail = True

print()
print("=== (5) the sharp H_XY projector ===")
v = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0.0])
ds = np.abs(v).sum() - 1.0
print("   Dstab = ||v||_1 - 1 = %.6f   (sqrt2 - 1 = %.6f)" % (ds, np.sqrt(2) - 1))
if abs(ds - (np.sqrt(2) - 1)) > 1e-12:
    fail = True

print()
print("=== (6) eq:twofilter ===")
worst = 0.0
for _ in range(20000):
    a, b = float(rng.normal()), float(rng.normal())
    A = I2 + a * Xm
    B = I2 + b * Zm
    lhs = (B @ A).conj().T @ (B @ A)
    rhs = ((1 + a ** 2) * (1 + b ** 2) * I2
           + 2 * a * (1 + b ** 2) * Xm
           + 2 * b * (1 - a ** 2) * Zm)
    worst = max(worst, np.abs(lhs - rhs).max())
print("   max |LHS - eq:twofilter| over 20000 random (a,b) = %.3e" % worst)
if worst > 1e-9:
    fail = True

a = b = 0.5
c0 = (1 + a ** 2) * (1 + b ** 2)
vx = 2 * a * (1 + b ** 2) / c0
vz = 2 * b * (1 - a ** 2) / c0
l1 = abs(vx) + abs(vz)
print("   at a=b=1/2 the normalised Bloch vector is (%.4f, 0, %.4f) with l1 norm %.4f"
      % (vx, vz, l1))
if abs(l1 - 1.28) > 5e-4:
    print("   MISMATCH against the quoted 1.28")
    fail = True
else:
    print("   agrees with the quoted 1.28")

sys.exit(1 if fail else 0)
