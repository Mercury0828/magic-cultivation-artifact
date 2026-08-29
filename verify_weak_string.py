"""The weak-string protocol: accepted effect, magic, and the d log d asymptotics.

Checks the closed forms the paper states for the protocol of Definition 8, one
layer of rotations exp(i theta_d Z_j) on a bare minimum-weight logical Z string,
followed by one syndrome round and the split readout.

  (1) eq:constr-effect   F_A = 1/2 [ (a+b) I + (a-b) Xbar + 2 alpha beta Ybar ]
                         with alpha = cos^d theta, beta = +- sin^d theta,
                         a = alpha^2, b = beta^2;
  (2) eq:constr-magic    p_acc = (a+b)/2,
                         Dstab = 2t(1-t)/(1+t^2) with t = |beta|/alpha,
                         magic_c = cos^d theta sin^d theta - sin^{2d} theta;
  (3) the worked value quoted for the distance-3 patch at kappa = 1/2;
  (4) eq:constr-asymp    -log magic_c = d log d + O(d), and p_acc tends to 1/2.
                         The ratio to d log d tends to 1 only like 1 + log(1/kappa)/log d,
                         which is too slow to see, so what is checked is the O(d)
                         coefficient, which must settle at log(1/kappa).

Item (4) is computed in log space. The direct double-precision value of magic_c
underflows to exactly zero from about d = 15 on at kappa = 1/2, so comparing it
against a nonzero closed form there would be meaningless; those rows are reported
from the stable form only and are excluded from the agreement test.

Exits with a nonzero status on any mismatch.
"""
import sys
from math import cos, sin, tan, log, log1p

import numpy as np

I2 = np.eye(2, dtype=complex)
Xm = np.array([[0, 1], [1, 0]], dtype=complex)
Ym = np.array([[0, -1j], [1j, 0]], dtype=complex)
Zm = np.array([[1, 0], [0, -1]], dtype=complex)

KAPPA = 0.5
TOL = 1e-9


def accepted_effect(d, theta):
    """F_A on the logical qubit, by direct matrix algebra from A = alpha I + i beta Z."""
    alpha = cos(theta) ** d
    beta = sin(theta) ** d * (1 if d % 4 == 1 else -1)
    A = alpha * I2 + 1j * beta * Zm
    return A.conj().T @ ((I2 + Xm) / 2) @ A, alpha, beta


def bloch(F):
    a = np.real(np.trace(F)) / 2
    v = np.array([np.real(np.trace(F @ Q)) / 2 for Q in (Xm, Ym, Zm)])
    return a, v


def dstab(F):
    a, v = bloch(F)
    if a <= 1e-300:
        return 0.0
    return max(0.0, np.abs(v).sum() / a - 1)


fail = False

print("=== (1),(2) closed forms for the accepted effect ===")
print("%5s %12s %14s %14s %14s %10s" %
      ("d", "p_acc", "Dstab", "2t(1-t)/(1+t^2)", "magic_c", "verdict"))
for d in (3, 5, 9, 15, 25, 41):
    theta = KAPPA / d
    F, alpha, beta = accepted_effect(d, theta)
    a, b = alpha ** 2, beta ** 2

    pred = 0.5 * ((a + b) * I2 + (a - b) * Xm + 2 * alpha * beta * Ym)
    e_effect = np.abs(F - pred).max()

    pacc_direct = np.real(np.trace(F)) / 2
    pacc_form = (a + b) / 2

    t = abs(beta) / alpha
    ds_direct = dstab(F)
    ds_form = 2 * t * (1 - t) / (1 + t * t)

    mg_direct = pacc_direct * ds_direct
    mg_form = cos(theta) ** d * sin(theta) ** d - sin(theta) ** (2 * d)

    underflow = (mg_direct == 0.0) or (ds_direct == 0.0)
    if underflow:
        verdict = "UNDERFLOW"
    else:
        # Dstab is computed by cancellation, so only a few relative digits survive
        ok = (e_effect < TOL
              and abs(pacc_direct - pacc_form) < TOL
              and abs(ds_direct - ds_form) <= 1e-4 * abs(ds_form)
              and abs(mg_direct - mg_form) <= 1e-4 * abs(mg_form))
        verdict = "ok" if ok else "MISMATCH"
        if not ok:
            fail = True
    print("%5d %12.6f %14.4e %14.4e %14.4e %10s"
          % (d, pacc_direct, ds_direct, ds_form, mg_direct, verdict))

print()
print("=== (3) the worked distance-3 value ===")
d, theta = 3, KAPPA / 3
mg = cos(theta) ** d * sin(theta) ** d - sin(theta) ** (2 * d)
F, _, _ = accepted_effect(d, theta)
print("   d=3, kappa=1/2, theta_3=1/6:  magic_c = %.6e   p_acc = %.6f" % (mg, bloch(F)[0]))
if abs(mg - 4.36e-3) > 5e-6:
    print("   MISMATCH against the quoted 4.36e-3")
    fail = True
else:
    print("   agrees with the quoted 4.36e-3")

print()
print("=== (4) asymptotics: -log magic_c vs d log d, and p_acc -> 1/2 ===")
print("The ratio to d log d tends to 1 only like 1 + log(1/kappa)/log d, which is far too")
print("slow to see directly, so the O(d) coefficient is what is checked: the paper's claim")
print("is -log magic_c = d log d + O(d), and the coefficient should settle at log(1/kappa).")
print()
print("%5s %16s %14s %10s %16s %12s"
      % ("d", "-log magic_c", "d log d", "ratio", "(diff)/d", "p_acc"))
coeffs = []
for d in (11, 21, 41, 81, 161, 321, 641, 1281):
    theta = KAPPA / d
    # magic_c = (cos t sin t)^d (1 - tan^d t); the direct value underflows, so use logs
    neglog = -d * log(cos(theta) * sin(theta)) - log1p(-tan(theta) ** d)
    ratio = neglog / (d * log(d))
    coeff = (neglog - d * log(d)) / d
    pacc = (cos(theta) ** (2 * d) + sin(theta) ** (2 * d)) / 2
    print("%5d %16.4f %14.4f %10.4f %16.6f %12.6f"
          % (d, neglog, d * log(d), ratio, coeff, pacc))
    coeffs.append(coeff)
target = log(1.0 / KAPPA)
if abs(coeffs[-1] - target) > 1e-3:
    print("   MISMATCH: the O(d) coefficient is %.6f, expected log(1/kappa) = %.6f"
          % (coeffs[-1], target))
    fail = True
else:
    print("   the O(d) coefficient settles at log(1/kappa) = %.6f, so" % target)
    print("   -log magic_c = d log d + O(d) with the linear term as claimed")

pacc_last = (cos(KAPPA / 321) ** (2 * 321) + sin(KAPPA / 321) ** (2 * 321)) / 2
if abs(pacc_last - 0.5) > 1e-3:
    print("   MISMATCH: p_acc does not approach 1/2")
    fail = True
else:
    print("   p_acc approaches 1/2")

sys.exit(1 if fail else 0)
