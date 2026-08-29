# Verification scripts for *The resource cost of magic in a code block*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22162937.svg)](https://doi.org/10.5281/zenodo.22162937)

Archived on Zenodo. The concept DOI
[10.5281/zenodo.22162937](https://doi.org/10.5281/zenodo.22162937) always resolves
to the latest version; [10.5281/zenodo.22162938](https://doi.org/10.5281/zenodo.22162938)
is v1.0.0, which is the version the paper cites and the one its quoted numbers
were produced from.

This repository holds the scripts that check the numerical statements made in the
paper. Every analytic result in the paper is proved there; nothing here is a
premise of any proof. What these scripts do is confirm the closed forms, the
constants, and the two combinatorial facts about the rotated surface code that
the paper quotes.

Each script is self-contained, asserts the code parameters it uses, fixes its
random seed where it samples, prints what it found, and exits with a nonzero
status on a mismatch.

## Running

```
pip install -r requirements.txt
python run_all.py            # run everything, print each script's output
python run_all.py --quiet    # run everything, print only the verdict table
```

The whole suite takes well under a minute. It needs only Python 3.9 or later and
NumPy. It was last run on Python 3.11.4 with NumPy 2.2.6.

## What each script checks

| Script | Paper statement | What is confirmed |
|---|---|---|
| `rotated_surface_code.py` | the code used throughout the appendices | builds the rotated surface code at `d = 3, 5` and **asserts** `n = d²`, `k = 1`, commuting checks, and `d_X = d_Z = d` |
| `verify_norm_constants.py` | normalisation lemma; the norm-conversion and binary-channel identities; the two-filter example | `magic_c(F) = p_acc · Δ_stab(Ê)`; `‖Q‖_c ≤ √3‖Q‖_∞` on one qubit with the constant attained at `a = 0, v = (1,1,1)/√3`; `‖B_F − B_G‖_◇ = 2‖F − G‖_∞`; `Δ_stab` of the sharp `H_XY` projector is `√2 − 1`; the expansion of `(BA)†(BA)` and its normalised Bloch ℓ¹ norm `1.28` at `a = b = ½` |
| `verify_dins_standard_round.py` | the standard-round proposition, within one insertion configuration | on a random sample of 300 separated configurations at each of `d = 3, 5`, every syndrome fibre with more than one member has all its members equal modulo `Z`-stabilizers, over **438** and **11 082** such fibres; and a weight-`d` column is a `Z`-logical, which is where the coset form stops |
| `verify_dins_cross_S.py` | the same proposition, across insertion configurations | prints the ranks that force the ambiguity into `{I, Z̄}` — with one logical qubit `dim ker H_X − rank H_Z = 1`, so no other class exists — and then samples to show the class `Z̄` **occurs**: over 21 and 120 configurations, **323** and **19 518** pairs differ by a stabilizer, **34** and **422** by `Z̄`, and none by anything else |
| `verify_split_readout_family.py` | the split-readout family's accepted effect | by direct 2⁹ state-vector simulation at `d = 3`: the zero-syndrome branch is `αI + βZ̄`, the accepted effect and its Bloch form match to `10⁻¹⁵`, a marked set with no logical support gives `β = 0` and `Δ_stab = 0`, the logical column gives `Δ_stab = 3.3×10⁻²`, all nine qubits give `Δ_stab = 0.38`, the logical column at `θ = π/4` gives `|α| = |β|` and `Δ_stab = 0` exactly, and over 3000 random angle vectors on a marked set with logical support none gave zero magic |
| `verify_weak_string.py` | the weak-string protocol | the closed forms for the accepted effect, `p_acc`, `Δ_stab` and `magic_c`; the quoted `4.36×10⁻³` at `d = 3, κ = ½`; and `−log magic_c = d log d + O(d)` with `p_acc → ½` |

## Two things worth knowing before reading the output

**Some rows say `UNDERFLOW`, and that is deliberate.** In the weak-string check
the direct double-precision value of `magic_c` falls below `10⁻³⁰⁸` from about
`d = 15` on, so it evaluates to exactly zero. Comparing that against a nonzero
closed form and calling it agreement would mean nothing, so those rows are
reported from the stable form only and are excluded from the agreement test. The
asymptotics are computed in log space for the same reason.

**One result is a rank fact, not a sampling result.** That the cross-configuration
ambiguity lies in `{I, Z̄}` follows from the code having one logical qubit and is
not something a sample could support; `verify_dins_cross_S.py` prints the ranks
that settle it. What the sample there contributes is the opposite direction, that
the class `Z̄` really occurs, so the ambiguity group is not trivial.

The samples are samples. They are drawn at fixed seeds and they are not
exhaustive, and the two distances tested are `d = 3` and `d = 5`.

## Layout

```
run_all.py                        runs every check, one verdict table
rotated_surface_code.py           the code construction, imported by the others
verify_norm_constants.py
verify_dins_standard_round.py
verify_dins_cross_S.py
verify_split_readout_family.py
verify_weak_string.py
requirements.txt
CITATION.cff
LICENSE
```

## Citing

If you use these scripts, please cite the paper, and cite the archive as

> Shen, J. and Zhong, H. *Verification scripts for "The resource cost of magic in
> a code block"*, v1.0.0, Zenodo, 2026. doi:10.5281/zenodo.22162938

`CITATION.cff` carries the machine-readable metadata.
