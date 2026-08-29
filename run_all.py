"""Run every check in this repository and report a single verdict.

    python run_all.py           run everything, print each script's output
    python run_all.py --quiet   run everything, print only the verdict table

Exits with a nonzero status if any check fails.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

CHECKS = [
    ("rotated_surface_code.py",
     "the [[d^2,1,d]] rotated surface code the other checks are run on"),
    ("verify_norm_constants.py",
     "normalisation lemma, one-qubit norm constants, two-filter example"),
    ("verify_dins_standard_round.py",
     "standard-round proposition, within one insertion configuration"),
    ("verify_dins_cross_S.py",
     "standard-round proposition, across insertion configurations"),
    ("verify_split_readout_family.py",
     "split-readout family: accepted effect and when it carries magic"),
    ("verify_weak_string.py",
     "weak-string protocol: closed forms and d log d asymptotics"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    results = []
    for script, what in CHECKS:
        path = os.path.join(HERE, script)
        if not os.path.exists(path):
            results.append((script, what, "MISSING", 0.0))
            continue
        t0 = time.time()
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True)
        dt = time.time() - t0
        if not args.quiet:
            print("=" * 78)
            print("%s  --  %s" % (script, what))
            print("=" * 78)
            sys.stdout.write(proc.stdout)
            if proc.stderr.strip():
                sys.stderr.write(proc.stderr)
            print()
        results.append((script, what, "PASS" if proc.returncode == 0 else "FAIL", dt))

    print("=" * 78)
    print("%-34s %-8s %8s   %s" % ("script", "verdict", "seconds", "checks"))
    print("-" * 78)
    for script, what, verdict, dt in results:
        print("%-34s %-8s %8.1f   %s" % (script, verdict, dt, what))
    print("-" * 78)

    bad = [r for r in results if r[2] != "PASS"]
    if bad:
        print("%d of %d checks did not pass." % (len(bad), len(results)))
        return 1
    print("All %d checks passed." % len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
