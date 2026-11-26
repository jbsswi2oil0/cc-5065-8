"""Run all transient-renderer tests with isolated Python.

Usage: python3 -I tests/run_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = ["test_kernels.py", "test_boundary.py", "test_two_gate.py",
         "test_origin_remap.py", "test_physics.py"]

rc = 0
for name in TESTS:
    print(f"\n================ {name} ================")
    proc = subprocess.run([sys.executable, "-I", os.path.join(HERE, name)])
    rc |= proc.returncode
print()
if rc == 0:
    print("ALL TEST SUITES PASSED")
else:
    print("SOME TESTS FAILED")
sys.exit(rc)
