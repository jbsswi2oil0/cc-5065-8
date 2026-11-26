import sys, math
sys.path.insert(0, "/workspace/transient")
from transient_renderer.kernel import Kernel, make_grid, deposit

grid = make_grid(0.0, 0.1, 100)
t = 5.0  # exactly on the boundary between bin 49 and bin 50

# 1. point deposition boundary rule
bins = [0.0] * 100
deposit(bins, t, 1.0, Kernel(Kernel.POINT), grid)
idx = [i for i, v in enumerate(bins) if v > 0]
print("point at t=5.0 lands in bins:", idx, "(rule: [t0+k*dt, t0+(k+1)*dt))")
# bin 50 covers [5.0, 5.1); the boundary instant belongs to the later bin index
assert idx == [50], idx
# just below the boundary lands in bin 49
bins2 = [0.0] * 100
deposit(bins2, 4.9999999, 1.0, Kernel(Kernel.POINT), grid)
assert bins2[49] == 1.0

# 2. box kernel width 0.4 ns straddling the boundary -> analytic 50/50 split
bins = [0.0] * 100
k = Kernel(Kernel.BOX, width=0.4)
deposit(bins, t, 1.0, k, grid)
left = sum(bins[:50])
right = sum(bins[50:])
print("box split: left=%.4f right=%.4f total=%.6f" % (left, right, sum(bins)))
assert abs(left - 0.5) < 1e-12 and abs(right - 0.5) < 1e-12
assert abs(sum(bins) - 1.0) < 1e-12

# 3. gate fractions directly
g1 = k.fraction_in_interval(t, 4.6, 5.0)
g2 = k.fraction_in_interval(t, 5.0, 5.4)
print("gate G1=%.3f G2=%.3f (probe width 0.4ns, pulse on boundary)" % (g1, g2))
assert abs(g1 - 0.5) < 1e-12 and abs(g2 - 0.5) < 1e-12

# 4. gaussian: truncated mass renormalized to exactly 1
bins = [0.0] * 400
g = make_grid(0.0, 0.05, 400)
gk = Kernel(Kernel.GAUSSIAN, width=0.2)
res = deposit(bins, t, 1.0, gk, g)
print("gaussian deposited=%.6f lost=%.2e" % (res["deposited"], res["lost_edge"]))
assert abs(sum(bins) - 1.0) < 1e-12

# 5. gaussian symmetric across the boundary
gl = make_grid(0.0, 0.05, 400)
b2 = [0.0]*400
deposit(b2, t, 1.0, gk, gl)
bi = int(t / 0.05)
lo = sum(b2[:bi]); hi = sum(b2[bi:])
print("gaussian boundary split: %.4f / %.4f" % (lo, hi))
assert abs(lo - hi) < 1e-9
print("ALL KERNEL TESTS PASS")
