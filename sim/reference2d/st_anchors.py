import sys, numpy as np
sys.path.insert(0, r"C:\Users\5d149\source\Aniso\sim\reference2d")
from pfflip2d import Sim

def fixed_run(sim, dt, steps, cb=None):
    for k in range(steps):
        sim.step(dt)
        if cb: cb(k, sim)

print("=== hydrostatic at 8x dt: ST vs the plain-FLIP control at the SAME dt")
# Giant steps on calm water are noisy for plain FLIP too (the gravity kick per
# step scales with dt), so the fair criterion is comparative: ST with
# velocity-adaptive jitter attenuation must stay within 20 percent of the
# plain-FLIP noise floor at the same dt. The absolute floor is a property of
# the step size, capped in production by the frame interval.
res = {}
for label, kw in (("plain", dict(sub_advect=True)), ("st", dict(st=True))):
    s = Sim(64, 64, **kw); s.seed(lambda x, y: y < 32); s.calibrate()
    vlog = []
    fixed_run(s, 0.2, 40, lambda k, s: vlog.append(float(np.abs(s.vel).max())))
    res[label] = max(vlog[-10:])
ratio = res["st"] / res["plain"]
print(f"  max |v| last 10: plain {res['plain']:.3f}, ST {res['st']:.3f}, ratio {ratio:.2f}")
print("  hydrostatic:", "PASS" if ratio < 1.2 else "FAIL")

print("=== air cushion, ST at 8x dt (0.08 vs trusted 0.01)")
mask = lambda x, y: (y < 20) | ((y > 30) & (y < 44))
def gap(sim):
    liq = sim.typ == 1
    mid = (sim.pos[:, 0] > 24) & (sim.pos[:, 0] < 72)
    pool = sim.pos[mid & liq & (sim.pos[:, 1] < 28), 1]
    slab = sim.pos[mid & liq & (sim.pos[:, 1] >= 28), 1]
    return np.percentile(slab, 2) - np.percentile(pool, 98)
s = Sim(96, 80, st=True); s.seed(mask); s.calibrate()
fixed_run(s, 0.08, 15)   # t = 1.2, the gate's horizon
g = gap(s)
print(f"  final gap: {g:.1f} cells (gate's trusted answer: 10.6; vacuum collapses to ~3)")
print("  air_cushion:", "PASS" if g > 4 else "FAIL")

print("=== rayleigh-taylor, ST at 8x dt (0.24 vs trusted 0.03)")
s = Sim(64, 192, rho_l=19.0, rho_g=1.0, alpha_l=0.97, alpha_g=0.97, st=True)
s.seed(lambda x, y: y > 96 + 3.0 * np.cos(2 * np.pi * x / 64))
s.calibrate()
tips = []
fixed_run(s, 0.24, 23, lambda k, s: tips.append(float(np.percentile(s.pos[s.typ == 1, 1], 0.5))))
print(f"  heavy tip: {tips[0]:.1f} -> {min(tips):.1f} (gate: fell by > 30; spurious fingers would show as early chaotic drop)")
print("  rayleigh_taylor:", "PASS" if min(tips) < tips[0] - 30 else "FAIL")
