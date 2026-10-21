"""Peak memory of the MaNTA objective's JVP against the number of tangents.

Runs desc-optimize-itg-flux.py's setup (everything before its ObjectiveFunction
is built), then one unchunked jvp_scaled of the MaNTA objective alone with N
random tangents. Run once per N, in a fresh process, so ru_maxrss is clean.

    python measure_jvp_memory.py N [driver.py] [chunk]

Measured on the doubled-NBI setup: +0.93 GB at N=1, +2.69 at 16, +8.14 at 64,
+15.06 at 126 (the optimiser's tangent count) -- linear, ~0.12 GB a tangent.
A chunk size runs DESC's lax.scan, which the MaNTA objective cannot go
through (its solve calls back into Python with traced arrays).
"""

import os
import resource
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1])
driver = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "desc-optimize-itg-flux.py")
chunk = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.chdir(os.path.dirname(os.path.abspath(driver)))


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


src = open(driver).read()
cut = src.index("# blocked: each objective's Jacobian")
ns = {"__name__": "__setup__", "__file__": os.path.abspath(driver)}
exec(compile(src[:cut], driver, "exec"), ns)

import numpy as np
from desc.objectives import ObjectiveFunction

manta_obj = ns["objectives"][-1]
of = ObjectiveFunction([manta_obj], deriv_mode="batched", jac_chunk_size=chunk)
of.build(use_jit=False)
x = of.x(*of.things)

# a primal at x first, as the optimiser does before a Jacobian
f = of.compute_scaled(x)
print(f"MEASURE primal f = {np.asarray(f)}  peak so far {rss_gb():.2f} GB", flush=True)

rng = np.random.default_rng(0)
v = rng.standard_normal((N, of.dim_x)) * 1e-6
base = rss_gb()
t0 = time.perf_counter()
jv = of.jvp_scaled(v, x)
jv = np.asarray(jv)
dt = time.perf_counter() - t0
peak = rss_gb()
print(
    f"MEASURE N={N} chunk={chunk} dim_x={of.dim_x} jvp shape {jv.shape} {dt:.1f} s  "
    f"peak before jvp {base:.2f} GB  after {peak:.2f} GB  jvp added {peak - base:.2f} GB",
    flush=True,
)
