"""Peak memory of the MaNTA objective's Jacobian in reverse mode, and a check of
it against forward-mode JVPs.

Runs desc-optimize-itg-flux.py's setup (everything before its ObjectiveFunction
is built), switches the MaNTA objective to deriv_mode="rev", builds its Jacobian
(one row: the objective is a scalar), then compares J v with forward-mode JVPs
on a few random tangents.

    python measure_rev_jacobian.py [driver.py] [n_check]

Measured: the full (1, 343) Jacobian adds 3.5 GB (7.0 GB peak) and matches
forward-mode JVPs to 4e-8. The driver uses reverse mode for this reason.
"""

import os
import resource
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
driver = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "paper", "desc-optimize-itg-flux.py")
n_check = int(sys.argv[2]) if len(sys.argv) > 2 else 4
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
manta_obj._deriv_mode = "rev"
of = ObjectiveFunction([manta_obj], deriv_mode="blocked")
of.build(use_jit=False)
x = of.x(*of.things)

f = of.compute_scaled(x)
print(f"MEASURE primal f = {np.asarray(f)}  peak so far {rss_gb():.2f} GB", flush=True)

base = rss_gb()
t0 = time.perf_counter()
J = np.asarray(of.jac_scaled(x))
dt = time.perf_counter() - t0
peak = rss_gb()
print(
    f"MEASURE rev jac shape {J.shape} {dt:.1f} s  peak before {base:.2f} GB  "
    f"after {peak:.2f} GB  jac added {peak - base:.2f} GB  |J| = {np.linalg.norm(J):.6e}",
    flush=True,
)

# forward mode on a few tangents, for the values
manta_obj._deriv_mode = "fwd"
of_fwd = ObjectiveFunction([manta_obj], deriv_mode="batched", jac_chunk_size=None)
of_fwd.build(use_jit=False)
rng = np.random.default_rng(0)
v = rng.standard_normal((n_check, of.dim_x))
jv_fwd = np.asarray(of_fwd.jvp_scaled(v, x)).ravel()
jv_rev = (v @ J.T).ravel()
rel = np.abs(jv_rev - jv_fwd) / np.maximum(np.abs(jv_fwd), 1e-300)
for a, b, r in zip(jv_rev, jv_fwd, rel):
    print(f"MEASURE check  rev {a: .9e}  fwd {b: .9e}  rel {r:.2e}", flush=True)
print(f"MEASURE max rel {rel.max():.2e}", flush=True)
