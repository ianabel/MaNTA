"""Finite-difference check of the custom JVP in `objective.py`.

`make_objective`'s JVP contracts MaNTA's spatial adjoint gradient `G_p` with
the tangent DESC hands it, after ravelling each node's tangent and prepending
`1 + 2 nzeta + 2 ntheta` zeros. Whether that dot product is the directional
derivative depends on three layouts agreeing element for element:

* the tangent, as DESC's trace actually produces it (which `Field` leaves come
  through as symbolic zeros, i.e. `None`, and are dropped by `ravel_pytree`);
* `G_p`'s explicit columns, `StellaratorAdjointProblem.dgFndp` (the field's
  float leaves, V' and V'' appended);
* `G_p`'s implicit columns, `StellaratorAdjointProblem.ComputePhysicsDerivatives`
  (the `jacrev` over `(field, vp, vpp)`, ravelled as it stands).

A mismatch is a gradient that is wrong but plausible, so this checks it two
ways:

1. `--layout-only` (cheap, no MaNTA solve): traces the real tangent structure
   through a probe with the same `filter_custom_jvp` signature and prints the
   three layouts side by side.
2. The default: for each DESC data quantity, `jax.jvp` of the objective along
   a smooth perturbation of that quantity against a central difference.

The perturbations are applied to DESC's *data* (|B|, B^theta, V_r(r), ...),
and the Fields are rebuilt from it with `yancc_data.from_data`, the same
construction as `objective_from_user_fun` in `desc-optimize-ambipolar.py`. So
the tangent has the structure the optimizer gives it, without differentiating
through DESC itself.

Under TEST_STELLARATOR the fluxes and sources depend on the field only through
V' and V'', so the expected derivative is nonzero for the V(r) quantities and
zero for everything else. That is what makes it an alignment test: a shifted
tangent puts the V'/V'' sensitivity on another quantity's row, or the dot
product fails on a shape mismatch.

Run only with TEST_STELLARATOR=true:

    TEST_STELLARATOR=true PYTHONPATH=$PWD/build/python \\
        python python-physics/stellarator/fd_check_objective.py --layout-only
"""

import os
import sys

if os.environ.get("TEST_STELLARATOR") != "true":
    sys.exit(
        "fd_check_objective.py: refusing to run without TEST_STELLARATOR=true "
        "(the real drift-kinetic solve is too expensive for this check)"
    )

import argparse
import tempfile
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import equinox as eqx
from jax.flatten_util import ravel_pytree

import desc
import desc.examples
import desc.io
from desc.grid import LinearGrid
from desc.backend import tree_unstack

import manta as MaNTA
from yancc_wrapper import yancc_data
from objective import make_objective

DATA_NAMES = [
    "B^theta",
    "B^zeta",
    "B_theta",
    "B_zeta",
    "|B|",
    "|B|_t",
    "|B|_z",
    "sqrt(g)",
    "Psi",
    "a",
    "R0",
    "iota",
    "V(r)",
    "V_r(r)",
    "V_rr(r)",
]

# what the test flux can see; everything else should difference to zero
EXPECTED_NONZERO = {"V(r)", "V_r(r)", "V_rr(r)"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--eq", default="precise_QA", help="DESC example name or .h5 path")
    p.add_argument("--cells", type=int, default=4)
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--M", type=int, default=None, help="DESC grid M (default eq.M_grid)")
    p.add_argument("--N", type=int, default=None, help="DESC grid N (default eq.N_grid)")
    p.add_argument("--keys", default=",".join(DATA_NAMES), help="data quantities to perturb")
    p.add_argument("--eps", type=float, default=1e-4, help="relative perturbation size")
    p.add_argument("--tol", type=float, default=1e-3, help="relative agreement required")
    p.add_argument("--ss-tol", type=float, default=1e-10, help="primal SteadyStateTolerance")
    p.add_argument("--evolve-density", action="store_true")
    p.add_argument("--layout-only", action="store_true")
    return p.parse_args()


def load_eq(name):
    if name.endswith(".h5"):
        eq = desc.io.load(name)
        return eq[-1] if hasattr(eq, "__len__") else eq
    return desc.examples.get(name)


def make_config(args):
    nodes = np.linspace(0.0, 1.0, args.cells + 1)
    st_config = {
        "ParticleSourceCenter": 0.0,
        "ParticleSourceHeight": 0.02,
        "ParticleSourceWidth": 0.4,
        "NBICenter": 0.0,
        "NBIPower": 0.5,
        "NBIWidth": 0.1,
        "ECHCenter": 0.0,
        "ECHPower": 0.5,
        "ECHWidth": 0.1,
        "EdgeTemperature": 0.5,
        "EdgeDensity": 0.4,
        "n0": 0.5,
        "T0": 1.0,
        "FusionFactor": 0.01,
        "evolveDensity": args.evolve_density,
        "useBatching": True,
        "adaptiveGrid": False,
    }
    solver_config = {
        "OutputFilename": "fd_check",
        "PolynomialDegree": args.degree,
        "GridPoints": nodes,
        "GridSize": args.cells,
        "tau": 0.5,
        "LowerBoundary": 0.0,
        "UpperBoundary": 1.0,
        "Relative_tolerance": 1e-8,
        "Absolute_tolerance": [1e-12],
        "delta_t": 1.0,
        "initialTimestep": 1e-2,
        "MinStepSize": 1e-12,
        "SteadyStateSolver": "PseudoTransient",
        "SteadyStateTest": "Residual",
        "SteadyStateTolerance": args.ss_tol,
        "MaxRejectedSteps": 10,
        "restart": False,
        "zeroFlux": True,
        "solveAdjoint": False,
    }
    return {"Stellarator": st_config, "Solver": solver_config}, nodes


YANCC_RES = {"nx": 3, "na": 9}  # unused by the test flux; kept small


def build_tree(data, grid):
    w = yancc_data.from_data(data, grid, **YANCC_RES)
    return (w.fields, w.Vp, w.Vpp)


def perturbation(data, key, rho_nodes):
    """A smooth relative perturbation of one quantity, zeros elsewhere.

    `name<r` restricts the perturbation to rho < r, which leaves the Dirichlet
    edge value (V'(x_R), through `InitialValue`) untouched.
    """
    key, _, cut = key.partition("<")
    mask = (rho_nodes < float(cut)) if cut else jnp.ones_like(rho_nodes)
    tangent = {k: jnp.zeros_like(v) for k, v in data.items()}
    v = data[key]
    if v.shape == rho_nodes.shape:
        tangent[key] = v * (1.0 + 0.5 * rho_nodes) * mask
    else:
        tangent[key] = v * jnp.ones_like(v)
    return tangent


# ---------------------------------------------------------------------------
# layout report
# ---------------------------------------------------------------------------


def _leaves_with_names(tree):
    """(name, size) for every array leaf, in ravel_pytree order."""
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [(jax.tree_util.keystr(path), int(np.size(leaf))) for path, leaf in flat]


def _labels(entries):
    out = []
    for name, size in entries:
        out.extend([name] * size)
    return out


def layout_report(data, grid):
    fields, _, _ = build_tree(data, grid)
    field0 = tree_unstack(fields)[-1]

    # StellaratorAdjointProblem.__init__: np_cell = len(flat) - 2 + 1 + 1
    flat_len = ravel_pytree(eqx.filter(field0, eqx.is_array))[0].size
    np_cell = flat_len - 2 + 1 + 1

    inexact = _leaves_with_names(eqx.filter(field0, eqx.is_inexact_array))
    # dgFndp: grad over the field, two zero columns appended
    explicit = list(inexact) + [("Vp", 1), ("Vpp", 1)]
    # ComputePhysicsDerivatives: jacrev over (field, vp, vpp), ravelled as is
    implicit = list(inexact) + [("Vp", 1), ("Vpp", 1)]

    # the tangent as the real trace produces it
    seen = {}

    @eqx.filter_custom_jvp
    def probe(tree_in, grid):
        return jnp.zeros(())

    @probe.def_jvp
    def probe_jvp(primals, tangents):
        v, _ = tangents
        f_t, vp_t, vpp_t = v
        node0 = jax.tree_util.tree_map(lambda x: x[0], f_t)
        seen["field"] = _leaves_with_names(node0)
        seen["vp"] = vp_t is not None
        seen["vpp"] = vpp_t is not None
        seen["ntheta"] = primals[0][0].ntheta
        seen["nzeta"] = primals[0][0].nzeta
        return jnp.zeros(()), jnp.zeros(())

    tangent = {k: jnp.ones_like(v) for k, v in data.items()}
    jax.jvp(lambda d: probe(build_tree(d, grid), grid), (data,), (tangent,))

    pad = 1 + 2 * seen["nzeta"] + 2 * seen["ntheta"]
    tangent_layout = [("<pad>", pad)] + seen["field"]
    tangent_layout += [("Vp", 1)] if seen["vp"] else []
    tangent_layout += [("Vpp", 1)] if seen["vpp"] else []

    all_names = [n for n, _ in _leaves_with_names(eqx.filter(field0, eqx.is_array))]
    present = {n for n, _ in seen["field"]}
    absent = [n for n in all_names if n not in present]

    print("\n=== layout report (one node) ===")
    print(f"ntheta = {seen['ntheta']}, nzeta = {seen['nzeta']}, pad = {pad}")
    print(f"np_cell (adjoint np)             = {np_cell}")
    print(f"rows from dgFndp (explicit)      = {sum(s for _, s in explicit)}")
    print(f"rows from ComputePhysicsDerivs   = {sum(s for _, s in implicit)}")
    print(f"tangent length after pad (JVP)   = {sum(s for _, s in tangent_layout)}")
    print(f"field leaves with None tangent   = {absent}")

    ok = True
    t_lab = _labels(tangent_layout)
    for label, layout in (("explicit", explicit), ("implicit", implicit)):
        g_lab = _labels(layout)
        if len(g_lab) != len(t_lab):
            print(f"  {label}: LENGTH MISMATCH, G_p {len(g_lab)} vs tangent {len(t_lab)}")
            ok = False
        bad = {}
        for i, (g, t) in enumerate(zip(g_lab, t_lab)):
            if t == "<pad>":
                if g in present:  # a leaf with a real tangent is being zeroed
                    bad.setdefault(g, "multiplied by pad")
            elif g != t:
                bad.setdefault(g, f"meets tangent of {t} at element {i}")
        if bad:
            ok = False
            print(f"  {label}: misaligned columns:")
            for g, why in bad.items():
                print(f"    {g:24s} {why}")
        else:
            print(f"  {label}: aligned with the tangent over the common length")
    print("layout:", "OK" if ok else "MISMATCH")
    return ok


# ---------------------------------------------------------------------------
# finite-difference check
# ---------------------------------------------------------------------------


def fd_check(args, data, grid, config):
    objective = make_objective(config, yancc_res=YANCC_RES)

    def f(d):
        G, _ = objective(build_tree(d, grid), grid)
        return jnp.ravel(G)[0]

    rho_nodes = jnp.asarray(grid.nodes[:, 0])
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]

    t0 = time.time()
    # the JVP rule restarts from this run's restart file, so the base primal
    # has to be the last solve before the JVPs
    G0 = float(f(data))
    print(f"\nbase G = {G0:.12e}  ({time.time() - t0:.1f} s)")
    if G0 == 1e-10:
        sys.exit("base solve did not converge (objective returned its 1e-10 sentinel)")

    jvps = {}
    for k in keys:
        t0 = time.time()
        _, dG = jax.jvp(f, (data,), (perturbation(data, k, rho_nodes),))
        jvps[k] = float(dG)
        print(f"  jvp  {k:10s} {jvps[k]: .6e}  ({time.time() - t0:.1f} s)")

    rows = []
    for k in keys:
        dk = perturbation(data, k, rho_nodes)
        plus = jax.tree_util.tree_map(lambda x, d: x + args.eps * d, data, dk)
        minus = jax.tree_util.tree_map(lambda x, d: x - args.eps * d, data, dk)
        t0 = time.time()
        Gp, Gm = float(f(plus)), float(f(minus))
        if 1e-10 in (Gp, Gm):
            rows.append((k, jvps[k], float("nan"), float("nan"), "SOLVE FAILED"))
            continue
        fd = (Gp - Gm) / (2 * args.eps)
        scale = max(abs(fd), abs(jvps[k]))
        floor = 1e-8 * abs(G0)
        if scale < floor:
            rel, verdict = 0.0, "ok (both ~0)"
        else:
            rel = abs(jvps[k] - fd) / scale
            verdict = "ok" if rel < args.tol else "FAIL"
        if k in EXPECTED_NONZERO and scale < floor:
            verdict = "FAIL (expected nonzero)"
        rows.append((k, jvps[k], fd, rel, verdict))
        print(f"  fd   {k:10s} {fd: .6e}  ({time.time() - t0:.1f} s)")

    print(f"\n{'quantity':10s} {'jvp':>14s} {'central FD':>14s} {'rel err':>10s}  verdict")
    for k, j, fd, rel, verdict in rows:
        print(f"{k:10s} {j: 14.6e} {fd: 14.6e} {rel: 10.2e}  {verdict}")
    failed = [r for r in rows if not r[4].startswith("ok")]
    print("\nFD check:", "OK" if not failed else f"{len(failed)} FAILED")
    return not failed


def main():
    args = parse_args()
    config, nodes = make_config(args)
    eq = load_eq(args.eq)

    points = jnp.asarray(MaNTA.getNodes(nodes, args.degree))
    grid = LinearGrid(
        rho=np.asarray(points),
        M=args.M if args.M is not None else eq.M_grid,
        N=args.N if args.N is not None else eq.N_grid,
        NFP=eq.NFP,
    )
    computed = eq.compute(DATA_NAMES, grid=grid)
    data = {k: jnp.asarray(computed[k]) for k in DATA_NAMES}

    # the JVP dots G_p (over MaNTA's nodes) with a tangent over the DESC grid's
    # rho, so the two must be the same points
    rho_desc = np.asarray(grid.compress(grid.nodes[:, 0], surface_label="rho"))
    assert np.allclose(rho_desc, np.asarray(points)), "DESC rho != MaNTA nodes"

    layout_ok = layout_report(data, grid)
    if args.layout_only:
        sys.exit(0 if layout_ok else 1)

    workdir = tempfile.mkdtemp(prefix="manta-fd-check-")
    print(f"\nrun output in {workdir}")
    os.chdir(workdir)
    fd_ok = fd_check(args, data, grid, config)
    sys.exit(0 if (layout_ok and fd_ok) else 1)


if __name__ == "__main__":
    main()
