"""Does a lower yancc resolution still point the optimizer the right way?

At one equilibrium and one MaNTA starting state, this builds the objective
paper/desc-optimize-stored-energy.py optimises -- the MaNTA stored-energy
term, optionally with the mirror-ratio and volume terms -- under the same
proximal projection and linear constraints that `eq.optimize(...,
optimizer="proximal-lsq-exact", x_scale="ess")` builds, and takes its scaled
residual f and Jacobian J at two yancc resolutions. It then compares what the
optimizer would do with each:

* the gradient g = J^T f in the optimizer's scaled variables (x_scale * g,
  what lsq-exact's trust region sees) -- the cosine between the two, and
  whether the low-resolution descent direction is a descent direction for the
  high-resolution objective at all (g_hi . g_lo > 0);
* the Gauss-Newton step, -pinv(J_s) f, which is the step lsq-exact takes when
  the trust region does not bind;
* the parameters that dominate each gradient, with their signs.

A resolution is na,nx,ntheta,nzeta. ntheta and nzeta must be odd: in the
optimisation yancc's angular grid is the DESC LinearGrid the objective is
evaluated on, and LinearGrid(M, N) has 2M+1 and 2N+1 points.

Each resolution starts MaNTA from the same --restart file (or cold) and
converges at its own resolution, so what is compared is what the optimizer
would see at each. Expensive: one MaNTA solve plus adjoint per resolution, as
one optimizer iteration. Run it where the optimisation runs.

    python gradient_direction_check.py --eq ../paper/eq8_all_equilibria.h5 \\
        --restart ../paper/stellarator_opteq8.restart.nc \\
        --low 45,7,17,27 --high 121,7,33,53 --out graddir

Results go to <out>_<res>.npz (one per resolution, reused if present unless
--recompute) and <out>.json (the comparison). The physics and solver
settings below are the driver's; keep them in step with it.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

RHO_UPPER = 0.9
NPOINTS = 9
DEGREE = 4
TAU = 0.5


def make_config(Spi=600.0, Sn=1.2):
    """paper/desc-optimize-stored-energy.py's make_config, as run there."""
    return {
        "ParticleSourceCenter": 0.2,
        "ParticleSourceHeight": Sn,
        "ParticleSourceWidth": 0.6,
        "NBICenter": 0.0,
        "NBIPower": Spi,
        "NBIWidth": 0.15,
        "ECHCenter": 0.0,
        "ECHPower": 0.0,
        "ECHWidth": 0.1,
        "EdgeTemperature": 1.5,
        "EdgeDensity": 0.5,
        "ADCoefficient": 0.0,
        "n0": 3.0,
        "T0": 3.0,
        "FusionFactor": 1.0,
        "evolveDensity": True,
        "ErDiffusion": 1e-6,
        "ErInertia": 5e-6,
        "useITGProxy": False,
        "ITGScale": 1.0,
        "useBatching": True,
        "batch_size": 16,
        "nchunks": 3,
    }


def solver_config(name, nodes, restart):
    """The driver's optimisation-time solver settings."""
    cfg = {
        "OutputFilename": name,
        "PolynomialDegree": DEGREE,
        "GridPoints": nodes,
        "GridSize": len(nodes) - 1,
        "tau": TAU,
        "LowerBoundary": 0.0,
        "UpperBoundary": RHO_UPPER,
        "Relative_tolerance": 1e-2,
        "Absolute_tolerance": [1e-5],
        "delta_t": 100.0,
        "initialTimestep": 1e4,
        "MinStepSize": 1e-10,
        "NewtonJacobianReuse": 5,
        "NewtonMaxIterations": 15,
        "zeroFlux": True,
        "SteadyStateTolerance": 2e-4,
        "SteadyStateStepTolerance": 0.33,
        "SteadyStateSolver": "Newton",
        "MaxRejectedSteps": 0,
        "MaxContinuationSteps": 30,
        "PseudoTransientSERFloor": 2.0,
    }
    if restart:
        cfg.update({"RestartFile": restart, "restart": True})
    else:
        cfg.update({"restart": False, "SteadyStateSolver": "PseudoTransient",
                    "initialTimestep": 0.1, "delta_t": 1.0, "MaxRejectedSteps": 10})
    return cfg


def parse_res(text):
    na, nx, nt, nz = (int(v) for v in text.split(","))
    if nt % 2 == 0 or nz % 2 == 0:
        sys.exit(f"ntheta and nzeta must be odd (LinearGrid has 2M+1, 2N+1): {text}")
    return na, nx, nt, nz


def jacobian_at(res, args, eq, V0, nodes):
    """(f, J, x_scale, G, names) of the projected, scaled problem at `res`."""
    import jax
    import jax.numpy as jnp
    import desc
    import yancc
    from desc.grid import Grid
    from desc.objectives import (
        FixBoundaryR, FixBoundaryZ, FixCurrent, FixPsi, ForceBalance,
        LinearObjectiveFromUser, ObjectiveFromUser, ObjectiveFunction, Volume,
    )
    from desc.optimize.optimizer import (
        _parse_x_scale, _project_x_scale, get_combined_constraint_objectives,
    )
    from desc.profiles import SplineProfile
    import manta as MaNTA
    from config import field_norm_from_eq
    from itg_wrapper import make_itg_builder, with_itg
    from objective import ObjectiveFromUserWithParams, make_objective

    na, nx, nt, nz = res
    tag = f"{na}_{nx}_{nt}_{nz}"
    points = MaNTA.getNodes(nodes, DEGREE)
    rho = jnp.array(points)
    st_config = make_config(args.Spi, args.Sn)
    normalization = {"n0": 1e19, **field_norm_from_eq(eq)}
    config = {
        "Stellarator": st_config,
        "Solver": solver_config(f"{args.out}_{tag}", nodes, args.restart),
        "Normalization": normalization,
    }
    manta_objective = make_objective(config, yancc_res={"na": na, "nx": nx},
                                     normalization=normalization)
    itg_builder = make_itg_builder(st_config, eq, RHO_UPPER)
    G_seen = []

    def objective_from_user_fun(grid, data, params=None):
        # the driver's, with the data keys written out (DESC reads them from
        # this function's source)
        yancc_dat = {
            "B_sup_t": data["B^theta"],
            "B_sup_z": data["B^zeta"],
            "B_sub_t": data["B_theta"],
            "B_sub_z": data["B_zeta"],
            "Bmag": data["|B|"],
            "dBdt": data["|B|_t"],
            "dBdz": data["|B|_z"],
            "sqrtg": data["sqrt(g)"],
        }
        yancc_dat = {k: grid.meshgrid_reshape(v, "rtz") for k, v in yancc_dat.items()}
        yancc_dat["Psi"] = grid.compress(data["Psi"] / grid.nodes[:, 0] ** 2,
                                         surface_label="rho")
        yancc_dat["a_minor"] = jnp.full(grid.num_rho, data["a"])
        yancc_dat["R_major"] = jnp.full(grid.num_rho, data["R0"])
        yancc_dat["iota"] = grid.compress(data["iota"], surface_label="rho")
        yancc_dat["rho"] = grid.compress(grid.nodes[:, 0], surface_label="rho")
        V = grid.compress(data["V(r)"])
        Vp = grid.compress(data["V_r(r)"]) / V[-1]
        Vpp = grid.compress(data["V_rr(r)"]) / V[-1]
        fields = jax.vmap(lambda d: yancc.field.Field(**d, NFP=grid.NFP))(yancc_dat)
        g, _ = manta_objective(with_itg((fields, Vp, Vpp), itg_builder, params), grid)
        jax.debug.callback(lambda v: G_seen.append(float(np.ravel(v)[0])), g)
        return 1 / g

    # yancc's angular grid in the optimisation is this grid
    grid = desc.grid.LinearGrid(rho=rho, M=(nt - 1) // 2, N=(nz - 1) // 2, NFP=eq.NFP)
    assert grid.num_theta == nt and grid.num_zeta == nz, (grid.num_theta, grid.num_zeta)
    terms = [
        (ObjectiveFromUserWithParams if itg_builder is not None else ObjectiveFromUser)(
            objective_from_user_fun, eq, target=0.0, weight=20.0, grid=grid,
            deriv_mode="fwd", name="MaNTA stored energy"),
    ]
    if args.all_terms:
        terms += [
            ObjectiveFromUser(lambda grid, data: grid.compress(data["mirror ratio"]),
                              thing=eq, grid=grid, bounds=(0.0, 0.2), weight=10.0,
                              name="mirror ratio"),
            Volume(eq=eq, target=V0, weight=100.0),
        ]
    objective = ObjectiveFunction(terms)
    objective.build(use_jit=False)

    pressure_rho = jnp.concatenate([jnp.zeros(1), rho, jnp.ones(1)])
    profile = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)

    def pressure_constraint_fun(params):
        p_l = params["p_l"]
        dp0 = profile(Grid(jnp.zeros((1, 3)), jitable=True), p_l, dr=1)
        p1 = profile(Grid(jnp.zeros((1, 3)).at[0, 0].set(1.0), jitable=True), p_l)
        return jnp.array([dp0, p1]).squeeze()

    k = args.fix_modes_above
    R_modes = np.vstack(([0, 0, 0], eq.surface.R_basis.modes[
        np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :]))
    Z_modes = eq.surface.Z_basis.modes[np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :]
    constraints = [
        ForceBalance(eq=eq),
        FixCurrent(eq=eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPsi(eq=eq),
        LinearObjectiveFromUser(pressure_constraint_fun, eq, target=jnp.array([0.0, 0.0])),
    ]
    # exactly what Optimizer.optimize builds for "proximal-lsq-exact"
    options = {}
    x_scale = _parse_x_scale("ess", [eq], options)
    proj, _ = get_combined_constraint_objectives(
        eq, constraints, objective, [eq], "proximal-lsq-exact", "lsq-exact", 1, options)
    x0 = proj.x(eq)
    x_scale = _project_x_scale(jnp.concatenate(x_scale), proj)
    f = np.asarray(proj.compute_scaled_error(x0, proj.constants))
    J = np.asarray(proj.jac_scaled_error(x0, proj.constants))
    names = [t.name for t in terms]
    dims = [t.dim_f for t in terms]
    return dict(f=f, J=J, x_scale=np.asarray(x_scale), G=np.array(G_seen[-1:]),
                term_dims=np.array(dims), term_names=np.array(names), x0=np.asarray(x0))


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


def compare(lo, hi, top):
    """What the optimizer would do with each, side by side."""
    out = {}
    for space, s_lo, s_hi in (("scaled", lo["x_scale"], hi["x_scale"]),
                              ("unscaled", 1.0, 1.0)):
        g_lo = (lo["J"] * s_lo).T @ lo["f"]
        g_hi = (hi["J"] * s_hi).T @ hi["f"]
        out[f"gradient_{space}"] = {
            "cosine": cosine(g_lo, g_hi),
            "low_is_descent_for_high": bool(g_lo @ g_hi > 0),
            "norm_ratio_low_over_high": float(np.linalg.norm(g_lo) / np.linalg.norm(g_hi)),
        }
    # Gauss-Newton steps in the scaled variables (lsq-exact without the trust
    # region), and the MaNTA term's gradient alone
    Js_lo, Js_hi = lo["J"] * lo["x_scale"], hi["J"] * hi["x_scale"]
    dx_lo = -np.linalg.pinv(Js_lo) @ lo["f"]
    dx_hi = -np.linalg.pinv(Js_hi) @ hi["f"]
    out["gauss_newton_step_scaled"] = {
        "cosine": cosine(dx_lo, dx_hi),
        # first-order change of the high-res objective 0.5|f|^2 along the
        # low-res step, against along its own
        "high_objective_change_along_low_step": float((Js_hi.T @ hi["f"]) @ dx_lo),
        "high_objective_change_along_high_step": float((Js_hi.T @ hi["f"]) @ dx_hi),
    }
    n0 = int(lo["term_dims"][0])
    gm_lo = (lo["J"][:n0] * lo["x_scale"]).T @ lo["f"][:n0]
    gm_hi = (hi["J"][:n0] * hi["x_scale"]).T @ hi["f"][:n0]
    out["manta_term_gradient_scaled"] = {
        "cosine": cosine(gm_lo, gm_hi),
        "low_is_descent_for_high": bool(gm_lo @ gm_hi > 0),
    }
    # the parameters that matter most at high resolution
    order = np.argsort(-np.abs(gm_hi))[:top]
    out["top_parameters"] = [
        {"index": int(i), "high": float(gm_hi[i]), "low": float(gm_lo[i]),
         "same_sign": bool(np.sign(gm_hi[i]) == np.sign(gm_lo[i]))}
        for i in order]
    w = np.abs(gm_hi)
    out["sign_agreement_weighted_by_high"] = float(
        w[np.sign(gm_hi) == np.sign(gm_lo)].sum() / w.sum())
    out["G"] = {"low": lo["G"].tolist(), "high": hi["G"].tolist()}
    out["f_manta"] = {"low": lo["f"][:n0].tolist(), "high": hi["f"][:n0].tolist()}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--eq", required=True, help="an Equilibrium or EquilibriaFamily .h5 "
                   "(a family's last member is used, its first sets the Volume target)")
    p.add_argument("--restart", help="MaNTA restart file both resolutions start from; "
                   "cold start (pseudo-transient) if omitted")
    p.add_argument("--low", type=parse_res, required=True, help="na,nx,ntheta,nzeta")
    p.add_argument("--high", type=parse_res, required=True, help="na,nx,ntheta,nzeta")
    p.add_argument("--all-terms", action="store_true",
                   help="include the driver's mirror-ratio and volume terms")
    p.add_argument("--fix-modes-above", type=int, default=5,
                   help="boundary modes with |m| or |n| above this are fixed (driver: 5)")
    p.add_argument("--Spi", type=float, default=600.0)
    p.add_argument("--Sn", type=float, default=1.2)
    p.add_argument("--keep-pressure", action="store_true",
                   help="keep the file's pressure profile; the driver resets it to zero")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--out", default="graddir")
    args = p.parse_args()

    if args.restart:
        args.restart = os.path.abspath(args.restart)
    import desc.io
    import jax.numpy as jnp
    from desc.equilibrium import EquilibriaFamily
    from desc.profiles import SplineProfile
    import manta as MaNTA

    loaded = desc.io.load(args.eq)
    family = loaded if isinstance(loaded, EquilibriaFamily) else [loaded]
    eq = family[-1].copy()
    V0 = family[0].compute("V")["V"]
    nodes = np.concatenate(([0.0], np.linspace(0.05, 0.99 * RHO_UPPER, NPOINTS - 1),
                            [RHO_UPPER]))
    if not args.keep_pressure:
        rho = jnp.array(MaNTA.getNodes(nodes, DEGREE))
        pressure_rho = jnp.concatenate([jnp.zeros(1), rho, jnp.ones(1)])
        eq.pressure = SplineProfile(jnp.zeros_like(pressure_rho), pressure_rho)

    results = {}
    for label, res in (("low", args.low), ("high", args.high)):
        path = f"{args.out}_{'_'.join(map(str, res))}.npz"
        if os.path.isfile(path) and not args.recompute:
            print(f"{label} {res}: reusing {path}", flush=True)
            results[label] = dict(np.load(path))
            continue
        print(f"{label} {res}: building the projected objective and its Jacobian",
              flush=True)
        r = jacobian_at(res, args, eq, V0, nodes)
        np.savez(path, **r)
        results[label] = r

    for k in ("x0", "x_scale"):
        if not np.allclose(results["low"][k], results["high"][k]):
            sys.exit(f"{k} differs between the resolutions: not the same point")
    cmp = compare(results["low"], results["high"], args.top)
    cmp["resolutions"] = {"low": list(args.low), "high": list(args.high)}
    with open(args.out + ".json", "w") as fh:
        json.dump(cmp, fh, indent=1)

    print(f"\nlow {args.low}  vs  high {args.high}   ({results['low']['J'].shape[1]} "
          f"free parameters after the projections)")
    print(f"  G: low {cmp['G']['low']}  high {cmp['G']['high']}")
    m = cmp["manta_term_gradient_scaled"]
    print(f"  MaNTA-term gradient (scaled):  cos = {m['cosine']:+.4f}   "
          f"low-res descent is descent for high-res: {m['low_is_descent_for_high']}")
    for space in ("scaled", "unscaled"):
        g = cmp[f"gradient_{space}"]
        print(f"  full gradient ({space:8s}):     cos = {g['cosine']:+.4f}   "
              f"|g_lo|/|g_hi| = {g['norm_ratio_low_over_high']:.3f}")
    gn = cmp["gauss_newton_step_scaled"]
    print(f"  Gauss-Newton step (scaled):    cos = {gn['cosine']:+.4f}   d(0.5|f_hi|^2) "
          f"along low step {gn['high_objective_change_along_low_step']:+.3e}, along own "
          f"{gn['high_objective_change_along_high_step']:+.3e}")
    print(f"  sign agreement, weighted by |g_hi|: "
          f"{cmp['sign_agreement_weighted_by_high']:.1%}")
    print("  largest high-res MaNTA-gradient components:")
    for t in cmp["top_parameters"]:
        print(f"    [{t['index']:4d}]  high {t['high']:+.3e}  low {t['low']:+.3e}"
              f"{'' if t['same_sign'] else '   <-- sign differs'}")
    print(f"\nwrote {args.out}.json")


if __name__ == "__main__":
    main()
