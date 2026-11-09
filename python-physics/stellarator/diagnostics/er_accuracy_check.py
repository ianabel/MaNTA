"""How accurate is a converged run's Er, as far as yancc's phase-space
resolution goes?

Reads the final block of a MaNTA .dat file from the stellarator case
(algebraic Er: transported n, (3/2)p_i, (3/2)p_e, all times V', then the aux
Er and <J||B>), rebuilds at chosen radii exactly what the model handed yancc
(dke_sol._species, StellaratorState.ambipolar), and then for each yancc
resolution

* evaluates J at the run's own Er -- at the run's resolution this should be
  ~0, which checks the reconstruction;
* solves J(Er) = 0 for Er by the secant method from the run's Er, which is
  the ambipolar Er that resolution would have given.

Only DESC and yancc are needed. The equilibrium is the one
paper/desc-optimize-stored-energy.py builds with append = False (cached as
.h5 beside --out after the first solve), and the normalisation is that
driver's: n0 = 1e19, T0 = 1000 eV.

    python er_accuracy_check.py ../paper/stellarator_opteq8_acc.dat \\
        --resolutions 45,7,17,27:81,7,17,27:45,11,17,27:45,7,25,37:81,11,25,37
"""

import argparse
import json
import os
import sys
import time

import numpy as np

T0EV, N0 = 1000.0, 1e19
ELEMENTARY_CHARGE = 1.60217663e-19
# cell edges of the run (from the DG jumps in the .dat); radii are kept off them
EDGES = np.array([0, 0.051, 0.171, 0.291, 0.411, 0.531, 0.651, 0.771, 0.891, 0.9])


def read_final_block(path):
    blocks, cur = [], None
    for line in open(path):
        if line.startswith("# t ="):
            cur = []
            blocks.append(cur)
        elif line.startswith("#") or not line.strip():
            continue
        elif cur is not None:
            cur.append([float(v) for v in line.split()])
    a = np.array(blocks[-1])
    if a.shape[1] != 18:
        sys.exit(f"expected 18 columns (3 variables, Er, <J||B>); got {a.shape[1]}")
    return a


def build_equilibrium(cache):
    import desc
    import desc.io
    from desc.equilibrium import Equilibrium
    from desc.geometry import FourierRZToroidalSurface
    from desc.profiles import SplineProfile

    if os.path.isfile(cache):
        print(f"equilibrium from {cache}", flush=True)
        return desc.io.load(cache)
    print("solving the driver's initial equilibrium with DESC", flush=True)
    rho = np.linspace(0, 1, 12)
    surf = FourierRZToroidalSurface(
        R_lmn=[1, 0.125, 0.1], Z_lmn=[-0.125, -0.1],
        modes_R=[[0, 0], [1, 0], [0, 1]], modes_Z=[[-1, 0], [0, -1]], NFP=4,
    )
    eq = Equilibrium(M=4, N=4, Psi=0.1, surface=surf,
                     pressure=SplineProfile(np.zeros_like(rho), rho))
    eq = desc.compat.rescale(eq, L=("a", 1.7), B=("<B>", 5.86),
                             scale_pressure=False, copy=True, verbose=0)
    eq = eq.solve(x_scale="ess", verbose=0)[0]
    eq.save(cache)
    return eq


def volume_derivatives(eq, rho, rho_last):
    """V', V'' as yancc_wrapper.from_eq makes them: over V at the last node."""
    import desc.grid

    pts = np.append(np.asarray(rho), rho_last)
    grid = desc.grid.LinearGrid(rho=pts, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    d = eq.compute(["V(r)", "V_r(r)", "V_rr(r)"], grid=grid)
    V = grid.compress(d["V(r)"])
    return (np.asarray(grid.compress(d["V_r(r)"]) / V[-1])[:-1],
            np.asarray(grid.compress(d["V_rr(r)"]) / V[-1])[:-1])


def plasma_at(row, vp, vpp):
    """StellaratorState.ambipolar, in SI / eV."""
    u = {"n": row[1], "Ei": row[6], "Ee": row[11]}
    q = {"n": row[2], "Ei": row[7], "Ee": row[12]}
    val = lambda k: u[k] / vp
    der = lambda k: (q[k] * vp - vpp * u[k]) / vp**2
    n, dn = val("n"), der("n")
    pi, dpi = 2 / 3 * val("Ei"), 2 / 3 * der("Ei")
    pe, dpe = 2 / 3 * val("Ee"), 2 / 3 * der("Ee")
    Ti, Te = pi / n, pe / n
    return dict(n=n * N0, dn=dn * N0, Ti=Ti * T0EV, dTi=(dpi - dn * Ti) / n * T0EV,
                Te=Te * T0EV, dTe=(dpe - dn * Te) / n * T0EV,
                Er=row[16] * T0EV, Er_norm=row[16], Jbs_norm=row[17])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dat")
    p.add_argument("--rho", default="0.06,0.12,0.20,0.26,0.32,0.38,0.45,0.60,0.75,0.86")
    p.add_argument("--resolutions", default="45,7,17,27:81,11,25,37",
                   help="na,nx,ntheta,nzeta, ':'-separated; the first is the run's")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--grids", type=int, default=3)
    p.add_argument("--smoother", default="banded")
    p.add_argument("--no-root", action="store_true", help="only J at the run's Er")
    p.add_argument("--max-iter", type=int, default=8)
    p.add_argument("--out", default="er_accuracy.json")
    p.add_argument("--degree", type=int, default=4, help="the run's PolynomialDegree")
    p.add_argument("--edge-density", type=float, default=0.5,
                   help="the run's EdgeDensity, only for the V' check")
    p.add_argument("--density-scale", type=float, default=1.0,
                   help="multiply n and n' (e.g. if the run's V' normalisation "
                        "differs from yancc_wrapper.from_eq's)")
    a = p.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    import equinox as eqx
    import jax.numpy as jnp
    from yancc import Field
    from yancc.solve import solve_dke
    from yancc.species import Electron, LocalMaxwellian, Species, _nustar
    from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid

    print(f"jax {jax.__version__} on {jax.devices()}", flush=True)
    data = read_final_block(a.dat)
    x = data[:, 0]
    # Er is an aux variable: J = 0 holds at the physics nodes only (k+1
    # Chebyshev points of the first kind per cell), not between them. So snap
    # each requested rho to the nearest node and evaluate the cell's degree-k
    # polynomial there, fitted to the .dat's samples (exact up to print
    # precision).
    k = a.degree
    cheb = np.cos((2 * np.arange(k + 1) + 1) * np.pi / (2 * k + 2))[::-1]
    nodes = [(c, lo + (cheb + 1) / 2 * (hi - lo))
             for c, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:]))]
    allnodes = np.array([(c, xn) for c, xs in nodes for xn in xs])
    rows, rhos = [], []
    for r in (float(v) for v in a.rho.split(",")):
        c, xn = allnodes[np.argmin(np.abs(allnodes[:, 1] - r))]
        c = int(c)
        lo, hi = EDGES[c], EDGES[c + 1]
        m = (x > lo + 1e-4) & (x < hi - 1e-4)
        if m.sum() < k + 2:
            sys.exit(f"too few samples in cell {c} to fit degree {k}")
        xi = lambda v: 2 * (v - lo) / (hi - lo) - 1
        row = np.array([np.polyval(np.polyfit(xi(x[m]), data[m, col], k), xi(xn))
                        for col in range(data.shape[1])])
        row[0] = xn
        rows.append(row)
        rhos.append(xn)
    rhos = np.array(rhos)
    data_nodes = np.array(rows)

    eq = build_equilibrium(os.path.splitext(a.out)[0] + "_eq.h5")
    # the last physics node of the run, where yancc_wrapper takes V[-1]
    rho_last = nodes[-1][1][-1]
    vp, vpp = volume_derivatives(eq, rhos, rho_last)
    vp_edge, _ = volume_derivatives(eq, [0.9], rho_last)
    print(f"V'(0.9) = {vp_edge[0]:.5f} from the equilibrium; the run's boundary "
          f"values give {data[-1, 1] / a.edge_density:.5f} (n_edge = {a.edge_density})",
          flush=True)

    ions = Species(2.5, 1)
    plasmas = [plasma_at(row, vp[j], vpp[j]) for j, row in enumerate(data_nodes)]
    for pl in plasmas:
        pl["n"] *= a.density_scale
        pl["dn"] *= a.density_scale

    def species_of(pl):
        return [
            LocalMaxwellian(ions, temperature=pl["Ti"], density=pl["n"],
                            dTdrho=pl["dTi"], dndrho=pl["dn"]),
            LocalMaxwellian(Electron, temperature=pl["Te"], density=pl["n"],
                            dTdrho=pl["dTe"], dndrho=pl["dn"]),
        ]

    resolutions = [tuple(int(v) for v in r.split(",")) for r in a.resolutions.split(":")]
    results = {"rho": rhos.tolist(), "plasma": plasmas, "resolutions": resolutions,
               "J_at_run_Er": {}, "Er_root": {}, "nustar": {}, "gamma_i_at_run_Er": {}}

    def save():
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)

    print(f"\n{'rho':>6} {'n[1e19]':>8} {'Ti[eV]':>8} {'Te[eV]':>8} {'Er_run':>9}")
    for r, pl in zip(rhos, plasmas):
        print(f"{r:6.3f} {pl['n'] / 1e19:8.3f} {pl['Ti']:8.1f} {pl['Te']:8.1f} "
              f"{pl['Er_norm']:9.4f}")

    for res in resolutions:
        na, nx, nt, nz = res
        tag = "{},{},{},{}".format(*res)
        pitch, speed = UniformPitchAngleGrid(na), MaxwellSpeedGrid(nx)

        @eqx.filter_jit
        def solve(field, species, Er):
            sol, info = solve_dke(
                field, pitch, speed, species, Erho=Er, rtol=a.rtol, throw=False,
                verbose=0,
                multigrid_options={"smooth_solver": a.smoother, "max_grids": a.grids})
            return (jnp.reshape(sol.get("J_rho"), ()), sol.get("<particle_flux>")[0],
                    info["res"], info["success"])

        print(f"\n=== {tag} ===", flush=True)
        print(f"{'rho':>6} {'nu*_i':>9} {'nu*_e':>9} {'J(Er_run)':>11} {'cancel':>9} "
              f"{'Er_root':>9} {'dEr/Er':>8} {'it':>3} {'s':>6}", flush=True)
        for j, (r, pl) in enumerate(zip(rhos, plasmas)):
            t0 = time.time()
            field = Field.from_desc(eq, float(r), nt, nz)
            sp = species_of(pl)
            nui = float(_nustar(sp[0], field, 1.0, sp[1]))
            nue = float(_nustar(sp[1], field, 1.0, sp[0]))
            def J(E):
                ts = time.time()
                out = [float(v) for v in solve(field, sp, jnp.asarray(E))]
                print(f"    [{tag} rho={r:.3f}] Er={E / T0EV:+.5f} J={out[0]:+.4e} "
                      f"res={out[2]:.2e} ok={bool(out[3])} {time.time() - ts:.1f}s",
                      flush=True)
                return out
            E0 = pl["Er"]
            J0, g0, res0, ok0 = J(E0)
            cancel = abs(J0) / (ELEMENTARY_CHARGE * abs(g0) + 1e-300)
            root, it = float("nan"), 0
            if not a.no_root:
                # secant from the run's Er
                Ea, Ja = E0, J0
                Eb = E0 * 1.02 if E0 != 0 else 1.0
                Jb = J(Eb)[0]
                for it in range(1, a.max_iter + 1):
                    if Jb == Ja:
                        break
                    Ec = Eb - Jb * (Eb - Ea) / (Jb - Ja)
                    Ea, Ja, Eb = Eb, Jb, Ec
                    Jb = J(Eb)[0]
                    if abs(Eb - Ea) < 1e-4 * abs(Eb):
                        break
                root = Eb
            results["J_at_run_Er"].setdefault(tag, []).append(J0)
            results["gamma_i_at_run_Er"].setdefault(tag, []).append(g0)
            results["Er_root"].setdefault(tag, []).append(root / T0EV)
            results["nustar"].setdefault(tag, []).append([nui, nue])
            save()
            print(f"{r:6.3f} {nui:9.2e} {nue:9.2e} {J0:11.3e} {cancel:9.2e} "
                  f"{root / T0EV:9.4f} {(root - E0) / E0:8.1%} {it:3d} "
                  f"{time.time() - t0:6.1f}{'' if ok0 else '  (not converged)'}",
                  flush=True)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
