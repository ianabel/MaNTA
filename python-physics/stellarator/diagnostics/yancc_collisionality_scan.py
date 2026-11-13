"""The radial current J from yancc's solve_dke against collisionality: fixed
density, temperature scanned, at two (or more) phase-space resolutions.

What it is for: telling apart the ways the ambipolarity row can go bad as nu*
falls (high power, low density). For each node and temperature it reports

* nu*_i, nu*_e -- yancc's own normalised collisionality at x = 1;
* J [A m^-3], and the ion and electron particle fluxes;
* cancel = |J| / (e Z_i |Gamma_i|) -- J is the small difference of two large
  charge fluxes near the ambipolar root, so a relative error eps in each flux is
  an error of about eps / cancel in J;
* yancc's Krylov `success`, relative residual `res`, and restarts `niter`.

Reading it:

* `success` false, or `res` above the rtol asked for: the solve is not
  converging, and nothing in dke_sol.py would say so (throw=False, info
  dropped).
* resolutions disagreeing at low nu*: the grid does not resolve the
  collisional boundary layers (they narrow like sqrt(nu)), however tight rtol.
* J jagged in T at one resolution: noise, which a Newton solve sees as a
  non-smooth residual.
* all smooth and agreeing: yancc is fine, and the trouble is elsewhere -- the
  initial guess, or several roots of J(Er).

The gradients are held at fixed scale lengths (dT/drho = T * dlnT/drho, the
same for n), and Er at a fixed Er/T, so that only the collisionality moves: the
ambipolar Er scales with T, and a fixed Er would sweep through the root as T
changes. Ti = Te. Only yancc, DESC and jax are needed, not MaNTA.

    python yancc_collisionality_scan.py W7-X --n 3e19 --rho 0.2,0.5,0.8
    python yancc_collisionality_scan.py ../paper/eq5_all_equilibria.h5 \\
        --tmin 300 --tmax 2e4 --ntemps 16 \\
        --resolutions 45,7,17,25:81,11,25,37 --out scan.npz --plot scan.png

A "g" after na (45g,7,17,25) uses the adaptive pitch grid MaNTA builds with
pitch_grid="adaptive" (pitch_grids.py): yancc's DKEGaussianPitchAngleGrid,
rebuilt for each solve's plasma and Er. It needs yancc's nonuniform branch.

A name that is not a file is taken as a DESC example, rescaled as
runStellarator.py does (a = 1.7 m, <B> = 5.86 T).

With --jacrev it measures memory instead of scanning: for each resolution, the
peak device memory of the jacrev the model's ComputePhysicsDerivatives takes --
the five outputs it uses (electron particle flux, both heat fluxes, J, <J||B>)
against the seven plasma inputs (n, n', Ti, Ti', Te, Te', Er) -- vmapped over
a batch of nodes, each with its own field. Each (resolution, batch) runs in a
subprocess of its own, because the peak is per process. Two batch sizes split
the peak into a fixed part and a per-node part, and project it to
--nodes-per-card:

    python yancc_collisionality_scan.py W7-X --jacrev \\
        --resolutions 45,7,17,25:45,9,17,25:55,7,17,25 --batch 1,2 \\
        --nodes-per-card 12 --temp 2000
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ELEMENTARY_CHARGE = 1.60217663e-19


def load_equilibrium(name):
    import desc.compat
    import desc.examples
    import desc.io

    if os.path.isfile(name):
        eq = desc.io.load(name)
        if hasattr(eq, "__len__"):  # an EquilibriaFamily: the final equilibrium
            eq = eq[-1]
        return eq
    eq = desc.examples.get(name)
    return desc.compat.rescale(eq, L=("a", 1.7), B=("<B>", 5.86), copy=True, verbose=0)


def parse_resolutions(text):
    """'na,nx,nt,nz:na,nx,nt,nz' -> [(na, nx, nt, nz), ...]; a "g" after na
    (adaptive_flags) is dropped here."""
    out = []
    for item in text.split(":"):
        vals = tuple(int(v.rstrip("g")) for v in item.split(","))
        if len(vals) != 4:
            sys.exit(f"a resolution is na,nx,ntheta,nzeta; got {item!r}")
        out.append(vals)
    return out


def adaptive_flags(text):
    """Which resolutions ask for the adaptive pitch grid: a "g" after na."""
    return [item.split(",")[0].endswith("g") for item in text.split(":")]


def scan(args):
    import jax

    jax.config.update("jax_enable_x64", True)
    import equinox as eqx
    import jax.numpy as jnp
    from yancc import Field
    from yancc.solve import solve_dke
    from yancc.species import Electron, LocalMaxwellian, Species, _nustar
    from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid

    print(f"jax {jax.__version__} on {jax.devices()}", flush=True)
    eq = load_equilibrium(args.eq)
    ions = Species(2.5, 1)  # ion_species.DeuteriumTritium, as in the model
    Z_i = 1.0

    rhos = [float(r) for r in args.rho.split(",")]
    temps = np.logspace(np.log10(args.tmin), np.log10(args.tmax), args.ntemps)
    resolutions = parse_resolutions(args.resolutions)
    adaptive = adaptive_flags(args.resolutions)
    if any(adaptive):
        # the grid MaNTA builds with pitch_grid="adaptive"
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pitch_grids import AdaptivePitchGrid, resolve_pitchgrid

    def plasma(T):
        n = args.n
        common = dict(density=n, dndrho=n * args.dlnn)
        return (
            LocalMaxwellian(ions, temperature=T, dTdrho=T * args.dlnt, **common),
            LocalMaxwellian(Electron, temperature=T, dTdrho=T * args.dlnt, **common),
        )

    shape = (len(resolutions), len(rhos), len(temps))
    out = {
        k: np.full(shape, np.nan)
        for k in ("J", "gamma_i", "gamma_e", "cancel", "res", "niter", "success",
                  "seconds")
    }
    nustar = np.full((2, len(rhos), len(temps)), np.nan)

    for ir, (na, nx, nt, nz) in enumerate(resolutions):
        speed = MaxwellSpeedGrid(nx)
        pitch = AdaptivePitchGrid(na) if adaptive[ir] else UniformPitchAngleGrid(na)

        @eqx.filter_jit
        def solve(field, T):
            ion, ele = plasma(T)
            pg = pitch
            if adaptive[ir]:
                # rebuilt for each solve's plasma and Er, as in MaNTA
                pg = resolve_pitchgrid(pitch, field, [ion, ele], args.er_over_t * T)
            sol, info = solve_dke(
                field, pg, speed, [ion, ele],
                Erho=args.er_over_t * T,
                rtol=args.rtol, throw=False, verbose=args.verbose,
                multigrid_options={"smooth_solver": args.smoother,
                                   "max_grids": args.grids},
            )
            gamma = sol.get("<particle_flux>")
            return (jnp.reshape(sol.get("J_rho"), ()), gamma[0], gamma[1],
                    info["res"], info["niter"], info["success"])

        g = " (adaptive pitch grid)" if adaptive[ir] else ""
        print(f"\n=== na={na} nx={nx} ntheta={nt} nzeta={nz}{g} ===", flush=True)
        print(f"{'rho':>6} {'T[eV]':>9} {'nu*_i':>9} {'nu*_e':>9} {'J[A/m3]':>11} "
              f"{'cancel':>9} {'res':>9} {'niter':>5} {'ok':>3} {'s':>6}", flush=True)
        for ip, rho in enumerate(rhos):
            field = Field.from_desc(eq, rho, nt, nz)
            for it, T in enumerate(temps):
                if ir == 0:
                    ion, ele = plasma(T)
                    nustar[0, ip, it] = float(_nustar(ion, field, 1.0, ele))
                    nustar[1, ip, it] = float(_nustar(ele, field, 1.0, ion))
                t0 = time.time()
                J, gi, ge, res, niter, ok = jax.block_until_ready(
                    solve(field, jnp.asarray(T)))
                idx = (ir, ip, it)
                out["seconds"][idx] = time.time() - t0
                out["J"][idx], out["gamma_i"][idx] = float(J), float(gi)
                out["gamma_e"][idx] = float(ge)
                out["cancel"][idx] = abs(float(J)) / (
                    ELEMENTARY_CHARGE * Z_i * abs(float(gi)) + 1e-300)
                out["res"][idx], out["niter"][idx] = float(res), float(niter)
                out["success"][idx] = float(bool(ok))
                print(f"{rho:6.3f} {T:9.1f} {nustar[0, ip, it]:9.2e} "
                      f"{nustar[1, ip, it]:9.2e} {float(J):11.3e} "
                      f"{out['cancel'][idx]:9.2e} {float(res):9.2e} "
                      f"{int(niter):5d} {'y' if ok else 'NO':>3} "
                      f"{out['seconds'][idx]:6.1f}", flush=True)

    if args.out:
        np.savez(args.out, rho=np.array(rhos), T=temps, nustar_i=nustar[0],
                 nustar_e=nustar[1], resolutions=np.array(resolutions),
                 adaptive=np.array(adaptive),
                 n=args.n, er_over_t=args.er_over_t, rtol=args.rtol, **out)
        print(f"\nwrote {args.out}")
    if args.plot:
        plot(args.plot, rhos, nustar, resolutions, out, args.rtol)


def memory_worker(args):
    """One jacrev at one resolution and batch size; prints a JSON line."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import equinox as eqx
    import jax.numpy as jnp
    from yancc import Field
    from yancc.solve import solve_dke
    from yancc.species import Electron, LocalMaxwellian, Species
    from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid

    (na, nx, nt, nz), = parse_resolutions(args.resolutions)
    batch = int(args.batch)
    eq = load_equilibrium(args.eq)
    ions = Species(2.5, 1)
    pitch, speed = UniformPitchAngleGrid(na), MaxwellSpeedGrid(nx)
    rhos = jnp.linspace(0.2, 0.8, batch) if batch > 1 else jnp.array([0.5])
    fields = jax.vmap(lambda r: Field.from_desc(eq, r, nt, nz))(rhos)

    def outputs(p, field):
        n, dn, Ti, dTi, Te, dTe, Er = p
        species = [
            LocalMaxwellian(ions, temperature=Ti, density=n, dTdrho=dTi, dndrho=dn),
            LocalMaxwellian(Electron, temperature=Te, density=n, dTdrho=dTe,
                            dndrho=dn),
        ]
        sol, _ = solve_dke(
            field, pitch, speed, species, Erho=Er, rtol=args.rtol, throw=False,
            verbose=args.verbose,
            multigrid_options={"smooth_solver": args.smoother,
                               "max_grids": args.grids},
        )
        # what dke_sol._normalized_fluxes keeps
        return jnp.stack([
            sol.get("<particle_flux>")[1],
            sol.get("<heat_flux>")[0],
            sol.get("<heat_flux>")[1],
            jnp.reshape(sol.get("J_rho"), ()),
            jnp.reshape(sol.get("<J||B>"), ()),
        ])

    T, n = args.temp, args.n
    p0 = jnp.array([n, n * args.dlnn, T, T * args.dlnt, T, T * args.dlnt,
                    args.er_over_t * T])
    # distinct per node, as a real batch is
    ps = p0[None, :] * (1.0 + 1e-3 * jnp.arange(batch))[:, None]
    fn = eqx.filter_jit(eqx.filter_vmap(jax.jacrev(outputs),
                                        in_axes=(0, eqx.if_array(0))))
    t0 = time.time()
    jac = jax.block_until_ready(fn(ps, fields))
    seconds = time.time() - t0

    stats = jax.devices()[0].memory_stats() or {}
    print("MEMORY " + json.dumps({
        "resolution": [na, nx, nt, nz], "batch": batch,
        "N": na * nx * nt * nz * 2,
        "peak": stats.get("peak_bytes_in_use"),
        "limit": stats.get("bytes_limit"),
        "seconds": seconds,
        "finite": bool(jnp.all(jnp.isfinite(jac))),
        "device": str(jax.devices()[0]),
    }), flush=True)


def memory(args):
    if any(adaptive_flags(args.resolutions)):
        sys.exit("--jacrev measures uniform pitch grids only; drop the g")
    """Run memory_worker per (resolution, batch); fit and project."""
    batches = [int(b) for b in args.batch.split(",")]
    resolutions = parse_resolutions(args.resolutions)
    passthrough = [args.eq] + [
        f"--{k.replace('_', '-')}={getattr(args, k)}"
        for k in ("n", "temp", "dlnt", "dlnn", "er_over_t", "rtol", "grids",
                  "smoother", "verbose")
    ]
    rows = {}
    for res in resolutions:
        for b in batches:
            tag = "{},{},{},{}".format(*res)
            print(f"--- {tag}, batch {b} ---", flush=True)
            proc = subprocess.run(
                [sys.executable, __file__, *passthrough, "--jacrev",
                 f"--resolutions={tag}", f"--batch={b}", "--memory-worker"],
                capture_output=True, text=True,
            )
            lines = (proc.stdout + proc.stderr).splitlines()
            found = [l for l in lines if l.startswith("MEMORY ")]
            if proc.returncode != 0 or not found:
                oom = any("RESOURCE_EXHAUSTED" in l or "out of memory" in l.lower()
                          for l in lines)
                print("  OUT OF MEMORY" if oom else
                      f"  FAILED (exit {proc.returncode})\n  " +
                      "\n  ".join(lines[-15:]), flush=True)
                rows[(tuple(res), b)] = None
                continue
            r = json.loads(found[0][len("MEMORY "):])
            rows[(tuple(res), b)] = r
            peak = r["peak"]
            print(f"  peak {peak / 1e9:.2f} GB" if peak is not None else
                  "  peak unavailable (memory_stats empty: not a GPU?)",
                  f" {r['seconds']:.1f}s  finite={r['finite']}  on {r['device']}",
                  flush=True)

    gb = 1e9
    print(f"\n{'resolution':>16} {'N':>9} " +
          " ".join(f"{'b=' + str(b):>8}" for b in batches) +
          f" {'per node':>9} {'fixed':>7} {'x' + str(args.nodes_per_card) + ' nodes':>10}")
    for res in resolutions:
        peaks = [rows[(tuple(res), b)] and rows[(tuple(res), b)]["peak"]
                 for b in batches]
        N = res[0] * res[1] * res[2] * res[3] * 2
        cells = " ".join(f"{p / gb:8.2f}" if p else f"{'--':>8}" for p in peaks)
        known = [(b, p) for b, p in zip(batches, peaks) if p]
        if len(known) >= 2:
            (b1, p1), (b2, p2) = known[0], known[-1]
            per = (p2 - p1) / (b2 - b1)
            fixed = p1 - per * b1
            proj = fixed + per * args.nodes_per_card
            tail = f" {per / gb:9.2f} {fixed / gb:7.2f} {proj / gb:10.1f}"
        else:
            tail = f" {'--':>9} {'--':>7} {'--':>10}"
        print(f"{'{},{},{},{}'.format(*res):>16} {N:9d} {cells}{tail}")
    print("\nGB = 1e9 bytes. 'x nodes' is fixed + per-node x --nodes-per-card, "
          "the estimate to compare with the card's memory (an A100-40GB has "
          "about 39.6e9 bytes usable).")


def plot(path, rhos, nustar, resolutions, out, rtol):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, len(rhos), figsize=(4 * len(rhos), 7),
                            squeeze=False, sharex="col")
    for ip, rho in enumerate(rhos):
        x = nustar[0, ip]
        for ir, res in enumerate(resolutions):
            label = "na={} nx={} nt={} nz={}".format(*res)
            J = out["J"][ir, ip]
            line, = axs[0, ip].loglog(x, np.abs(J), "-", label=label)
            neg = J < 0
            axs[0, ip].loglog(x[neg], np.abs(J[neg]), "v", color=line.get_color())
            axs[0, ip].loglog(x[~neg], np.abs(J[~neg]), "^", color=line.get_color())
            bad = out["success"][ir, ip] == 0
            axs[1, ip].loglog(x, out["res"][ir, ip], "-o", color=line.get_color())
            axs[1, ip].loglog(x[bad], out["res"][ir, ip][bad], "x", color="r", ms=10)
        axs[1, ip].axhline(rtol, color="k", ls="--", lw=0.8)
        axs[0, ip].set_title(rf"$\rho = {rho}$")
        axs[1, ip].set_xlabel(r"$\nu^*_i$")
    axs[0, 0].set_ylabel(r"$|J_\rho|$ [A m$^{-3}$]  ($\blacktriangle$ +, $\blacktriangledown$ -)")
    axs[1, 0].set_ylabel("Krylov residual (x = not converged)")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("eq", help="DESC .h5 (equilibrium or family), or an example name")
    p.add_argument("--n", type=float, default=3e19, help="density [m^-3], fixed")
    p.add_argument("--tmin", type=float, default=200.0, help="[eV]")
    p.add_argument("--tmax", type=float, default=1e4, help="[eV]")
    p.add_argument("--ntemps", type=int, default=12)
    p.add_argument("--rho", default="0.2,0.5,0.8")
    p.add_argument("--dlnt", type=float, default=-1.0, help="dlnT/drho, fixed")
    p.add_argument("--dlnn", type=float, default=-0.5, help="dlnn/drho, fixed")
    p.add_argument("--er-over-t", type=float, default=-2.0,
                   help="E_rho / T [1/rho], fixed; -2 is the state "
                        "yancc_smoother_test.py uses")
    p.add_argument("--resolutions", default="45,7,17,25:81,11,25,37",
                   help="na,nx,ntheta,nzeta per resolution, ':'-separated")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--grids", type=int, default=3, help="multigrid max_grids")
    p.add_argument("--smoother", default="banded")
    p.add_argument("--out", help=".npz to write the results to")
    p.add_argument("--plot", help="image to write |J| and the residual to")
    p.add_argument("--verbose", type=int, default=0, help="yancc's verbose")
    p.add_argument("--jacrev", action="store_true",
                   help="measure the jacrev's peak device memory instead of scanning")
    p.add_argument("--batch", default="1,2",
                   help="--jacrev: nodes vmapped together, ','-separated; two or "
                        "more split the peak into fixed and per-node parts")
    p.add_argument("--nodes-per-card", type=int, default=12,
                   help="--jacrev: nodes one card evaluates at once, to project to")
    p.add_argument("--temp", type=float, default=2000.0,
                   help="--jacrev: the temperature [eV] of the point measured")
    p.add_argument("--memory-worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if os.path.sep in args.eq or args.eq.endswith(".h5"):
        args.eq = os.path.abspath(args.eq)
        if not os.path.isfile(args.eq):
            sys.exit(f"no equilibrium file at {args.eq}: the path is taken "
                     f"relative to the directory you run from ({os.getcwd()})")
    if args.memory_worker:
        memory_worker(args)
    elif args.jacrev:
        memory(args)
    else:
        scan(args)


if __name__ == "__main__":
    main()
