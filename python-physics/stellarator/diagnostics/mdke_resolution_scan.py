"""Where does each yancc resolution put the 1/nu regime?

The optimisation is after a reduction of 1/nu transport, which a resolution can
get right in trend while getting the radial current wrong by O(1). This sweeps
yancc's monoenergetic D11 over collisionality at E* = 0 -- no electric field,
so nothing cuts the 1/nu regime off into sqrt(nu) -- on DESC example
equilibria, at several pitch-angle and surface resolutions, and reports for
each:

* the local slope s = d ln D11 / d ln nu between neighbouring nu, which is -1
  in the 1/nu regime and ~0 on the plateau;
* the 1/nu range: the nu over which s stays within --band of -1. Where a low
  resolution's range ends at a larger nu than a high one's, it stops resolving
  the regime there -- the boundary layer at the trapped-passing boundary
  narrows as nu falls, and na is what has to resolve it;
* nu * D11 across that range, which is flat in the 1/nu regime (it is the
  eps_eff^(3/2) coefficient), and its ratio to the reference resolution's;
* yancc's relative residual and convergence flag for every solve: a solve that
  did not converge is no evidence either way.

The last resolution given is the reference. Collisionality is yancc's nuhat =
nu / v [1/m]; nu* = nuhat R0 / iota is printed beside it as the size-free
measure. Monoenergetic only: no speed grid, so a resolution here is
na,ntheta,nzeta (the stellarator objective's na,nx,ntheta,nzeta less nx).

    python mdke_resolution_scan.py --resolutions 45,17,27:121,33,53
    python mdke_resolution_scan.py --eqs W7-X --rho 0.2,0.5,0.8 \\
        --nu-min 1e-5 --nu-max 1e-1 --nnu 9 --out mdke.npz --plot mdke.png

Equilibria are DESC example names (W7-X, ARIES-CS, precise_QH, ...) or .h5
files (the last of a family). Needs only yancc, DESC and jax.
"""

import argparse
import os
import time

import numpy as np


def parse_resolutions(text):
    out = []
    for part in text.split(":"):
        na, nt, nz = (int(v) for v in part.split(","))
        if nt % 2 == 0 or nz % 2 == 0:
            raise SystemExit(f"ntheta and nzeta must be odd, got {part}")
        out.append((na, nt, nz))
    return out


def load_eq(name):
    import desc.examples
    import desc.io

    if os.path.isfile(name):
        eq = desc.io.load(name)
        return eq[-1] if hasattr(eq, "__len__") else eq
    return desc.examples.get(name)


def regime_range(nu, s, band):
    """The nu interval, as (high, low), of the longest run of slopes within
    band of -1, or None. Slope k sits between nu[k] and nu[k+1]."""
    ok = np.abs(s + 1.0) <= band
    best, start = None, None
    for k, v in enumerate(list(ok) + [False]):
        if v and start is None:
            start = k
        elif not v and start is not None:
            if best is None or k - start > best[1] - best[0]:
                best = (start, k)
            start = None
    if best is None:
        return None
    i, j = best  # slopes i..j-1, so points i..j
    return nu[i], nu[j]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--eqs", default="HELIOTRON,ATF,W7-X,DSHAPE")
    p.add_argument("--rho", default="0.5")
    p.add_argument(
        "--resolutions",
        default="45,17,27:171,43,73",
        help="na,ntheta,nzeta[:...]; the last is the reference",
    )
    p.add_argument("--nu-min", type=float, default=5e-5, help="nuhat = nu/v [1/m]")
    p.add_argument("--nu-max", type=float, default=1e-1)
    p.add_argument("--nnu", type=int, default=9)
    p.add_argument(
        "--erhohat",
        type=float,
        default=0.0,
        help="E_rho / v [V s/m]; 0 keeps the 1/nu regime uncut",
    )
    p.add_argument(
        "--band", type=float, default=0.15, help="|s + 1| within this counts as 1/nu"
    )
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--grids", type=int, default=4, help="multigrid max_grids")
    p.add_argument("--smoother", default="dense")
    p.add_argument("--out", help=".npz to write the results to")
    p.add_argument("--plot", help="image of nu*D11 against nu")
    args = p.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    import equinox as eqx
    import jax.numpy as jnp
    from yancc import Field, UniformPitchAngleGrid, solve_mdke

    names = args.eqs.split(",")
    rhos = [float(r) for r in args.rho.split(",")]
    resolutions = parse_resolutions(args.resolutions)
    nus = np.logspace(np.log10(args.nu_max), np.log10(args.nu_min), args.nnu)
    mg = {"smooth_solver": args.smoother, "max_grids": args.grids}

    @eqx.filter_jit
    def d11(field, pitchgrid, nu, er):
        sol, info = solve_mdke(
            field, pitchgrid, er, nu, multigrid_options=mg, rtol=args.rtol
        )
        return sol.get("Dij")[0, 0], info["res1"], info["success1"]

    shape = (len(names), len(rhos), len(resolutions), len(nus))
    D = np.full(shape, np.nan)
    res = np.full(shape, np.nan)
    ok = np.zeros(shape, bool)
    wall = np.full(shape, np.nan)
    nustar_per_nuhat = np.full((len(names), len(rhos)), np.nan)

    for a, name in enumerate(names):
        eq = load_eq(name)
        for b, rho in enumerate(rhos):
            for c, (na, nt, nz) in enumerate(resolutions):
                field = Field.from_desc(eq, rho, nt, nz)
                pitchgrid = UniformPitchAngleGrid(na)
                if c == len(resolutions) - 1:
                    nustar_per_nuhat[a, b] = float(field.R_major / jnp.abs(field.iota))
                for d, nu in enumerate(nus):
                    t = time.perf_counter()
                    out = d11(
                        field, pitchgrid, jnp.asarray(nu), jnp.asarray(args.erhohat)
                    )
                    D[a, b, c, d], res[a, b, c, d], ok[a, b, c, d] = (
                        float(out[0]),
                        float(out[1]),
                        bool(out[2]),
                    )
                    wall[a, b, c, d] = time.perf_counter() - t
                    print(
                        f"{name} rho={rho} res={na},{nt},{nz} nuhat={nu:.2e}: "
                        f"D11={D[a, b, c, d]:.4e} res={res[a, b, c, d]:.1e} "
                        f"{'ok' if ok[a, b, c, d] else 'NOT CONVERGED'} "
                        f"{wall[a, b, c, d]:.1f}s",
                        flush=True,
                    )

    lnu = np.log(nus)
    slope = np.diff(np.log(np.abs(D)), axis=-1) / np.diff(lnu)
    tag = [f"{na},{nt},{nz}" for na, nt, nz in resolutions]
    width = max(len(t) for t in tag)

    print("\nslope d ln D11 / d ln nu between neighbouring nuhat (-1 = 1/nu)")
    for a, name in enumerate(names):
        for b, rho in enumerate(rhos):
            k = nustar_per_nuhat[a, b]
            print(f"\n{name}, rho = {rho}   (nu* = {k:.3g} nuhat)")
            mids = np.sqrt(nus[:-1] * nus[1:])
            print(f"  {'nuhat':>{width}s} " + " ".join(f"{m:9.1e}" for m in mids))
            print(f"  {'nu*':>{width}s} " + " ".join(f"{m * k:9.1e}" for m in mids))
            for c in range(len(resolutions)):
                flags = "" if ok[a, b, c].all() else "   (some solves NOT converged)"
                print(
                    f"  {tag[c]:>{width}s} "
                    + " ".join(f"{s:+9.3f}" for s in slope[a, b, c])
                    + flags
                )
            print(f"  1/nu range (|s+1| <= {args.band}), and nu*D11 there:")
            ref = D[a, b, -1] * nus
            for c in range(len(resolutions)):
                r = regime_range(nus, slope[a, b, c], args.band)
                if r is None:
                    print(f"    {tag[c]:>{width}s}: none")
                    continue
                hi, lo = r
                sel = (nus <= hi) & (nus >= lo)
                nd = D[a, b, c, sel] * nus[sel]
                ratio = (nd / ref[sel]).mean()
                print(
                    f"    {tag[c]:>{width}s}: nuhat {lo:.1e} .. {hi:.1e} "
                    f"(nu* {lo * k:.1e} .. {hi * k:.1e})   "
                    f"nu*D11 {nd.min():.3e} .. {nd.max():.3e}   "
                    f"/ reference {ratio:.3f}"
                )

    if args.out:
        np.savez(
            args.out,
            names=np.array(names),
            rho=np.array(rhos),
            resolutions=np.array(resolutions),
            nuhat=nus,
            D11=D,
            slope=slope,
            residual=res,
            converged=ok,
            wall=wall,
            nustar_per_nuhat=nustar_per_nuhat,
            erhohat=args.erhohat,
        )
        print(f"\nwrote {args.out}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(
            len(rhos),
            len(names),
            figsize=(4 * len(names), 3 * len(rhos)),
            squeeze=False,
            layout="constrained",
        )
        for a, name in enumerate(names):
            for b, rho in enumerate(rhos):
                ax = axs[b, a]
                for c in range(len(resolutions)):
                    (line,) = ax.loglog(nus, nus * D[a, b, c], "o-", ms=3, label=tag[c])
                    bad = ~ok[a, b, c]
                    ax.loglog(
                        nus[bad],
                        (nus * D[a, b, c])[bad],
                        "x",
                        color=line.get_color(),
                        ms=8,
                    )
                ax.set_title(f"{name}, rho = {rho}")
                ax.set_xlabel("nuhat = nu / v [1/m]")
                ax.set_ylabel("nuhat * D11")
                ax.legend(fontsize=7)
        fig.suptitle("flat = 1/nu regime; x = solve not converged")
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
