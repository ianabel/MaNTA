"""Does yancc converge at a lower surface resolution in Boozer angles than in
DESC's?

yancc upwinds theta and zeta with separate stencils, and the numerical
diffusion that adds grows with the grid spacing, acting like extra collisions
(Conlin et al., the yancc paper). The stellarator optimisation builds its
fields with Field.from_desc, in DESC's angles, where field lines are curved on
the grid. This builds the same surface two ways --

* desc:   Field.from_desc(eq, rho, nt, nz)
* boozer: DESC's differentiable Boozer transform (|B|_mn_B, I, G -- what
          QuasisymmetryBoozer uses), |B| and its derivatives evaluated on a
          uniform (theta_B, zeta_B) grid, then Field.from_boozer

-- and solves each at a list of resolutions, reporting every solve's error
against its own angle system's reference (the last resolution) and against the
other system's. The two references should agree with each other; if they do
not, one of them is not converged and the comparison says nothing.

Two modes:

* --mode mdke: the monoenergetic DKE (solve_mdke) over nuhat = nu/v at a fixed
  erhohat; reports D11 and D31. Resolutions are na,ntheta,nzeta.
* --mode dke:  the full DKE (solve_dke), ions (D-T, A = 2.5) and electrons at
  fixed density, scanned in temperature with Er at fixed Er/T, as
  yancc_collisionality_scan.py does; reports J_rho, the ion and electron
  particle and heat fluxes and <J||B>. Resolutions are na,nx,ntheta,nzeta.

Before any solve it checks the two fields against each other -- iota, I = <B_theta>,
G = <B_zeta>, Psi, <|B|>, <|B|^2> and the Jacobian's sign, all independent of
the poloidal and toroidal angles -- and how far the Boozer |B| moves when M_booz, N_booz
grow by one eq.M, eq.N. A failed check stops the run (--force to go on), since a
Boozer field with a flipped sign or an unconverged spectrum would make every
comparison below meaningless. --check-only runs only these checks and needs no
GPU.

Pitch grid: --pitch adaptive (the default) clusters the pitch nodes where the
solution needs them, as the optimisation does with pitch_grid="adaptive":
yancc's MDKEGaussianPitchAngleGrid for --mode mdke, rebuilt per nuhat, and
pitch_grids.AdaptivePitchGrid (yancc's DKEGaussianPitchAngleGrid) for --mode
dke, rebuilt per plasma and Er. Both need yancc's nonuniform branch.
--pitch uniform uses UniformPitchAngleGrid.

Device memory: the "peak" column is jax's peak_bytes_in_use, which never
falls, so give resolutions in increasing order; each resolution runs both
angle systems before the next.

    python yancc_angle_scan.py precise_QH --check-only
    python yancc_angle_scan.py precise_QH --mode mdke --rho 0.5 \\
        --resolutions 45,13,17:45,17,25:45,27,33:121,43,73 --out angle_mdke.npz
    python yancc_angle_scan.py ../paper/eq5_all_equilibria.h5 --mode dke \\
        --rho 0.5 --resolutions 45,7,13,17:45,7,27,33:81,11,43,65

Equilibria are DESC example names, rescaled as runStellarator.py does
(a = 1.7 m, <B> = 5.86 T), or .h5 files (the last of a family, file.h5:0 for
the first). Needs yancc, DESC and jax; not MaNTA.
"""

import argparse
import os
import sys
import time

import numpy as np

COORDS = ("desc", "boozer")


def load_equilibrium(name):
    import desc.compat
    import desc.examples
    import desc.io

    path, _, index = name.rpartition(":") if ":" in name else (name, "", "")
    if os.path.isfile(path):
        eq = desc.io.load(path)
        if hasattr(eq, "__len__"):
            eq = eq[int(index) if index else -1]
        elif index:
            sys.exit(f"{path} holds one equilibrium; drop the :{index}")
        return eq
    eq = desc.examples.get(name)
    return desc.compat.rescale(eq, L=("a", 1.7), B=("<B>", 5.86), copy=True, verbose=0)


def parse_resolutions(text, mode):
    """mdke: na,nt,nz; dke: na,nx,nt,nz. ':'-separated, the last the reference."""
    want = 3 if mode == "mdke" else 4
    out = []
    for item in text.split(":"):
        vals = tuple(int(v) for v in item.split(","))
        if len(vals) != want:
            sys.exit(
                f"--mode {mode} takes " + ("na,ntheta,nzeta" if want == 3
                                           else "na,nx,ntheta,nzeta")
                + f" per resolution; got {item!r}"
            )
        if vals[-1] % 2 == 0 or vals[-2] % 2 == 0:
            sys.exit(f"ntheta and nzeta must be odd, got {item!r}")
        out.append(vals)
    return out


class BoozerSurface:
    """DESC's Boozer spectrum of |B| on one surface, and the surface's I, G,
    iota, evaluated onto any uniform (theta_B, zeta_B) grid."""

    def __init__(self, eq, rho, mfac):
        from desc.compute import get_transforms
        from desc.grid import LinearGrid

        self.eq, self.rho = eq, rho
        Mb, Nb = mfac * eq.M, mfac * eq.N
        grid = LinearGrid(rho=[rho], M=2 * Mb, N=2 * Nb, NFP=eq.NFP, sym=False)
        transforms = get_transforms(
            ["|B|_mn_B"], obj=eq, grid=grid, M_booz=Mb, N_booz=Nb
        )
        data = eq.compute(
            ["|B|_mn_B", "I", "G", "iota"], grid=grid, transforms=transforms
        )
        self.basis = transforms["B"].basis
        self.coeffs = np.asarray(data["|B|_mn_B"])
        self.I = float(data["I"][0])
        self.G = float(data["G"][0])
        self.iota = float(data["iota"][0])

    def on_grid(self, nt, nz):
        """|B|, d|B|/dtheta_B, d|B|/dzeta_B on yancc's (nt, nz) grid: theta in
        [0, 2 pi), zeta in [0, 2 pi / NFP), no endpoint -- yancc's Field.theta
        and Field.zeta."""
        import jax.numpy as jnp
        from desc.grid import Grid
        from desc.transform import Transform

        theta = np.linspace(0, 2 * np.pi, nt, endpoint=False)
        zeta = np.linspace(0, 2 * np.pi / self.eq.NFP, nz, endpoint=False)
        grid = Grid.create_meshgrid([self.rho, theta, zeta], NFP=self.eq.NFP)
        tr = Transform(grid, self.basis, derivs=1, build=True)
        c = jnp.asarray(self.coeffs)
        # theta fastest, as from_desc reshapes
        shape = lambda f: np.asarray(f).reshape((nt, nz), order="F")
        return (shape(tr.transform(c)), shape(tr.transform(c, dt=1)),
                shape(tr.transform(c, dz=1)))


def build_fields(eq, rho, nt, nz, boozer, scale):
    """{"desc": Field, "boozer": Field} for one surface and grid."""
    from yancc import Field

    fd = Field.from_desc(eq, rho, nt, nz)
    Bmag, dBdt, dBdz = boozer.on_grid(nt, nz)
    fb = Field.from_boozer(
        rho, Bmag, boozer.I, boozer.G, boozer.iota, eq.Psi,
        R_major=scale["R0"], a_minor=scale["a"], NFP=eq.NFP,
        dBdt=dBdt, dBdz=dBdz,
    )
    return {"desc": fd, "boozer": fb}


def check_fields(eq, rho, boozer, boozer_finer, tol):
    """Angle-independent quantities of the two fields; returns True if all agree."""
    import jax.numpy as jnp

    nt, nz = 4 * eq.M + 1, 4 * eq.N + 1  # fine enough that the averages are exact
    scale = surface_scale(eq)
    f = build_fields(eq, rho, nt, nz, boozer, scale)
    fd, fb = f["desc"], f["boozer"]

    def flux(field):  # Psi from the field geometry, as yancc's Field does
        return float(jnp.pi * (field.sqrtg * field.B_sup_z).mean() / field.rho)

    rows = [
        ("iota", float(fd.iota), float(fb.iota)),
        ("I = <B_theta> [T m]", float(fd.I), float(fb.I)),
        ("G = <B_zeta> [T m]", float(fd.G), float(fb.G)),
        ("Psi from sqrtg B^zeta [Wb]", flux(fd), flux(fb)),
        ("<|B|> [T]", float(fd.Bmag_fsa), float(fb.Bmag_fsa)),
        ("<|B|^2> [T^2]", float(fd.B2mag_fsa), float(fb.B2mag_fsa)),
        ("sign of sqrt(g)", float(np.sign(fd.sqrtg.mean())),
         float(np.sign(fb.sqrtg.mean()))),
    ]
    ok = True
    print(f"\nfield checks at rho = {rho}  (desc vs boozer, tolerance {tol:g})")
    for name, a, b in rows:
        # I vanishes for a current-free equilibrium: compare it against G's size
        ref = abs(float(fd.G)) if name.startswith("I ") else max(abs(a), 1e-300)
        err = abs(a - b) / ref
        good = err <= tol
        ok &= good
        print(f"  {name:28s} {a:+.6e} {b:+.6e}  rel {err:.1e}"
              f"{'' if good else '   <-- FAILS'}")

    Bc = boozer.on_grid(nt, nz)[0]
    Bf = boozer_finer.on_grid(nt, nz)[0]
    conv = float(np.max(np.abs(Bf - Bc)) / np.max(np.abs(Bf)))
    good = conv <= tol
    ok &= good
    print(f"  {'Boozer |B| vs one more M,N':28s} max change {conv:.1e}"
          f"{'' if good else '   <-- FAILS: raise --mbooz'}")
    return ok


def surface_scale(eq):
    """a and R0, as Field.from_desc takes them, so both fields carry the same."""
    from desc.grid import QuadratureGrid

    d = eq.compute(["a", "R0"], grid=QuadratureGrid(eq.L_grid, eq.M_grid, eq.N_grid, eq.NFP))
    return {"a": float(d["a"]), "R0": float(d["R0"])}


def make_solver(args, res):
    """The jitted solve for one resolution: field, scan value -> (outputs, info)."""
    import equinox as eqx
    import jax.numpy as jnp
    from yancc import UniformPitchAngleGrid
    from yancc.solve import solve_dke, solve_mdke

    mg = {"smooth_solver": args.smoother, "max_grids": args.grids}
    adaptive = args.pitch == "adaptive"
    if args.mode == "mdke":
        na = res[0]

        @eqx.filter_jit
        def solve(field, nu):
            er = jnp.asarray(args.erhohat)
            if adaptive:
                from yancc.velocity_grids import MDKEGaussianPitchAngleGrid

                # depends on nu and E*, so built per solve, inside the jit
                pitch = MDKEGaussianPitchAngleGrid(na, field, er, nu)
            else:
                pitch = UniformPitchAngleGrid(na)
            sol, info = solve_mdke(
                field, pitch, er, nu, multigrid_options=mg, rtol=args.rtol,
            )
            D = sol.get("Dij")
            return (jnp.stack([D[0, 0], D[2, 0]]),
                    info["res1"], info["success1"], info["nmv1"])

        return solve

    from yancc.species import Electron, LocalMaxwellian, Species
    from yancc.velocity_grids import MaxwellSpeedGrid

    na, nx = res[0], res[1]
    ions = Species(2.5, 1)
    if adaptive:
        # the grid MaNTA builds with pitch_grid="adaptive"
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pitch_grids import AdaptivePitchGrid, resolve_pitchgrid

    @eqx.filter_jit
    def solve(field, T):
        common = dict(density=args.n, dndrho=args.n * args.dlnn)
        species = [
            LocalMaxwellian(ions, temperature=T, dTdrho=T * args.dlnt, **common),
            LocalMaxwellian(Electron, temperature=T, dTdrho=T * args.dlnt, **common),
        ]
        pitch = (resolve_pitchgrid(AdaptivePitchGrid(na), field, species,
                                   args.er_over_t * T)
                 if adaptive else UniformPitchAngleGrid(na))
        sol, info = solve_dke(
            field, pitch, MaxwellSpeedGrid(nx), species,
            Erho=args.er_over_t * T, rtol=args.rtol, throw=False,
            multigrid_options=mg,
        )
        gamma, q = sol.get("<particle_flux>"), sol.get("<heat_flux>")
        out = jnp.stack([jnp.reshape(sol.get("J_rho"), ()), gamma[0], gamma[1],
                         q[0], q[1], jnp.reshape(sol.get("<J||B>"), ())])
        return out, info["res"], info["success"], info["niter"]

    return solve


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("eq", help="DESC .h5 (equilibrium or family), or an example name")
    p.add_argument("--mode", choices=("mdke", "dke"), default="mdke")
    p.add_argument("--coords", default="desc,boozer",
                   help="angle systems to solve in, of desc,boozer")
    p.add_argument("--rho", default="0.5")
    p.add_argument("--pitch", choices=("adaptive", "uniform"), default="adaptive",
                   help="pitch-angle grid; adaptive needs yancc's nonuniform branch")
    p.add_argument("--resolutions", default=None,
                   help="mdke na,ntheta,nzeta / dke na,nx,ntheta,nzeta, ':'-separated, "
                        "increasing; the last is the reference")
    p.add_argument("--mbooz", type=int, default=3,
                   help="M_booz, N_booz = this times eq.M, eq.N (DESC's default is 2)")
    p.add_argument("--check-tol", type=float, default=1e-4,
                   help="relative tolerance of the field checks")
    p.add_argument("--check-only", action="store_true",
                   help="run the field checks and stop; no yancc solve")
    p.add_argument("--force", action="store_true", help="solve even if a check fails")
    # mdke
    p.add_argument("--nu", default="1e-4,1e-3,1e-2",
                   help="mdke: nuhat = nu/v [1/m], ','-separated")
    p.add_argument("--erhohat", type=float, default=0.0,
                   help="mdke: E_rho / v [V s/m]; 0 keeps the 1/nu regime uncut")
    # dke
    p.add_argument("--temps", default="500,2000,8000", help="dke: T [eV], ','-separated")
    p.add_argument("--n", type=float, default=3e19, help="dke: density [m^-3]")
    p.add_argument("--dlnt", type=float, default=-1.0, help="dke: dlnT/drho")
    p.add_argument("--dlnn", type=float, default=-0.5, help="dke: dlnn/drho")
    p.add_argument("--er-over-t", type=float, default=-2.0, help="dke: E_rho / T")
    # solver
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--grids", type=int, default=3, help="multigrid max_grids")
    p.add_argument("--smoother", default="banded")
    p.add_argument("--out", help=".npz to write the results to")
    args = p.parse_args()
    if args.resolutions is None:
        args.resolutions = ("45,13,17:45,27,33:121,43,73" if args.mode == "mdke"
                            else "45,7,13,17:45,7,27,33:81,11,43,65")

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    if os.path.sep in args.eq or args.eq.endswith(".h5"):
        args.eq = os.path.abspath(args.eq)
    eq = load_equilibrium(args.eq)
    rhos = [float(r) for r in args.rho.split(",")]
    coords = args.coords.split(",")
    if not set(coords) <= set(COORDS):
        sys.exit(f"--coords takes some of {','.join(COORDS)}")
    resolutions = parse_resolutions(args.resolutions, args.mode)
    scale = surface_scale(eq)
    print(f"jax {jax.__version__} on {jax.devices()}; {args.eq}: NFP {eq.NFP}, "
          f"M {eq.M}, N {eq.N}, M_booz {args.mbooz * eq.M}, N_booz {args.mbooz * eq.N}, "
          f"{args.pitch} pitch grid",
          flush=True)

    boozers, all_ok = {}, True
    for rho in rhos:
        boozers[rho] = BoozerSurface(eq, rho, args.mbooz)
        finer = BoozerSurface(eq, rho, args.mbooz + 1)
        all_ok &= check_fields(eq, rho, boozers[rho], finer, args.check_tol)
    if args.check_only:
        sys.exit(0 if all_ok else 1)
    if args.pitch == "adaptive":
        import yancc.velocity_grids as vg

        need = ("MDKEGaussianPitchAngleGrid" if args.mode == "mdke"
                else "DKEGaussianPitchAngleGrid")
        if not hasattr(vg, need):
            sys.exit(f"--pitch adaptive needs yancc's {need}, from its nonuniform "
                     f"branch; this yancc ({vg.__file__}) lacks it. Install that "
                     "branch, or pass --pitch uniform")
    if not all_ok and not args.force:
        sys.exit("\na field check failed: the comparison would mean nothing "
                 "(--force to solve anyway)")

    if args.mode == "mdke":
        scan = [float(v) for v in args.nu.split(",")]
        scan_name, names = "nuhat", ["D11", "D31"]
    else:
        scan = [float(v) for v in args.temps.split(",")]
        scan_name, names = "T[eV]", ["J_rho", "Gamma_i", "Gamma_e", "Q_i", "Q_e",
                                     "<J||B>"]
    shape = (len(rhos), len(resolutions), len(coords), len(scan))
    vals = np.full(shape + (len(names),), np.nan)
    res_ = np.full(shape, np.nan)
    ok = np.zeros(shape, bool)
    work = np.zeros(shape, int)
    wall = np.full(shape, np.nan)
    peak = np.full(shape[:3], np.nan)
    tags = [",".join(map(str, r)) for r in resolutions]

    for ir, res in enumerate(resolutions):
        solve = make_solver(args, res)
        nt, nz = res[-2], res[-1]
        for ip, rho in enumerate(rhos):
            fields = build_fields(eq, rho, nt, nz, boozers[rho], scale)
            for ic, c in enumerate(coords):
                for k, s in enumerate(scan):
                    t0 = time.perf_counter()
                    out = jax.block_until_ready(solve(fields[c], jnp.asarray(s)))
                    idx = (ip, ir, ic, k)
                    wall[idx] = time.perf_counter() - t0
                    vals[idx] = np.asarray(out[0])
                    res_[idx], ok[idx], work[idx] = float(out[1]), bool(out[2]), int(out[3])
                    print(f"rho={rho} res={tags[ir]} {c:6s} {scan_name}={s:.2e}: "
                          + " ".join(f"{n}={v:+.4e}" for n, v in zip(names, vals[idx]))
                          + f"  res={res_[idx]:.1e} work={work[idx]}"
                          f" {'ok' if ok[idx] else 'NOT CONVERGED'} {wall[idx]:.1f}s",
                          flush=True)
                stats = jax.devices()[0].memory_stats() or {}
                if stats.get("peak_bytes_in_use") is not None:
                    peak[ip, ir, ic] = stats["peak_bytes_in_use"] / 1e9

    # relative error of each output against the references
    ref_own = vals[:, -1:]                      # (rho, 1, coords, scan, out)
    with np.errstate(invalid="ignore", divide="ignore"):
        err_own = np.abs(vals - ref_own) / np.abs(ref_own)
        if len(coords) == 2:
            ref_other = vals[:, -1:, ::-1]
            err_other = np.abs(vals - ref_other) / np.abs(ref_other)
            ref_gap = np.abs(vals[:, -1, 0] - vals[:, -1, 1]) / np.abs(vals[:, -1, 1])

    print("\nrelative error against the reference (last) resolution: "
          "own angle system" + (" / the other's" if len(coords) == 2 else ""))
    width = max(len(t) for t in tags)
    for ip, rho in enumerate(rhos):
        for k, s in enumerate(scan):
            print(f"\nrho = {rho}, {scan_name} = {s:.2e}")
            print(f"  {'resolution':>{width}s} {'coords':>6s} "
                  + " ".join(f"{n:>17s}" for n in names) + f" {'work':>6s} {'peak GB':>8s}")
            for ir in range(len(resolutions)):
                for ic, c in enumerate(coords):
                    cells = []
                    for o in range(len(names)):
                        e = f"{err_own[ip, ir, ic, k, o]:.1e}"
                        if len(coords) == 2:
                            e += f"/{err_other[ip, ir, ic, k, o]:.1e}"
                        cells.append(f"{e:>17s}")
                    flag = "" if ok[ip, ir, ic, k] else "  NOT CONVERGED"
                    pk = peak[ip, ir, ic]
                    print(f"  {tags[ir]:>{width}s} {c:>6s} " + " ".join(cells)
                          + f" {work[ip, ir, ic, k]:6d} "
                          + (f"{pk:8.2f}" if np.isfinite(pk) else f"{'--':>8s}") + flag)
            if len(coords) == 2:
                print(f"  references disagree by "
                      + " ".join(f"{n} {ref_gap[ip, k, o]:.1e}" for o, n in enumerate(names))
                      + "  (large: a reference is not converged)")

    if args.out:
        np.savez(args.out, mode=args.mode, eq=args.eq, rho=np.array(rhos),
                 coords=np.array(coords), resolutions=np.array(tags),
                 scan=np.array(scan), scan_name=scan_name, outputs=np.array(names),
                 values=vals, residual=res_, converged=ok, work=work, wall=wall,
                 peak_gb=peak, mbooz=args.mbooz, pitch=args.pitch)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
