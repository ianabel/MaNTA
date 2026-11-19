"""The sources that hold a given profile steady, fitted at the ambipolar Er.

Given target profiles of n, Ti and Te (the config's initial profiles) on an
equilibrium, this finds the source heights for which those profiles are a
steady state of the multichannel case. Two things set it apart from a
pointwise fit of d(flux)/drho:

* **Er is ambipolar.** The fluxes are evaluated at the Er a MaNTA run would
  settle on, found per surface by yancc's own root finder,
  `solve_dke_ambipolar`. At any other Er the electron particle flux -- which is
  what the density channel carries -- and the heat fluxes are those of a state
  the solver never reaches, and so is the source fitted to them.

* **The fit is to the integrated balance.** At a steady state

      F(rho) = int_{rho_L}^{rho} S drho',      F = V' Gamma (outward),

  with S the V'-weighted source and no flux through the inner boundary. The fit
  matches that at every node, and *exactly* at the edge: the total source is
  what sets the edge flux, and with Dirichlet edge data the edge flux sets the
  gradient that the whole profile hangs from. Nothing is differentiated, so the
  noise of the drift-kinetic fluxes is not amplified, and nodes where the
  required source is negative are kept rather than dropped.

  Each channel's source is its config Gaussian (centre fixed, width optionally
  fitted) plus a source uniform in volume (`UniformParticleSource`,
  `UniformIonHeating`, `UniformElectronHeating`). With fixed widths the heights
  are a linear least-squares problem with one equality constraint. The
  integrals use each cell's degree-k interpolant through the MaNTA nodes, which
  is what the residual integrates (`I_h S`).

The sources already present at the target state -- the collisional exchange,
the fusion terms -- are kept and the fitted ones are added to them, so the
fitted heights are the config values to use directly.

Run under `TEST_STELLARATOR=true STELLARATOR_TEST_FLUX=physical` the fluxes and
the radial current come from the SI-units mock in `dke_sol.py`, and the root
find is a bracketing search on that mock's current, so the whole script runs
without a drift-kinetic solve; nothing else about it changes.
"""

import os

os.environ.setdefault("JAX_ENABLE_PINNED_HOST_TRANSFER", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import json
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.optimize import brentq, minimize_scalar

import desc
import desc.io
from desc.equilibrium import EquilibriaFamily
import manta as MaNTA
from manta.jax import State

from dke_sol import _species
from pitch_grids import AdaptivePitchGrid
from sources import _gaussian
from stellarator_multichannel import StellaratorTransport
from stellarator_state import Channel, StellaratorState
from yancc_wrapper import yancc_data

MOCK = "TEST_STELLARATOR" in os.environ

CHANNELS = (Channel.Density, Channel.IonEnergy, Channel.ElectronEnergy)
# per channel: the Gaussian's (height, centre, width) keys, and the uniform key
SOURCE_KEYS = {
    Channel.Density: (
        ("ParticleSourceHeight", "ParticleSourceCenter", "ParticleSourceWidth"),
        "UniformParticleSource",
    ),
    Channel.IonEnergy: (("NBIPower", "NBICenter", "NBIWidth"), "UniformIonHeating"),
    Channel.ElectronEnergy: (
        ("ECHPower", "ECHCenter", "ECHWidth"),
        "UniformElectronHeating",
    ),
}

# ------------------------------------------------------------------------- #
# Configuration                                                             #
# ------------------------------------------------------------------------- #

npoints = 4
degree = 4
nodes = np.linspace(0.0, 1.0, npoints + 1)

solver_config = {
    "OutputFilename": "source_fit",
    "PolynomialDegree": degree,
    "GridPoints": nodes,
    "GridSize": len(nodes) - 1,
    "tau": 10.0,
    "LowerBoundary": float(nodes[0]),
    "UpperBoundary": float(nodes[-1]),
    "RelativeTolerance": 1e-2,
    "AbsoluteTolerance": [1e-5],
    "delta_t": 1.0,
    "zeroFlux": True,
    "solveAdjoint": False,
}

# The target profiles are this table's initial profiles. The heights of the
# sources being fitted are ignored (set to zero before the fit); their centres
# and widths are the starting point.
st_config = {
    "ParticleSourceCenter": 0.0,
    "ParticleSourceHeight": 0.0,
    "ParticleSourceWidth": 0.5,
    "NBICenter": 0.0,
    "NBIPower": 0.0,
    "NBIWidth": 0.3,
    "ECHCenter": 0.0,
    "ECHPower": 0.0,
    "ECHWidth": 0.3,
    "EdgeTemperature": 0.5,
    "EdgeDensity": 0.4,
    "ADCoefficient": 0.0,
    "n0": 2.0,
    "T0": 5.0,
    "alpha": 6,
    "FusionFactor": 0.0,
    "evolveDensity": True,
    "useITGProxy": False,
    "useBatching": False,
    "nchunks": 1,
}


# ------------------------------------------------------------------------- #
# The physics at the target state                                           #
# ------------------------------------------------------------------------- #


def build_transport(eq, points, nt, nz, na, nx):
    st_cfg = dict(st_config)
    for channel in CHANNELS:
        (height, _, _), uniform = SOURCE_KEYS[channel]
        st_cfg[height] = 0.0
        st_cfg[uniform] = 0.0
    if st_cfg.get("useITGProxy", False):
        raise NotImplementedError("the ITG term is not wired into this script")
    yw = yancc_data.from_eq(points, eq=eq, nt=nt, nz=nz, na=na, nx=nx)
    st = StellaratorTransport(
        {"Stellarator": st_cfg, "Solver": solver_config}, yancc_wrapper=yw
    )
    st.layout = {len(points): (1, *yw.get_fields())}
    return st, yw


def target_state(st, points, Er):
    """The batched state MaNTA would hand ComputePhysics at the target
    profiles, with `Er` (normalised, one per node) in the Er aux slot."""
    Variable = jnp.stack([st.InitialValue(i, points) for i in range(3)]).T
    Derivative = jnp.stack(
        [jax.vmap(partial(st.InitialDerivative, i))(points) for i in range(3)]
    ).T
    Aux = jnp.stack([jnp.asarray(Er), jnp.zeros_like(points)]).T
    return {
        "Variable": Variable,
        "Derivative": Derivative,
        "Flux": jnp.zeros(Variable.shape),
        "Aux": Aux,
        "Scalars": [],
    }


def node_species(st, yw, points):
    """yancc's species at each node, exactly as the flux evaluation builds them."""
    params = st.params
    raw = target_state(st, points, jnp.zeros_like(points))
    states = StellaratorState.from_state_vmap(
        State.from_manta(raw), points, yw.Vp, yw.Vpp, params
    )
    return [
        _species(jax.tree.map(lambda a: a[i], states), params, True)
        for i in range(len(points))
    ]


def _bounds(field, species):
    """yancc's default search interval, E* = +-0.1 for the first species."""
    Escale = field.a_minor * species[0].v_thermal * field.Bmag_fsa
    return -0.1 * float(Escale), 0.1 * float(Escale)


def _mock_roots(field, species, num_roots, bounds, samples=201):
    """Every sign change of the mock's radial current, by bracketing."""
    from dke_sol import physical_test_moments

    def current(E):
        return float(physical_test_moments(species, jnp.array(E), field, None, None, True)[2])

    lo, hi = bounds if bounds is not None else _bounds(field, species)
    grid = np.linspace(lo, hi, samples)
    J = np.array([current(E) for E in grid])
    roots = [
        brentq(current, a, b, xtol=1e-12 * max(abs(lo), abs(hi)))
        for a, b, ja, jb in zip(grid[:-1], grid[1:], J[:-1], J[1:])
        if ja == 0.0 or ja * jb < 0.0
    ]
    out = np.full(num_roots, np.inf)
    out[: min(num_roots, len(roots))] = roots[:num_roots]
    return out


def ambipolar_roots(st, yw, points, num_roots=3, bounds=None, ftol=1e-4, rtol=1e-4, verbose=0):
    """Every ambipolar Er yancc finds at each node, in volts: shape
    (nodes, num_roots), inf where fewer roots were found."""
    if isinstance(yw.pitchgrid, AdaptivePitchGrid):
        raise NotImplementedError(
            "the adaptive pitch grid depends on Er; use a uniform one for the root find"
        )
    species = node_species(st, yw, points)
    out = np.full((len(points), num_roots), np.inf)
    for i, (field, sp) in enumerate(zip(yw.fields_unstacked, species)):
        if MOCK:
            out[i] = _mock_roots(field, sp, num_roots, bounds)
            continue
        from yancc import solve_dke_ambipolar

        Erho, _, info = solve_dke_ambipolar(
            field,
            yw.pitchgrid,
            yw.speedgrid,
            sp,
            num_roots,
            bounds=bounds,
            rtol=rtol,
            root_options={"ftol": ftol},
            multigrid_options={"smooth_solver": "dense", "max_grids": 3},
            verbose=verbose,
        )
        ok = np.asarray(info["success"])
        out[i] = np.where(ok, np.asarray(Erho), np.inf)
        print(
            f"  rho = {float(points[i]):.3f}: roots {np.sort(out[i][np.isfinite(out[i])])} V"
        )
    return out


def choose_root(roots, rule, guess):
    """One root per node. `nearest` takes the one closest to `guess` (volts):
    MaNTA starts its algebraic Er at zero and Newton goes to a nearby root,
    so with guess 0 this is the usual choice, not a guaranteed one. `ion`
    takes the most negative, `electron` the most positive."""
    chosen = np.empty(len(roots))
    for i, r in enumerate(roots):
        r = r[np.isfinite(r)]
        if r.size == 0:
            raise RuntimeError(f"no ambipolar root at node {i}")
        if rule == "ion":
            chosen[i] = r.min()
        elif rule == "electron":
            chosen[i] = r.max()
        else:
            chosen[i] = r[np.argmin(np.abs(r - guess))]
    return chosen


# ------------------------------------------------------------------------- #
# Integrals of nodal data, the way the residual sees them                   #
# ------------------------------------------------------------------------- #


class CellInterpolant:
    """Per-cell degree-k interpolation of values at MaNTA's nodes, and its
    cumulative integral from the lower boundary."""

    def __init__(self, faces, k, points):
        self.faces = np.asarray(faces, dtype=float)
        self.k = k
        self.points = np.asarray(points, dtype=float)
        n = k + 1
        assert self.points.size == n * (self.faces.size - 1)
        self.cells = [slice(i * n, (i + 1) * n) for i in range(self.faces.size - 1)]

    def _t(self, i, x):
        a, b = self.faces[i], self.faces[i + 1]
        return 2.0 * (np.asarray(x) - a) / (b - a) - 1.0

    def _coeffs(self, values):
        values = np.asarray(values, dtype=float)
        return [
            cheb.chebfit(self._t(i, self.points[c]), values[c], self.k)
            for i, c in enumerate(self.cells)
        ]

    def _cell_of(self, x):
        return np.clip(np.searchsorted(self.faces, x, side="right") - 1, 0, len(self.cells) - 1)

    def evaluate(self, values, x):
        co = self._coeffs(values)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.array([cheb.chebval(self._t(i, xi), co[i]) for xi, i in zip(x, self._cell_of(x))])

    def derivative(self, values, x):
        co = self._coeffs(values)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = []
        for xi, i in zip(x, self._cell_of(x)):
            h = self.faces[i + 1] - self.faces[i]
            out.append(cheb.chebval(self._t(i, xi), cheb.chebder(co[i])) * 2.0 / h)
        return np.array(out)

    def cumulative(self, values, x):
        """int_{faces[0]}^{x} of the interpolant, at each x."""
        co = self._coeffs(values)
        anti = [cheb.chebint(c, lbnd=-1.0) for c in co]
        half = 0.5 * np.diff(self.faces)
        whole = np.concatenate(
            [[0.0], np.cumsum([cheb.chebval(1.0, a) * h for a, h in zip(anti, half)])]
        )
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.array(
            [
                whole[i] + cheb.chebval(self._t(i, xi), anti[i]) * half[i]
                for xi, i in zip(x, self._cell_of(x))
            ]
        )


# ------------------------------------------------------------------------- #
# The fit                                                                   #
# ------------------------------------------------------------------------- #


def _solve_heights(A, a_edge, R, R_edge, w):
    """min ||W (A h - R)|| subject to a_edge . h = R_edge."""
    Aw, Rw = A * w[:, None], R * w
    nb = A.shape[1]
    K = np.zeros((nb + 1, nb + 1))
    K[:nb, :nb] = Aw.T @ Aw
    K[:nb, nb] = K[nb, :nb] = a_edge
    rhs = np.concatenate([Aw.T @ Rw, [R_edge]])
    h = np.linalg.lstsq(K, rhs, rcond=None)[0][:nb]
    return h, np.linalg.norm(Aw @ h - Rw)


def fit_channel(interp, vp, R, R_edge, centre, width, use_uniform, fit_width, width_bounds):
    """Heights (and width) of V' [h_g g(rho) + h_u] whose integral matches R."""
    x, edge = interp.points, interp.faces[-1]
    w = np.full(x.size, 1.0 / max(abs(R_edge), np.max(np.abs(R)), 1e-300))

    def columns(wd):
        cols = [vp * np.asarray(_gaussian(x, 1.0, centre, wd))]
        if use_uniform:
            cols.append(vp)
        A = np.stack([interp.cumulative(c, x) for c in cols], axis=1)
        a = np.array([interp.cumulative(c, [edge])[0] for c in cols])
        return A, a

    def misfit(wd):
        A, a = columns(wd)
        return _solve_heights(A, a, R, R_edge, w)[1]

    if fit_width:
        width = minimize_scalar(misfit, bounds=width_bounds, method="bounded").x
    A, a = columns(width)
    h, _ = _solve_heights(A, a, R, R_edge, w)
    fitted = A @ h
    return {
        "height": float(h[0]),
        "uniform": float(h[1]) if use_uniform else 0.0,
        "width": float(width),
        "centre": float(centre),
        "fitted_cumulative": fitted,
        "misfit": float(np.max(np.abs(fitted - R)) / max(abs(R_edge), 1e-300)),
    }


# ------------------------------------------------------------------------- #
# Driver                                                                    #
# ------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eq", default="eq_qa.h5", help="DESC equilibrium file")
    ap.add_argument("--resolution", type=int, nargs=4, default=(17, 33, 55, 7),
                    metavar=("NT", "NZ", "NA", "NX"))
    ap.add_argument("--root", choices=("nearest", "ion", "electron"), default="nearest")
    ap.add_argument("--er-guess", type=float, default=0.0,
                    help="volts; the root `nearest` picks is closest to this")
    ap.add_argument("--num-roots", type=int, default=3)
    ap.add_argument("--ftol", type=float, default=1e-4)
    ap.add_argument("--no-uniform", action="store_true",
                    help="fit the Gaussians alone (the edge total then fixes each height)")
    ap.add_argument("--fit-width", action="store_true")
    ap.add_argument("--plot", default="required_source.png")
    args = ap.parse_args()

    eq = desc.io.load(args.eq)
    if isinstance(eq, EquilibriaFamily):
        eq = eq[-1]
    points = np.asarray(MaNTA.getNodes(nodes, degree))
    st, yw = build_transport(eq, jnp.asarray(points), *args.resolution)
    c = st.params.constants
    vp = np.asarray(yw.Vp)

    print("Ambipolar Er at the target profiles")
    roots = ambipolar_roots(st, yw, jnp.asarray(points), args.num_roots, ftol=args.ftol)
    Erho = choose_root(roots, args.root, args.er_guess)
    multiple = np.sum(np.isfinite(roots), axis=1) > 1
    if multiple.any():
        print(f"  several roots at rho = {np.round(points[multiple], 3)}; took '{args.root}'")
    Er = Erho / float(c.T0eV)

    out = st.ComputePhysics(target_state(st, jnp.asarray(points), Er), points, 0.0)
    flux, existing, aux = out[0], out[1], out[2]
    current = np.asarray(aux[0])
    print(f"  max |V' J| at the chosen Er: {np.max(np.abs(current)):.3e} (normalised)")

    interp = CellInterpolant(nodes, degree, points)
    lower, edge = float(nodes[0]), float(nodes[-1])
    results, fitted_config = {}, {}
    for channel in CHANNELS:
        F = -np.asarray(flux[channel])  # outward V' Gamma
        S0 = np.asarray(existing[channel])
        F_edge = float(interp.evaluate(F, [edge])[0])
        F_lower = float(interp.evaluate(F, [lower])[0])
        R = F - interp.cumulative(S0, points)
        R_edge = F_edge - float(interp.cumulative(S0, [edge])[0])
        (height_key, centre_key, width_key), uniform_key = SOURCE_KEYS[channel]
        res = fit_channel(
            interp, vp, R, R_edge,
            centre=st_config[centre_key], width=st_config[width_key],
            use_uniform=not args.no_uniform, fit_width=args.fit_width,
            width_bounds=(0.05, 2.0),
        )
        res.update(R=R, R_edge=R_edge, F_lower=F_lower, F_edge=F_edge, S0=S0)
        results[channel] = res

        fitted_config[height_key] = res["height"]
        fitted_config[width_key] = res["width"]
        if not args.no_uniform:
            fitted_config[uniform_key] = res["uniform"]

        print(f"\n{channel.name}")
        print(f"  edge flux {F_edge:.4e}, existing sources supply {F_edge - R_edge:.4e}")
        if lower > 0.0 and abs(F_lower) > 1e-3 * abs(F_edge):
            print(f"  WARNING: the target carries {F_lower / F_edge:.1%} of the edge flux"
                  f" through rho = {lower}, which a zero-flux lower boundary cannot")
        print(f"  {height_key} = {res['height']:.6e}   {width_key} = {res['width']:.4f}")
        if not args.no_uniform:
            print(f"  {uniform_key} = {res['uniform']:.6e}")
        print(f"  worst cumulative misfit: {res['misfit']:.2%} of the edge total")
        if res["height"] < 0 or res["uniform"] < 0:
            print("  NOTE: a negative height -- that term is a sink")

    print("\nFitted source keys:")
    print(json.dumps(fitted_config, indent=4))

    if args.plot:
        plot(results, interp, vp, points, Erho, args.plot)


def plot(results, interp, vp, points, Erho, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    # from the first node out: V' vanishes on the axis, and the pointwise
    # source is the derivative of the integral divided by it
    fine = np.linspace(points[0], interp.faces[-1], 400)
    vp_fine = interp.evaluate(vp, fine)
    for j, channel in enumerate(CHANNELS):
        r = results[channel]
        ax[0, j].plot(points, r["R"], "o", label="required")
        ax[0, j].plot(points, r["fitted_cumulative"], "-", label="fitted")
        ax[0, j].plot(interp.faces[-1], r["R_edge"], "k*", label="edge (exact)")
        ax[0, j].set_title(f"{channel.name}: cumulative")
        ax[0, j].legend()
        # pointwise, for looking at only: the fit never differentiates
        required = interp.derivative(r["R"], fine) / np.maximum(vp_fine, 1e-12)
        fitted = np.asarray(_gaussian(fine, r["height"], r["centre"], r["width"])) + r["uniform"]
        ax[1, j].plot(fine, required, label="d(required)/drho / V'")
        ax[1, j].plot(fine, fitted, label="fitted source")
        ax[1, j].set_title(f"{channel.name}: source per volume")
        ax[1, j].legend()
    ax[0, 3].plot(points, Erho, "o-")
    ax[0, 3].set_title("ambipolar Er [V]")
    ax[1, 3].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
