#!/usr/bin/env python
"""Geometric features the GX/ML ITG paper found the heat flux most sensitive to.

Landreman et al. (the gyrokinetics machine learning paper behind the NN ITG proxy)
ran sequential feature selection over ~10^6 candidate features built from the GX
geometry coefficients, and the twelve it kept for the XGBoost fit (their Fig. 13,
the SHAP figure) are a/LT, a/Ln and the ten geometric ones below. Their Fig. 14
adds a single hand-optimised feature that alone correlates well with Q. This
script builds the same flux tube the training set used for each equilibrium in a
DESC file, evaluates those features, and shows where each equilibrium sits in the
training distribution.

The definitions are transcribed from the zenodo archive's `gx_features`
(feature_sets.unary_funcs_20241227 / reductions_20241227 / compute_fn_20241214,
combinations.add_local_shear, calculations.differentiate) and the feature names
in figure13/20241228-04-GX_ML_figure_of_Shapley_values.py. They act on the GX
geometry arrays with the dataset's shat-free variants

    gbdrift0_over_shat = gbdrift0 / shat
    gds21_over_shat = gds21 / shat
    gds22_over_shat_squared = gds22 / shat^2      (= |grad x|^2)

on 96 points along the tube (the eik file's nz = 97, the repeated end point
dropped). Several features (the FFT coefficient, the L1 norm, the derivative in
the local shear) depend on that grid, so the tube is built exactly as the
training set's was: length 75.4 a (2 pi * 12 m at a = 1 m), sigma_Bxy = -1,
nz = 97. Change those and the numbers are no longer comparable with the paper's.

With --dataset (the archive's 20250102-01_GX_stellarator_dataset.h5) the same
functions are applied to the training set's geometry tensor, and each value is
reported as a percentile of that distribution, with histograms.
"""

import argparse
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))  # the stellarator modules
from gx_inputs_from_manta import DEFAULT_GX_GEOMETRY  # noqa: E402

ZENODO = os.path.join(
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")),
    "random",
    "20250119-01-gyrokinetics_machine_learning_zenodo",
)
DEFAULT_DATASET = os.path.join(ZENODO, "20250102-01_GX_stellarator_dataset.h5")

# the order of the dataset's raw_feature_tensor (z_functions_GX)
Z_FUNCTIONS = [
    "bmag",
    "gbdrift",
    "cvdrift",
    "gbdrift0_over_shat",
    "gds2",
    "gds21_over_shat",
    "gds22_over_shat_squared",
]


def differentiate(f):
    """gx_features.calculations.differentiate: centred, periodic, dz = 2 pi / nz."""
    dz = 2 * np.pi / f.shape[1]
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dz)


def heaviside(f):
    return np.heaviside(f, 0.0)


def relu(f):
    return np.where(f > 0, f, 0.0)


def features(T):
    """The paper's features of a geometry tensor T of shape (n_tubes, n_z, 7)."""
    B, gb, cv, gb0, gds2, gds21, gds22 = (T[:, :, j] for j in range(7))
    S = differentiate(gds21 / gds22)  # combinations.add_local_shear
    out = {}
    # Fig. 13, in the order sequential feature selection picked them
    out["mean(Θ(B×κ·∇y) |∇x|⁴/B²)"] = np.mean((gds22 / B * heaviside(cv)) ** 2, axis=1)
    out["median((B×∇B·∇x)² |∇x|⁸/B⁸)"] = np.median((gds22**2 * gb0 / B) ** 2, axis=1)
    out["|FFT₁|(ReLU(B×∇B·∇x)/B⁵)"] = np.abs(np.fft.fft(relu(gb0) / B**2, axis=1))[:, 1]
    out["mean((B×∇B·∇y)/B⁴)"] = np.mean(gb / B, axis=1)
    out["var(ReLU(B×∇B·∇y)/(|∇y|² B⁴))"] = np.var(relu(gb) / gds2 / B, axis=1)
    out["mean(|∇x|⁴/B⁶)"] = np.mean((gds22 / B**3) ** 2, axis=1)
    out["mean((B×∇B·∇x)(∇x·∇y)/B⁶)"] = np.mean(gds21 * gb0 / B**3, axis=1)
    out["var(Θ(B×∇B·∇y)/|∇y|⁴)"] = np.var((heaviside(gb) / gds2) ** 2, axis=1)
    out["‖∂z(∇x·∇y/|∇x|²)/B³‖₁"] = np.sum(np.abs(S / B**3), axis=1)
    out["mean((B×∇B·∇x)²/B⁴)"] = np.mean((gb0 * B) ** 2, axis=1)
    # Fig. 14's single optimised feature
    out["Fig.14: mean([Θ(B×κ·∇y)+0.2] |∇x|³/B)"] = np.mean(
        (heaviside(cv) + 0.2) * gds22**1.5 / B, axis=1
    )
    return out


def tube_tensor(gx_geometry, eq, rho, length, nz, alpha, workdir):
    """One flux tube as a (1, nz - 1, 7) tensor, built as the training set's were."""
    fl = gx_geometry.create_eik_from_desc_given_length(
        eq,
        rho**2,
        theta0=alpha,
        zeta0=0.0,
        nz=nz,
        length=length,
        sigma_Bxy=-1.0,
        eik_filename=os.path.join(workdir, "eik.nc"),
    )
    shat = float(fl.shat)
    cols = {
        "bmag": fl.bmag,
        "gbdrift": fl.gbdrift,
        "cvdrift": fl.cvdrift,
        "gbdrift0_over_shat": fl.gbdrift0 / shat,
        "gds2": fl.gds2,
        "gds21_over_shat": fl.gds21 / shat,
        "gds22_over_shat_squared": fl.gds22 / shat**2,
    }
    T = np.stack([np.ravel(cols[k])[:-1] for k in Z_FUNCTIONS], axis=-1)[None]
    return T, {"iota": float(fl.iota), "shat": shat}


def load_dataset(path):
    import h5py

    with h5py.File(path, "r") as f:
        T = f["raw_feature_tensor"][()]
        Q = f["fixed_gradient_simulations/Q_avgs"][()]
        names = [n.decode() for n in f["z_functions_GX"][()]]
    assert names == Z_FUNCTIONS, names
    return T, Q


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("equilibrium", help="DESC .h5 (an Equilibrium or EquilibriaFamily)")
    p.add_argument("--rho", type=float, nargs="+", default=[0.85])
    p.add_argument(
        "--eq-index",
        type=int,
        nargs="+",
        default=None,
        help="members of a family (default all)",
    )
    p.add_argument("--length", type=float, default=75.4)
    p.add_argument("--nz", type=int, default=97)
    p.add_argument("--alpha", type=float, default=0.0, help="field line label theta0")
    p.add_argument("--gx-geometry", default=DEFAULT_GX_GEOMETRY)
    p.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="the archive's GX dataset .h5, for percentiles ('' to skip)",
    )
    p.add_argument("--out", default="figs/geometric_features.pdf")
    p.add_argument("--show", action="store_true")
    args = p.parse_args(argv)

    if args.nz != 97 or args.length != 75.4:
        print(
            "warning: the training set used nz = 97 and length = 75.4; grid-dependent "
            "features will not be comparable"
        )

    sys.path.insert(0, os.path.abspath(args.gx_geometry))
    import desc.io
    import gx_geometry

    loaded = desc.io.load(args.equilibrium)
    try:
        members = list(loaded)
    except TypeError:
        members = [loaded]
    indices = args.eq_index if args.eq_index is not None else range(len(members))
    indices = [i % len(members) for i in indices]

    rows = []  # (label, eq index, rho, scalars, features)
    with tempfile.TemporaryDirectory() as work:
        for i in indices:
            for r in args.rho:
                print(f"equilibrium {i}, rho = {r:g}: building the flux tube", flush=True)
                T, scal = tube_tensor(
                    gx_geometry, members[i], r, args.length, args.nz, args.alpha, work
                )
                F = {k: float(v[0]) for k, v in features(T).items()}
                rows.append((f"eq{i} ρ={r:g}", i, r, scal, F))

    names = list(rows[0][4])
    ds = None
    if args.dataset:
        if os.path.isfile(args.dataset):
            print(f"\nreading the training set's geometry from {args.dataset}")
            T_ds, Q_ds = load_dataset(args.dataset)
            ds = features(T_ds)
        else:
            print(f"\nno dataset at {args.dataset}; skipping percentiles")

    # table: one column per tube
    w = max(len(n) for n in names) + 2
    labels = [r[0] for r in rows]
    print()
    print(" " * w + "".join(f"{lab:>16}" for lab in labels))
    for key in ("iota", "shat"):
        print(f"{key:<{w}}" + "".join(f"{r[3][key]:>16.4g}" for r in rows))
    for n in names:
        line = f"{n:<{w}}"
        for r in rows:
            v = r[4][n]
            if ds is not None:
                pct = 100.0 * np.mean(ds[n] < v)
                line += f"{v:>10.3g} ({pct:3.0f}%)"
            else:
                line += f"{v:>16.4g}"
        print(line)
    if ds is not None:
        print("\n(percentile of the value among the training set's 100705 tubes)")

    plot(rows, names, ds, Q_ds if ds is not None else None, args.out, args.show)


def plot(rows, names, ds, Q_ds, out, show):
    import matplotlib.pyplot as plt

    n = len(names)
    ncol = 4
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.0 * nrow))
    axes = axes.ravel()
    eqs = sorted({r[1] for r in rows})
    rhos = sorted({r[2] for r in rows})
    cmap = plt.get_cmap("viridis")
    colours = {e: cmap(k / max(len(eqs) - 1, 1)) for k, e in enumerate(eqs)}
    markers = dict(zip(rhos, "o^sDv<>p"))

    for ax, name in zip(axes, names):
        vals = np.array([r[4][name] for r in rows])
        # log axis when the feature is non-negative (zeros, e.g. no bad curvature, dropped)
        positive = ds is not None and np.all(ds[name] >= 0) and np.all(vals > 0)
        if ds is not None:
            # Q against the feature, as the paper's Fig. 14
            x = ds[name]
            ok = (Q_ds > 0) & ((x > 0) if positive else True)
            ax.scatter(
                x[ok], Q_ds[ok], s=1, c="0.6", alpha=0.08, edgecolors="none", rasterized=True
            )
            ax.set_yscale("log")
            if positive:
                ax.set_xscale("log")
            lo, hi = np.quantile(x[ok], [0.001, 0.999])
            lo, hi = min(lo, vals.min()), max(hi, vals.max())
            if positive:
                ax.set_xlim(lo / 1.3, hi * 1.3)
            else:
                pad = 0.05 * (hi - lo)
                ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(1e-2, 1e3)
        for r, v in zip(rows, vals):
            ax.axvline(v, color=colours[r[1]], lw=1.0, alpha=0.8)
            if ds is not None:
                ax.plot(v, 2e2, markers[r[2]], color=colours[r[1]], ms=5)
            else:
                ax.plot(r[1], v, markers[r[2]], color=colours[r[1]])
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    handles = [
        plt.Line2D([], [], color=colours[e], marker="o", lw=0, label=f"eq {e}") for e in eqs
    ]
    if len(rhos) > 1:
        handles += [
            plt.Line2D([], [], color="k", marker=markers[r], lw=0, label=f"ρ = {r:g}")
            for r in rhos
        ]
    axes[n - 1].legend(handles=handles, fontsize=7, loc="lower right")
    if ds is not None:
        fig.supylabel("training-set Q (fixed gradients)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
