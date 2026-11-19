#!/usr/bin/env python
"""ITG critical gradient of the NN proxy against radius, for each member of a family.

At each radius the proxy (built per-rho, as compare_gx_nn.py builds it) is
scanned in a/LT over the training box at fixed a/Ln. The critical gradient is
the smallest a/LT where the network's flux rises `--threshold` above its floor
(Q_raw - Q_floor > threshold, in Q_gB). Because the network bottoms out at
Q_floor ~ e^-2 rather than at zero, this is a threshold in the proxy's output,
not a linear-stability boundary.

Scans run at a/Ln = 0 and at the MaNTA run's a/Ln(rho); with a MaNTA file the
run's a/LTi(rho) is drawn over the critical gradients, for the member it was
run on (--manta-eq-index, default the last).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))  # the stellarator modules
from compare_gx_nn import evaluate  # noqa: E402
from gx_inputs_from_manta import (  # noqa: E402
    A_OVER_LN_RANGE,
    A_OVER_LT_RANGE,
    DENSITY_PROFILE,
    manta_profiles,
)

DEFAULT_MODEL_DIR = os.environ.get(
    "ITG_MODEL_DIR",
    os.environ.get(
        "ITG_NN_DIR",
        os.path.expanduser("~/projects/itg-proxy/demo/neural_networks"),
    ),
)


def critical(aLT, dQ, threshold):
    """First a/LT where dQ crosses threshold, linearly interpolated; nan if never."""
    above = np.nonzero(dQ > threshold)[0]
    if len(above) == 0:
        return np.nan
    j = above[0]
    if j == 0:
        return aLT[0]
    t = (threshold - dQ[j - 1]) / (dQ[j] - dQ[j - 1])
    return aLT[j - 1] + t * (aLT[j] - aLT[j - 1])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("equilibrium")
    p.add_argument("--manta", default=None, help="MaNTA output, for a/LTi and a/Ln")
    p.add_argument("--manta-eq-index", type=int, default=-1)
    p.add_argument("--edge-density", type=float, default=DENSITY_PROFILE["EdgeDensity"])
    p.add_argument("--n0", type=float, default=DENSITY_PROFILE["n0"])
    p.add_argument("--density-exponent", type=float, default=DENSITY_PROFILE["alpha"])
    p.add_argument("--eq-index", type=int, nargs="+", default=None)
    p.add_argument("--rho", type=float, nargs="+",
                   default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--n-scan", type=int, default=146)
    p.add_argument("--length", type=float, default=75.4)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--n-ensemble", type=int, default=10)
    p.add_argument("--out", default="figs/critical_gradient.pdf")
    args = p.parse_args(argv)

    import desc.io
    from compare_gx_nn import nn_fluxes

    loaded = desc.io.load(args.equilibrium)
    try:
        members = list(loaded)
    except TypeError:
        members = [loaded]
    idx = [i % len(members) for i in (args.eq_index or range(len(members)))]
    rho = np.array(sorted(args.rho))
    aLT = np.linspace(1e-3, A_OVER_LT_RANGE[1], args.n_scan)

    prof = None
    if args.manta:
        density = {"EdgeDensity": args.edge_density, "n0": args.n0,
                   "alpha": args.density_exponent}
        prof = manta_profiles(args.manta, rho, members[args.manta_eq_index],
                              density=density)
        aLn_manta = np.clip(prof["a/Ln"], *A_OVER_LN_RANGE)
    scans = {"a/Ln = 0": np.zeros_like(rho)}
    if prof is not None:
        scans["MaNTA a/Ln"] = aLn_manta

    results = {}  # (eq, scan) -> crit(rho)
    for i in idx:
        print(f"equilibrium {i}: building the proxy on {len(rho)} surfaces", flush=True)
        raw, _ = nn_fluxes(members[i], rho, args.length, 0.0, True,
                           args.model_dir, args.n_ensemble)
        floor = float(raw.Q_floor) if hasattr(raw, "Q_floor") else np.exp(-2.0)
        R, A = np.meshgrid(rho, aLT, indexing="ij")
        for name, aLn in scans.items():
            N = np.broadcast_to(aLn[:, None], R.shape)
            Q = evaluate(raw, R.ravel(), A.ravel(), N.ravel()).reshape(R.shape)
            results[i, name] = np.array(
                [critical(aLT, Q[k] - floor, args.threshold) for k in range(len(rho))]
            )

    print(f"\ncritical a/LT (Q_raw - floor > {args.threshold}); nan = none in the box")
    header = "  rho  " + "".join(f"{f'eq{i} {n}':>18}" for i in idx for n in scans)
    if prof is not None:
        header += f"{'MaNTA a/LTi':>14}{'MaNTA a/Ln':>12}"
    print(header)
    for k, r in enumerate(rho):
        line = f"{r:5.2f}  " + "".join(f"{results[i, n][k]:>18.3g}" for i in idx for n in scans)
        if prof is not None:
            line += f"{prof['a/LTi'][k]:>14.3g}{prof['a/Ln'][k]:>12.3g}"
        print(line)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(scans), figsize=(5.5 * len(scans), 4.2), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax, name in zip(axes[0], scans):
        for c, i in enumerate(idx):
            ax.plot(rho, results[i, name], "o-", ms=3,
                    color=cmap(c / max(len(idx) - 1, 1)), label=f"eq {i}")
        if prof is not None:
            ax.plot(rho, prof["a/LTi"], "k--", lw=2,
                    label=f"MaNTA a/LTi (eq {args.manta_eq_index % len(members)})")
        ax.set_xlabel("ρ")
        ax.set_ylabel("critical a/LT")
        ax.set_ylim(0, A_OVER_LT_RANGE[1])
        ax.set_title(f"scan at {name}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
