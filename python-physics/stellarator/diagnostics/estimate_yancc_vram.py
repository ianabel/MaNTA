"""Estimate the per-card GPU memory a stellarator_multichannel run needs, from
its yancc resolution and how many nodes each card evaluates at once.

    python estimate_yancc_vram.py ../runStellarator.py
    python estimate_yancc_vram.py --na 61 --nx 7 --ntheta 19 --nzeta 29 --batch-size 8
    python estimate_yancc_vram.py ../runStellarator.py --calibrate 45,7,17,25:5.21

Nothing here imports jax, yancc or the model: it is arithmetic on a
measurement, and runs anywhere.

The model. The peak is set by the jacrev ComputePhysicsDerivatives takes
through yancc's solve_dke, and that is linear in the number of nodes vmapped
together on a card, with no fixed part worth the name:

    per card = OTHER + nodes_per_card * BYTES_PER_UNKNOWN * N
    N        = na * nx * ntheta * nzeta * species

Calibration (2026-11-01, A100-40GB on Perlmutter, yancc_collisionality_scan.py
--jacrev, banded smoother, rtol 1e-5, max_grids 3, GMRES m = 150, T = 2 keV):

    45,7,17,25 (N = 267750):  batch 1 -> 5.22 GB, batch 2 -> 10.43 GB
                              i.e. 5.21 GB per node, 0.02 GB fixed

and the production run at that resolution, batch_size 16 over 4 cards (4 nodes
per card), used about 26 GB per card. 4 * 5.21 = 20.8, so OTHER, everything
that is not the jacrev (fields, MaNTA state, XLA scratch), is about 5 GB.

What that does and does not license:

* ONE resolution was measured. Linear-in-N is the model, not an observation:
  GMRES's Krylov basis (m + 1 vectors per cotangent) is linear in N, and so is
  the banded smoother (bandwidth fixed). With smooth_solver = "dense" the
  smoother stores one M x M block per line, so memory grows faster than N
  when na, ntheta or nzeta grow -- this script will UNDERESTIMATE dense.
  Measure two resolutions with --jacrev and pass both to --calibrate to fit a
  real slope.
* The cost per unknown scales with the number of outputs jacrev pulls back (5:
  Gamma_e, Q_i, Q_e, J, <J||B>) and with the Krylov restart length m. Change
  either and the calibration is stale.
* OTHER was inferred from one total, not measured.
"""

import argparse
import ast
import math
import sys

GB = 1e9
# 2026-11-01 measurement, see the docstring
CAL_RES = (45, 7, 17, 25)
CAL_PER_NODE_GB = 5.21
CAL_FIXED_GB = 0.02
OTHER_GB = 26.0 - (4 * CAL_PER_NODE_GB + CAL_FIXED_GB)  # ~5.1
SPECIES = 2  # ions + electrons
USABLE_GB = {"a100-40": 39.6, "a100-80": 79.2, "h100-80": 79.2}
CONFIG_DEFAULT_BATCH_SIZE = 16  # config.py's StellaratorConfig default


def unknowns(na, nx, nt, nz, species=SPECIES):
    return na * nx * nt * nz * species


def read_driver(path):
    """Pull the resolution and batching out of a driver's top-level literals.

    Reads the last plain assignment of yancc_ntheta, yancc_nzeta, yancc_res,
    npoints and degree, and the keys useBatching / batch_size / nchunks from any
    dict literal. Anything computed rather than written as a literal is
    skipped; pass it on the command line instead.
    """
    tree = ast.parse(open(path).read(), path)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("yancc_ntheta", "yancc_nzeta", "yancc_res", "npoints",
                        "degree"):
                try:
                    found[name] = ast.literal_eval(node.value)
                except ValueError:
                    pass
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value in (
                        "useBatching", "batch_size", "nchunks", "Superconvergent"):
                    try:
                        found[k.value] = ast.literal_eval(v)
                    except ValueError:
                        pass
    out = {}
    if "yancc_ntheta" in found:
        out["ntheta"] = found["yancc_ntheta"]
    if "yancc_nzeta" in found:
        out["nzeta"] = found["yancc_nzeta"]
    if isinstance(found.get("yancc_res"), dict):
        out.update({k: v for k, v in found["yancc_res"].items() if k in ("na", "nx")})
    if "npoints" in found:
        out["cells"] = found["npoints"]
    if "degree" in found:
        out["degree"] = found["degree"]
    for k in ("useBatching", "batch_size", "Superconvergent"):
        if k in found:
            out[k] = found[k]
    return out


def parse_calibration(text):
    """'na,nx,nt,nz:per_node_GB[;...]' -> [(N, bytes per node), ...]"""
    pts = []
    for item in text.split(";"):
        res, gb = item.split(":")
        r = tuple(int(v) for v in res.split(","))
        if len(r) != 4:
            sys.exit(f"--calibrate wants na,nx,ntheta,nzeta:GB; got {item!r}")
        pts.append((unknowns(*r), float(gb) * GB))
    return pts


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("driver", nargs="?",
                   help="a driver script (runStellarator.py, desc-optimize-*.py) "
                        "to read the resolution and batching from")
    p.add_argument("--na", type=int)
    p.add_argument("--nx", type=int)
    p.add_argument("--ntheta", type=int)
    p.add_argument("--nzeta", type=int)
    p.add_argument("--devices", type=int, default=4, help="GPUs the run shards over")
    p.add_argument("--batch-size", type=int,
                   help="nodes per round over all cards (config batch_size)")
    p.add_argument("--no-batching", action="store_true",
                   help="useBatching = False: every node at once")
    p.add_argument("--cells", type=int, help="grid cells (no-batching only)")
    p.add_argument("--degree", type=int, help="polynomial degree (no-batching only)")
    p.add_argument("--card", default="a100-40", choices=sorted(USABLE_GB),
                   help="card, for the usable memory")
    p.add_argument("--usable-gb", type=float, help="usable memory per card, overrides --card")
    p.add_argument("--other-gb", type=float, default=OTHER_GB,
                   help=f"non-jacrev memory per card (default {OTHER_GB:.1f}, inferred)")
    p.add_argument("--calibrate",
                   help="measured per-node GB from --jacrev, 'na,nx,nt,nz:GB', "
                        "';'-separated; two or more fit a slope and intercept in N")
    a = p.parse_args()

    cfg = read_driver(a.driver) if a.driver else {}
    na = a.na or cfg.get("na", CAL_RES[0])
    nx = a.nx or cfg.get("nx", CAL_RES[1])
    nt = a.ntheta or cfg.get("ntheta", CAL_RES[2])
    nz = a.nzeta or cfg.get("nzeta", CAL_RES[3])
    batching = (not a.no_batching) and cfg.get("useBatching", True)
    batch_size = a.batch_size or cfg.get("batch_size", CONFIG_DEFAULT_BATCH_SIZE)

    if batching:
        per_card = math.ceil(batch_size / a.devices)
        how = f"batch_size {batch_size} over {a.devices} cards"
        if batch_size % a.devices:
            how += " (not a multiple of the card count: the busiest card is shown)"
    else:
        cells, k = a.cells or cfg.get("cells"), a.degree or cfg.get("degree")
        if cells is None or k is None:
            sys.exit("without batching every node is evaluated at once: "
                     "give --cells and --degree (or a driver that sets them)")
        nodes = cells * (k + 2 if cfg.get("Superconvergent") else k + 1)
        per_card = math.ceil(nodes / a.devices)
        how = f"no batching: all {nodes} nodes over {a.devices} cards"

    # bytes per node as a + b * N
    if a.calibrate:
        pts = parse_calibration(a.calibrate)
        if len(pts) == 1:
            slope, icpt = pts[0][1] / pts[0][0], 0.0
        else:
            (n1, b1), (n2, b2) = min(pts), max(pts)
            slope = (b2 - b1) / (n2 - n1)
            icpt = b1 - slope * n1
        source = f"--calibrate ({len(pts)} point{'s' * (len(pts) > 1)})"
    else:
        slope, icpt = CAL_PER_NODE_GB * GB / unknowns(*CAL_RES), 0.0
        source = "2026-11-01 measurement at 45,7,17,25 (one point, banded smoother)"

    usable = (a.usable_gb if a.usable_gb else USABLE_GB[a.card]) * GB
    N = unknowns(na, nx, nt, nz)
    per_node = icpt + slope * N
    total = a.other_gb * GB + per_card * per_node + CAL_FIXED_GB * GB
    budget = usable - a.other_gb * GB - CAL_FIXED_GB * GB

    print(f"resolution   na={na} nx={nx} ntheta={nt} nzeta={nz}   "
          f"N = {N:,} unknowns per node ({N / unknowns(*CAL_RES):.2f}x the calibration)")
    print(f"nodes/card   {per_card}   ({how})")
    print(f"calibration  {slope / 1e3:.1f} kB per unknown per node; {source}")
    print()
    print(f"  jacrev     {per_card} x {per_node / GB:.2f} GB = {per_card * per_node / GB:6.1f} GB")
    print(f"  other      {a.other_gb:6.1f} GB")
    print(f"  total      {total / GB:6.1f} GB  of {usable / GB:.1f} usable  "
          f"-> {'FITS' if total <= usable else 'DOES NOT FIT'} "
          f"({100 * total / usable:.0f}%)")

    # what would fit
    max_nodes = int(budget // per_node) if per_node > 0 else 0
    max_N = (budget / per_card - icpt) / slope if slope > 0 else float("inf")
    print()
    print(f"  at this resolution: at most {max_nodes} node(s) per card "
          f"(batch_size <= {max_nodes * a.devices})" if max_nodes else
          "  at this resolution: not even one node per card fits")
    print(f"  at {per_card} nodes per card: N up to {max_N:,.0f} "
          f"({max_N / N:.2f}x this resolution)")
    for bs in (16, 8, 4):
        npc = math.ceil(bs / a.devices)
        n_ok = (budget / npc - icpt) / slope
        print(f"    batch_size {bs:2d} ({npc} per card): N up to {n_ok:>12,.0f} "
              f"({n_ok / N:.2f}x this resolution)")
    print()
    print("Linear-in-N extrapolation from the calibration: underestimates a dense "
          "smoother, and is stale if GMRES m or the outputs differentiated change. "
          "Check a new resolution with yancc_collisionality_scan.py --jacrev.")


if __name__ == "__main__":
    main()
