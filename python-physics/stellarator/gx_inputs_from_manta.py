#!/usr/bin/env python
"""GX inputs at a set of rho, from a DESC equilibrium and a MaNTA run.

The geometry of each flux tube comes from the DESC equilibrium through
`gx_geometry` -- the pipeline the NN ITG proxy's training database was built
with -- and the gradients come from the MaNTA profiles, so the GX runs check
the heat flux the proxy predicted inside the optimisation. Every other setting
is copied from a template `gx.in`, whose path must be given.

    python gx_inputs_from_manta.py eq_itg_flux_all_equilibria.h5 \\
        stellarator_opteq_itg_flux.nc path/to/sample/gx.in \\
        --rho 0.3 0.5 0.7 --out gx_runs

writes, for each rho, a directory `gx_runs/rho0.500/` holding `gx.in` and the
geometry file `eik.nc` it names (a relative name: run GX from that directory),
plus `gx_runs/summary.csv` with the gradients and the tube's scalars.

What is changed in the template, and nothing else:

* `[Geometry] geo_file` -- the new eik file, and `geo_option = "nc"`, since it
  is written in netCDF.
* `[Geometry] desired_normalized_toroidal_flux` -- `rho^2`. GX does not read it
  from an eik file; it is set so the input says which surface it is.
* `[species] tprim`, `fprim` -- `[a/LTi, a/LTe]` and `[a/Ln, a/Ln]`.
* `[Boltzmann] tau_fac` -- 1.0: `Ti = Te`, as the proxy was trained.

Conventions, all the proxy's (`itg_wrapper.py`):

* `a/LT = -d ln T / d rho`, `a/Ln = -d ln n / d rho`, DESC's
  `rho = sqrt(psi/psi_b)`. GX's `tprim` is `-(a/T) dT/dx`, `x = a rho`: the same.
* The tube has a fixed length in units of `L_ref = a`: 75.4 by default, the
  proxy's `target_flux_tube_length`. Each rho is solved for its own number of
  poloidal turns, which is what the training database did. The proxy instead
  solves the turns at its innermost rho and holds the toroidal extent fixed, so
  its tubes drift from 75.4 with iota (up to a few percent in Q).
* `alpha = 0`, `sigma_Bxy = -1`, as the database and the proxy.

MaNTA's variables are `V' n` and `(3/2) V' p`, `V'` being DESC's `V_r` over the
total volume. Under `evolveDensity` every ratio that matters is a ratio of two
of them, so `V'` cancels:

    Ti = (2/3) u_i / u_n,   a/LTi = -(q_i/u_i - q_n/u_n),   a/Ln = -(q_n/u_n - V''/V')

With the density held fixed (only `IonEnergy` in the file, `Te = Ti`), `n` is
`StellaratorState.initial_profile`, whose parameters are not in the output and
are set in `DENSITY_PROFILE` below -- keep it in step with the `st_config` of
the run being read. Its `rhoUpper` is the grid's upper boundary, as
`StellaratorTransport` sets it. Only the ratio EdgeDensity/n0 matters, so the
values may be given in either normalisation.

Needs DESC and a `gx_geometry` that runs against it. The copy in the zenodo
archive calls `eq.compute_theta_coords` and `np.trapz`, both gone from the DESC
and numpy here, so the default is the itg-proxy demo's copy, which differs only
in those two calls; `--gx-geometry` points elsewhere.
"""

import argparse
import csv
import os
import re
import sys
import tomllib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECTS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_GX_GEOMETRY = os.path.join(PROJECTS, "itg-proxy", "demo", "gx_geometry")

# The fixed density profile of an evolveDensity = False run: the st_config keys
# StellaratorState.initial_profile reads. These are
# paper/desc-optimize-itg-flux.py's.
DENSITY_PROFILE = {
    "EdgeDensity": 0.4,
    "n0": 1.0,
    "alpha": 4.0,
}

# the box the proxy's training set covers (itg_wrapper.py)
A_OVER_LT_RANGE = (0.0, 14.5)
A_OVER_LN_RANGE = (-0.5, 9.0)


# --- MaNTA profiles -----------------------------------------------------------


def initial_profile(x, edge_value, peak_value, xr, exponent):
    """StellaratorState.initial_profile, and its derivative."""
    a = 1.0 / xr**exponent
    value = (peak_value - edge_value) * (1 - a * x**exponent) + edge_value
    deriv = -(peak_value - edge_value) * a * exponent * x ** (exponent - 1)
    return value, deriv


def volume_derivatives(eq, rho):
    """V' and V'' in rho, normalised to the total volume as yancc_wrapper does."""
    from desc.grid import LinearGrid

    grid = LinearGrid(rho=np.asarray(rho), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    data = eq.compute(["V", "V_r(r)", "V_rr(r)"], grid=grid)
    V = float(data["V"])
    return (
        np.asarray(grid.compress(data["V_r(r)"])) / V,
        np.asarray(grid.compress(data["V_rr(r)"])) / V,
    )


def manta_profiles(nc_file, rho, eq, time_index=-1, density=DENSITY_PROFILE):
    """n, Ti, Te and the GX gradients at `rho`, from a MaNTA netCDF output.

    Returns a dict of arrays the length of `rho`. n, Ti, Te are in MaNTA's
    units (n0, T0); only their ratios and logarithmic derivatives are used.
    """
    from netCDF4 import Dataset
    from scipy.interpolate import CubicSpline

    rho = np.asarray(rho, dtype=float)
    with Dataset(nc_file) as f:
        x = np.asarray(f.variables["x"][:])
        t = np.asarray(f.variables["t"][:])
        bounds = np.asarray(f.groups["Grid"].variables["CellBoundaries"][:])
        groups = {
            name: (
                np.asarray(g.variables["u"][time_index, :]),
                np.asarray(g.variables["q"][time_index, :]),
            )
            for name, g in f.groups.items()
            if "u" in g.variables
        }
    print(f"MaNTA output {nc_file}: t = {t[time_index]:g} (slice {time_index} of {len(t)})")
    print(f"  variables: {list(groups)}; grid [{bounds[0]:g}, {bounds[-1]:g}]")

    if np.any(rho < x[0]) or np.any(rho > x[-1]):
        raise ValueError(f"rho must lie in MaNTA's domain [{x[0]:g}, {x[-1]:g}]")
    if "IonEnergy" not in groups:
        raise ValueError(f"no IonEnergy in {nc_file}: is this a stellarator run?")

    def at_rho(name):
        u, q = groups[name]
        # u and q are both stored, so interpolate each rather than differentiate
        return CubicSpline(x, u)(rho), CubicSpline(x, q)(rho)

    u_i, q_i = at_rho("IonEnergy")
    out = {}
    if "Density" in groups:
        u_n, q_n = at_rho("Density")
        Vp, Vpp = volume_derivatives(eq, rho)
        out["n"] = u_n / Vp
        out["Ti"] = 2.0 / 3.0 * u_i / u_n
        out["a/LTi"] = -(q_i / u_i - q_n / u_n)
        out["a/Ln"] = -(q_n / u_n - Vpp / Vp)
        if "ElectronEnergy" in groups:
            u_e, q_e = at_rho("ElectronEnergy")
            out["Te"] = 2.0 / 3.0 * u_e / u_n
            out["a/LTe"] = -(q_e / u_e - q_n / u_n)
        else:
            out["Te"], out["a/LTe"] = out["Ti"], out["a/LTi"]
    else:
        edge, core, exponent = density["EdgeDensity"], density["n0"], density["alpha"]
        n, dndrho = initial_profile(rho, edge, core, bounds[-1], exponent)
        Vp, Vpp = volume_derivatives(eq, rho)
        out["n"] = n
        out["Ti"] = 2.0 / 3.0 * u_i / (Vp * n)
        out["a/LTi"] = -(q_i / u_i - Vpp / Vp - dndrho / n)
        out["a/Ln"] = -dndrho / n
        # the constant-density case has Te = Ti (StellaratorState.from_state)
        out["Te"], out["a/LTe"] = out["Ti"], out["a/LTi"]
        print(
            f"  density fixed: initial_profile(edge={edge:g}, core={core:g}, "
            f"rhoUpper={bounds[-1]:g}, exponent={exponent:g})"
        )

    if np.any(out["n"] <= 0) or np.any(out["Ti"] <= 0) or np.any(out["Te"] <= 0):
        raise ValueError("non-positive density or temperature at a requested rho")
    return out


# --- GX input file ------------------------------------------------------------


def _format(value):
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        return "[ " + ", ".join(_format(v) for v in value) + " ]"
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(float(value)) if isinstance(value, float) else str(value)


def set_key(text, section, key, value):
    """Set `key` in `[section]` of a TOML-ish GX input, keeping its comment.

    Edits the line in place so the rest of the template, comments included,
    is untouched; adds the key at the end of the section if it is absent.
    """
    lines = text.splitlines(keepends=True)
    header = re.compile(r"^\s*\[([^\]]+)\]")
    assign = re.compile(rf"^(\s*){re.escape(key)}(\s*=\s*)([^#\n]*?)(\s*#.*)?$")
    current, last_in_section = None, None
    for i, line in enumerate(lines):
        m = header.match(line)
        if m:
            current = m.group(1).strip()
            continue
        if current != section:
            continue
        last_in_section = i
        m = assign.match(line.rstrip("\n"))
        if m:
            comment = m.group(4) or ""
            lines[i] = f"{m.group(1)}{key}{m.group(2)}{_format(value)}{comment}\n"
            return "".join(lines)
    if last_in_section is None:
        raise KeyError(f"no [{section}] section in the template")
    lines.insert(last_in_section + 1, f" {key} = {_format(value)}\n")
    return "".join(lines)


def make_gx_input(template, geo_file, s, tprim, fprim):
    text = template
    text = set_key(text, "Geometry", "geo_option", "nc")
    text = set_key(text, "Geometry", "geo_file", geo_file)
    text = set_key(text, "Geometry", "desired_normalized_toroidal_flux", float(s))
    text = set_key(text, "species", "tprim", [float(v) for v in tprim])
    text = set_key(text, "species", "fprim", [float(v) for v in fprim])
    text = set_key(text, "Boltzmann", "tau_fac", 1.0)
    tomllib.loads(text)  # refuse to write something GX's parser would not read
    return text


# --- driver -------------------------------------------------------------------


def load_equilibrium(path, index):
    import desc.io

    eq = desc.io.load(path)
    try:
        eq = eq[index]  # an EquilibriaFamily
    except TypeError:
        pass
    return eq


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("equilibrium", help="DESC .h5 (an Equilibrium or EquilibriaFamily)")
    p.add_argument("manta", help="MaNTA netCDF output (.nc or .restart.nc)")
    p.add_argument("template", help="sample gx.in whose settings are copied")
    p.add_argument("--rho", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    p.add_argument("--out", default="gx_runs", help="output directory")
    p.add_argument("--eq-index", type=int, default=-1, help="member of a family")
    p.add_argument("--time-index", type=int, default=-1, help="MaNTA output slice")
    p.add_argument(
        "--length",
        type=float,
        default=75.4,
        help="flux tube length / a (the proxy's target_flux_tube_length)",
    )
    p.add_argument(
        "--nz",
        type=int,
        default=None,
        help="points along the tube (odd); default the template's ntheta + 1",
    )
    p.add_argument("--alpha", type=float, default=0.0, help="field line label")
    p.add_argument("--gx-geometry", default=DEFAULT_GX_GEOMETRY)
    args = p.parse_args(argv)

    sys.path.insert(0, os.path.abspath(args.gx_geometry))
    import gx_geometry

    with open(args.template) as f:
        template = f.read()
    params = tomllib.loads(template)
    nz = args.nz or params["Dimensions"]["ntheta"] + 1
    if nz % 2 == 0:
        raise ValueError(f"nz must be odd (write_eik requires it), got {nz}")

    rho = np.array(sorted(args.rho))
    eq = load_equilibrium(args.equilibrium, args.eq_index)
    prof = manta_profiles(
        args.manta, rho, eq, time_index=args.time_index
    )

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for i, r in enumerate(rho):
        s = r**2
        run_dir = os.path.join(args.out, f"rho{r:.3f}")
        os.makedirs(run_dir, exist_ok=True)
        eik = "eik.nc"
        print(f"\nrho = {r:g} (s = {s:g}) -> {run_dir}")

        fl = gx_geometry.create_eik_from_desc_given_length(
            eq,
            s,
            theta0=args.alpha,
            zeta0=0.0,
            nz=nz,
            length=args.length,
            sigma_Bxy=-1.0,
            eik_filename=os.path.join(run_dir, eik),
        )

        tprim = [prof["a/LTi"][i], prof["a/LTe"][i]]
        fprim = [prof["a/Ln"][i], prof["a/Ln"][i]]
        text = make_gx_input(template, eik, s, tprim, fprim)
        with open(os.path.join(run_dir, "gx.in"), "w") as f:
            f.write(text)

        lt, ln = prof["a/LTi"][i], prof["a/Ln"][i]
        outside = not (
            A_OVER_LT_RANGE[0] < lt <= A_OVER_LT_RANGE[1]
            and A_OVER_LN_RANGE[0] <= ln <= A_OVER_LN_RANGE[1]
        )
        print(
            f"  a/LTi = {lt:.4g}, a/Ln = {ln:.4g}, "
            f"iota = {fl.iota:.4g}, shat = {fl.shat:.4g}, "
            f"poloidal turns = {fl.poloidal_turns:.4g}"
            + ("  [outside the proxy's training box]" if outside else "")
        )
        rows.append(
            {
                "rho": r,
                "s": s,
                "dir": os.path.relpath(run_dir, args.out),
                "a/LTi": lt,
                "a/LTe": prof["a/LTe"][i],
                "a/Ln": ln,
                "n": prof["n"][i],
                "Ti": prof["Ti"][i],
                "Te": prof["Te"][i],
                "iota": fl.iota,
                "shat": fl.shat,
                "poloidal_turns": fl.poloidal_turns,
                "length": args.length,
                "nz": nz,
                "in_training_box": not outside,
            }
        )

    summary = os.path.join(args.out, "summary.csv")
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"\nwrote {len(rows)} GX inputs under {args.out}/ and {summary}")


if __name__ == "__main__":
    main()
