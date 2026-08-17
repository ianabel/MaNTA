#!/usr/bin/env python3
"""Plot a MaNTA netCDF output file.

    Tools/plot.py run.nc                       # u for every variable, final time
    Tools/plot.py run.nc --list                # what is in the file
    Tools/plot.py run.nc -t 0 --grid           # initial condition, cells marked
    Tools/plot.py run.nc -f u q sigma u_star   # every spatial field
    Tools/plot.py run.nc --times               # every timeslice, overlaid
    Tools/plot.py run.nc --aux --scalars       # auxiliary variables and scalars
    Tools/plot.py run.nc -o run.png            # write a file instead of showing

Nothing here is specific to a physics case: the file is inspected rather than
assumed, so a case that declares its own variable names, auxiliary variables or
scalars is plotted without this script being told about it.


What a run writes
-----------------

`SystemSolver::initialiseNetCDF` and `WriteTimeslice` (`NetCDFIO.cpp`) produce:

* root variables `t` (t) and `x` (x) -- the output grid, `OutputPoints` evenly
  spaced points across the domain, *not* the cell boundaries;
* a **group per solution variable**, named whatever the case called it, holding
  `u`, `q`, `sigma` and -- for any `PolynomialDegree >= 1` -- `u_star`, each
  shaped (t, x);
* a root **2-D variable per auxiliary variable** (t, x), named by the case;
* a root **1-D time series per scalar** (t,), named by the case;
* the `Grid` group: `CellBoundaries`, `Index` and `PolyOrder`;
* `nVariables`, and a case's own diagnostics if it writes any.

Groups and root variables are therefore discovered by *shape*, never by name --
see `variable_groups` and `root_series` below for why that is the only rule that
works.

`u_star` is written for every run with `PolynomialDegree >= 1` regardless of
whether `Superconvergent` was set: the flag controls whether the *method* uses
the reconstruction, not whether it is computed. So its presence here says
nothing about which scheme ran.

Adjoint output (`ng`, `G<i>`, the `G<i>_p` groups) is not plotted because no run
emits it -- `WriteAdjoints()` is commented out at `Solver.cpp:350`. `--list`
reports the groups it does not recognise rather than ignoring them silently.
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

# The spatial fields a variable group holds, in the order they are plotted.
# u_star is last because it is the only optional one.
FIELDS = ("u", "q", "sigma", "u_star")


def variable_groups(nc):
    """The groups holding a solution variable, sorted by name.

    Identified by holding a ``u``, which is what separates them from ``Grid``,
    from the adjoint groups and from a case's own diagnostics. The rule is not a
    convenience: a physics case *names its own variables*, so these groups are
    called "u", "Density", "IonEnergy" -- whatever the case declared -- and only
    the three cases that use the numbered placeholders still have a "Var0".

    That is what this script used to assume, and it is why it stopped working:
    ``groups["Var0"]`` is a KeyError on `LinearDiffusion.ref.nc`, on
    `AuxVarTest.ref.nc` (groups "u" and "v") and on every other case that took a
    name. `Tests/RegressionTests/TestSolutions.py:variable_groups` applies the
    same rule for the same reason; keep the two in step.

    Sorted by name because the file does not record declaration order.
    """
    return sorted(g for g, grp in nc.groups.items() if "u" in grp.variables)


def root_series(nc):
    """Root variables split by shape: (auxiliary, scalar).

    Auxiliary variables are 2-D (t, x) and scalars are 1-D (t,), and both are
    named by the case -- so, again, shape is the only thing that identifies
    them. `AuxVarTest.ref.nc` calls its auxiliary variable "a";
    `PIDTest.ref.nc` calls its scalars "E", "J", "I" and "Mass". A rule keyed on
    an "AuxVariable"/"Scalar" prefix sees neither.

    `t` is excluded by name because it is the axis, and 0-d variables
    (`nVariables`, and the adjoint counts) are excluded by shape.
    """
    aux, scalars = [], []
    for name, var in nc.variables.items():
        if name in ("t", "x"):
            continue
        if var.dimensions == ("t", "x"):
            aux.append(name)
        elif var.dimensions == ("t",):
            scalars.append(name)
    return sorted(aux), sorted(scalars)


def attr(var, name, default=""):
    return getattr(var, name, default)


def label(name, var):
    units = attr(var, "units")
    return f"{name} [{units}]" if units else name


def cell_boundaries(nc):
    if "Grid" not in nc.groups:
        return None
    return np.asarray(nc.groups["Grid"].variables["CellBoundaries"][:])


def mark_cells(ax, boundaries):
    if boundaries is None:
        return
    for edge in boundaries:
        ax.axvline(edge, color="red", linestyle="--", alpha=0.25, linewidth=0.8)


def describe(nc, path):
    """--list: everything the file holds, and anything this script cannot place."""
    t = np.asarray(nc.variables["t"][:])
    x = np.asarray(nc.variables["x"][:])
    groups = variable_groups(nc)
    aux, scalars = root_series(nc)
    boundaries = cell_boundaries(nc)

    print(f"{path}")
    print(f"  {len(t)} timeslices, t = {t[0]:g} .. {t[-1]:g}")
    print(f"  {len(x)} output points, x = {x[0]:g} .. {x[-1]:g}")
    if boundaries is not None:
        order = nc.groups["Grid"].variables["PolyOrder"][...]
        print(f"  {len(boundaries) - 1} cells, PolynomialDegree = {int(order)}")

    for name in groups:
        grp = nc.groups[name]
        fields = " ".join(f for f in FIELDS if f in grp.variables)
        print(f"  variable {name!r}: {fields}"
              f"{'   -- ' + attr(grp, 'description') if attr(grp, 'description') else ''}")
    for name in aux:
        print(f"  auxiliary {name!r}   -- {attr(nc.variables[name], 'description')}")
    for name in scalars:
        print(f"  scalar    {name!r}   -- {attr(nc.variables[name], 'description')}")

    # Said rather than skipped. A case's own diagnostics land in groups of their
    # own, and the adjoint groups would too if WriteAdjoints() were enabled; this
    # script plots neither, and silence would read as "the file has nothing else".
    unplotted = sorted(set(nc.groups) - set(groups) - {"Grid"})
    if unplotted:
        print(f"  not plotted: {', '.join(unplotted)} "
              "(a case's diagnostics, or adjoint output)")


def plot_fields(nc, groups, fields, indices, mark, times):
    """One figure per field, one subplot per variable."""
    x = np.asarray(nc.variables["x"][:])
    boundaries = cell_boundaries(nc) if mark else None

    for field in fields:
        present = [g for g in groups if field in nc.groups[g].variables]
        if not present:
            # u_star is absent at PolynomialDegree = 0, where there is no
            # reconstruction to compute. Anything else missing is a real
            # surprise and worth the same one line.
            print(f"note: no variable has a {field!r}; skipping", file=sys.stderr)
            continue

        fig, axes = plt.subplots(len(present), 1, sharex=True, squeeze=False,
                                 figsize=(8, 2.6 * len(present)))
        fig.canvas.manager.set_window_title(field)
        for ax, name in zip(axes[:, 0], present):
            var = nc.groups[name].variables[field]
            for i in indices:
                ax.plot(x, np.asarray(var[i, :]),
                        label=f"t = {times[i]:g}",
                        alpha=1.0 if len(indices) == 1 else 0.7)
            mark_cells(ax, boundaries)
            ax.set_ylabel(label(name, var))
            # The legend is the timeslice list, so it is noise for a single one:
            # the time is in the figure title instead.
            if len(indices) > 1:
                ax.legend(fontsize="small", ncols=2)
        axes[-1, 0].set_xlabel("x")
        title = f"{field}" if len(indices) > 1 else f"{field} at t = {times[indices[0]]:g}"
        fig.suptitle(title)
        fig.tight_layout()


def plot_aux(nc, names, indices, mark, times):
    if not names:
        print("note: this run has no auxiliary variables", file=sys.stderr)
        return
    x = np.asarray(nc.variables["x"][:])
    boundaries = cell_boundaries(nc) if mark else None

    fig, axes = plt.subplots(len(names), 1, sharex=True, squeeze=False,
                             figsize=(8, 2.6 * len(names)))
    fig.canvas.manager.set_window_title("auxiliary")
    for ax, name in zip(axes[:, 0], names):
        var = nc.variables[name]
        for i in indices:
            ax.plot(x, np.asarray(var[i, :]), label=f"t = {times[i]:g}")
        mark_cells(ax, boundaries)
        ax.set_ylabel(label(name, var))
        if len(indices) > 1:
            ax.legend(fontsize="small", ncols=2)
    axes[-1, 0].set_xlabel("x")
    fig.suptitle("auxiliary variables")
    fig.tight_layout()


def plot_scalars(nc, names, times):
    """Scalars against time -- they have no x, so they are their own figure."""
    if not names:
        print("note: this run has no scalars", file=sys.stderr)
        return
    fig, axes = plt.subplots(len(names), 1, sharex=True, squeeze=False,
                             figsize=(8, 2.2 * len(names)))
    fig.canvas.manager.set_window_title("scalars")
    for ax, name in zip(axes[:, 0], names):
        var = nc.variables[name]
        ax.plot(times, np.asarray(var[:]), marker=".")
        ax.set_ylabel(label(name, var))
    axes[-1, 0].set_xlabel("t")
    fig.suptitle("scalars")
    fig.tight_layout()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Plot a MaNTA netCDF output file.",
        epilog="With no options: u for every variable, at the final timeslice.",
    )
    p.add_argument("file", help="a run's <stem>.nc (or <stem>.restart.nc)")
    p.add_argument("-t", "--time", type=int, default=-1, metavar="INDEX",
                   help="timeslice index; negative counts from the end "
                        "(default: -1, the final one)")
    p.add_argument("--times", action="store_true",
                   help="overlay every timeslice instead of one")
    p.add_argument("-f", "--fields", nargs="+", default=["u"], choices=FIELDS,
                   metavar="FIELD",
                   help=f"which of {', '.join(FIELDS)} to plot (default: u)")
    p.add_argument("--vars", nargs="+", metavar="NAME",
                   help="only these variables (default: all of them)")
    p.add_argument("--aux", action="store_true", help="also plot auxiliary variables")
    p.add_argument("--scalars", action="store_true", help="also plot scalars against t")
    p.add_argument("--grid", action="store_true",
                   help="mark the cell boundaries from the Grid group")
    p.add_argument("--list", action="store_true",
                   help="report what the file holds and exit")
    p.add_argument("-o", "--output", metavar="FILE",
                   help="save to FILE instead of opening a window; with more "
                        "than one figure, the field name is inserted before the "
                        "suffix")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    with Dataset(args.file, "r") as nc:
        if args.list:
            describe(nc, args.file)
            return 0

        times = np.asarray(nc.variables["t"][:])
        groups = variable_groups(nc)
        aux, scalars = root_series(nc)

        if args.vars:
            unknown = [v for v in args.vars if v not in groups]
            if unknown:
                print(f"error: no variable named {', '.join(repr(u) for u in unknown)}. "
                      f"This file has: {', '.join(groups) or '(none)'}. "
                      "Run with --list.", file=sys.stderr)
                return 1
            groups = [v for v in groups if v in args.vars]

        if not groups and not (args.aux or args.scalars):
            print(f"error: {args.file} holds no solution variables -- no group "
                  "contains a 'u'. Run with --list.", file=sys.stderr)
            return 1

        if args.times:
            indices = list(range(len(times)))
        else:
            # Checked rather than left to raise from inside netCDF4, whose
            # IndexError does not mention how many timeslices there are.
            if not -len(times) <= args.time < len(times):
                print(f"error: --time {args.time} is out of range; this file has "
                      f"{len(times)} timeslices (0 .. {len(times) - 1}, or -1 "
                      "for the last).", file=sys.stderr)
                return 1
            indices = [args.time % len(times)]

        if groups:
            plot_fields(nc, groups, args.fields, indices, args.grid, times)
        if args.aux:
            plot_aux(nc, aux, indices, args.grid, times)
        if args.scalars:
            plot_scalars(nc, scalars, times)

        if args.output:
            figures = plt.get_fignums()
            for number in figures:
                fig = plt.figure(number)
                if len(figures) == 1:
                    path = args.output
                else:
                    stem, _, suffix = args.output.rpartition(".")
                    tag = fig.canvas.manager.get_window_title()
                    path = f"{stem}-{tag}.{suffix}" if stem else f"{args.output}-{tag}"
                fig.savefig(path, dpi=150)
                print(f"wrote {path}")
        else:
            plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
