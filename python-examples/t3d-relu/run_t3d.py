"""Run T3D on an input file, optionally with a one-sided ReLU heat flux.

    python run_t3d.py input.in [--one-sided] [--seed N]

Needs T3D importable. Changes nothing in T3D: `--one-sided` replaces the
module-level `ReLU` its ReLU flux model looks up at call time with one that is
zero for kappa_T < a rather than for |kappa_T| < a (see relu_ke._below).
`--seed` overrides the model's `random_seed`. `--fix-axis-grho` re-interpolates
<|grad rho|> from VMEC without its axis surface, where VMEC stores 0: T3D's
linear interpolation from that 0 leaves <|grad rho|> too small, and so V' =
area/grho too large, everywhere inside VMEC's first surface (rho = 0.071 here),
at every resolution.
"""
import argparse
import random
import sys

import numpy as np

import t3d.flux_models.ReLU as relu_model
import t3d.Geometry as geometry
import t3d.Profiles as pf
from scipy.interpolate import interp1d
from t3d.__main__ import main


def fix_axis_grho():
    init = geometry.Geometry.__init__

    def fixed(self, inputs, grid, *a, **k):
        init(self, inputs, grid, *a, **k)
        # The reader T3D kept: building another costs a second VMEC read, and
        # one read peaks at ~3.6 GB for this equilibrium.
        vm = self.VmecReader
        interp = interp1d(vm.rho[1:], vm.grad_rho[1:], fill_value="extrapolate")
        self.grho = pf.FluxProfile(interp(grid.midpoints) * self.a_minor_vmec, grid)
        self.grho_grid = pf.GridProfile(interp(grid.rho) * self.a_minor_vmec, grid)
        self.G_fac = self.grho_grid / self.area_grid
    geometry.Geometry.__init__ = fixed


def one_sided(x, a=1, m=1, p=1, noise=0):
    if x < a:
        return 1e-16
    return m * (x - a) ** p * (1 + noise * random.uniform(-1, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--one-sided", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--fix-axis-grho", action="store_true")
    args = ap.parse_args()
    if args.fix_axis_grho:
        fix_axis_grho()
    if args.one_sided:
        relu_model.ReLU = np.vectorize(one_sided, otypes=[np.float64])
    if args.seed is not None:
        init = relu_model.ReLU_FluxModel.__init__

        def seeded(self, pars, grid, time):
            init(self, dict(pars, random_seed=args.seed), grid, time)
        relu_model.ReLU_FluxModel.__init__ = seeded
    sys.argv = [sys.argv[0], args.input]
    main()
