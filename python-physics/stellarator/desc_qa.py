import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# uncomment to run test flux
# os.environ["TEST_STELLARATOR"] = "true"

# os.environ.pop("TEST_STELLARATOR", None)
import matplotlib.pyplot as plt
from desc.profiles import SplineProfile
from desc.plotting import plot_boozer_surface, plot_boundaries, plot_qs_error
from desc.plotting import plot_comparison
from desc.objectives import (
    AspectRatio,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
    QuasisymmetryTwoTerm,
    GenericObjective,
    ObjectiveFromUser,
    Volume,
)

import desc
from desc.grid import Grid, LinearGrid
from desc.geometry import FourierRZToroidalSurface
from desc.equilibrium import Equilibrium, EquilibriaFamily
import numpy as np


# %%
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


eq_name = "eq_qa"
# # create initial equilibrium. Psi chosen to give B ~ 1 T. Could also give profiles here,
# # default is zero pressure and zero current
# eq = Equilibrium(M=4, N=4, Psi=0.1, surface=surf, pressure=desc_pressure)
# # this is usually all you need to solve a fixed boundary equilibrium
# eq = eq.solve(x_scale="ess")[0]
# print(pressure_rho)
#
surf = FourierRZToroidalSurface(
    R_lmn=[1, 0.125, 0.1],
    Z_lmn=[-0.125, -0.1],
    modes_R=[[0, 0], [1, 0], [0, 1]],
    modes_Z=[[-1, 0], [0, -1]],
    NFP=4,
)
# # create initial equilibrium. Psi chosen to give B ~ 1 T. Could also give profiles here,
# # default is zero pressure and zero current
eq = Equilibrium(M=4, N=4, Psi=0.1, surface=surf)
#
# eqs = desc.io.load(eq_name + "_all_equilibria.h5")
#
eq = desc.compat.rescale(eq, L=("R0", 10), B=("B0", 5.0))
# #
# eq.change_resolution(M=4, N=4, L_grid=len(points), M_grid=8, N_grid=8)
# eq.pressure = desc_pressure
# eq = eq.solve(x_scale="ess")[0]
# # desc_pressure = eq.get_profile('p')
# eqs = EquilibriaFamily(eq)
eq_init = eq.copy()

V0 = eq.compute("V")["V"]
# yancc_wrapper = yancc_data.from_eq(points, grid = yancc_grid,rho = yancc_rho, Density=Density, eq=eq_init, nt = yancc_ntheta, nz = yancc_nzeta)


def run_qh_step(k, eq):
    """Run a step of the precise QH optimization example from Landreman & Paul."""
    # this step will only optimize boundary modes with |m|,|n| <= k

    # create grid where we want to minimize QS error. Here we do it on 3 surfaces
    grid = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True
    )

    # Mirror ratio, manually computed from "|B|"
    def fun_mirror_ratio(grid, data):
        # alternatively, "mirror ratio" is something that can be computed in the data index
        # directly (see List of Variables docs), so can replace entirety of the above function code with this return statement
        return grid.compress(data["mirror ratio"])
        # or can just use GenericObjective(f="mirror ratio", thing=eq) or the already-existing objective, MirrorRatio(eq)

    obj_mirror_ratio = ObjectiveFromUser(
        fun=fun_mirror_ratio,
        thing=eq,
        grid=grid,
        bounds=(0.0, 0.3),
        weight=2.0,
        name="my mirror ratio",
    )

    # we create an ObjectiveFunction, in this case made up of multiple objectives
    # which will be combined in a least squares sense
    objective = ObjectiveFunction(
        (
            # pass in the grid we defined, and don't forget the target helicity!
            QuasisymmetryTwoTerm(eq=eq, helicity=(1, eq.NFP), grid=grid),
            # try to keep the aspect ratio about the same
            Volume(eq=eq, target=V0, weight=10.0),
            AspectRatio(eq=eq, target=8, weight=100),
            obj_mirror_ratio,
        ),
    )
    objective.build()
    # as opposed to SIMSOPT and STELLOPT where variables are assumed fixed, in DESC
    # we assume variables are free. Here we decide which ones to fix, starting with
    # the major radius (R mode = [0,0,0]) and all modes with m,n > k
    R_modes = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :
            ],
        )
    )
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :
    ]
    # next we create the constraints, using the mode number arrays just created
    # if we didn't pass those in, it would fix all the modes (like for the profiles)
    constraints = (
        ForceBalance(eq=eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        # AspectRatio(eq=eq, target=8, weight=100),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    )
    # this is the default optimizer, which re-solves the equilibrium at each step

    eq_new, info_out = eq.optimize(
        objective=objective,
        constraints=constraints,
        optimizer="proximal-lsq-exact",
        x_scale="ess",
        maxiter=20,
        ftol=1e-6,  # stopping tolerance on the function value
        xtol=1e-6,  # stopping tolerance on the step size
        gtol=1e-6,  # stopping tolerance on the gradient
        # options={
        #     "initial_trust_radius": 1.0,
        #     # "perturb_options": {"order": 2, "verbose": 3},  # use 2nd-order perturbations
        #     #     # "solve_options": {
        #     #     #     "ftol": 5e-3,
        #     #     #     "xtol": 1e-6,
        #     #     #     "gtol": 1e-6,
        #     #     #     "verbose": 3,
        #     # },  # for equilibrium subproblem
        # },
        verbose=3,
        copy=True,
    )

    return eq_new


for k in range(1, 4):
    eq = run_qh_step(k, eq)
# %%

# %%

fig, ax = plot_comparison(eqs=[eq_init, eq], labels=["Initial", "optimized"])

fig.savefig("figs/" + eq_name + "comparison")
eq.save(eq_name + ".h5")
fig.savefig("figs/" + eq_name + "final_comparison.png")
fig, ax = plot_boundaries(eqs=[eq_init, eq], labels=["Initial", "optimized"])
fig.savefig("figs/" + eq_name + "final_comparison_boundary.png")
# %%
# eq = desc.io.load("../python/eq2optimized_equilibrium.h5")#desc.examples.get("ESTELL")
# plot_boozer_surface(eq_init, fieldlines=8)
fig, ax = plot_boozer_surface(eq, fieldlines=8)
fig.savefig("figs/" + eq_name + "final_boozer_surface.png")
fig, ax = plot_qs_error(eq)
fig.savefig("figs/" + eq_name + "final_qs_error.png")


#
