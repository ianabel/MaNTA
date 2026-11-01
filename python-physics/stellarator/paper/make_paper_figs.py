"""
Script for putting together plots of equilibria
"""

import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 10})
import desc
import desc.io
from desc.plotting import plot_boozer_surface, plot_boundaries, plot_qs_error, plot_1d


# eq_names = ["eq_random0", "eq_random2", "eq_random3"]
# output_name = "random"
# eq_names = ["eq_itg_flux"]
eq_names = ["eq_itg_flux_no_nc"]
# eq_names = ["eq4", "eq5"]
# names = ["A", "B"]
# output_name = "nc_regimes"

names = None
output_name = eq_names[0]

fig, axs = plt.subplots(len(eq_names), 2, layout="compressed")

fig = plt.figure(figsize=(5, 2.5 * len(eq_names)), layout="constrained")
subfigs = fig.subfigures(len(eq_names), 1, wspace=0, hspace=0.1)

# fig.subplots_adjust(wspace=0, hspace=0)
ax_lists = []
if len(eq_names) == 1:
    subfigs = [subfigs]


for eq_name,  i in zip(eq_names, range(0, len(eq_names))):
    eqs = desc.io.load(eq_name + "_all_equilibria.h5")

    axs = subfigs[i].subplots(1, 2)
    plot_boundaries(eqs=[eqs[0], eqs[-1]], labels=["Initial", "Optimized"], ax=axs[0])
    plot_boozer_surface(eqs[-1], fieldlines=4, ax=axs[1], fill=True)
    if names:
        subfigs[i].suptitle(f"Configuration {names[i]}")


# fig.tight_layout()


fig.savefig("figs/" + output_name + "_fields.pdf", dpi=500)
plt.show()
