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
# eq_names = ["eq_itg_flux_no_nc"]
eq_names = ["eq4", "eq5"]
output_name = "nc_regimes"


fig, axs = plt.subplots(len(eq_names), 2, layout="compressed")

# fig.subplots_adjust(wspace=0, hspace=0)
ax_lists = []
if len(eq_names) == 1:
    ax_lists = [axs]
else:
    ax_lists = [axs[i, :] for i in range(0, len(eq_names))]

for eq_name, ax in zip(eq_names, ax_lists):
    eqs = desc.io.load(eq_name + "_all_equilibria.h5")

    plot_boundaries(eqs=[eqs[0], eqs[-1]], labels=["Initial", "Optimized"], ax=ax[0])
    plot_boozer_surface(eqs[-1], fieldlines=4, ax=ax[1], fill=True)


fig.set_figwidth(5.0)
fig.set_figheight(2.5 * len(eq_names))

# fig.tight_layout()


fig.savefig("figs/" + output_name + "_fields.pdf", dpi=500)
plt.show()
