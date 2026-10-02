import desc
from desc.plotting import plot_boundaries
from desc.equilibrium import Equilibrium, EquilibriaFamily
import matplotlib.pyplot as plt

eqs = EquilibriaFamily()
for i in range(0, 5):
    surf = desc.random.random_surface(
        M=4, N=4, R0=10, NFP=4, R_scale=(1.0, 2.0), Z_scale=(1.0, 2.0)
    )
    eq = Equilibrium(M=4, N=4, Psi=0.1, surface=surf)
    eq = desc.compat.rescale(eq, B=("<B>", 5.0))
    eq = eq.solve(x_scale="ess")[0]
    eqs.append(eq)
eqs.save("random_equilibria.h5")
fig, ax = plot_boundaries(eqs)
fig.savefig("random_equilibrium.png", dpi=300)
plt.show()
