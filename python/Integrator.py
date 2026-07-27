import equinox as eqx
from jaxtyping import ArrayLike, Float, Int
import jax
import jax.numpy as jnp


class Integrator(eqx.Module):
    weights: Float[ArrayLike, "..."]
    phi_boundary: Float[ArrayLike, "..."]
    k: Int = eqx.field(static=True)
    nCells: Int = eqx.field(static=True)

    def __init__(self, k_, nCells_, weights_, phi_boundary_):
        self.weights = weights_
        self.phi_boundary = phi_boundary_
        self.k = k_
        self.nCells = nCells_

    def __call__(self, f):
        # jax.debug.print("weights = {val}", val=self.weights)
        return jnp.dot(f, self.weights)

    def computeCellProducts(self, f):
        return f * self.weights

    def phiL(self):
        return self.phi_boundary[:, 0]

    def phiR(self):
        return self.phi_boundary[:, 1]
