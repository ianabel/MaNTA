import equinox as eqx
from jaxtyping import ArrayLike, Float, Int
import jax
import jax.numpy as jnp


class Integrator(eqx.Module):
    weights: Float[ArrayLike, "..."]
    phis: Float[ArrayLike, "..."]
    phi_boundary: Float[ArrayLike, "..."]
    k: Int = eqx.field(static=True)
    nCells: Int = eqx.field(static=True)

    def __init__(self, k_, nCells_, weights_, phis_, phi_boundary_):
        self.weights = weights_
        self.phis = phis_
        self.phi_boundary = phi_boundary_
        self.k = k_
        self.nCells = nCells_

    def __call__(self, f):
        return jnp.dot(self.weights, f)

    def computeCellProducts(self, f):
        def cellProduct(v, w):
            return jnp.dot(v, w)

        _cell_weights = jnp.reshape(
            jnp.atleast_2d(self.weights), (self.k + 1, self.nCells)
        )
        _rep_weights = jnp.repeat(_cell_weights, repeats=self.k + 1, axis=1)
        _vin = jnp.multiply(f, self.phis)
        return jax.vmap(cellProduct, in_axes=(1, 1))(_vin, _rep_weights)

    def phiL(self):
        return self.phi_boundary[:, 0]

    def phiR(self):
        return self.phi_boundary[:, -1]
