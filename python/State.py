import equinox as eqx
import jax
from jaxtyping import Array, ArrayLike, Float, Int
import jax.numpy as jnp
import numpy as np

class State(eqx.Module):
    Variable: Float[ArrayLike, '...']
    Derivative: Float[ArrayLike, '...']
    Flux: Float[ArrayLike, '...']
    Aux: Float[ArrayLike, '...']
    Scalars: Float[ArrayLike, '...']

    def __init__(self,
                 Variable_,
                 Derivative_,
                 Flux_, 
                 Aux_,
                 Scalars_):
        self.Variable = Variable_
        self.Derivative = Derivative_
        self.Flux = Flux_
        self.Aux = Aux_
        self.Scalars = Scalars_

    @classmethod 
    def from_manta(cls, manta_state):
        return cls(Variable_=jnp.array(manta_state["Variable"]),
                   Derivative_=jnp.array(manta_state["Derivative"]),
                   Flux_=jnp.array(manta_state["Flux"]),
                   Aux_=jnp.array(manta_state["Aux"]),
                   Scalars_=jnp.array(manta_state["Scalars"]))
    
    def to_manta(self):

        return {
            "Variable": np.asarray(self.Variable),
            "Derivative": np.asarray(self.Derivative),
            "Flux": np.asarray(self.Flux),
            "Aux": np.asarray(self.Aux),
            "Scalars": np.asarray([])
        }
    
    @staticmethod
    def vmap_axes():
        return State(0,0,0,0,None)
    

