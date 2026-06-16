import equinox as eqx
import jax
from jaxtyping import Array, ArrayLike, Float, Int
import jax.numpy as jnp
import numpy as np

"""
Wrapper class for MaNTA State
"""
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
        shape = manta_state["Variable"].shape
        dp = shape[0]
        nscalars = 0 if manta_state["Scalars"] is None else manta_state["Scalars"].shape[0]
        return cls(Variable_=jnp.array(manta_state["Variable"]),
                   Derivative_=jnp.array(manta_state["Derivative"]),
                   Flux_=jnp.array(manta_state["Flux"]),
                   Aux_=jnp.array(manta_state["Aux"]),
                   Scalars_=jnp.repeat(jnp.expand_dims(jnp.array(manta_state["Scalars"]),axis=0),repeats=dp-nscalars, axis=0))
    
    def to_manta(self):
        Scalars_out = self.Scalars if self.Scalars.size==0 else self.Scalars[0]
        return {
            "Variable":   np.asarray(self.Variable),
            "Derivative": np.asarray(self.Derivative),
            "Flux":       np.asarray(self.Flux),
            "Aux":        np.asarray(self.Aux),
            "Scalars":    np.asarray(Scalars_out)
        }
    
    @staticmethod
    def vmap_axes():
        return 0
    
def MaNTA_Decorator(func):
    def wrapper(self, index, states, positions, *args):
        states_ = State.from_manta(states)
        positions_ = jnp.array(positions)
        res = func(self, index, states_, positions_, *args)

        if (isinstance(res, State)):
            return res.to_manta()
        else: 
            return res
    return wrapper
    

