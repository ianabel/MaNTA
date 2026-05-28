from typing import NamedTuple

from functools import partial

import MaNTA
from VectorizedTransportSystem import VectorizedTransportSystem
from JAXAdjointProblem import JAXAdjointProblem
import jax.numpy as jnp
import numpy as np
import jax
from jaxtyping import Array, ArrayLike, Float, Int
import equinox as eqx
class NonlinearDiffusionParams(NamedTuple):
    T_s: Float[ArrayLike, '...'] 
    D: Float[ArrayLike, '...'] 
    a: Float[ArrayLike, '...']
    SourceWidth: Float[ArrayLike, '...']
    SourceCentre: Float[ArrayLike, '...']
   
    @classmethod
    def make(cls, config) -> 'NonlinearDiffusionParams':
        return cls(
             T_s = 50.0,
             SourceCentre = config["SourceCentre"],
             D = config["D"],
             a = config["a"],
             SourceWidth = 0.02
        )
    
    @classmethod
    def makeSpatial(cls, config) -> 'NonlinearDiffusionParams':
        ones = jnp.ones_like(config["T_s"])
        return cls(
             T_s = config["T_s"],
             SourceCentre = config["SourceCentre"] * ones,
             D = config["D"] * ones,
             a = config["a"] * ones,
             SourceWidth = 0.02 * ones
        )
class JAXNonlinearDiffusion(VectorizedTransportSystem):
    def __init__(self, config):
        super().__init__(spatialParameters=True)
        self.nVars = 1
        self.isUpperDirichlet  = True
        self.isLowerDirichlet  = False
        
        solver_config = config["solver"]
        
        self.points = MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])

        self.params = NonlinearDiffusionParams.makeSpatial(config["ts"])
        self.adjointProblem = JAXAdjointProblem(self, self.g, spatialParameters=True)
        self.runner = MaNTA.Runner(self)

        self.runner.configure(solver_config)


        # This object will be passed to sigma and source functions
    
    def run(self, tFinal = None):
        if (tFinal is not None):
            sFinal = self.runner.run(tFinal)
        else: 
            sFinal = self.runner.run_ss()

        return sFinal

    def getAdjointGradients(self):
        G, G_p = self.runner.getAdjointGradients()
        return G, G_p["G_p"]

    def g(self, state, x, params):
        u = state.Variable[0]
        return 0.5 * u * u 

    def sigma( self, index, state, x, t, params ):
        
        u = state.Variable[0]
        q = state.Derivative[0]
        return params.D*(u ** params.a) * q

    def source( self, index, state, x, t, params ):
        return params.T_s

    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.3
    
    def InitialValue(self, index, x):
        return 0.3
    
    def createAdjointProblem(self):
        return self.adjointProblem

def runMaNTA(func, solver_config):

    ts_config = {
        "SourceCentre" : 0.3,
        "D" : 2.0,
        "a" : 0.0,
        "T_s" : func 
    }
    config = {"solver": solver_config, "ts": ts_config}
    transportSystem = JAXNonlinearDiffusion(config)

    transportSystem.run()
    return transportSystem.getAdjointGradients()
    
 
solver_config = {
    "OutputFilename": "out",
    "Polynomial_degree": 3,
    "Grid_size": 5,
    "tau": 1.0, 
    "Lower_boundary": 0.0,
    "Upper_boundary": 1.0,
    "Relative_tolerance": 0.01,
    "delta_t": 1.0,
    "restart": False,
    "solveAdjoint": True, 
}

points = MaNTA.getNodes(solver_config["Lower_boundary"], solver_config["Upper_boundary"], solver_config["Grid_size"], solver_config["Polynomial_degree"])

f = partial(runMaNTA, solver_config=solver_config)    
T = lambda x : 50.0 * jnp.sin(2 * jnp.pi * x) ** 2
G, G_p_adj = f(T(points))
T_p = T(points)

G_p_fd = []

def fd_jvp(tangent):
    dT = 0.001 + 0.1*jnp.linalg.norm(T_p)

    T_in = T_p + dT * (tangent / jnp.linalg.norm(tangent))
    G_2 = f(T_in)[0]
    return (G_2 - G)/ dT 
def adj_jvp(tangent):
    return jnp.dot(G_p_adj[:,0], tangent)
# for i in range(0,len(points)):
#     x = points[i]
#     dT = 0.001 + 0.1* T(x)
#     T_pert = T_p
#     T_pert = T_pert.at[i].set(T_p[i] + dT)
#     G_1 = f(T_pert)[0]
#     G_2 = f(T_p)[0]
    
#     print(G_1)
#     print(G_2)
#     G_p_fd.append((G_1-G_2) / (dT))
# G_d_fd_arr = jnp.array(G_p_fd)
# fig, ax = plt.subplots()
# ax.plot(points, G_p_fd, 'bo', label="finite differences")
# ax.plot(points, G_p_adj[:,0], 'rx', label="adjoints")
# ax.legend()
# fig, ax = plt.subplots()
# ax.semilogy(points,jnp.abs(jnp.array(G_p_fd)- G_p_adj[:,0])**2)
# plt.show()

import matplotlib.pyplot as plt
nTangents = 10

rng_key = jax.random.PRNGKey(10)
def make_random_tangent(primal):
    global rng_key
    rng_key, key = jax.random.split(rng_key)
    # key = jax.random.key(69)
    def map_fn(leaf):
        if jnp.isscalar(leaf) or jnp.isdtype(leaf, "integral"):
            return leaf
        else: 
            v = jax.random.normal(key=key, shape=leaf.shape, dtype=leaf.dtype)
            return v/jnp.linalg.norm(v) 
    tangent_field = jax.tree.map(
        map_fn,
        primal 
    )
    return tangent_field

fig, ax = plt.subplots()

for i in range(0, nTangents):
    t = make_random_tangent(T_p)
    adj = adj_jvp(t)
    fd = fd_jvp(t)
    ax.plot(i, adj, 'bo')
    ax.plot(i, fd, 'rx')

plt.show()
"""
IDEA: 

Source term f(x) = some sort of series

need three test cases 
1) adjoints are f(x) at each point
2) adjoints are fourier coefficients
Need a spatial paramter version, finite difference version, and a global version

How do I compare the two cases? 

What is the derivative that I need? 
"""