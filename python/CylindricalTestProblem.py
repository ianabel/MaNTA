import matplotlib.pyplot as plt
from FFIRunner import FFIRunner
from VectorizedTransportSystem import VectorizedTransportSystem
from JAXAdjointProblem import JAXAdjointProblem
import MaNTA
import scipy.special
import scipy.optimize
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple
from jaxtyping import Array, ArrayLike, Float, Int
import equinox as eqx
import numpy as np
a = 1.0
n = 10
chi = 0.1
kappa = 2./3. * chi
tau_d = 2.0
b = 1/tau_d
roots = scipy.special.jn_zeros(0, n) / a

S = lambda i, r: scipy.special.jv(0, roots[i] * r)
def Sjax(i, r):
    return jax.pure_callback(S, 
                        jax.ShapeDtypeStruct(r.shape,r.dtype),
                         i, r, vmap_method="expand_dims")
T = lambda i , r: 1 / (kappa * roots[i]**2 + b) * S(i, r)
r = np.linspace(0, a)

T_out = 0
for i in range(0,n):
    T_out += T(i, r)

# add r as a parameter
class DiffusionParams(NamedTuple):
    kappa: Float[ArrayLike, '...'] 
    b: Float[ArrayLike, '...'] 
   
    @classmethod
    def make(cls, config) -> 'DiffusionParams':
        return cls(
             kappa = config["kappa"],
             b = config["b"],
        )

class CylindricalTestProblem(VectorizedTransportSystem):
    def __init__(self, params, coord_type="rT"):
            super().__init__()

            self.nVars = 1
            self.params = params

            self.isUpperDirichlet  = True
            self.isLowerDirichlet  = False
            self.coord_type = coord_type
            # %%
            config = {
                "OutputFilename": "CylindricalTestProblem",
                "Polynomial_degree": 4,
                "Grid_size": 5,
                "Lower_boundary": 0.0,
                "Upper_boundary":  self.r_to_x(a),
                "Relative_tolerance" : 0.001,
                "tFinal": 1.0,
                "delta_t": 0.5,
                "solveAdjoint": True, 
                "restart" : False,
                "zeroFlux": True,
                "SteadyStateTolerance": 1e-3
            }

            self.points = MaNTA.getNodes(config["Lower_boundary"], config["Upper_boundary"], config["Grid_size"], config["Polynomial_degree"])

            self.adjointProblem = JAXAdjointProblem(self, self.g)
            
            self.runner = FFIRunner(self, self.points, 1, self.adjointProblem.np)
            self.runner.configure(config)

    def run(self, tFinal = None):
        if (tFinal is not None):
            self.runner.Run(tFinal)
        else:
            self.runner.Run_ss()

    def getAdjointGradients(self):
        G, G_p = self.runner.Get_adjoint_gradients()
        return G, G_p

    def getTemperature(self, points = None):
        if points is None:
            points = self.points

        return self.u_to_T(self.runner.Get_profile(0, points), points)
    def g(self, state, x, params):
        u = state.Variable[0]
        return u
 
    def setParams(self, params):
        self.params = params

    def sigma( self, index, state, x, t, params ):
        tprime = self.up_to_Tp(state.Derivative[index], state.Variable[index], x)
        return self.geometricFactor(x) * params.kappa * tprime
    
    def source( self, index, state, x, t, params ):
        s = 0.0
        for i in range(0,n):
            s+=Sjax(i, self.x_to_r(x))
        s -= self.u_to_T(state.Variable[index], x) *params.b
        return self.sourceFactor(x)* s
    
    def geometricFactor(self, x):
        if (self.coord_type == "rT"):
            return x
        else:
            return self.x_to_r(x) ** 2

    def sourceFactor(self,x):
        if (self.coord_type == "rT"):
            return x
        else: 
            return 1.0

    def x_to_r(self, x):
        if (self.coord_type == "rT"):
            return x
        else: 
            return jnp.sqrt(2 * x)

    def r_to_x(self,r):
        if (self.coord_type == "rT"):
            return r
        else:
            return r * r / 2

    def u_to_T(self, u, x):
        if (self.coord_type == "rT"):
            return u / x
        else:
            return u

    def up_to_Tp(self, up, u,  x):
        if (self.coord_type == "rT"):
            return (up - self.u_to_T(u, x))/x
        else:
            return up
    def LowerBoundary(self, index, t):
        return 0.0

    def UpperBoundary(self, index, t):
        return 0.0
    
    def InitialValue( self, index, x ):
        return 0.0

    def createAdjointProblem(self):
        return self.adjointProblem


def make_jvp(coord_type):
    def f(params, coord_type):
        ctp = CylindricalTestProblem(params, coord_type)
        ctp.run()
        G, G_p = ctp.getAdjointGradients()
        return G[0], G_p

    f_wrapper = partial(f, coord_type=coord_type)
    @eqx.filter_custom_jvp
    def obj(params):
        G = f_wrapper(params)[0]
        
        return G

    @obj.def_jvp
    def obj_jvp(primals,tangents):
        params, = primals
        t, = tangents
        G, G_p = f_wrapper(params)
        t_out = jnp.dot(G_p, jax.flatten_util.ravel_pytree(t)[0])
        return G, t_out[0]
    
    return obj

k1 = jnp.linspace(0.1, 0.5, 10)
dk = k1[1] - k1[0]
gfunc = make_jvp("adf")
g_out = []
grad_g = []
for k in k1:
    params = DiffusionParams(k, b)
    g, gp = jax.value_and_grad(gfunc)(params)
    g_out.append(g)
    grad_g.append(gp.kappa)

plt.figure()
plt.plot(k1,g_out)
plt.figure()
plt.plot(k1, jnp.gradient(jnp.array(g_out)/dk),'ro')
plt.plot(k1, grad_g, 'bx')
plt.show()
# ctp = CylindricalTestProblem(coord_type="rT")

# ctp.run()
# Te = ctp.getTemperature(ctp.r_to_x(r))
# G, G_p = ctp.getAdjointGradients()
# print(G_p)

# ctp = CylindricalTestProblem(coord_type="asdf")

# ctp.run()
# Te = ctp.getTemperature(ctp.r_to_x(r))
# G, G_p = ctp.getAdjointGradients()
# print(G_p)
# # fig, ax = plt.subplots()
# # ax.plot(r,T_out, label="analytic")
# # ax.plot(r,Te, label="numerical")
# # ax.legend()
# # plt.show()