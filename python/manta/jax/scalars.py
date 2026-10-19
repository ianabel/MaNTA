"""Global scalar constraints for the JAX base classes, derived from one function.

A JAX case writes its physics as pure functions of the parameters -- `sigma`,
`source`, `aux` -- and the base classes derive every derivative MaNTA asks for
from them. `scalar` is the same thing for a global scalar constraint G_s:

    def scalar(self, index, states, states_dot, integrator, t, params): ...

`states` and `states_dot` are the State on the k+1 basis nodes (Scalars is the
(nScalars,) vector of global unknowns), `integrator` the quadrature over them,
and the return is G_s for scalar `index`. From it the base classes provide
`ScalarG` and `ScalarGPrime`, and `JAXAdjointProblem` provides `dScalarG_dp` --
the one derivative MaNTA has no default for.

`scalar` is optional, so a case that writes `ScalarG` and `ScalarGPrime` itself
goes on working: its methods override these. Such a case's adjoint has no
`params` to differentiate, and `dScalarG_dp` says so rather than assume zero.
"""

import jax

from .state import ScalarG_Decorator, ScalarGPrime_Decorator


class ScalarConstraints:
    """ScalarG and ScalarGPrime from a case's `scalar`, for either JAX base."""

    def _check_scalar_hooks(self):
        # At construction rather than at the first residual: without either form
        # there is nothing to evaluate, and the trampoline's own check for the
        # hook would be satisfied by the defaults below.
        if self.nScalars == 0 or hasattr(self, "scalar"):
            return
        if type(self).ScalarG is ScalarConstraints.ScalarG:
            raise TypeError(
                f"{type(self).__name__} declares {self.nScalars} global scalar(s) "
                "but defines neither scalar(index, states, states_dot, integrator, "
                "t, params) nor ScalarG and ScalarGPrime."
            )

    @ScalarG_Decorator
    def ScalarG(self, index, states, states_dot, integrator, t):
        return self.scalar(index, states, states_dot, integrator, t, self.params)

    @ScalarGPrime_Decorator
    def ScalarGPrime(self, states, states_dot, integrator, t):
        """d G_s / d state and d G_s / d state_dot, for every scalar.

        A pair of lists, each of nScalars States: the gradient of `scalar` with
        respect to the state, which carries the weights `scalar` integrates
        with, and with respect to its time derivative (identically zero for an
        algebraic scalar).
        """
        grad = jax.grad(self.scalar, argnums=(1, 2))
        dG, dGdot = [], []
        for s in range(self.nScalars):
            d, ddot = grad(s, states, states_dot, integrator, t, self.params)
            dG.append(d)
            dGdot.append(ddot)
        return [dG, dGdot]
