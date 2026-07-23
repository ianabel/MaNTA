import jax.numpy as jnp
from MirrorPlasma.PlasmaState import (
    MirrorPlasmaParams,
    MirrorPlasmaState,
)
import optimistix as optx
import jax
import equinox as eqx


@jax.jit
def InitialPhiValue(state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
    def func(phi, args):
        state_new = eqx.tree_at(lambda s: s.phi, state, phi)
        return ParallelCurrent(state_new, x, t, params)

    solver = optx.Newton(rtol=1e-4, atol=1e-4)
    sol = optx.root_find(func, solver, 0.0)
    phi_g = sol.value
    return phi_g


def ParallelCurrent(state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
    return (
        IonPastukhovLossRate(state, x, t, params)
        / params.Constants.DensityEquationNormalization()
        - ElectronPastukhovLossRate(state, x, t, params)
        / params.Constants.DensityEquationNormalization()
    )


def CentrifugalPotential(state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
    tau = state.Ti / state.Te
    return (
        1.0
        / (1.0 + tau)
        * (1.0 - 1.0 / params.MagneticField.MirrorRatio(x))
        * state.M**2
        / 2
    )


def Xi_i(state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
    return CentrifugalPotential(state, x, t, params) + state.phi


def Xi_e(state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
    return CentrifugalPotential(state, x, t, params) - state.Ti / state.Te * state.phi


def ElectronPastukhovLossRate(
    state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams
):
    tau_ee = (
        params.Constants.ElectronCollisionTime(state.n, state.Te)
        * params.Constants.ReferenceElectronCollisionTime()
    )
    Sigma = 1 + params.Constants.Z_eff
    Xi = Xi_e(state, x, t, params)
    PastukhovFactor = jnp.exp(-Xi) / Xi

    n = params.Constants.n0 * state.n

    return (
        2
        * n
        * Sigma
        / (
            jnp.sqrt(jnp.pi)
            * tau_ee
            * jnp.log(params.MagneticField.MirrorRatio(x) * Sigma)
        )
    ) * PastukhovFactor


def IonPastukhovLossRate(state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
    tau_i = (
        params.Constants.IonCollisionTime(state.n, state.Ti)
        * params.Constants.ReferenceIonCollisionTime()
    )
    Sigma = 1.0
    Xi = Xi_i(state, x, t, params)

    PastukhovFactor = jnp.exp(-Xi) / Xi

    n = params.Constants.n0 * state.n

    return (
        2
        * n
        * Sigma
        / (
            jnp.sqrt(jnp.pi)
            * tau_i
            * jnp.log(params.MagneticField.MirrorRatio(x) * Sigma)
        )
    ) * PastukhovFactor
