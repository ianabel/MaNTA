import jax.numpy as jnp
import equinox as eqx
from stellarator_state import StellaratorParams, StellaratorState
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid
from yancc.species import LocalMaxwellian, Electron
from yancc.solve import solve_dke


# we keep this function pure and not dependent on class data so jax can save it in persistent cache
def compute_dke_sol(
    _state,
    x,
    t,
    field,
    vp,
    vpp,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: MaxwellSpeedGrid,
    params: StellaratorParams,
    evolveDensity=False,
):
    """
    Computes neoclassical information for stellarator model

    Parameters
    ----------

    state : eqx.Module
        Object containing state information
    x : float
        Spatial location
    t : float
        Time
    field: yancc.Field
        Magnetic field object
    vp: float
        V'
    vpp: float
        V''
    params : NamedTuple
        Transport system parameters, passed for JAX PyTree compatibility
    pitchgrid: yancc.UniformPitchAngleGrid
    speedgrid: yancc.MaxwellSpeedGrid
    evolveDensity: bool
        whether to compute multichannel fluxes
    Returns
    -------
    float
        Computed flux and aux terms (normalized)
    """

    state = StellaratorState.from_state(_state, x, vp, vpp, params)

    def constant_density(
        state: StellaratorState,
        x,
        t,
        field,
        vp,
        vpp,
        pitchgrid,
        speedgrid,
        params: StellaratorParams,
    ):
        Erho = jnp.array(0.0)

        species = [
            LocalMaxwellian(
                params.constants.IonSpecies.yancc_species,
                temperature=state.Ti * params.constants.T0eV,
                density=state.n * params.constants.n0,
                dTdrho=state.dTidrho * params.constants.T0eV,
                dndrho=state.dndrho * params.constants.n0,
            ),
        ]

        sol, info = solve_dke(
            field,
            pitchgrid,
            speedgrid,
            species,
            Erho=Erho,
            # m=50,
            rtol=1e-3,
            throw=False,
            verbose=0,
            multigrid_options={"smooth_solver": "banded", "max_grids": 3},
        )
        flux = (
            -sol.get("<heat_flux>")[0]
            * vp
            * (params.constants.a / params.constants.HeatEquationNormalization())
        )

        return [flux], []

    def ambipolar(
        state: StellaratorState,
        x,
        t,
        field,
        vp,
        vpp,
        pitchgrid,
        speedgrid,
        params: StellaratorParams,
    ):
        species = [
            LocalMaxwellian(
                params.constants.IonSpecies.yancc_species,
                temperature=state.Ti * params.constants.T0eV,
                density=state.n * params.constants.n0,
                dTdrho=state.dTidrho * params.constants.T0eV,
                dndrho=state.dndrho * params.constants.n0,
            ),
            LocalMaxwellian(
                Electron,
                temperature=state.Te * params.constants.T0eV,
                density=state.n * params.constants.n0,
                dTdrho=state.dTedrho * params.constants.T0eV,
                dndrho=state.dndrho * params.constants.n0,
            ),
        ]

        sol, info = solve_dke(
            field,
            pitchgrid,
            speedgrid,
            species,
            Erho=state.Er * params.constants.T0eV,
            # m=50,
            rtol=1e-2,
            throw=False,
            verbose=0,
            multigrid_options={"smooth_solver": "banded", "max_grids": 2},
        )

        # particle flux is in units of [m^-2 s^-1], dn/dt is [m^-3 s^-1], so correct normalization is (a / (n_0/t_0))
        particle_flux = (
            -sol.get("<particle_flux>")[1]
            * vp
            * (params.constants.a / params.constants.DensityEquationNormalization())
        )

        heat_flux = (
            -sol.get("<heat_flux>")
            * vp
            * (params.constants.a / params.constants.HeatEquationNormalization())
        )

        # scale current up for stricter enforcement of ambipolarity
        aux_g_out = (
            10.0
            * sol.get("J_rho")
            * (params.constants.a / params.constants.CurrentNormalization())
        )

        return [particle_flux, heat_flux[0], heat_flux[1]], [aux_g_out]

    if evolveDensity:
        return ambipolar(
            state,
            x,
            t,
            field,
            vp,
            vpp,
            pitchgrid,
            speedgrid,
            params,
        )
    else:
        return constant_density(
            state,
            x,
            t,
            field,
            vp,
            vpp,
            pitchgrid,
            speedgrid,
            params,
        )


def dke_field_jac(
    states,
    positions,
    t,
    field,
    vp,
    vpp,
    pitchgrid,
    speedgrid,
    params,
    evolveDensity=False,
):
    def _dke_sol(tree_in):
        _field, _vp, _vpp = tree_in
        return compute_dke_sol(
            states,
            positions,
            t,
            _field,
            _vp,
            _vpp,
            pitchgrid,
            speedgrid,
            params,
            evolveDensity,
        )

    return eqx.filter_jacrev(_dke_sol)((field, vp, vpp))


"""
Linear diffusion with the same signature as above for testing purposes
"""


def test_flux(
    _state,
    x,
    t,
    field,
    vp,
    vpp,
    pitchgrid: UniformPitchAngleGrid,
    speedgrid: MaxwellSpeedGrid,
    params: StellaratorParams,
    evolveDensity=False,
):

    state = StellaratorState.from_state(_state, x, vp, vpp, params)

    def ambipolar(state, x, t, field, vp, vpp, *args):

        particle_flux = vp * 0.1 * state.dndrho
        ion_heat_flux = vp * 0.1 * state.dTidrho

        electron_heat_flux = vp * 0.1 * state.dTedrho

        aux_g_out = 0.0 * vp

        return [particle_flux, ion_heat_flux, electron_heat_flux], [aux_g_out]

    def constant_density(state, x, t, field, vp, vpp, *args):
        return [vp * 0.01 * state.dTidrho], []

    if evolveDensity:
        return ambipolar(state, x, t, field, vp, vpp, pitchgrid, speedgrid, params)
    else:
        return constant_density(
            state, x, t, field, vp, vpp, pitchgrid, speedgrid, params
        )


def test_flux_field_jac(
    states,
    positions,
    t,
    field,
    vp,
    vpp,
    pitchgrid,
    speedgrid,
    params,
    evolveDensity=False,
):
    def _dke_sol(tree_in):
        _field, _vp, _vpp = tree_in
        return test_flux(
            states,
            positions,
            t,
            _field,
            _vp,
            _vpp,
            pitchgrid,
            speedgrid,
            params,
            evolveDensity,
        )

    return eqx.filter_jacrev(_dke_sol)((field, vp, vpp))
