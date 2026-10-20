import jax.numpy as jnp
import equinox as eqx
from stellarator_state import StellaratorParams, StellaratorState
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid
from yancc.species import LocalMaxwellian, Electron
from yancc.solve import solve_dke
import jax


def _species(state: StellaratorState, params: StellaratorParams, evolveDensity):
    """The plasma at one point as yancc wants it: SI densities, eV temperatures,
    gradients per unit rho. Ions only without evolveDensity."""
    c = params.constants
    ions = LocalMaxwellian(
        c.IonSpecies.yancc_species,
        temperature=state.Ti * c.T0eV,
        density=state.n * c.n0,
        dTdrho=state.dTidrho * c.T0eV,
        dndrho=state.dndrho * c.n0,
    )
    if not evolveDensity:
        return [ions]
    electrons = LocalMaxwellian(
        Electron,
        temperature=state.Te * c.T0eV,
        density=state.n * c.n0,
        dTdrho=state.dTedrho * c.T0eV,
        dndrho=state.dndrho * c.n0,
    )
    return [ions, electrons]


def yancc_moments(species, Erho, field, pitchgrid, speedgrid, evolveDensity):
    """Fluxes from the drift-kinetic solve, in yancc's (SI) units.

    Returns per-species `<particle_flux>` [m^-3 s^-1] and `<heat_flux>`
    [W m^-3], both the flux-surface average of the flux dotted with grad(rho),
    and the radial current `J_rho` [A m^-3].
    """
    if evolveDensity:
        rtol, max_grids = 1e-2, 2
    else:
        rtol, max_grids = 1e-3, 3

    sol, info = solve_dke(
        field,
        pitchgrid,
        speedgrid,
        species,
        Erho=Erho,
        # m=50,
        rtol=rtol,
        throw=False,
        verbose=0,
        multigrid_options={"smooth_solver": "banded", "max_grids": max_grids},
    )
    return sol.get("<particle_flux>"), sol.get("<heat_flux>"), sol.get("J_rho")


def _normalized_fluxes(
    moments,
    _state,
    x,
    t,
    field,
    vp,
    vpp,
    pitchgrid,
    speedgrid,
    params: StellaratorParams,
    evolveDensity,
):
    """Everything between the state MaNTA holds and the fluxes it wants back,
    around a `moments` function that works in SI units.

    The physics in `moments` sees only physical quantities and returns only
    physical ones, so this function is the whole of the normalisation of the
    fluxes -- which is what lets a mock in SI units (`physical_test_flux`) test
    that the normalisation drops out.
    """
    state = StellaratorState.from_state(_state, x, vp, vpp, params)
    c = params.constants
    species = _species(state, params, evolveDensity)

    if not evolveDensity:
        _, heat_flux, _ = moments(
            species, jnp.array(0.0), field, pitchgrid, speedgrid, evolveDensity
        )
        flux = -heat_flux[0] * vp / c.HeatEquationNormalization()
        return [flux], []

    particle, heat_flux, current = moments(
        species, state.Er * c.T0eV, field, pitchgrid, speedgrid, evolveDensity
    )

    # add a little extra diffusion to the density
    x0 = 0.9
    extra_diffusion = params.config.ADCoefficient * jax.lax.cond(
        jax.lax.ge(x, x0),
        lambda s: vp * (x - x0) / (1 - x0) * s.dndrho,
        lambda s: s.dndrho * 0.0,
        state,
    )

    # The fluxes are already per unit rho ([m^-3 s^-1], [W m^-3]), the units of
    # dn/dt and dp/dt, so they normalise exactly as the sources do. (yancc's
    # docs site still says per m^2, from before its move to rho in 980ae59;
    # see radial_magnetic_drift, which computes v_m . grad(rho).) Had they been
    # per m^2, the conversion would be a division by the device's field.a_minor,
    # never a factor of the normalising length constants.a.
    particle_flux = (
        -particle[1] * vp / c.DensityEquationNormalization()
    ) + extra_diffusion

    heat_flux = -heat_flux * vp / c.HeatEquationNormalization()

    aux_g_out = current / c.CurrentNormalization()

    return [particle_flux, heat_flux[0], heat_flux[1]], [aux_g_out]


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
    return _normalized_fluxes(
        yancc_moments,
        _state,
        x,
        t,
        field,
        vp,
        vpp,
        pitchgrid,
        speedgrid,
        params,
        evolveDensity,
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


"""
A mock drift-kinetic solve in SI units, for checking the normalisation.

`test_flux` is written in normalised units, so its diffusivity *is* a
normalised number and changing the normalisation changes the physics it
describes. This one stands in for `yancc_moments` instead: physical species in,
physical fluxes out, through the same `_normalized_fluxes` as the real solve.
A result that moves with the normalisation under it is therefore a leak in the
normalisation, not in the mock.

Diffusive in each species, D ~ T^(7/2) / (n B^2) -- the 1/nu scaling, and so
the stiff case -- with B the flux-surface average of |B| from the field, and
with the neoclassical form of the particle
flux,

    Gamma = -D n (n'/n - Z Er/T + (D2/D1 - 3/2) T'/T),

so the Er drive gives the ambipolarity row a root that depends on the state,
and T' drives particles (thermodiffusion). D2/D1 = 7/2 is the 1/nu value. The
reference values are fixed physical constants, deliberately not
`params.constants`.
"""

_MOCK_D_ION = 0.03  # [1/s], at the temperature, density and field below
_MOCK_D_ELECTRON = 0.015
_MOCK_T_REF = 1000.0  # [eV]
_MOCK_N_REF = 1e20  # [m^-3]
_MOCK_B_REF = 5.0  # [T]
_MOCK_T_EXPONENT = 3.5
_MOCK_N_EXPONENT = -1.0
_MOCK_D2_OVER_D1 = 3.5
_ELEMENTARY_CHARGE = 1.60217663e-19


def physical_test_moments(species, Erho, field, pitchgrid, speedgrid, evolveDensity):
    # state.Er is the one-entry aux vector; yancc reduces it to a scalar itself
    Erho = jnp.reshape(Erho, ())
    B = field.Bmag_fsa
    particle, heat = [], []
    for s, D_ref in zip(species, (_MOCK_D_ION, _MOCK_D_ELECTRON)):
        n, T = s.density, s.temperature
        Z = s.species.charge / _ELEMENTARY_CHARGE
        D = (
            D_ref
            * (T / _MOCK_T_REF) ** _MOCK_T_EXPONENT
            * (n / _MOCK_N_REF) ** _MOCK_N_EXPONENT
            * (_MOCK_B_REF / B) ** 2
        )
        gamma = -D * (
            s.dndrho - Z * n * Erho / T + (_MOCK_D2_OVER_D1 - 1.5) * n * s.dTdrho / T
        )
        q = 2.5 * T * _ELEMENTARY_CHARGE * gamma - D * n * _ELEMENTARY_CHARGE * s.dTdrho
        particle.append(gamma)
        heat.append(q)
    particle, heat = jnp.stack(particle), jnp.stack(heat)
    charges = jnp.stack([s.species.charge for s in species])
    return particle, heat, jnp.sum(charges * particle)


def physical_test_flux(
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
    return _normalized_fluxes(
        physical_test_moments,
        _state,
        x,
        t,
        field,
        vp,
        vpp,
        pitchgrid,
        speedgrid,
        params,
        evolveDensity,
    )


def physical_test_flux_field_jac(
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
        return physical_test_flux(
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
