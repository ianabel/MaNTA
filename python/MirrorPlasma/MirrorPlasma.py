import jax
import jax.numpy as jnp
import equinox as eqx
from numpy.random import geometric
from scipy.integrate import quad
from MirrorPlasma.PlasmaState import (
    MirrorPlasmaDecorator,
    MirrorPlasmaParams,
    MirrorPlasmaState,
    MirrorPlasmaConfig,
    Channel,
    Scalar,
)
import sys

from MirrorPlasma.ParallelPhysics import (
    InitialPhiValue,
    ParallelCurrent,
    ElectronPastukhovLossRate,
    IonPastukhovLossRate,
    Xi_i,
    Xi_e,
)

from functools import partial

sys.path.append("..")
from VectorizedTransportSystem import VectorizedTransportSystem
from State import State, MaNTA_Decorator, ScalarG_Decorator, ScalarGPrime_Decorator
import MaNTA


class MirrorPlasma(VectorizedTransportSystem):
    def __init__(self, config: MirrorPlasmaConfig, solver_config):
        super().__init__()
        self.nVars = 4
        self.nAux = 1
        if config.useConstantVoltage:
            self.nScalars = 3
        else:
            self.nScalars = 0
        self.isUpperDirichlet = True
        self.isLowerDirichlet = True
        self.params = MirrorPlasmaParams.make(config)
        self.nCells = solver_config["Grid_size"]
        self.k = solver_config["Polynomial_degree"]
        self.points = MaNTA.getNodes(
            solver_config["Lower_boundary"],
            solver_config["Upper_boundary"],
            self.nCells,
            self.k,
        )
        self.nPoints = len(self.points)
        self.runner = MaNTA.Runner(self)
        self.runner.configure(solver_config)

    def run(self, tFinal=None):
        if tFinal is not None:
            self.runner.run(tFinal)
        else:
            self.runner.run_ss()

    @partial(jax.jit, static_argnames=("self",))
    def LowerBoundary(self, index, t):
        return self.InitialValue(index, 0.0)

    @partial(jax.jit, static_argnames=("self",))
    def UpperBoundary(self, index, t):
        return self.InitialValue(index, 1.0)

    @partial(jax.jit, static_argnames=("self",))
    def InitialValue(self, index, x):
        Rmin = self.params.Config.Rmin
        Rmax = self.params.Config.Rmax
        Rmid = 0.5 * (Rmin + Rmax)
        R = self.params.MagneticField.R_x(x)
        v = jnp.cos(jnp.pi * (R - Rmid) / (Rmax - Rmin))

        def n0():
            return (
                self.params.Config.EdgeDensity
                + (
                    self.params.Config.InitialDensityHeight
                    - self.params.Config.EdgeDensity
                )
                * v**2
            )

        def ui0():
            return (
                3.0
                / 2.0
                * (
                    self.params.Config.EdgeIonTemperature
                    + (
                        self.params.Config.InitialIonTemperatureHeight
                        - self.params.Config.EdgeIonTemperature
                    )
                    * v
                )
                * n0()
            )

        def ue0():
            return (
                3.0
                / 2.0
                * (
                    self.params.Config.EdgeElectronTemperature
                    + (
                        self.params.Config.InitialElectronTemperatureHeight
                        - self.params.Config.EdgeElectronTemperature
                    )
                    * v
                )
                * n0()
            )

        def L0():
            Te = 2.0 / 3.0 * ue0() / n0()
            M0 = (
                self.params.Config.EdgeMachNumber
                + (
                    self.params.Config.InitialMachNumber
                    - self.params.Config.EdgeMachNumber
                )
                * v
            )
            omega = jnp.sqrt(Te) * M0 / R

            return omega * R * R * n0()

        return jax.lax.switch(index, [n0, L0, ui0, ue0])

    def InitialAuxValue(self, index, x):

        n0 = self.InitialValue(Channel.Density, x)
        L0 = self.InitialValue(Channel.AngularMomentum, x)
        ui0 = self.InitialValue(Channel.IonEnergy, x)
        ue0 = self.InitialValue(Channel.ElectronEnergy, x)

        dn0 = self.InitialDerivative(Channel.Density, x)
        dL0 = self.InitialDerivative(Channel.AngularMomentum, x)
        dui0 = self.InitialDerivative(Channel.IonEnergy, x)
        due0 = self.InitialDerivative(Channel.ElectronEnergy, x)
        s0 = State(
            jnp.stack([n0, L0, ui0, ue0]).transpose(),
            jnp.stack([dn0, dL0, dui0, due0]).transpose(),
            Flux_=jnp.zeros((4,)),
            Aux_=jnp.zeros((1,)),
            Scalars_=jnp.zeros((self.nScalars,)),
        )
        s1 = MirrorPlasmaState.from_state(s0, x, self.params)

        return InitialPhiValue(s1, x, 0.0, self.params)

    @MirrorPlasmaDecorator
    def sigma(self, index, state, x, t, params):
        return jax.lax.switch(
            index, [self.Gamma, self.Pi, self.qi, self.qe], state, x, t, params
        )

    @MirrorPlasmaDecorator
    def source(self, index, state, x, t, params):
        return jax.lax.switch(
            index, [self.Sn, self.Somega, self.Spi, self.Spe], state, x, t, params
        )

    @MirrorPlasmaDecorator
    def aux(self, index, state, x, t, params):
        return ParallelCurrent(state, x, t, params)

    def Gamma(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        GeometricFactor = (state.R * state.VPrime) ** 2

        Uei = (
            state.dpedx / state.pe
            + state.Ti / (state.Te * params.Constants.Z_eff)
            + (state.omega * state.R**2) / state.Ti * state.domegadx
        )

        G = (
            GeometricFactor
            * state.pe
            / params.Constants.ElectronCollisionTime(state.n, state.Te)
            * (Uei - 3.0 / 2.0 * state.dTedx / state.Te)
        )
        return (
            G
            * params.Constants.Gamma0()
            / params.Constants.DensityEquationNormalization()
        )

    def Pi(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        GeometricFactor = (state.R * state.VPrime**2) ** 2

        IonClassicalViscosity = (
            GeometricFactor
            * 3.0
            / 10.0
            * state.pi
            / params.Constants.IonCollisionTime(state.n, state.Ti)
            * state.domegadx
        )

        Pi_out = (
            params.Constants.Pi0() * IonClassicalViscosity
            + (
                params.IonSpecies.IonMass
                * params.Constants.omega0
                * params.Constants.a**2
            )
            * state.omega
            * state.R**2
            * self.Gamma(state, x, t, params)
            * params.Constants.DensityEquationNormalization()
        )
        return Pi_out / params.Constants.MomentumEquationNormalization()

    def qi(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        GeometricFactor = (state.R * state.VPrime) ** 2

        HeatFlux = (
            2
            * GeometricFactor
            * state.pi
            * state.Ti
            / params.Constants.IonCollisionTime(state.n, state.Ti)
            * state.dTidx
            / state.Ti
        )
        qi_out = (
            params.Constants.qi0() * HeatFlux
            - (params.IonSpecies.IonMass * params.Constants.omega0**2)
            * 0.5
            * (state.omega * state.R) ** 2
            * self.Gamma(state, x, t, params)
            * params.Constants.DensityEquationNormalization()
        )
        return qi_out / params.Constants.HeatEquationNormalization()

    def qe(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        GeometricFactor = (state.R * state.VPrime) ** 2
        Uei = (
            state.dpedx / state.pe
            + state.Ti / (state.Te * params.Constants.Z_eff)
            + (state.omega * state.R * state.R) / state.Ti * state.domegadx
        )

        HeatFlux = (
            GeometricFactor
            * state.pe
            * state.Te
            / params.Constants.ElectronCollisionTime(state.n, state.Te)
            * (4.66 * state.dTedx / state.Te - Uei)
        )
        return (
            params.Constants.qe0()
            / params.Constants.HeatEquationNormalization()
            * HeatFlux
        )

    def Sn(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        return self.ParticleSource(state, x, t, params) - self.ParallelParticleLosses(
            state, x, t, params
        )

    def Somega(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        return self.JxBForce(state, x, t, params) - (
            self.ParallelAngularMomentumLosses(state, x, t, params)
            + self.ChargeExchangeMomentumLosses(state, x, t, params)
        )

    def Spi(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        return (
            self.ViscousHeating(state, x, t, params)
            + self.IonPotentialHeating(state, x, t, params)
            + params.Constants.IonElectronEnergyExchange(state.n, state.pe, state.pi)
        ) - (
            self.IonParallelHeatLosses(state, x, t, params)
            + self.ChargeExchangeHeatLosses(state, x, t, params)
        )

    def Spe(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        return self.AlphaHeating(state, x, t, params) - (
            self.RadiationHeatLosses(state, x, t, params)
            + self.ElectronParallelHeatLosses(state, x, t, params)
            + params.Constants.IonElectronEnergyExchange(state.n, state.pe, state.pi)
        )

    # =======================================================================
    # Particle Sources
    # =======================================================================

    def ParticleSource(self, state, x, t, params):
        Center = params.Config.ParticleSourceCenter
        Width = params.Config.ParticleSourceWidth
        return jnp.exp(-((state.R - Center) ** 2) / Width)

    def ParallelParticleLosses(self, state, x, t, params):
        return (
            ElectronPastukhovLossRate(state, x, t, params)
            / params.Constants.DensityEquationNormalization()
        )

    # =======================================================================
    # Momentum Sources
    # =======================================================================

    def ParallelAngularMomentumLosses(self, state, x, t, params):
        return (
            state.L
            / state.n
            * IonPastukhovLossRate(state, x, t, params)
            / params.Constants.DensityEquationNormalization()
        )

    def ChargeExchangeMomentumLosses(self, state, x, t, params):
        R = params.MagneticField.R_x(x)

        def true_fun():
            return (
                state.L
                / state.n
                * params.Constants.ChargeExchangeLossRate(
                    state.n, params.Config.NeutralDensity, R * state.omega, state.Ti
                )
                / params.Constants.DensityEquationNormalization()
            )

        def false_fun():
            return 0.0

        return jax.lax.cond(params.Config.useNeutralsModel, true_fun, false_fun)

    def JxBForce(self, state, x, t, params):
        return -state.Current / params.MagneticField.VPrime(x)

    # =======================================================================
    # Heat sources
    # =======================================================================

    """
    Ion heat sources
    """

    def ViscousHeating(
        self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams
    ):
        return state.domegadx * state.Pi

    def IonPotentialHeating(
        self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams
    ):
        return (
            -0.5
            * (params.MagneticField.R_x(x) * state.omega) ** 2
            * self.Sn(state, x, t, params)
        )

    def ChargeExchangeHeatLosses(
        self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams
    ):
        def true_fun():
            return (
                state.Ti
                * params.Constants.T0
                * params.Constants.ChargeExchangeLossRate(
                    state.n,
                    params.Config.NeutralDensity,
                    state.R * state.omega,
                    state.Ti,
                )
                / params.Constants.HeatEquationNormalization()
            )

        def false_fun():
            return 0.0

        return jax.lax.cond(params.Config.useNeutralsModel, true_fun, false_fun)

    def IonParallelHeatLosses(
        self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams
    ):
        ParticleEnergy = state.Ti * (1 + Xi_i(state, x, t, params))
        return (
            ParticleEnergy
            * IonPastukhovLossRate(state, x, t, params)
            / params.Constants.DensityEquationNormalization()
        )

    """
    Electron heat sources
    """

    def RadiationHeatLosses(
        self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams
    ):
        return params.Constants.BremsstrahlungLosses(
            state.n, state.pe
        ) + params.Constants.CyclotronLosses(x, state.n, state.Te)

    def AlphaHeating(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        return (
            params.Constants.TotalAlphaPower(state.n, state.pi)
            / params.Constants.HeatEquationNormalization()
        )

    def ElectronParallelHeatLosses(
        self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams
    ):
        ParticleEnergy = state.Te * (1 + Xi_e(state, x, t, params))
        return (
            ParticleEnergy
            * ElectronPastukhovLossRate(state, x, t, params)
            / params.Constants.DensityEquationNormalization()
        )

    # =======================================================================
    #  Scalars
    # =======================================================================

    def InitialCurrent(self, t):
        return self.params.Config.Current * (
            1 + jnp.tanh(-t / self.params.Config.CurrentDecay)
        )

    def TotalCurrent(self, states: MirrorPlasmaState, integrator, t):
        Vp = jax.vmap(self.params.MagneticField.VPrime)(self.points)
        dPsi = integrator(1.0 / Vp)
        deltaPi = states.Pi[0] - states.Pi[-1]

        S = jax.vmap(
            self.Somega, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None)
        )(states, self.points, t, self.params)

        Itot = 1 / dPsi * (deltaPi - integrator(S))
        print(Itot)
        return Itot

    def InitialScalarValue(self, s):
        def omega(x):
            R = self.params.MagneticField.R_x(x)
            return self.InitialValue(Channel.AngularMomentum, x) / (
                R**2 * self.InitialValue(Channel.Density, x)
            )

        phi = quad(omega, 0.0, 1.0)[0]
        match s:
            case Scalar.Error:
                return (
                    self.params.Config.PlasmaVoltage / self.params.Constants.cs0 - phi
                )
            case Scalar.Integral:
                return 0.0
            case Scalar.Current:
                return self.InitialCurrent(0.0)

    @ScalarG_Decorator
    def InitialScalarDerivative(
        self, i, states_: State, states_dot_: State, integrator
    ):
        states = jax.vmap(
            MirrorPlasmaState.from_state, in_axes=(State.vmap_axes(), 0, None)
        )(states_, self.points, self.params)
        states_dot = jax.vmap(
            MirrorPlasmaState.from_state, in_axes=(State.vmap_axes(), 0, None)
        )(states_dot_, self.points, self.params)
        match i:
            case Scalar.Error:
                domegadt = (
                    1.0
                    / (states.R**2 * states.VPrime)
                    * (
                        states_dot.L / states.n
                        - states.L * states_dot.n / (states.n**2)
                    )
                )

                return -integrator(domegadt)
            case Scalar.Integral:
                return self.InitialScalarValue(i)

    @partial(jax.jit, static_argnames=("self",))
    def isScalarDifferential(self, s) -> bool:
        def sError():
            return True

        def sIntegral():
            return True

        def sCurrent():
            return False

        return jax.lax.switch(s, [sError, sIntegral, sCurrent])

    @ScalarG_Decorator
    @partial(jax.jit, static_argnames=("self",))
    def ScalarG(self, i, states_, states_dot_, integrator, t):
        states = jax.vmap(
            MirrorPlasmaState.from_state, in_axes=(State.vmap_axes(), 0, None)
        )(states_, self.points, self.params)
        states_dot = jax.vmap(
            MirrorPlasmaState.from_state, in_axes=(State.vmap_axes(), 0, None)
        )(states_dot_, self.points, self.params)

        states = eqx.tree_at(lambda s: s.Current, states, states.Current[0])
        states_dot = eqx.tree_at(lambda s: s.Current, states_dot, states_dot.Current[0])

        states = eqx.tree_at(lambda s: s.Scalars, states, states.Scalars[0, :])
        states_dot = eqx.tree_at(
            lambda s: s.Scalars, states_dot, states_dot.Scalars[0, :]
        )
        phi = integrator(states.omega / states.VPrime)

        tfac = jnp.tanh(t / self.params.Config.CurrentDecay)

        def sError():
            return states.Scalars[Scalar.Error] - (
                self.params.Config.PlasmaVoltage / self.params.Constants.cs0 - phi
            )

        def sIntegral():
            return states_dot.Scalars[Scalar.Integral] - states.Scalars[Scalar.Error]

        def sCurrent():
            return (
                states.Current
                - self.InitialCurrent(t)
                - tfac
                * (
                    self.TotalCurrent(states, integrator, t)
                    + self.params.Config.gamma * states.Scalars[Scalar.Error]
                    + self.params.Config.gamma_d * states_dot.Scalars[Scalar.Error]
                    + self.params.Config.gamma_h * states.Scalars[Scalar.Integral]
                )
            )

        return jax.lax.switch(i, [sError, sIntegral, sCurrent])

    @ScalarGPrime_Decorator
    @partial(jax.jit, static_argnames=("self",))
    def ScalarGPrime(self, states_, states_dot_, integrator, t):
        states = jax.vmap(
            MirrorPlasmaState.from_state, in_axes=(State.vmap_axes(), 0, None)
        )(states_, self.points, self.params)
        states_dot = jax.vmap(
            MirrorPlasmaState.from_state, in_axes=(State.vmap_axes(), 0, None)
        )(states_dot_, self.points, self.params)

        states = eqx.tree_at(lambda s: s.Current, states, states.Current[0])
        states_dot = eqx.tree_at(lambda s: s.Current, states_dot, states_dot.Current[0])

        states = eqx.tree_at(lambda s: s.Scalars, states, states.Scalars[0, :])
        states_dot = eqx.tree_at(
            lambda s: s.Scalars, states_dot, states_dot.Scalars[0, :]
        )

        tfac = jnp.tanh(t / self.params.Config.CurrentDecay)
        sArgs = (self.nVars, self.nAux, self.nScalars, len(self.points))
        sZero = State.make_zero(*sArgs)

        varshape = (len(self.points),)
        _zeros = jnp.zeros(varshape)
        """
        Error
        """

        P_L = integrator.computeCellProducts(
            1.0 / (states.VPrime * states.n * states.R**2)
        )
        P_n = integrator.computeCellProducts(
            -1.0 / (states.VPrime) * states.L / (states.n**2 * states.R**2)
        )
        _scalar = jnp.array([1.0, 0.0, 0.0])

        derivs_Error = eqx.tree_at(
            lambda s: s.Variable,
            sZero,
            jnp.stack([P_n, P_L, _zeros, _zeros]).transpose(),
        )
        derivs_Error = eqx.tree_at(lambda s: s.Scalars, derivs_Error, _scalar)

        """
        Integral
        """
        # Error, Integral, Current
        _scalar = jnp.array([-1.0, 0.0, 0.0])
        _scalar_dt = jnp.array([0.0, 1.0, 0.0])

        derivs_Integral = eqx.tree_at(lambda s: s.Scalars, sZero, _scalar)
        derivs_dt_Integral = eqx.tree_at(lambda s: s.Scalars, sZero, _scalar_dt)

        """
        Current
        """

        dPsi = integrator(1.0 / states.VPrime)
        dSource = self.dSources(Channel.AngularMomentum, states_, self.points, t)
        _dvariable = (
            tfac
            * jax.vmap(integrator.computeCellProducts, in_axes=1)(dSource.Variable)
            / dPsi
        )
        _daux = (
            tfac
            * jax.vmap(integrator.computeCellProducts, in_axes=1)(dSource.Aux)
            / dPsi
        )

        _scalar = tfac * jnp.array(
            [-self.params.Config.gamma, -self.params.Config.gamma_h, 1.0]
        )
        _scalar_dt = tfac * jnp.array([-self.params.Config.gamma_d, 0.0, 0.0])

        _flux_L = 1.0 / dPsi * integrator.phiL()
        _flux_R = -1.0 / dPsi * integrator.phiR()
        _flux = jnp.concatenate(
            [_flux_L, jnp.zeros(((self.nCells - 2) * (self.k + 1),)), _flux_R]
        )
        _flux = jnp.stack([_zeros, _flux, _zeros, _zeros]).transpose()

        derivs_Current = eqx.tree_at(
            lambda s: s.Variable, sZero, _dvariable.transpose()
        )
        derivs_Current = eqx.tree_at(lambda s: s.Aux, sZero, _daux.transpose())
        derivs_Current = eqx.tree_at(
            lambda s: s.Flux, derivs_Current, derivs_Current.Flux + _flux
        )
        derivs_Current = eqx.tree_at(lambda s: s.Scalars, derivs_Current, _scalar)

        derivs_dt_Current = eqx.tree_at(lambda s: s.Scalars, sZero, _scalar_dt)

        return [
            [derivs_Error, derivs_Integral, derivs_Current],
            [sZero, derivs_dt_Integral, derivs_dt_Current],
        ]

    @partial(jax.jit, static_argnames=("self",))
    def dSources_dScalars(self, i, _state, x, t):
        state = State.from_manta(_state)
        return jax.grad(self.source, argnums=1)(i, state, x, t, self.params).Scalars
