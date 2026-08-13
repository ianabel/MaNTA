import jax
import jax.numpy as jnp
import equinox as eqx
from .config import MirrorPlasmaConfig
from .plasma_state import (
    MirrorPlasmaDecorator,
    MirrorPlasmaParams,
    MirrorPlasmaState,
    Channel,
    Scalar,
)
from .parallel_physics import (
    InitialPhiValue,
    ParallelCurrent,
    ElectronPastukhovLossRate,
    IonPastukhovLossRate,
    Xi_i,
    Xi_e,
)

from functools import partial

from manta.jax import (
    InitialScalarDerivative_Decorator,
    ScalarG_Decorator,
    ScalarGPrime_Decorator,
    State,
    VectorizedTransportSystem,
)
import manta as MaNTA


def buildSpec(config: MirrorPlasmaConfig) -> MaNTA.SystemSpec:
    """The four channels, the ambipolar potential, and the voltage controller.

    Whether the controller's three scalars exist depends on the config, so the
    spec has to be built before the base class does -- `nVars`, `nAux` and
    `nScalars` are read-only, derived from what is passed to `__init__`, and
    cannot be assigned in the constructor body the way they were.

    The variable order is `Channel`'s, which is *not* the C++ MirrorPlasma's:
    this one is (Density, AngularMomentum, IonEnergy, ElectronEnergy), that one
    (Density, IonEnergy, ElectronEnergy, AngularMomentum). The two cases are
    independent implementations and no index is shared between them.
    """
    variables = [
        # Density is Neumann at both ends; the other three are Dirichlet.
        MaNTA.Field(
            "Density",
            "particle density",
            "n0",
            lower=MaNTA.Neumann,
            upper=MaNTA.Neumann,
        ),
        MaNTA.Field("AngularMomentum", "angular momentum density", "n0 T0 / c_s0"),
        MaNTA.Field("IonEnergy", "ion energy density", "n0 T0"),
        MaNTA.Field("ElectronEnergy", "electron energy density", "n0 T0"),
    ]

    aux = [
        MaNTA.Aux(
            "AmbipolarPhi",
            "electrostatic potential enforcing zero parallel current",
            "T0/e",
        )
    ]

    # Scalar's order. Error and Integral are differential -- G depends on their
    # time derivatives -- and Current is algebraic, which is what
    # isScalarDifferential used to report.
    scalars = (
        [
            MaNTA.Scalar("VoltageError", "V0 minus the achieved voltage", "", True),
            MaNTA.Scalar(
                "VoltageErrorIntegral", "time integral of the error", "", True
            ),
            MaNTA.Scalar("RadialCurrent", "radial current", "I0", False),
        ]
        if config.useConstantVoltage
        else []
    )

    return MaNTA.SystemSpec(variables=variables, scalars=scalars, aux=aux)


class MirrorPlasma(VectorizedTransportSystem):
    def __init__(self, config: MirrorPlasmaConfig, solver_config):
        spec = buildSpec(config)
        super().__init__(spec)

        # The same booleans the spec carries, as a jnp array. LowerBoundary and
        # UpperBoundary are jitted, so `index` reaches them as a tracer, and the
        # base class's isLowerBoundaryDirichlet is a bound C++ method that wants
        # a concrete int. Derived from the spec rather than written out again,
        # so there is still one source of truth.
        self.lower_bcs = jnp.array(
            [f.lower == MaNTA.Dirichlet for f in spec.variables]
        )
        self.upper_bcs = jnp.array(
            [f.upper == MaNTA.Dirichlet for f in spec.variables]
        )

        self.params = MirrorPlasmaParams.make(config)
        self.nCells = solver_config["Grid_size"]
        self.k = solver_config["Polynomial_degree"]
        if "Grid_points" in solver_config:
            self.points = MaNTA.getNodes(solver_config["Grid_points"], self.k)
        else:
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
        def dirichlet(index):
            return self.InitialValue(index, 0.0)

        def neumann(index):
            return self.InitialDerivative(index, 0.0)

        return jax.lax.cond(self.lower_bcs[index], dirichlet, neumann, index)

    @partial(jax.jit, static_argnames=("self",))
    def UpperBoundary(self, index, t):
        def dirichlet(index):
            return self.InitialValue(index, 1.0)

        def neumann(index):
            return self.InitialDerivative(index, 1.0)

        return jax.lax.cond(self.upper_bcs[index], dirichlet, neumann, index)

    @partial(jax.jit, static_argnames=("self",))
    def InitialValue(self, index, x):
        a = self.params.Constants.a
        Rmin = self.params.Config.Rmin / a
        Rmax = self.params.Config.Rmax / a
        Rmid = 0.5 * (Rmin + Rmax)
        R = self.params.MagneticField.R_x(x) / a
        v = jnp.cos(jnp.pi * (R - Rmid) / (Rmax - Rmin))

        def n0():
            return (
                self.params.Config.EdgeDensity
                + (
                    self.params.Config.InitialDensityHeight
                    - self.params.Config.EdgeDensity
                )
                * v
                * v
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

    # ======================================================================= #
    # Fluxes                                                                  #
    # ======================================================================= #

    def Gamma(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        GeometricFactor = state.R**2 * state.VPrime
        Uei = (
            state.dpedpsi / state.pe
            + state.Ti / (state.Te * params.Constants.Z_eff) * state.dpidpsi / state.pi
            + (state.omega * state.R**2) / state.Te * state.domegadpsi
        )
        D = (
            GeometricFactor
            * state.pe
            / params.Constants.ElectronCollisionTime(state.n, state.Te)
        )

        G = D * (
            Uei - 3.0 / 2.0 * state.dTedpsi / state.Te
        ) + GeometricFactor * params.Config.ADCoefficient * state.dndpsi * (
            jnp.exp(-t / params.Config.ADDecayRates[Channel.Density])
            + params.Config.ADFinalCoeffs[Channel.Density]
        )

        return (
            G
            * params.Constants.Gamma0()
            / params.Constants.DensityEquationNormalization()
        )

    def Pi(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        GeometricFactor = state.R**4 * state.VPrime

        IonClassicalViscosity = (
            GeometricFactor
            * 3.0
            / 10.0
            * state.pi
            / params.Constants.IonCollisionTime(state.n, state.Ti)
            * state.domegadpsi
        ) + GeometricFactor * params.Config.ADCoefficient * state.domegadpsi * (
            jnp.exp(-t / params.Config.ADDecayRates[Channel.AngularMomentum])
            + params.Config.ADFinalCoeffs[Channel.AngularMomentum]
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
        GeometricFactor = state.R**2 * state.VPrime

        HeatFlux = (
            2
            * GeometricFactor
            * state.pi
            * state.Ti
            / params.Constants.IonCollisionTime(state.n, state.Ti)
            * state.dTidpsi
            / state.Ti
        ) + GeometricFactor * params.Config.ADCoefficient * state.dTidpsi * (
            jnp.exp(-t / params.Config.ADDecayRates[Channel.IonEnergy])
            + params.Config.ADFinalCoeffs[Channel.IonEnergy]
        )

        qi_out = (
            params.Constants.qi0() * HeatFlux
            - (
                params.IonSpecies.IonMass
                * params.Constants.a**2
                * params.Constants.omega0**2
            )
            * 0.5
            * (state.omega * state.R) ** 2
            * self.Gamma(state, x, t, params)
            * params.Constants.DensityEquationNormalization()
        )
        return qi_out / params.Constants.HeatEquationNormalization()

    def qe(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        GeometricFactor = state.R**2 * state.VPrime

        Uei = (
            state.dpedpsi / state.pe
            + state.Ti / (state.Te * params.Constants.Z_eff) * state.dpidpsi / state.pi
            + (state.omega * state.R**2) / state.Te * state.domegadpsi
        )

        HeatFlux = (
            GeometricFactor
            * state.pe
            * state.Te
            / params.Constants.ElectronCollisionTime(state.n, state.Te)
            * (4.66 * state.dTedpsi / state.Te - 3.0 / 2.0 * Uei)
        ) + GeometricFactor * params.Config.ADCoefficient * state.dTedpsi * (
            jnp.exp(-t / params.Config.ADDecayRates[Channel.ElectronEnergy])
            + params.Config.ADFinalCoeffs[Channel.ElectronEnergy]
        )

        return (
            params.Constants.qe0()
            / params.Constants.HeatEquationNormalization()
            * HeatFlux
        )

    # ======================================================================= #
    # Sources                                                                 #
    # ======================================================================= #

    def Sn(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        return (
            self.ParticleSource(state, x, t, params)
            + self.IonizationSource(state, x, t, params)
            - self.ParallelParticleLosses(state, x, t, params)
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
            + self.UniformHeatSource(state, x, t, params)
        ) - (
            self.IonParallelHeatLosses(state, x, t, params)
            + self.ChargeExchangeHeatLosses(state, x, t, params)
        )

    def Spe(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        return (
            self.AlphaHeating(state, x, t, params)
            + self.UniformHeatSource(state, x, t, params)
            - (
                self.RadiationHeatLosses(state, x, t, params)
                + self.ElectronParallelHeatLosses(state, x, t, params)
                + params.Constants.IonElectronEnergyExchange(
                    state.n, state.pe, state.pi
                )
            )
        )

    # ======================================================================= #
    # Particle Sources                                                        #
    # ======================================================================= #

    def ParticleSource(self, state, x, t, params):
        Center = params.Config.ParticleSourceCenter / params.Constants.a
        Width = params.Config.ParticleSourceWidth / params.Constants.a
        Height = params.Config.ParticleSourceHeight * params.Constants.a**2
        return (
            Height * jnp.exp(-(((state.R - Center) / Width) ** 2)) * jnp.exp(-t / 0.01)
        )

    def IonizationSource(self, state, x, t, params):
        return (
            params.Constants.IonizationRate(
                state.n,
                params.Config.NeutralDensity,
                state.R * params.Constants.a * state.omega * params.Constants.omega0,
                state.Te,
                state.Ti,
            )
            / params.Constants.DensityEquationNormalization()
        )

    def ParallelParticleLosses(self, state, x, t, params):
        return (
            ElectronPastukhovLossRate(state, x, t, params)
            / params.Constants.DensityEquationNormalization()
        )

    # ======================================================================= #
    # Momentum Sources                                                        #
    # ======================================================================= #

    def ParallelAngularMomentumLosses(self, state, x, t, params):
        return (
            state.omega
            * state.R**2
            * IonPastukhovLossRate(state, x, t, params)
            * (
                params.Constants.IonSpecies.IonMass
                * params.Constants.omega0
                * params.Constants.a**2
            )
            / params.Constants.MomentumEquationNormalization()
        )

    def ChargeExchangeMomentumLosses(self, state, x, t, params):

        def true_fun():
            return (
                state.omega
                * state.R**2
                * params.Constants.ChargeExchangeLossRate(
                    state.n,
                    params.Config.NeutralDensity,
                    state.R
                    * params.Constants.a
                    * state.omega
                    * params.Constants.omega0,
                    state.Ti,
                )
                * (
                    params.Constants.IonSpecies.IonMass
                    * params.Constants.omega0
                    * params.Constants.a**2
                )
                / params.Constants.MomentumEquationNormalization()
            )

        def false_fun():
            return 0.0

        return jax.lax.cond(params.Config.useNeutralsModel, true_fun, false_fun)

    def JxBForce(self, state, x, t, params):
        return (
            state.Current * params.Constants.I0() / state.VPrime
        ) / params.Constants.MomentumEquationNormalization()

    # ======================================================================= #
    # Heat sources                                                            #
    # ======================================================================= #

    # Decaying uniform source to help solution along
    def UniformHeatSource(
        self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams
    ):
        return params.Constants.a**2 * 200.0 * jnp.exp(-t / 0.01)

    """
    Ion heat sources
    """

    def ViscousHeating(
        self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams
    ):
        return (
            -1
            * (state.domegadpsi * state.Pi / state.VPrime)
            * (
                params.Constants.omega0
                * params.Constants.MomentumEquationNormalization()
            )
            / params.Constants.HeatEquationNormalization()
        )

    def IonPotentialHeating(
        self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams
    ):
        return (
            -0.5
            * params.Constants.IonSpecies.IonMass
            * (params.Constants.a * params.Constants.omega0) ** 2
            * (state.R * state.omega) ** 2
            * self.Sn(state, x, t, params)
            * params.Constants.DensityEquationNormalization()
        ) / params.Constants.HeatEquationNormalization()

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
                    state.R
                    * params.Constants.a
                    * state.omega
                    * params.Constants.omega0,
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
            * params.Constants.T0
            * IonPastukhovLossRate(state, x, t, params)
            / params.Constants.HeatEquationNormalization()
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
        return params.Constants.TotalAlphaPower(state.n, state.pi)

    def ElectronParallelHeatLosses(
        self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams
    ):
        ParticleEnergy = state.Te * (1 + Xi_e(state, x, t, params))
        return (
            ParticleEnergy
            * params.Constants.T0
            * ElectronPastukhovLossRate(state, x, t, params)
            / params.Constants.HeatEquationNormalization()
        )

    # ======================================================================= #
    # Scalars                                                                 #
    # ======================================================================= #

    def InitialCurrent(self, t):
        return (
            self.params.Config.Current
            / self.params.Constants.I0()
            * (1 + jnp.tanh(-t / self.params.Config.CurrentDecay))
        )

    def TotalCurrent(self, states: MirrorPlasmaState, integrator, t):
        Vp = jax.vmap(self.params.MagneticField.VPrime)(self.points)
        dPsi = integrator(1.0 / Vp)
        deltaPi = states.Pi[-1] - states.Pi[0]

        sin = eqx.tree_at(lambda s: s.Scalars, states, jnp.zeros(states.Scalars.shape))

        S = jax.vmap(
            self.Somega, in_axes=(MirrorPlasmaState.vmap_axes(), 0, None, None)
        )(sin, self.points, t, self.params)

        Itot = 1 / dPsi * (deltaPi - integrator(S))
        return Itot

    def InitialScalarValue(self, s):
        def omega(x):
            R = self.params.MagneticField.R_x(x) / self.params.Constants.a
            VPrime = self.params.Constants.MagneticField.VPrime(x)
            L = self.InitialValue(Channel.AngularMomentum, x)
            n = self.InitialValue(Channel.Density, x)
            return L / (n * R**2 * VPrime)

        # phi = quad(omega, 0.0, 1.0)[0]
        integrand = jax.vmap(omega)(self.points)
        phi = jax.scipy.integrate.trapezoid(integrand, self.points)

        # jax.debug.print(
        #     "V0: {val}",
        #     val=self.params.Config.PlasmaVoltage / self.params.Constants.omega0,
        # )

        match s:
            case Scalar.Error:
                return (
                    self.params.Config.PlasmaVoltage / self.params.Constants.omega0
                    - phi
                )
            case Scalar.Integral:
                return 0.0
            case Scalar.Current:
                return self.InitialCurrent(0.0)

    @InitialScalarDerivative_Decorator
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
                return self.InitialScalarValue(Scalar.Error)

    # isScalarDifferential is gone: `differential` is a field of MaNTA.Scalar
    # now, and buildSpec above sets it.

    @ScalarG_Decorator
    def ScalarG(self, i, states_, states_dot_, integrator, t):
        return self._ScalarG(i, states_, states_dot_, integrator, t)

    @partial(jax.jit, static_argnames=("self",))
    def _ScalarG(self, i, states_, states_dot_, integrator, t):
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
                self.params.Config.PlasmaVoltage / self.params.Constants.omega0 - phi
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

        derivs_Error = jax.grad(self._ScalarG, argnums=1)(
            Scalar.Error, states_, states_dot_, integrator, t
        )
        """
        Integral
        """

        derivs_Integral = jax.grad(self._ScalarG, argnums=1)(
            Scalar.Integral, states_, states_dot_, integrator, t
        )

        derivs_dt_Integral = jax.grad(self._ScalarG, argnums=2)(
            Scalar.Integral, states_, states_dot_, integrator, t
        )
        """
        Current
        """
        sin = eqx.tree_at(
            lambda s: s.Scalars, states_, jnp.zeros(states_.Scalars.shape)
        )
        derivs_Current = jax.grad(self._ScalarG, argnums=1)(
            Scalar.Current, sin, states_dot_, integrator, t
        )

        derivs_dt_Current = jax.grad(self._ScalarG, argnums=2)(
            Scalar.Current, sin, states_dot_, integrator, t
        )

        dPsi = integrator(1.0 / states.VPrime)
        _flux_L = -1.0 / dPsi * integrator.phiL() * tfac
        _flux_R = 1.0 / dPsi * integrator.phiR() * tfac
        _flux = jnp.concatenate(
            [_flux_L, jnp.zeros(((self.nCells - 2) * (self.k + 1),)), _flux_R]
        )
        _flux = jnp.stack([_zeros, _flux, _zeros, _zeros]).transpose()
        # set fluxes
        derivs_Current = eqx.tree_at(lambda s: s.Flux, derivs_Current, _flux)

        return [
            [derivs_Error, derivs_Integral, derivs_Current],
            [sZero, derivs_dt_Integral, derivs_dt_Current],
        ]

    # The conversion happens outside the jit, the way ScalarG/_ScalarG above are
    # split. A pointwise hook is handed a manta.State -- a non-owning view, not
    # a pytree of arrays -- so tracing through State.from_manta is what jax
    # rejects with "Error interpreting argument ... as an abstract array".
    def dSources_dScalars(self, i, _state, x, t):
        return self._dSources_dScalars(i, State.from_manta(_state), x, t)

    @partial(jax.jit, static_argnames=("self",))
    def _dSources_dScalars(self, i, state, x, t):
        return jax.grad(self.source, argnums=1)(i, state, x, t, self.params).Scalars
