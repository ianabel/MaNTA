import equinox as eqx
from constants import PlasmaConstants
import ion_species


class StellaratorConfig(eqx.Module):
    ParticleSourceCenter: float
    ParticleSourceWidth: float
    ParticleSourceHeight: float
    NBICenter: float
    NBIPower: float
    NBIWidth: float
    ECHCenter: float
    ECHPower: float
    ECHWidth: float
    EdgeTemperature: float
    EdgeDensity: float
    ADCoefficient: float
    rhoUpper: float
    n0: float
    T0: float
    alpha: float
    FusionFactor: float
    evolveDensity: bool
    useSharding: bool
    useBatching: bool
    useITGProxy: bool
    ITGScale: float
    batch_size: int
    nchunks: int
    adaptiveGrid: bool

    def __init__(
        self,
        EdgeTemperature,
        EdgeDensity,
        NBICenter,
        NBIPower,
        NBIWidth,
        ECHCenter=0.2,
        ECHPower=0.0,
        ECHWidth=0.05,
        ParticleSourceCenter=0.0,
        ParticleSourceWidth=0.1,
        ParticleSourceHeight=1.0,
        ADCoefficient=0.0,
        rhoUpper=1.0,
        T0=None,
        n0=None,
        alpha=2.0,
        FusionFactor=1.0,
        evolveDensity=False,
        useSharding=True,
        useBatching=False,
        useITGProxy=False,
        ITGScale=1.0,
        batch_size=16,
        nchunks=2,
        adaptiveGrid=False,
    ):
        self.ParticleSourceCenter = ParticleSourceCenter
        self.ParticleSourceWidth = ParticleSourceWidth
        self.ParticleSourceHeight = ParticleSourceHeight
        self.NBICenter = NBICenter
        self.NBIPower = NBIPower
        self.NBIWidth = NBIWidth
        self.ECHCenter = ECHCenter
        self.ECHPower = ECHPower
        self.ECHWidth = ECHWidth
        self.EdgeTemperature = EdgeTemperature
        self.EdgeDensity = EdgeDensity
        self.ADCoefficient = ADCoefficient
        self.rhoUpper = rhoUpper
        if T0 is None:
            self.T0 = self.EdgeTemperature
        else:
            self.T0 = T0

        if n0 is None:
            self.n0 = self.EdgeDensity
        else:
            self.n0 = n0
        self.alpha = alpha
        self.FusionFactor = FusionFactor
        self.evolveDensity = evolveDensity
        self.useSharding = useSharding
        self.useBatching = useBatching
        self.useITGProxy = useITGProxy
        # multiplies the NN ITG heat flux; 1 is the network's prediction
        self.ITGScale = ITGScale
        self.batch_size = batch_size
        self.nchunks = nchunks
        self.adaptiveGrid = adaptiveGrid


class StellaratorParams(eqx.Module):
    config: StellaratorConfig
    constants: PlasmaConstants

    def __init__(
        self, _config, ion_species=ion_species.DeuteriumTritium(), **constant_args
    ):
        self.config = _config
        self.constants = PlasmaConstants(ion_species, **constant_args)


# The reference values PlasmaConstants normalises to, under the names a
# config's "Normalization" table uses: density [m^-3], temperature [eV], length
# [m] and field [T]. The last two enter only the time unit, NormalizingTime.
DEFAULT_NORMALIZATION = {"n0": 1e20, "T0": 1000.0, "a": 1.0, "B0": 5.0}


def field_norm_from_eq(eq):
    """Length and field normalisations from a DESC equilibrium: its minor
    radius "a" and its volume-rms field "<|B|>_rms", as "a" and "B0".

    Merge the result into a "Normalization" table. For an optimisation, take it
    once from the initial equilibrium and keep it fixed: a normalisation that
    followed the geometry would move the physical sources with it, through a
    path the adjoint gradient does not carry.
    """
    data = eq.compute(["a", "<|B|>_rms"])
    return {"a": float(data["a"]), "B0": float(data["<|B|>_rms"])}


def normalization_args(normalization):
    """PlasmaConstants keyword arguments from a "Normalization" table."""
    unknown = set(normalization) - set(DEFAULT_NORMALIZATION)
    if unknown:
        raise ValueError(
            f"unknown Normalization keys {sorted(unknown)}; "
            f"expected some of {sorted(DEFAULT_NORMALIZATION)}"
        )
    values = {**DEFAULT_NORMALIZATION, **normalization}
    return {"_" + k: v for k, v in values.items()}


def plasma_constants(normalization=None):
    return PlasmaConstants(
        ion_species.DeuteriumTritium(), **normalization_args(normalization or {})
    )


# What each normalised entry of the "Stellarator" table is measured in. A rate
# is per normalising time, so ADCoefficient (a diffusivity in rho) is one.
STELLARATOR_UNITS = {
    "EdgeDensity": "density",
    "n0": "density",
    "EdgeTemperature": "temperature",
    "T0": "temperature",
    "ParticleSourceHeight": "particle_rate",
    "NBIPower": "heat_rate",
    "ECHPower": "heat_rate",
    "ADCoefficient": "rate",
}

# The same for the "Solver" table. tau is added to a flux per unit of the
# variable, so it is a rate as well; Absolute_tolerance mixes the units of every
# block and is left alone.
SOLVER_UNITS = {
    "tau": "rate",
    "t_final": "time",
    "delta_t": "time",
    "initialTimestep": "time",
    "MinStepSize": "time",
}


def reference_values(constants: PlasmaConstants):
    """The physical value of one normalised unit of each kind."""
    t0 = constants.NormalizingTime()
    return {
        "density": constants.n0,
        "temperature": constants.T0,
        "particle_rate": constants.DensityEquationNormalization(),
        "heat_rate": constants.HeatEquationNormalization(),
        "rate": 1.0 / t0,
        "time": t0,
    }


def _rescale(table, units, old, new):
    ref_old, ref_new = reference_values(old), reference_values(new)
    out = dict(table)
    for key, kind in units.items():
        if key in out:
            out[key] = float(out[key] * ref_old[kind] / ref_new[kind])
    return out


def rescale_config(config, new_normalization):
    """The same physical problem, expressed in another normalisation.

    `config` is a full {"Stellarator", "Solver", "Normalization"} config, and
    the result is a copy with "Normalization" replaced and every normalised
    entry converted to it, so the two describe identical physics and their
    solutions should agree once unnormalised.
    """
    old = plasma_constants(config.get("Normalization", {}))
    new = plasma_constants(new_normalization)
    return {
        **config,
        "Stellarator": _rescale(config["Stellarator"], STELLARATOR_UNITS, old, new),
        "Solver": _rescale(config["Solver"], SOLVER_UNITS, old, new),
        "Normalization": dict(new_normalization),
    }
