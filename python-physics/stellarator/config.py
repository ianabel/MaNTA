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
