import equinox as eqx
from constants import PlasmaConstants
from yancc.species import Hydrogen


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
    n0: float
    evolveDensity: bool
    useSharding: bool
    useBatching: bool

    def __init__(
        self,
        EdgeTemperature,
        EdgeDensity,
        n0,
        NBICenter,
        NBIPower,
        NBIWidth,
        ECHCenter=0.2,
        ECHPower=0.0,
        ECHWidth=0.05,
        ParticleSourceCenter=0.0,
        ParticleSourceWidth=0.1,
        ParticleSourceHeight=1.0,
        evolveDensity=False,
        useSharding=True,
        useBatching=False,
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
        self.n0 = n0
        self.evolveDensity = evolveDensity
        self.useSharding = useSharding
        self.useBatching = useBatching


class StellaratorParams(eqx.Module):
    config: StellaratorConfig
    constants: PlasmaConstants

    def __init__(self, _config, ion_species=Hydrogen, **constant_args):
        self.config = _config
        self.constants = PlasmaConstants(ion_species, **constant_args)
