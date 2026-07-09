import equinox as eqx
from MirrorPlasmaState import (
    MirrorPlasmaDecorator,
    MirrorPlasmaParams,
    MirrorPlasmaState,
)
import sys

sys.path.append("..")
from VectorizedTransportSystem import VectorizedTransportSystem


class MirrorPlasma(VectorizedTransportSystem):
    def __init__(self):
        pass

    @MirrorPlasmaDecorator
    def Gamma(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass

    @MirrorPlasmaDecorator
    def Pi(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass

    @MirrorPlasmaDecorator
    def qi(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass

    @MirrorPlasmaDecorator
    def qe(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass

    @MirrorPlasmaDecorator
    def Sn(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass

    @MirrorPlasmaDecorator
    def Somega(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass

    @MirrorPlasmaDecorator
    def Spi(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass

    @MirrorPlasmaDecorator
    def Spe(self, state: MirrorPlasmaState, x, t, params: MirrorPlasmaParams):
        pass
