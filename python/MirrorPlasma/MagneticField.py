import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, ArrayLike
from abc import abstractmethod
from typing import override


class _MagneticField(eqx.Module):
    @abstractmethod
    def Psi_x(self, x):
        raise NotImplementedError("Psi_x not implemented")

    @abstractmethod
    def B(self, x, s=0):
        raise NotImplementedError("B not implemented")

    @abstractmethod
    def R_x(self, x, s=0):
        raise NotImplementedError("R_x not implemented")

    @abstractmethod
    def VPrime(self, x):
        return 1.0 / jax.grad(self.Psi_x)(x)

    @abstractmethod
    def MirrorRatio(self, x, s):
        raise NotImplementedError("MirrorRatio not implemented")


class StraightMagneticField(_MagneticField):
    L_z: Float
    B_z: Float
    Rm: Float
    Vmin: Float
    Vmax: Float
    dV: Float
    m: Float

    def __init__(self, _L_z=0.6, _B_z=0.3, _Rm=10.0, _Vmin=0.0, _Vmax=1.0, _m=0.0):
        self.L_z = _L_z
        self.B_z = _B_z
        self.Rm = _Rm
        self.Vmin = _Vmin
        self.Vmax = _Vmax
        self.dV = _Vmax - _Vmin
        self.m = _m

    @override
    def Psi_x(self, x):
        return self.B * self.V_x(x) / (2 * jnp.pi * self.L_z)

    @override
    def B(self, x, s=0):
        return self.B_z - self.m * self.dV * x

    @override
    def R_x(self, x, s=0):
        return jnp.sqrt(self.V_x(x) / (jnp.pi * self.L_z))

    @override
    def MirrorRatio(self, x, s=0):
        return self.Rm * self.B_z * self.B(x)

    def V_x(self, x):
        return self.Vmin + self.dV * x
