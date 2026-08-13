"""A centrifugal mirror plasma, written against ``manta.jax``.

Four transport channels, an auxiliary variable for the ambipolar potential and
-- when the config asks for it -- three scalars implementing a voltage
controller. ``configs`` holds the machine parameters the case has been run
with.

    from mirror_plasma import MirrorPlasma, configs

    MirrorPlasma(configs.CMFX, solver_config).run(2.0)

Importing this package pulls in JAX, equinox, optimistix and matplotlib; see
the README for what to install. ``constants`` also needs ``land.pkl``, which
``landremann.py`` generates.
"""

from . import configs  # noqa: F401
from .config import MirrorPlasmaConfig  # noqa: F401
from .mirror_plasma import MirrorPlasma, buildSpec  # noqa: F401
from .plasma_state import Channel, MirrorPlasmaState, Scalar  # noqa: F401

__all__ = [
    "Channel",
    "MirrorPlasma",
    "MirrorPlasmaConfig",
    "MirrorPlasmaState",
    "Scalar",
    "buildSpec",
    "configs",
]
