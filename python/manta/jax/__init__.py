"""JAX physics cases for MaNTA.

A case written against this layer supplies its flux and sources as JAX
functions and has every derivative hook supplied by ``jax.grad`` rather than
written by hand:

    import manta
    from manta.jax import JAXTransportSystem

    class MyCase(JAXTransportSystem):
        ...

This subpackage is the only part of ``manta`` that needs JAX, which is why it
is an optional extra rather than a dependency of the package. ``import manta``
stays numpy-only, and nothing in ``manta/__init__.py`` may import this module
-- doing so would make JAX mandatory for every user of the solver and would
turn the relative imports below into a cycle.

Worked examples are under ``python-examples/``: ``jax-diffusion``,
``jax-linear-diffusion`` and ``jax-nonlinear-adjoint``.
"""

try:
    import equinox as _equinox  # noqa: F401
    import jax as _jax  # noqa: F401
except ImportError as _e:  # pragma: no cover -- depends on the environment
    raise ImportError(
        "manta.jax needs the JAX extra: pip install manta[jax]"
    ) from _e

from .state import (  # noqa: F401
    MaNTA_Decorator,
    Physics_Decorator,
    ScalarG_Decorator,
    ScalarGPrime_Decorator,
    State,
)
from .integrator import Integrator  # noqa: F401
from .adjoint_problem import JAXAdjointProblem  # noqa: F401
from .transport_system import JAXTransportSystem  # noqa: F401
from .vectorized import VectorizedTransportSystem  # noqa: F401

__all__ = [
    "FFIRunner",
    "Integrator",
    "JAXAdjointProblem",
    "JAXTransportSystem",
    "MaNTA_Decorator",
    "Physics_Decorator",
    "Platform",
    "ScalarG_Decorator",
    "ScalarGPrime_Decorator",
    "State",
    "VectorizedTransportSystem",
    "register_ffi_cpu",
    "register_ffi_gpu",
]

# ffi_runner is loaded on demand, not with the rest. It registers XLA FFI
# targets at module scope and raises RuntimeError when it cannot find them, and
# the bindings it looks for -- runner_ffi_ops, runner_ffi_ops_cuda -- exist only
# in an XLA_FFI build (Python.cpp:361). Imported eagerly it would take
# `from manta.jax import State` down with it on every default build.
_LAZY = {
    "FFIRunner": "ffi_runner",
    "Platform": "ffi_runner",
    "register_ffi_cpu": "ffi_runner",
    "register_ffi_gpu": "ffi_runner",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
