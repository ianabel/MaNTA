"""Pitch-angle grids for the yancc solves: fixed, or adapted to each solve.

A uniform grid in pitch angle resolves the trapped-passing boundary layer only
down to a collisionality set by nalpha, and the layer that carries the 1/nu
radial flux narrows as the suprathermal particles that carry it become less
collisional. yancc's DKEGaussianPitchAngleGrid clusters the nodes in that layer,
more tightly at lower collisionality, from the species, field and Er of the
solve -- so it is rebuilt for every solve, at every node (about 1 ms warm, and
no recompilation: only the node positions change, never a shape).

`AdaptivePitchGrid` stands where a yancc pitch grid goes -- yancc_data holds it,
and every function that passes a pitch grid on passes it unchanged -- and
`resolve_pitchgrid` turns it into the real grid at the solve, where the plasma
it depends on is known.

The grid is built from stop_gradient'd inputs, so derivatives are taken at
fixed node placement. That drops d(flux)/d(nodes) d(nodes)/d(state), which is
the sensitivity of the discretisation error to where the nodes sit: negligible
where the grid resolves the flux, and a resolution artefact where it does not.
It is also, at present, the only choice that works: reverse mode through the
grid's Newton-Cotes weights is NaN on yancc's nonuniform branch (a
where-division at velocity_grids.py:669), and both the transport Jacobian and
the optimiser's field Jacobian are jacrev's. The steady state is unaffected; a
finite-difference check of a Jacobian will differ from it by the dropped term.

Needs yancc's nonuniform branch until it merges.
"""

import equinox as eqx
import jax
import jax.numpy as jnp


class AdaptivePitchGrid(eqx.Module):
    """A pitch grid to be built per solve by `resolve_pitchgrid`.

    nalpha : number of pitch-angle nodes, as for a uniform grid.
    floor : yancc's minimum relative node density away from the clusters.
    """

    nalpha: int = eqx.field(static=True)
    floor: float = eqx.field(static=True, default=0.2)


def _constant(tree):
    """`tree` with no derivative flowing through its float arrays."""
    dynamic, static = eqx.partition(tree, eqx.is_inexact_array)
    return eqx.combine(jax.lax.stop_gradient(dynamic), static)


def resolve_pitchgrid(pitchgrid, field, species, Erho):
    """The pitch grid for one solve: `pitchgrid` itself if it is a yancc grid,
    or, for an AdaptivePitchGrid, yancc's DKEGaussianPitchAngleGrid for these
    species, field and Er, built at fixed (non-differentiated) inputs."""
    if not isinstance(pitchgrid, AdaptivePitchGrid):
        return pitchgrid
    try:
        from yancc.velocity_grids import DKEGaussianPitchAngleGrid
    except ImportError as e:
        raise ImportError(
            "pitch_grid='adaptive' needs DKEGaussianPitchAngleGrid, from yancc's "
            "nonuniform branch"
        ) from e
    field, species = _constant(field), _constant(species)
    # Er arrives as the one-entry aux vector; the grid wants a scalar
    Erho = jax.lax.stop_gradient(jnp.reshape(jnp.asarray(Erho), ()))
    return DKEGaussianPitchAngleGrid(
        pitchgrid.nalpha, field, species, Erho, floor=pitchgrid.floor
    )
