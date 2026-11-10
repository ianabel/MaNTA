"""Bringing results back to the host when the device mesh spans processes.

Under ``jax.distributed`` every process runs the whole program -- MaNTA's
solver included -- and a jitted call over a mesh of several processes is a
collective that each computes its own shards of. Two directions of transfer
follow, and only one of them needs anything:

* host to device works for a numpy array or an *uncommitted* process-local
  ``jax.Array``: ``jax.device_put`` onto a sharding that spans processes keeps
  the local shards, provided every process passes the same data. MaNTA hands
  every process the same state, so that holds. A *committed* local array -- one
  ``device_put`` to a device, or computed from one -- is refused ("device_put's second argument must be
  a Device or a Sharding which represents addressable devices"), and
  `put_global` exists for that case.
* device to host does not. ``np.asarray`` of an array with shards on another
  process raises ("spans non-addressable devices"), and so does slicing it to
  a length the device count does not divide, which is what unpadding a batch
  does. `gather_to_host` makes every such array whole and local again on every
  process, so what follows runs exactly as it would in a single process.

Both were measured on jax 0.9.2 with two CPU processes and a four-device mesh.

The gather is itself a collective, so every process has to reach it, in the
same order -- which they do, because every process makes the same calls.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils


def _gather_leaf(x):
    if isinstance(x, jax.Array) and not x.is_fully_addressable:
        return jnp.asarray(np.asarray(multihost_utils.process_allgather(x, tiled=True)))
    return x


def put_global(x, sharding):
    """`jax.device_put(x, sharding)`, also when `x` is a committed array local
    to this process and `sharding` spans several.

    `device_put` takes such an array only if it is uncommitted, and which
    arrays are committed depends on how the caller made them -- the solver's
    state is numpy, but a module's arrays, or anything a jit computed from a
    committed input, need not be. Built from the host copy instead, every
    process contributing the shards it addresses; like `device_put`, every
    process must pass the same value.
    """
    if (
        jax.process_count() == 1
        or sharding.is_fully_addressable
        or (isinstance(x, jax.Array) and not x.is_fully_addressable)
    ):
        return jax.device_put(x, sharding)
    host = np.asarray(x)
    return jax.make_array_from_callback(host.shape, sharding, lambda idx: host[idx])


def gather_to_host(tree):
    """Every array leaf of `tree` with shards on another process, gathered and
    made a local array on every process. The identity in a single process."""
    if jax.process_count() == 1:
        return tree
    return jax.tree.map(_gather_leaf, tree)


def to_numpy(x):
    """`np.asarray`, gathering first if `x` has shards on another process."""
    return np.asarray(_gather_leaf(x) if jax.process_count() > 1 else x)
