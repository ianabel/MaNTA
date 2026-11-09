"""`vmap_batched(..., use_shard_map=True)` on a mesh of several devices.

A vmapped `while_loop` over a sharded batch is one program across every
device, and XLA decides whether the loop goes on with an all-reduce on each
iteration -- across nodes, a network round trip per iteration of an iterative
solve. Under shard_map each device runs its own loop on its own points. What
is pinned here: the two give the same answer, through the paths the
stellarator case uses (rounds, a carry, a module holding arrays, a pytree
argument mapped on some leaves only), and only the shard_map program is free
of collectives.

The checks run in a subprocess, because the number of CPU devices is fixed
when JAX starts and the rest of the suite has started it with one.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("equinox")

SCRIPT = textwrap.dedent(
    """
    import os
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import equinox as eqx
    from jax.sharding import NamedSharding, PartitionSpec as P
    from manta.jax import vmap_batched
    from manta.jax.batching import _place, _shard_mapped_vmap

    assert len(jax.devices()) == 4
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("axis",))
    data = NamedSharding(mesh, P("axis"))

    class Point(eqx.Module):
        # a pytree argument, as State is; every leaf is per point, which the
        # rounds' reshape needs of a mapped argument
        x: jax.Array
        y: jax.Array
        def axes():
            return Point(0, 0)

    class Solve(eqx.Module):
        # a module holding an array, as FieldJac and WarmFlux do
        scale: jax.Array
        def __call__(self, p, w, f1):
            # Newton for sqrt(x), from the warm start: a trip count that
            # depends on the point, as an iterative solve's does
            def cond(s):
                return jnp.abs(s[0] ** 2 - p.x) > 1e-12 * p.x
            def body(s):
                return (0.5 * (s[0] + p.x / s[0]), s[1] + 1)
            r, n = jax.lax.while_loop(cond, body, (f1, 0))
            return (self.scale * r * p.y + jnp.sum(w), n), r

    n = 16
    x = jnp.linspace(1.0, 1e6, n)
    pts = Point(x, jnp.full(n, 2.0))
    w = jnp.arange(3.0)  # shared, not mapped: replicated, any length
    f1 = jnp.ones(n)
    func = Solve(jnp.array(3.0))
    axes = (Point.axes(), None, 0)

    def run(**kw):
        return vmap_batched(func, axes, sharding=data, has_carry=True, **kw)(pts, w, f1)

    for nchunks, chunk in ((0, 0), (2, 8)):
        (ref, nref), cref = run(nchunks=nchunks, chunk_size=chunk)
        (out, nout), cout = run(nchunks=nchunks, chunk_size=chunk, use_shard_map=True)
        assert np.array_equal(np.asarray(ref), np.asarray(out)), (nchunks, ref, out)
        assert np.array_equal(np.asarray(nref), np.asarray(nout))
        assert np.array_equal(np.asarray(cref), np.asarray(cout))
        # the carry stays sharded, to go straight back in on the next call
        assert cout.sharding.is_equivalent_to(data, 1), cout.sharding
        expect = 3.0 * np.sqrt(np.asarray(x)) * 2.0 + 3.0
        assert np.allclose(np.asarray(out), expect, rtol=1e-12)

    # the warm start goes back in sharded and is used: one iteration or none
    (_, nwarm), _ = vmap_batched(
        func, axes, sharding=data, has_carry=True, use_shard_map=True
    )(pts, w, cout)
    assert int(np.max(np.asarray(nwarm))) <= 1, nwarm

    # the point of it: no collective in the shard_map program, and one in
    # the plain one (the loop's convergence test across the whole batch)
    args = (pts, w, f1)
    placed = _place(args, axes, mesh, "axis")
    def hlo(f, *a):
        return jax.jit(f).lower(*a).compile().as_text()
    hlo_sm = hlo(lambda pl: _shard_mapped_vmap(func, axes, mesh, "axis", pl), placed)
    sharded = (eqx.filter_shard(pts, data), w, eqx.filter_shard(f1, data))
    hlo_vmap = hlo(lambda *a: eqx.filter_vmap(func, in_axes=axes)(*a), *sharded)
    for op in ("all-reduce", "all-gather", "collective-permute", "all-to-all"):
        assert op not in hlo_sm, op
    assert "all-reduce" in hlo_vmap

    # anything but the leading axis is refused rather than mis-split
    try:
        vmap_batched(lambda a: a, (1,), sharding=data, use_shard_map=True)(
            jnp.ones((2, 4))
        )
    except ValueError as e:
        assert "leading axis" in str(e)
    else:
        raise AssertionError("in_axes 1 was accepted")

    print("OK")
    """
)


def test_shard_map_batching_matches_vmap_and_has_no_collectives():
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout + proc.stderr
