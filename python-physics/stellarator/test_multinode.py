import os
import jax
from jax.sharding import PartitionSpec, NamedSharding
import jax.numpy as jnp
import equinox as eqx

jax.distributed.initialize()
P = jax.sharding.PartitionSpec

# Always print this to verify correct setup
print(
    f"[{jax.process_index()}/{jax.process_count()}] devices:",
    jax.devices(),
    flush=True,
)
print(
    f"[{jax.process_index()}/{jax.process_count()}] local devices:",
    jax.local_devices(),
    flush=True,
)
print(
    f"I will be running a calculation among {jax.process_count()} tasks, "
    f"using a total of {len(jax.devices())} devices "
    f"({len(jax.local_devices())} per slurm task). "
    f"If this does not match your expected number of total devices, "
    f"something is misconfigured",
    flush=True,
)


P = PartitionSpec
devices = jax.devices()
print(devices)
mesh = jax.make_mesh(
    (jax.device_count(),), ("axis",), axis_types=(jax.sharding.AxisType.Auto,)
)
data_sharding = NamedSharding(
    mesh,
    P(
        "axis",
    ),
)


arr = jnp.ones((32, 3))
arr2 = eqx.filter_shard(arr, data_sharding)
if jax.process_index() == 0:
    jax.debug.visualize_array_sharding(arr2)
    print(arr2)
