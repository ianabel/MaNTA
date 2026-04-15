import jax
import MaNTA
import jax.numpy as jnp
jax.ffi.register_ffi_type(
    "runner", MaNTA.runner_type(), platform="cpu")

jax.ffi.register_ffi_target("runner", MaNTA.handler(), platform="cpu", api_version=1)

runner = jax.ffi.ffi_call("runner", None)()
print(runner)