import jax
import os
import MaNTA

# jax.config.update('jax_enable_x64', True)

ffi_ops = dict.fromkeys(["get_solution", "run_adjoint_solve", "run", "run_ss"])
def register_ffi_cpu(ops_dict):
    jax.config.update('jax_enable_x64', True)
    for (name, target), dict_entry in zip(MaNTA.runner_ffi_ops().items(), list(ops_dict.keys())):
        ops_dict[dict_entry] = name
        jax.ffi.register_ffi_target(name, target, platform="cpu")

def register_ffi_gpu(ops_dict):
    platform = "gpu"
    for (name, target), dict_entry in zip(MaNTA.runner_ffi_ops_cuda().items(), list(ops_dict.keys())):
        print(dict_entry)
        ops_dict[dict_entry] = name
        jax.ffi.register_ffi_target(name, target, platform="CUDA")
  
jax.lax.platform_dependent(ffi_ops, cpu=register_ffi_cpu, cuda=register_ffi_gpu)

platform = ""

platform = jax.lax.platform_dependent("", cpu=(lambda x : "cpu") , cuda=(lambda x : "gpu")) # very silly syntax
import jax.numpy as jnp

class FFIRunner(MaNTA.Runner):
    def __init__(self, transport_system, points, ng, np, spatialParameters = False):
        MaNTA.Runner.__init__(self, transport_system)
        self.points = points
        
        self.dtype = jnp.float32 if platform == "gpu" else jnp.float64# change if using 64 bit jax
        #jax.lax.platform_dependent(self.dtype, cpu=output_dtype("cpu"),cuda=output_dtype("gpu"))
        print(ng)
        print(np)
        fac = 1
        if (spatialParameters):
            fac = len(self.points)
        self.adjoint_output = [
            jax.ShapeDtypeStruct((ng,), self.dtype),
            jax.ShapeDtypeStruct((ng * fac, np), self.dtype)
        ]  
        self.sol_output = jax.ShapeDtypeStruct((len(self.points),), self.dtype)
    
    def run(self, tFinal):
        return jax.ffi.ffi_call(ffi_ops["run"], [], has_side_effect=True)(self.dtype(tFinal), obj=self.get_address())
    def run_ss(self):
        return jax.ffi.ffi_call(ffi_ops["run_ss"], [], has_side_effect=True)(obj=self.get_address())
    def run_adjoint_solve(self):
        return jax.ffi.ffi_call(ffi_ops["run_adjoint_solve"], self.adjoint_output, has_side_effect=True)(obj=self.get_address())
    def get_profile(self, var, points = None):
        if (points is None):
            points = self.points
        return jax.ffi.ffi_call(ffi_ops["get_solution"], self.sol_output)(var, points, obj=self.get_address())
