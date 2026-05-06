
import jax

import MaNTA
import jax.numpy as jnp
# jax.config.update('jax_enable_x64', True)
CPU = 0
GPU = 1

# MaNTA has to run on cpu so we only have cpu implementation for the run functions
# (op_name, use_gpu)
ffi_ops_names = [("get_solution", True), ("run_adjoint_solve", True), ("run", False), ("run_ss", False)]
ffi_ops = {}

def register_ffi_cpu(op_name):
    print("Using cpu implementation for operation " + op_name)
    jax.config.update('jax_enable_x64', True)
    for name, target in MaNTA.runner_ffi_ops().items():
        if (name.startswith(op_name)):
            jax.ffi.register_ffi_target(name, target, platform="cpu")
            return name

def register_ffi_gpu(op_name):
    print("Using gpu implementation for operation " + op_name)
    for name, target in MaNTA.runner_ffi_ops_cuda().items():
        if (name.startswith(op_name)):
            jax.ffi.register_ffi_target(name, target, platform="CUDA")
            return name

platform = jax.lax.platform_dependent(None, cpu=(lambda x : CPU) , cuda=(lambda x : GPU)) # very silly syntax

for name, use_gpu in ffi_ops_names:
    if (use_gpu):
        ffi_ops[name] = jax.lax.platform_dependent(name, cpu=register_ffi_cpu, cuda=register_ffi_gpu)
    else:
        ffi_ops[name] = register_ffi_cpu(name)

dtype   = jnp.float32 if jax.lax.eq(platform, GPU) else jnp.float64
i_dtype = jnp.int32   if jax.lax.eq(platform, GPU) else jnp.int64

class FFIRunner(MaNTA.Runner):
    def __init__(self, transport_system, points, ng, np, spatialParameters = False):
        MaNTA.Runner.__init__(self, transport_system)
    
        self.points = jnp.array(points, dtype=dtype)
        fac = 1
        if (spatialParameters):
            fac = len(self.points)

        self.adjoint_output = [
            jax.ShapeDtypeStruct((ng,),dtype),
            jax.ShapeDtypeStruct((ng * fac, np),dtype)
        ]  
        self.sol_output = jax.ShapeDtypeStruct((len(self.points),),dtype)
    
    def run(self, *args, **kwargs):
        raise NotImplementedError("run method is disabled when using FFI; use Run")
    def run_ss(self, *args, **kwargs):
        raise NotImplementedError("run method is disabled when using FFI; use Run_ss")
    def runAdjointSolve(self, *args, **kwargs):
        raise NotImplementedError("runAdjointSolve method is disabled when using FFI; use Run_adjoint_solve")
    def getSolution(self, *args, **kwargs):
        raise NotImplementedError("getSolution method is disabled when using FFI; use get_profile")

    def Run(self, tFinal):
        jax.ffi.ffi_call(ffi_ops["run"], [], has_side_effect=True)(tFinal, obj=self.get_address())
    def Run_ss(self):
        return jax.ffi.ffi_call(ffi_ops["run_ss"], [],  has_side_effect=True)(obj=self.get_address())
    def Run_adjoint_solve(self):
        return jax.ffi.ffi_call(ffi_ops["run_adjoint_solve"], self.adjoint_output, has_side_effect=True)(obj=self.get_address())
    def get_profile(self, var, points = None):
        if (points is None):
            points = self.points

        sol_output = jax.ShapeDtypeStruct((len(points),),dtype)
        return jax.ffi.ffi_call(ffi_ops["get_solution"], sol_output, has_side_effect=True)(i_dtype(var), points, obj=self.get_address())
