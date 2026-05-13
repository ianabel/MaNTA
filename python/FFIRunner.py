import jax
import MaNTA
import jax.numpy as jnp

import enum

class Platform(enum.IntEnum):
    CPU = 0
    GPU = 1

cpu_device = jax.devices('cpu')[0]

# MaNTA has to run on cpu so we only have cpu implementation for the run functions
ffi_ops_names = [("get_solution", True), ("get_adjoint_gradients", True), ("run", False), ("run_ss", False)] # (op_name, use_gpu)
ffi_ops = {}

def register_ffi_cpu(op_name):
    jax.config.update('jax_enable_x64', True)
    for name, target in MaNTA.runner_ffi_ops().items():
        if (name.startswith(op_name)):
            print("Registering cpu implementation for operation " + op_name)
            jax.ffi.register_ffi_target(name, target, platform="cpu")
            return name

def register_ffi_gpu(op_name):
    for name, target in MaNTA.runner_ffi_ops_cuda().items():
        if (name.startswith(op_name)):
            print("Registering gpu implementation for operation " + op_name)
            jax.ffi.register_ffi_target(name, target, platform="CUDA")
            return name

platform = jax.lax.platform_dependent(None, cpu=(lambda x : Platform.CPU) , cuda=(lambda x : Platform.GPU)) # very silly syntax

for name, has_gpu in ffi_ops_names:
    if (has_gpu):
        ffi_ops[name] = [register_ffi_cpu(name), register_ffi_gpu(name)] 
    else:
        ffi_ops[name] = [register_ffi_cpu(name)]

dtype   = jnp.float32 if jax.lax.eq(platform, Platform.GPU) else jnp.float64
i_dtype = jnp.int32   if jax.lax.eq(platform, Platform.GPU) else jnp.int64

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
    
    """
    Assume that the user would want to call the ffi versions of the Runner class methods if they're using this class, so we disable the regular versions
    """
    def run(self, *args, **kwargs):
        raise NotImplementedError("run method is disabled when using FFI; use Run")
    def run_ss(self, *args, **kwargs):
        raise NotImplementedError("run method is disabled when using FFI; use Run_ss")
    def getAdjointGradients(self, *args, **kwargs):
        raise NotImplementedError("runAdjointSolve method is disabled when using FFI; use Run_adjoint_solve")
    def getSolution(self, *args, **kwargs):
        raise NotImplementedError("getSolution method is disabled when using FFI; use get_profile")

    """
    Runner functions using the ffi api 
    """
    def Run(self, tFinal):
        with jax.default_device(cpu_device):
            jax.ffi.ffi_call(ffi_ops["run"][Platform.CPU], [], has_side_effect=True)(tFinal, obj=self.get_address())
    def Run_ss(self):
        with jax.default_device(cpu_device):
            jax.ffi.ffi_call(ffi_ops["run_ss"][Platform.CPU], [],  has_side_effect=True)(obj=self.get_address())
    def Get_adjoint_gradients(self):
        op_name = jax.lax.platform_dependent(ffi_ops["get_adjoint_gradients"], cpu=(lambda op : op[Platform.CPU]) , cuda=(lambda op : op[Platform.GPU]))
        return jax.ffi.ffi_call(op_name, self.adjoint_output)(obj=self.get_address())
    def Get_profile(self, var, points = None):
        if (points is None):
            points = self.points
        sol_output = jax.ShapeDtypeStruct((len(points),),dtype)
        op_name = jax.lax.platform_dependent(ffi_ops["get_solution"], cpu=(lambda op : op[Platform.CPU]) , cuda=(lambda op : op[Platform.GPU]))
        return jax.ffi.ffi_call(op_name, sol_output)(i_dtype(var), points, obj=self.get_address())
