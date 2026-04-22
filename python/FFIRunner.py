
import jax
import MaNTA
import jax.numpy as jnp
# jax.config.update('jax_enable_x64', True)
CPU = 0
GPU = 1

ffi_ops = dict.fromkeys(["get_solution", "run_adjoint_solve", "run", "run_ss"])
def register_ffi_cpu(ops_dict):
    print("Using cpu implementation")
    jax.config.update('jax_enable_x64', True)
    for (name, target), dict_entry in zip(MaNTA.runner_ffi_ops().items(), list(ops_dict.keys())):
        ops_dict[dict_entry] = name
        jax.ffi.register_ffi_target(name, target, platform="cpu")

def register_ffi_gpu(ops_dict):
    print("Using gpu implementation")

    # ops = MaNTA.runner_ffi_ops_cuda()

    # ops_dict["get_solution"] = "get_solution_ffi_cuda"
    # ops_dict["run_adjoint_solve"] = "run_adjoint_solve_ffi_cuda"
    # ops_dict["run"] = "run_ffi_cuda"
    # ops_dict["run_ss"] = "run_ss_ffi_cuda"

    # name = "get_solution_ffi_cuda"
    # jax.ffi.register_ffi_target(name, ops[name], platform="CUDA")
    # name = "run_adjoint_solve_ffi_cuda"
    # jax.ffi.register_ffi_target(name, ops[name], platform="CUDA")
    # name = "run_ffi_cuda"
    # jax.ffi.register_ffi_target(name, ops[name], platform="CUDA")
    # name = "run_ss_ffi_cuda"
    # jax.ffi.register_ffi_target(name, ops[name], platform="CUDA")
    for (name, target), dict_entry in zip(MaNTA.runner_ffi_ops_cuda().items(), list(ops_dict.keys())):
        ops_dict[dict_entry] = name
        jax.ffi.register_ffi_target(name, target, platform="CUDA")

platform = 0

platform = jax.lax.platform_dependent(None, cpu=(lambda x : CPU) , cuda=(lambda x : GPU)) # very silly syntax

jax.lax.platform_dependent(ffi_ops, cpu=register_ffi_cpu, cuda=register_ffi_gpu)

dtype =  jnp.float32 if jax.lax.eq(platform, GPU) else jnp.float64
i_dtype = jnp.int32 if jax.lax.eq(platform, GPU) else jnp.int64

class FFIRunner(MaNTA.Runner):
    def __init__(self, transport_system, points, ng, np, spatialParameters = False):
        MaNTA.Runner.__init__(self, transport_system)
    
        #self.dtype = jnp.float32 if jax.lax.eq(platform, GPU) else jnp.float64# change if using 64 bit jax
        #jax.lax.platform_dependent(self.dtype, cpu=output_dtype("cpu"),cuda=output_dtype("gpu"))
        self.points = jnp.array(points, dtype=dtype)
        print(ng)
        fac = 1
        if (spatialParameters):
            fac = len(self.points)

        print(np*fac)
        self.adjoint_output = [
            jax.ShapeDtypeStruct((ng,),dtype),
            jax.ShapeDtypeStruct((ng * fac, np),dtype)
        ]  
        self.sol_output = jax.ShapeDtypeStruct((len(self.points),),dtype)
        self.run_output = jax.ShapeDtypeStruct((), i_dtype)
    
    def Run(self, tFinal):
        jax.ffi.ffi_call(ffi_ops["run"], [], has_side_effect=True)(dtype(tFinal), obj=self.get_address())
    def Run_ss(self):
        return jax.ffi.ffi_call(ffi_ops["run_ss"], self.run_output, has_side_effect=True)(obj=self.get_address())
    def Run_adjoint_solve(self):
        return jax.ffi.ffi_call(ffi_ops["run_adjoint_solve"], self.adjoint_output)(obj=self.get_address())
    def get_profile(self, var, points = None):
        if (points is None):
            points = self.points
        return jax.ffi.ffi_call(ffi_ops["get_solution"], self.sol_output)(i_dtype(var), points, obj=self.get_address())
