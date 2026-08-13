import enum

# The three os.environ writes that used to sit here -- TF_CPP_MIN_LOG_LEVEL,
# HDF5_USE_FILE_LOCKING and XLA_PYTHON_CLIENT_MEM_FRACTION -- moved out to the
# drivers that use them (now python-physics/stellarator/) when this became
# library code. Process-wide policy set as a side effect of importing a library
# is a trap: it applies to every caller, including ones that wanted the
# opposite.
import jax
import manta as MaNTA
import jax.numpy as jnp

# The FFI bindings this module registers are compiled in only under XLA_FFI
# (Python.cpp:361). Without them the registration loop below dies on an
# AttributeError naming `runner_ffi_ops`, which reads as a bug in the package
# rather than as a build that was not asked for the feature.
if not hasattr(MaNTA, "runner_ffi_ops"):
    raise ImportError(
        "manta.jax.ffi_runner needs an XLA_FFI build of the extension: "
        "manta.runner_ffi_ops is absent. Rebuild with `make python XLA_FFI=on` "
        "-- it needs the jaxlib headers -- and add CUDA=on for the GPU targets. "
        "The rest of manta.jax does not require it."
    )


class Platform(enum.IntEnum):
    CPU = 0
    GPU = 1

cpu_device = jax.devices('cpu')[0]

cpu_fp_dtype = jnp.float32
cpu_i_dtype = jnp.int32

if (cpu_fp_dtype == jnp.float64):
    print("Enabling 64-bit math")
    jax.config.update('jax_enable_x64', True)
# MaNTA has to run on cpu so we only have cpu implementation for the run functions
ffi_ops_names = ["get_solution", "get_adjoint_gradients","run", "run_ss"] # (op_name, use_gpu)
ffi_ops = {}

def register_ffi_cpu(op_name):
    for name, target in MaNTA.runner_ffi_ops().items():
        if (name.startswith(op_name)):
            print("Registering cpu implementation for operation " + op_name)
            jax.ffi.register_ffi_target(name, target, platform="cpu")
            return name
    return False

def register_ffi_gpu(op_name):
    for name, target in MaNTA.runner_ffi_ops_cuda().items():
        if (name.startswith(op_name)):
            print("Registering gpu implementation for operation " + op_name)
            jax.ffi.register_ffi_target(name, target, platform="CUDA")
            return name
    return False

has_gpu = hasattr(MaNTA, "runner_ffi_ops_cuda")

for name in ffi_ops_names:
    cpu_op = register_ffi_cpu(name)
    if (not cpu_op):
        raise RuntimeError("Could not find cpu implementation for operation " + name)
    ffi_ops[name] = [cpu_op]
    if (has_gpu):
        gpu_op = register_ffi_gpu(name)
        if (gpu_op):
            ffi_ops[name].append(gpu_op)

class FFIRunner(MaNTA.Runner):
    def __init__(self, transport_system, points, ng, np, spatialParameters = False):
        MaNTA.Runner.__init__(self, transport_system)
    
        self.points = jnp.array(points)
        self.fac = 1
        if (spatialParameters):
            self.fac = len(self.points)
        self.ng = ng
        self.np = np
    
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
        raise NotImplementedError("getSolution method is disabled when using FFI; use Get_profile")

    """
    Runner functions using the ffi api 

    Run and Run_ss can only be called on cpu, but Get_profile and Get_adjoint_gradients can be called on either cpu or gpu and will select the appropriate implementation
    """
    def Run(self, tFinal):
        with jax.default_device(cpu_device):
            jax.ffi.ffi_call(ffi_ops["run"][Platform.CPU], [], has_side_effect=True)(cpu_fp_dtype(tFinal), obj=self.get_address())
    def Run_ss(self):
        with jax.default_device(cpu_device):
            jax.ffi.ffi_call(ffi_ops["run_ss"][Platform.CPU], [],  has_side_effect=True)(obj=self.get_address())

    def Get_adjoint_gradients(self):
        def cpu_call():
            adjoint_output = [
                jax.ShapeDtypeStruct((self.ng,),cpu_fp_dtype),
                jax.ShapeDtypeStruct((self.ng * self.fac, self.np),cpu_fp_dtype)
            ]  
            G, G_p = jax.ffi.ffi_call(ffi_ops["get_adjoint_gradients"][Platform.CPU], adjoint_output)(obj=self.get_address())
            return G, G_p
           
        def gpu_call():
            adjoint_output = [
                jax.ShapeDtypeStruct((self.ng,),jnp.float32),
                jax.ShapeDtypeStruct((self.ng * self.fac, self.np),jnp.float32)
            ] 
            G, G_p = jax.ffi.ffi_call(ffi_ops["get_adjoint_gradients"][Platform.GPU], adjoint_output)(obj=self.get_address()) 
            return cpu_fp_dtype(G), cpu_fp_dtype(G_p)

        if (has_gpu):
            return jax.lax.platform_dependent(cpu=cpu_call, cuda=gpu_call)
        else:
            return cpu_call()

    def Get_profile(self, var, points = None):
        def cpu_call(points, var):
            sol_output = jax.ShapeDtypeStruct((len(points),),cpu_fp_dtype)
            return jax.ffi.ffi_call(ffi_ops["get_solution"][Platform.CPU], sol_output)(cpu_i_dtype(var), points.astype(cpu_fp_dtype), obj=self.get_address())
        
        def gpu_call(points, var):
             sol_output = jax.ShapeDtypeStruct((len(points),),jnp.float32)
             return cpu_fp_dtype(jax.ffi.ffi_call(ffi_ops["get_solution"][Platform.GPU], sol_output)(jnp.int32(var), points.astype(jnp.float32), obj=self.get_address()))

        if (has_gpu):
            return jax.lax.platform_dependent(points if points is not None else self.points, var, cpu=cpu_call, cuda=gpu_call)
        else: 
            return cpu_call(points if points is not None else self.points, var)