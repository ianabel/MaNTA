"""
Compiled core of the MaNTA Python package; import `manta` instead.
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['AdjointProblem', 'Aux', 'BoundaryCondition', 'BoundaryKind', 'Dirichlet', 'Field', 'Grid', 'Mixed', 'Neumann', 'Runner', 'Scalar', 'State', 'StateField', 'SteadyOutcome', 'SystemSpec', 'TomlValue', 'TransportSystem', 'getNodes', 'load_physics_plugin', 'numbered_spec', 'physics_cases', 'registerPhysicsCase', 'run']
class AdjointProblem:
    spatialParameters: bool
    def __init__(self) -> None:
        ...
    def computeLowerBoundarySensitivity(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    def computeUpperBoundarySensitivity(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    def dAux(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: dict[typing.Sequence[float]], arg2: dict[typing.Sequence[float]], arg3: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    def dAux_dp(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsFloat | typing.SupportsIndex, arg3: State, arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def dSigma(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: dict[typing.Sequence[float]], arg2: dict[typing.Sequence[float]], arg3: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    def dSources(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: dict[typing.Sequence[float]], arg2: dict[typing.Sequence[float]], arg3: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    def dg(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: dict[typing.Sequence[float]], arg2: dict[typing.Sequence[float]], arg3: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> None:
        ...
    def dgFn_dphi(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], arg2: State, arg3: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def dgFndp(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: dict[typing.Sequence[float]], arg2: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
    def gFn(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: dict[typing.Sequence[float]], arg2: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def getName(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> str:
        ...
    @property
    def ng(self) -> int:
        ...
    @ng.setter
    def ng(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def np(self) -> int:
        ...
    @np.setter
    def np(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def np_boundary(self) -> int:
        ...
    @np_boundary.setter
    def np_boundary(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class Aux:
    description: str
    name: str
    units: str
    def __init__(self, name: str, description: str = '', units: str = '') -> None:
        ...
class BoundaryCondition:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: BoundaryKind) -> bool:
        ...
    def __init__(self, kind: BoundaryKind) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def a(self) -> float:
        ...
    @property
    def b(self) -> float:
        ...
    @property
    def d(self) -> float:
        ...
    @property
    def kind(self) -> BoundaryKind:
        ...
class BoundaryKind:
    """
    Members:
    
      Dirichlet
    
      Neumann
    
      Mixed
    """
    Dirichlet: typing.ClassVar[BoundaryKind]  # value = <BoundaryKind.Dirichlet: 0>
    Mixed: typing.ClassVar[BoundaryKind]  # value = <BoundaryKind.Mixed: 2>
    Neumann: typing.ClassVar[BoundaryKind]  # value = <BoundaryKind.Neumann: 1>
    __members__: typing.ClassVar[dict[str, BoundaryKind]]  # value = {'Dirichlet': <BoundaryKind.Dirichlet: 0>, 'Neumann': <BoundaryKind.Neumann: 1>, 'Mixed': <BoundaryKind.Mixed: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Field:
    description: str
    lower: BoundaryCondition
    name: str
    units: str
    upper: BoundaryCondition
    def __init__(self, name: str, description: str = '', units: str = '', lower: BoundaryCondition = ..., upper: BoundaryCondition = ...) -> None:
        ...
class Grid:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: bool, arg4: typing.SupportsFloat | typing.SupportsIndex, arg5: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def getNCells(self) -> int:
        ...
class Runner:
    def G(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    @typing.overload
    def __init__(self, arg0: TransportSystem) -> None:
        ...
    @typing.overload
    def __init__(self, physics_case: str) -> None:
        ...
    def abandon_steady(self) -> None:
        ...
    def configure(self, arg0: dict) -> None:
        ...
    def continue_steady(self, estimate: bool = True) -> SteadyOutcome:
        ...
    def finish_steady(self) -> None:
        ...
    def getAdjointGradients(self) -> tuple:
        ...
    def getDerivative(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def getPostprocessedSolution(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def getSolution(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex] | None) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def get_address(self) -> int:
        ...
    def objectiveEstimate(self) -> dict:
        ...
    @typing.overload
    def run(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def run(self) -> None:
        ...
    def run_ss(self) -> None:
        ...
    def start_steady(self, estimate: bool = True) -> SteadyOutcome:
        ...
    def steadyStats(self) -> dict:
        ...
    @property
    def physics_case(self) -> str:
        """
        The registered C++ case name this Runner was built from, or "" when it was handed a transport system object.
        """
class Scalar:
    description: str
    differential: bool
    name: str
    units: str
    def __init__(self, name: str, description: str = '', units: str = '', differential: bool = False) -> None:
        ...
class State:
    """
    A view of the solution at one point. Valid only inside the hook it was passed to.
    """
    def __repr__(self) -> str:
        ...
    @property
    def geom(self) -> StateField:
        """
        the field model's geometry (derived, not an unknown)
        """
    @property
    def phi(self) -> StateField:
        """
        the auxiliary variables
        """
    @property
    def q(self) -> StateField:
        """
        d(variable)/dx
        """
    @property
    def scalars(self) -> StateField:
        """
        the global scalars
        """
    @property
    def sigma(self) -> StateField:
        """
        the stored flux, sigma = -sigma_hat
        """
    @property
    def sigmaHat(self) -> StateField:
        """
        the physical flux, the quantity SigmaFn returns (read-only)
        """
    @property
    def u(self) -> StateField:
        """
        the variables
        """
class StateField:
    """
    One field of a State: indexable by position or by declared name.
    """
    def __array__(self, dtype: typing.Any = None, copy: typing.Any = None) -> numpy.ndarray:
        ...
    def __getitem__(self, arg0: typing.Any) -> float:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setitem__(self, arg0: typing.Any, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class SteadyOutcome:
    """
    Why a steady solve, or one slice of one, stopped.
    
    Members:
    
      NotRun : No steady solve has been taken on this solver.
    
      Converged : ||F|| fell below SteadyStateTolerance.
    
      OutOfSteps : The MaxContinuationSteps budget was spent. Not a failure: the state and the pseudo-time step reached are both good, and continue_steady() resumes from them.
    
      SolverFailed : KINSol failed in a way pseudo-transient damping cannot answer.
    """
    Converged: typing.ClassVar[SteadyOutcome]  # value = <SteadyOutcome.Converged: 1>
    NotRun: typing.ClassVar[SteadyOutcome]  # value = <SteadyOutcome.NotRun: 0>
    OutOfSteps: typing.ClassVar[SteadyOutcome]  # value = <SteadyOutcome.OutOfSteps: 2>
    SolverFailed: typing.ClassVar[SteadyOutcome]  # value = <SteadyOutcome.SolverFailed: 3>
    __members__: typing.ClassVar[dict[str, SteadyOutcome]]  # value = {'NotRun': <SteadyOutcome.NotRun: 0>, 'Converged': <SteadyOutcome.Converged: 1>, 'OutOfSteps': <SteadyOutcome.OutOfSteps: 2>, 'SolverFailed': <SteadyOutcome.SolverFailed: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SystemSpec:
    def __init__(self, variables: collections.abc.Sequence[Field], scalars: collections.abc.Sequence[Scalar] = [], aux: collections.abc.Sequence[Aux] = []) -> None:
        ...
    def validate(self) -> None:
        ...
    @property
    def aux(self) -> list[Aux]:
        ...
    @aux.setter
    def aux(self, arg0: collections.abc.Sequence[Aux]) -> None:
        ...
    @property
    def scalars(self) -> list[Scalar]:
        ...
    @scalars.setter
    def scalars(self, arg0: collections.abc.Sequence[Scalar]) -> None:
        ...
    @property
    def variables(self) -> list[Field]:
        ...
    @variables.setter
    def variables(self, arg0: collections.abc.Sequence[Field]) -> None:
        ...
class TomlValue:
    def __getitem__(self, arg0: str) -> typing.Any:
        ...
    def __init__(self) -> None:
        ...
class TransportSystem:
    def AuxG(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: State, arg2: typing.SupportsFloat | typing.SupportsIndex, arg3: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def AuxGPrime(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: State, arg2: State, arg3: typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def AuxGPrime_v(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: dict[typing.Sequence[float]], arg2: dict[typing.Sequence[float]], arg3: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def AuxG_v(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: dict[typing.Sequence[float]], arg2: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], arg3: typing.SupportsFloat | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def ComputePhysics(self, arg0: dict[typing.Sequence[float]], arg1: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], arg2: typing.SupportsFloat | typing.SupportsIndex) -> typing.Annotated[list[list[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]], "FixedSize(3)"]:
        ...
    def InitialAuxValue(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def InitialDerivative(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def InitialScalarValue(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> float:
        ...
    def InitialValue(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def LowerBoundary(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def ScalarG(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: dict[typing.Sequence[float]], arg2: dict[typing.Sequence[float]], arg3: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], arg4: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], arg5: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], arg6: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def SigmaFn(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: State, arg2: typing.SupportsFloat | typing.SupportsIndex, arg3: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def SigmaFn_v(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: dict[typing.Sequence[float]], arg2: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], arg3: typing.SupportsFloat | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def Sources(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: State, arg2: typing.SupportsFloat | typing.SupportsIndex, arg3: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def Sources_v(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: dict[typing.Sequence[float]], arg2: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], arg3: typing.SupportsFloat | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        ...
    def UpperBoundary(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    @typing.overload
    def __init__(self, spec: SystemSpec) -> None:
        ...
    @typing.overload
    def __init__(self, variables: collections.abc.Sequence[Field], scalars: collections.abc.Sequence[Scalar] = [], aux: collections.abc.Sequence[Aux] = []) -> None:
        ...
    def aFn(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex) -> float:
        ...
    def createAdjointProblem(self) -> AdjointProblem:
        ...
    def dAuxG_dGeometry(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], arg2: State, arg3: typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def dSigma(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: dict[typing.Sequence[float]], arg2: dict[typing.Sequence[float]], arg3: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def dSigmaFn_dGeometry(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], arg2: State, arg3: typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def dSigmaFn_dq(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], arg2: State, arg3: typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def dSigmaFn_du(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], arg2: State, arg3: typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def dSigma_dPhi(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], arg2: State, arg3: typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def dSources(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: dict[typing.Sequence[float]], arg2: dict[typing.Sequence[float]], arg3: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def dSources_dGeometry(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], arg2: State, arg3: typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def dSources_dPhi(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], arg2: State, arg3: typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def dSources_dScalars(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], arg2: State, arg3: typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def dSources_dq(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], arg2: State, arg3: typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def dSources_dsigma(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], arg2: State, arg3: typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def dSources_du(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"], arg2: State, arg3: typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def isLowerBoundaryDirichlet(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    def isScalarDifferential(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    def isUpperBoundaryDirichlet(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> bool:
        ...
    @property
    def nAux(self) -> int:
        ...
    @property
    def nScalars(self) -> int:
        ...
    @property
    def nVars(self) -> int:
        ...
    @property
    def spec(self) -> SystemSpec:
        ...
def Mixed(a: typing.SupportsFloat | typing.SupportsIndex = 0.0, b: typing.SupportsFloat | typing.SupportsIndex = 0.0, d: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> BoundaryCondition:
    """
    A mixed/Robin boundary condition a u + b q + d sigma = c, where c is what LowerBoundary/UpperBoundary returns. sigma is the stored flux, which is -sigma_hat. At least one of b and d must be nonzero.
    """
def _test_dSigmaFn_dGeometry(sys: TransportSystem, i: typing.SupportsInt | typing.SupportsIndex, geom: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], x: typing.SupportsFloat | typing.SupportsIndex, t: typing.SupportsFloat | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Test support only: builds a State carrying the given geometry and calls the pointwise dSigmaFn_dGeometry dispatcher directly.
    """
@typing.overload
def getNodes(arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], arg1: typing.SupportsInt | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Get the points of a grid
    """
@typing.overload
def getNodes(arg0: typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex, arg3: typing.SupportsInt | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Get the points of a grid
    """
def load_physics_plugin(path: str) -> None:
    """
    Load a physics case built outside the MaNTA tree, so that manta.Runner(name) can reach it. The dict equivalent of a config file's PhysicsPlugins key. Compile the plugin with the flags `pkg-config --cflags manta` reports, and do not link it against -lmanta; see the out-of-tree section of the docs.
    """
def numbered_spec(nVars: typing.SupportsInt | typing.SupportsIndex, nScalars: typing.SupportsInt | typing.SupportsIndex = 0, nAux: typing.SupportsInt | typing.SupportsIndex = 0, lower: BoundaryCondition = ..., upper: BoundaryCondition = ..., differential: bool = False) -> SystemSpec:
    """
    A SystemSpec using the historical placeholder names (Var0, Scalar0, AuxVariable0).
    """
def physics_cases() -> list[str]:
    """
    Every physics case name manta.Runner(name) will accept, ascending. Includes the C++ cases compiled into this extension, anything a loaded plugin registered, and anything registerPhysicsCase was called with.
    """
def registerPhysicsCase(name: str, factory: collections.abc.Callable[[TomlValue, Grid], TransportSystem]) -> None:
    """
    Register a physics case under the name a config file can ask for.
    """
def run(arg0: str) -> int:
    """
    Runs the MaNTA suite using given configuration file
    """
Dirichlet: BoundaryKind  # value = <BoundaryKind.Dirichlet: 0>
Neumann: BoundaryKind  # value = <BoundaryKind.Neumann: 1>
