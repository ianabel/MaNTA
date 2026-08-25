#ifndef PYRUNNER_HPP
#define PYRUNNER_HPP

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string_view>
#include <variant>

#include "SolverConfig.hpp"
#include "SystemSolver.hpp"

#include "PhysicsCases.hpp"

namespace py = pybind11;

// Generic parameter
template <typename T> struct Parameter {
  bool required;
  T _default = T();
};

using ParameterType =
    std::variant<Parameter<double>, Parameter<std::string>, Parameter<int>,
                 Parameter<unsigned int>, Parameter<bool>,
                 Parameter<std::vector<double>>>;

using map_t = std::map<std::string_view, ParameterType>;

class PyRunner {
public:
  /*
      Creates runner object for running MaNTA from Python given a config
     dictionary Takes constructed transport system as input
  */
  explicit PyRunner(std::shared_ptr<TransportSystem> problem)
      : pProblem(problem) {};

  // ...or the name of a C++ physics case, as a config file's TransportSystem
  // key gives it. The object cannot be handed in the way a Python case is,
  // because a C++ case's constructor takes the Grid -- which configure() is
  // what builds. So the name is kept and the case is instantiated there, at
  // the point the grid exists. Its own configuration table travels in the same
  // dict; see PyToml.hpp.
  //
  // The name is checked here rather than at configure() so that a typo fails
  // where it was written, with the list of what is registered.
  explicit PyRunner(std::string physicsCase);

  // Not defaulted: a sliced solve the driver walked away from still owns live
  // SUNDIALS objects, and ~SystemSolver frees none of them.
  ~PyRunner();

  // Configure solver from Python
  void configure(const py::dict &);

  // Runs solver to time tFinal
  void run(double tFinal);
  // Runs to the configuration's t_final. Throws if it had none.
  void run();

  // Runs solver to steady state
  void run_ss(void);

  // A steady solve driven in slices, so a driver can look at the state, the
  // cost and the objective between them and decide whether to go on.
  //
  // start_steady() initialises and takes the first slice; continue_steady()
  // resumes from the state *and* the pseudo-time step the last one reached, so
  // slicing costs no extra continuation steps. Each returns why the slice
  // stopped. OutOfSteps is the ordinary exit -- the step budget
  // (MaxContinuationSteps) is spent, nothing is wrong -- and is *returned*; a
  // genuine solver failure throws, having written the last state it reached.
  //
  // `estimate` chooses whether the slice estimates the objective on its way
  // out. That costs one residual, one Jacobian build and one solve per slice,
  // which is worth suppressing on slices whose answer nobody reads.
  //
  // finish_steady() ends the solve: the output files, the restart file and the
  // adjoint solve, then teardown. It must be called, and calling it twice is an
  // error rather than a second write.
  SystemSolver::SteadyOutcome start_steady(bool estimate);
  SystemSolver::SteadyOutcome continue_steady(bool estimate);
  void finish_steady(void);

  // End a sliced solve without writing anything: no output slice, no restart
  // file, no adjoint solve. For a driver abandoning a parameter point, and for
  // unwinding after its own exception. A no-op when no slice loop is live, so
  // it is safe in a finally block.
  void abandon_steady(void) { abandonSlices(); };

  // What the last slice cost and what residual it ended on. Per slice, not
  // cumulative -- a driver wanting totals sums them, which keeps the more
  // informative number the primitive one.
  py::dict steadyStats(void) const;


  // The objective alone, without an adjoint solve. Needs solveAdjoint = True
  // (that is what constructs the AdjointProblem that defines G) but not the
  // gradient machinery.
  Vector G(void);

  // The objective with a first-order correction to the fixed point, and a bound
  // on what is left. A steady solve stops when ||F|| is small, not when G is, so
  // a sweep comparing two parameter points needs to know how much of the
  // difference is the answer moving and how much is each solve stopping short.
  //
  // Returned as a dict of "value", "corrected" and "uncertainty", each one entry
  // per objective. Empty when the run had no AdjointProblem, or when it was not
  // a steady solve.
  py::dict objectiveEstimate(void) const;

  // Run adjoint solver and return tuple (G, G_p)
  py::tuple getAdjointGradients(void);

  Vector getSolution(Index var,
                     std::optional<std::vector<Position>> const &points);

  // q, sampled exactly as getSolution samples u. q is an unknown of the system
  // in its own right, not a derivative to be recovered after the fact, and
  // there was no way to read it from Python at all -- so a caller wanting
  // d_x u had to differentiate a fit to getSolution's output, which is neither
  // the solver's q nor as accurate as it.
  Vector getDerivative(Index var,
                       std::optional<std::vector<Position>> const &points);

  // The element-local postprocessed solution u* in P_{k+1}, sampled the same way
  // getSolution samples u. Available for any k >= 1 regardless of whether the
  // superconvergent scheme is switched on.
  Vector getPostprocessedSolution(Index var,
                                  std::optional<std::vector<Position>> const &points);

  // The registered name of the C++ case this Runner drives, or empty when it
  // was handed a transport system object. Exposed because a driver that was
  // given a Runner has otherwise no way to ask which of the two it is.
  std::string const &physicsCase() const { return caseName; }

private:
  // Empty unless constructed from a name. Non-empty is what makes configure()
  // build `pProblem` itself, so it is also the flag distinguishing the two
  // constructors -- there is no separate bool that could disagree with it.
  std::string caseName;

  std::shared_ptr<TransportSystem> pProblem = nullptr;
  std::unique_ptr<AdjointProblem> adjoint = nullptr;
  // Built on demand by G() when solveAdjoint is false, purely to evaluate the
  // objective. Kept separate from `adjoint` so that its presence cannot be
  // mistaken for "the gradients have been computed".
  std::unique_ptr<AdjointProblem> objectiveOnlyAdjoint = nullptr;

  // Ownership of objects handled by C++.
  //
  // configure() builds the first `system`; with DegreeAdaptation armed, run()
  // and run_ss() *replace* it, once per polynomial degree. Only this pointer
  // moves -- `grid`, `adjoint` and `pProblem` are the same objects at every
  // level, which is what makes adapting the degree cheap where adapting the
  // mesh would not be. Everything crossing to Python does so by value, so no
  // caller can be left holding a reference into a replaced solver.
  //
  // The order of these two matters on teardown and in configure(): a
  // SystemSolver holds Grid const&, so `system` is nulled before `grid`.
  std::unique_ptr<SystemSolver> system;
  std::unique_ptr<Grid> grid;

  // Kept, not just consumed. Every set* in applySolverConfig configures the
  // *solver*, so a solver built later in run() needs the configuration replayed
  // against it; a fresh one has defaults.
  SolverConfig cfg;

  // The degree configure() built at, which is where an adaptive run starts.
  unsigned int k = 1;

  // True between start_steady() and finish_steady(). A sliced solve owns live
  // SUNDIALS objects that nothing else frees -- ~SystemSolver does not call
  // destroySundials() -- so this is what lets configure() and the destructor
  // clean up after a loop the driver walked away from.
  bool slicing = false;

  // Shared by start_steady and continue_steady, so the two cannot differ on the
  // outcome policy or on refreshing yJac.
  SystemSolver::SteadyOutcome runSlice(bool resume, bool estimate);

  // Tear down a slice loop that will not be finished: write nothing, free
  // everything. Safe when no loop is live.
  void abandonSlices(void);

  // Hand off to runAdaptiveDegree and adopt the solver it settles on. Shared by
  // run() and run_ss() so the two cannot diverge on the sequencing.
  void adaptDegree(double tFinal);

  // Build the case `caseName` names, from the config dict, against `grid`.
  // Only called when caseName is non-empty, and only from configure(), which
  // has built the grid by then.
  void instantiatePhysicsCase(const py::dict &config);

  bool configured = false;
  double steady_state_tolerance;

  // t_final from the configuration, if it had one. A dict need not carry it --
  // run(tFinal) is the usual way in, and a driver legitimately runs one
  // configuration to many end times -- so run() with no argument is what needs
  // it, and says so when it is absent.
  std::optional<double> configured_t_final;
};

#endif
