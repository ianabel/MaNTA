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
  ~PyRunner() = default;

  // Configure solver from Python
  void configure(const py::dict &);

  // Runs solver to time tFinal
  void run(double tFinal);
  // Runs to the configuration's t_final. Throws if it had none.
  void run();

  // Runs solver to steady state
  void run_ss(void);

  // Whether the last run was abandoned by the dG/dt gate instead of integrated,
  // i.e. whether ObjectiveDecreaseTolerance was set and the objective was already
  // falling faster than it at the initial condition. Always false when the gate
  // is not configured.
  //
  // run() and run_ss() stay void, so an existing driver is unaffected; one that
  // cares asks this. Note that a rejected step deliberately does not synthesise
  // an objective value -- G() reports G at the initial condition, which is the
  // state the solver is actually in, and what a rejected step means for the
  // search is the driver's decision, not this class's.
  bool wasRejected(void) const;

  // The dG/dt values behind that decision, one per objective.
  Vector lastDGdt(void) const;

  // The objective alone, without an adjoint solve. Needs solveAdjoint = True
  // (that is what constructs the AdjointProblem that defines G) but not the
  // gradient machinery.
  Vector G(void);

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

private:
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

  // Hand off to runAdaptiveDegree and adopt the solver it settles on. Shared by
  // run() and run_ss() so the two cannot diverge on the sequencing.
  void adaptDegree(double tFinal);

  bool configured = false;
  double steady_state_tolerance;

  // t_final from the configuration, if it had one. A dict need not carry it --
  // run(tFinal) is the usual way in, and a driver legitimately runs one
  // configuration to many end times -- so run() with no argument is what needs
  // it, and says so when it is absent.
  std::optional<double> configured_t_final;
};

#endif
