#ifndef PYRUNNER_HPP
#define PYRUNNER_HPP

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string_view>
#include <variant>

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

  // Runs solver to steady state
  void run_ss(void);

  // The objective alone, without an adjoint solve. Needs solveAdjoint = True
  // (that is what constructs the AdjointProblem that defines G) but not the
  // gradient machinery.
  Vector G(void);

  // Run adjoint solver and return tuple (G, G_p)
  py::tuple getAdjointGradients(void);

  Vector getSolution(Index var,
                     std::optional<std::vector<Position>> const &points);

private:
  std::shared_ptr<TransportSystem> pProblem = nullptr;
  std::unique_ptr<AdjointProblem> adjoint = nullptr;
  // Built on demand by G() when solveAdjoint is false, purely to evaluate the
  // objective. Kept separate from `adjoint` so that its presence cannot be
  // mistaken for "the gradients have been computed".
  std::unique_ptr<AdjointProblem> objectiveOnlyAdjoint = nullptr;

  // Ownership of objects handled by C++
  std::unique_ptr<SystemSolver> system;
  std::unique_ptr<Grid> grid;

  bool configured = false;
  double steady_state_tolerance;
};

#endif
