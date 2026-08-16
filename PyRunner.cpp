#include "PyRunner.hpp"
#include "Logging.hpp"
#include "PyConfigSource.hpp"
#include <pybind11/eigen.h>
#include <string>
#include <print>
// Load restart data into vectors. Defined in MaNTA.cpp; `nField` comes back as
// how many trailing entries of Y are a field model's psi.
int LoadFromFile(netCDF::NcFile &restart_file, std::vector<double> &Y,
                 std::vector<double> &dYdt, Index &nField);

void PyRunner::configure(const py::dict &config) {
  if (!pProblem)
    throw std::runtime_error("Transport system not set. Please set transport "
                             "system before configuring solver.");
  // Set stored problem to null to allow reconfiguration after object creation
  system = nullptr;
  grid = nullptr;

  // Every key this accepts is declared in ConfigSchema.cpp, the same table
  // runManta reads. This function used to carry its own `params` list and its
  // own reader, which is how the dict came to want `tZero` where a config file
  // wanted `t_initial`, and to default Absolute_tolerance to a different
  // number.
  DictConfigSource source(config);
  SolverConfig cfg = [&] {
    try
    {
      return loadSolverConfig(source, ConfigSchema::Reader::Dict);
    }
    catch (std::invalid_argument const &e)
    {
      // std::invalid_argument is the right C++ type and pybind maps it to
      // ValueError -- but configure() has raised RuntimeError for a bad
      // configuration since it existed, and a driver catching that is entitled
      // to go on working. Translated here rather than weakening the loader's
      // type, which the C++ tests check.
      throw std::runtime_error(e.what());
    }
  }();

  netCDF::NcFile restart_file;
  if (cfg.restart) {
    std::string fileName = cfg.RestartFile.empty()
                               ? cfg.OutputFilename + ".restart.nc"
                               : cfg.RestartFile;
    try {
      restart_file.open(fileName, netCDF::NcFile::FileMode::read);
    } catch (...) {
      throw std::runtime_error(
          "Failed to open restart netCDF file at: " +
          std::string(std::filesystem::absolute(std::filesystem::path(fileName))));
    }
  }

  unsigned int k = 1;
  grid = makeGrid(cfg, cfg.restart ? &restart_file : nullptr, k);

  if (cfg.solveAdjoint)
    adjoint = pProblem->createAdjointProblem();

  if (cfg.restart) {
    std::vector<double> Y, dYdt;
    Index nField_file = 0;
    Index nDOF_file = LoadFromFile(restart_file, Y, dYdt, nField_file);

    // This surface cannot attach a field model at all: FieldModel is
    // Category::ProblemSelection and so is an *error* in a dict, and there is no
    // pybind11 class for a FieldModel to hand over instead. So a coupled restart
    // file is refused rather than silently read with psi landing in the last
    // nField entries of a vector that has no field block -- which is a length
    // mismatch reported as an nVars/nAux/nScalars disagreement, three names none
    // of which is the problem.
    if (nField_file != 0)
      throw std::runtime_error(
          "This restart file was written by a run with a field model (" +
          std::to_string(nField_file) +
          " field unknowns), and Runner cannot attach one: FieldModel names a "
          "registered model and is a config-file key. Resume it with the MaNTA "
          "binary.");

    // Make sure degrees of freedom are consistent with restart file
    const Index nCells = grid->getNCells();
    const Index nDOF = pProblem->getNumVars() * 3 * nCells * (k + 1) +
                       pProblem->getNumVars() * (nCells + 1) +
                       pProblem->getNumScalars() +
                       pProblem->getNumAux() * nCells * (k + 1);

    if (nDOF_file != nDOF)
      throw std::invalid_argument(
          "nVars/nAux/nScalars in restart file inconsistent with physics case");

    pProblem->setRestartValues(Y, dYdt, *grid, k);
  }

  system = std::make_unique<SystemSolver>(*grid, k, pProblem.get());

  applySolverConfig(cfg, *system);
  if (cfg.solveAdjoint)
    system->setAdjointProblem(adjoint.get());

  // run_ss() arms steady-state termination itself, so it needs the value
  // whether or not the key was present. applySolverConfig has already armed it
  // when it *was* present, and run() clears that with a warning -- which is the
  // behaviour this surface has always had.
  steady_state_tolerance = cfg.SteadyStateTolerance.value_or(1e-3);

  // run() with no argument uses the configured end time; run(tFinal) overrides.
  configured_t_final = cfg.t_final;

  configured = true;
  logmsg<LOG_LEVEL::INFO>("Configuration done.");
}

void PyRunner::run(double tFinal) {
  if (!configured) {
    throw std::runtime_error(
        "Error: Runner must be configured before running solver.");
  }
  if (system->TerminateOnSteadyState) {
    logmsg<LOG_LEVEL::WARNING>(
        "\"run\" called but TerminateOnSteadyState is set to true. If you"
        " intended to run to steady-state, call \"run_ss\"");
    system->TerminateOnSteadyState = false;
  }
  system->runSolver(tFinal);

  std::println("Done.");
}

void PyRunner::run() {
  if (!configured) {
    throw std::runtime_error(
        "Error: Runner must be configured before running solver.");
  }
  if (!configured_t_final)
    throw std::runtime_error(
        "run() with no argument needs t_final in the configuration; "
        "pass run(tFinal) instead.");
  run(*configured_t_final);
}

void PyRunner::run_ss() {
  if (!configured) {
    throw std::runtime_error(
        "Error: Runner must be configured before running solver.");
  }
  system->setSteadyStateTolerance(steady_state_tolerance);
  system->runSolver(0);

  std::println("Done.");
}

bool PyRunner::wasRejected() const {
  if (!configured)
    throw std::runtime_error(
        "Error: Runner must be configured before asking about the dG/dt gate.");
  return system->wasRejected();
}

Vector PyRunner::lastDGdt() const {
  if (!configured)
    throw std::runtime_error(
        "Error: Runner must be configured before asking about the dG/dt gate.");
  return system->lastDGdt();
}

Vector PyRunner::G(void) {
  if (!configured)
    throw std::runtime_error(
        "Error: Runner must be configured before evaluating G.");

  // The objective without the gradient, for a driver that only needs G: a
  // finite-difference reference, a line search, a gradient-free optimiser.
  //
  // The saving is in the *run*, not here. SystemSolver::integrate calls
  // runAdjointSolve() whenever solveAdjoint is set, so with solveAdjoint = True
  // the gradients are already computed by the time the run returns and
  // getAdjointGradients() merely reads G_p. Configure with solveAdjoint = False
  // and the run skips the adjoint solve entirely -- and this is then the way to
  // get the objective out.
  //
  // Which means G() has to be able to work without a configured adjoint. The
  // AdjointProblem is what *defines* G, so build one on demand.
  if (adjoint == nullptr && objectiveOnlyAdjoint == nullptr)
    objectiveOnlyAdjoint = pProblem->createAdjointProblem();

  // Deliberately not handed to the SystemSolver via setAdjointProblem: that
  // would make getAdjointGradients() pass its null check and hand back a G_p
  // that was never computed.
  AdjointProblem *ap =
      adjoint != nullptr ? adjoint.get() : objectiveOnlyAdjoint.get();

  // GFn reads yJac, which holds the initial condition from initialize() and the
  // final solution after a run. So this does not run the solver -- call run() or
  // run_ss() first, exactly as for getAdjointGradients().
  Vector Gout(ap->getNg());
  for (Index i = 0; i < ap->getNg(); i++)
    Gout(i) = ap->GFn(i, system->yJac);

  return Gout;
}

py::tuple PyRunner::getAdjointGradients(void) {
  if (adjoint == nullptr)
    throw std::runtime_error(
        "\"getAdjointGradients\" called but adjoint problem not set");

  // system->runAdjointSolve();

  auto np_internal = adjoint->getNpInternal();

  Matrix G_p = system->G_p.leftCols(np_internal);

  // Create output to pass back to Python
  using namespace pybind11::literals;
  py::dict gp("G_p"_a = G_p);
  if (adjoint->getNpBoundary() > 0) {
    Matrix G_p_boundary =
        system->G_p.middleCols(np_internal, adjoint->getNp() - np_internal);
    gp["G_p_boundary"] = G_p_boundary;
  }

  Vector G(adjoint->getNg());
  if (system->isSuperconvergent()) {
    // G_p above is the gradient of the u*-based objective, so the value reported
    // alongside it has to be that same objective -- otherwise a finite-difference
    // check compares the derivative of one functional against differences of
    // another.
    system->postprocessor->computeUStar(system->yJac);
    for (Index i = 0; i < adjoint->getNg(); i++)
      G(i) = adjoint->GFn(i, system->yJac, *system->postprocessor);
  } else {
    for (Index i = 0; i < adjoint->getNg(); i++)
      G(i) = adjoint->GFn(i, system->yJac);
  }

  return py::make_tuple(G, gp);
}

Vector
PyRunner::getSolution(Index var,
                      std::optional<std::vector<Position>> const &points) {
  if (points) {
    Vector sol(points.value().size());

    for (size_t i = 0; i < points.value().size(); i++) {
      const auto &p = points.value()[i];

      if (p < grid->lowerBoundary() || p > grid->upperBoundary())
        throw std::out_of_range("Requested point outside of grid boundaries");

      sol(i) = system->yJac.u(var)(p);
    }
    return sol;
  } else {
    // Read yJac, not y. `y` is a non-owning DGSoln view over the N_Vector that
    // runSolver allocates and then destroys before returning, so sampling it
    // after a run reads freed memory -- and it disagreed with the branch above,
    // which has always used yJac. yJac is owned by the SystemSolver (yJacMem)
    // and stays valid.
    //
    // yJac holds the state as of the last Jacobian evaluation rather than the
    // final step, so both branches can lag the very last correction slightly;
    // that is pre-existing and applies equally to getAdjointGradients.
    const auto points = system->yJac.getPoints();
    Vector sol(points.size());
    for (size_t i = 0; i < points.size(); i++) {
      const auto &p = points[i];

      if (p < grid->lowerBoundary() || p > grid->upperBoundary())
        throw std::out_of_range("Requested point outside of grid boundaries");

      sol(i) = system->yJac.u(var)(p);
    }
    return sol;
  }
}

Vector PyRunner::getPostprocessedSolution(
    Index var, std::optional<std::vector<Position>> const &points) {
  Postprocessor const *pp = system->getPostprocessor();
  if (pp == nullptr)
    throw std::runtime_error("No postprocessed solution is available: it "
                             "requires Polynomial_degree >= 1 and a solver that "
                             "has been run at least once");

  // computeUStar is what fills the reconstruction, and it is driven by output
  // writing -- so it has already run against yJac's contents only if this run
  // wrote output. Recompute from yJac here so the answer does not depend on
  // whether WriteOutput was set. Same yJac-vs-final-step caveat as getSolution.
  system->postprocessor->computeUStar(system->yJac);

  const std::vector<Position> xs =
      points ? points.value() : system->yJac.getPoints();

  Vector sol(xs.size());
  for (size_t i = 0; i < xs.size(); i++) {
    const auto &p = xs[i];
    if (p < grid->lowerBoundary() || p > grid->upperBoundary())
      throw std::out_of_range("Requested point outside of grid boundaries");
    sol(i) = pp->uStar(var)(p);
  }
  return sol;
}
