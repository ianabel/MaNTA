#include "PyRunner.hpp"
#include "Logging.hpp"
#include "DegreeAdaptation.hpp"
#include "PyConfigSource.hpp"
#include "PyToml.hpp"
#include <algorithm>
#include <pybind11/eigen.h>
#include <string>
#include <print>
// Load restart data into vectors
int LoadFromFile(netCDF::NcFile &restart_file, std::vector<double> &Y,
                 std::vector<double> &dYdt);

PyRunner::PyRunner(std::string physicsCase) : caseName(std::move(physicsCase)) {
  // Rejected here, at the point the name was written, rather than at the first
  // configure(): the registry is populated by static initialisation, so
  // whether a name is available is already settled by the time any Python runs
  // -- barring a later registerPhysicsCase or plugin load, which a caller does
  // before building a Runner because there is no other useful order.
  auto const names = PhysicsCases::RegisteredNames();
  if (std::find(names.begin(), names.end(), caseName) == names.end()) {
    std::string available;
    for (auto const &n : names)
      available += (available.empty() ? "" : ", ") + n;
    throw std::invalid_argument(
        "There is no physics case named '" + caseName +
        "'. Available cases: " +
        (available.empty() ? "(none -- no physics case object files are linked "
                             "in)"
                           : available) +
        ". manta.physics_cases() lists them, manta.load_physics_plugin() adds "
        "a case built out of tree, and manta.Runner(system) takes a Python "
        "case as an object instead.");
  }
}

// Build the C++ case named at construction, from the same dict the solver's
// configuration came out of. Called by configure() once the grid exists.
//
// Rebuilt on every configure() rather than once, and that is the point of doing
// it here at all: a C++ case reads its table in its constructor, so a driver
// sweeping a physics parameter -- which is the reason to want a C++ case under
// a Python optimiser -- changes the dict and reconfigures. Instantiating once
// would silently pin the first call's parameters.
//
// Everything derived from the old case goes first. The AdjointProblem an
// autodiff case hands out holds a raw pointer back to it
// (AutodiffAdjointProblem::PhysicsProblem), so an adjoint outliving its problem
// dangles; `system` was already nulled by configure() before this runs.
void PyRunner::instantiatePhysicsCase(const py::dict &config) {
  adjoint = nullptr;
  objectiveOnlyAdjoint = nullptr;
  pProblem = nullptr;

  try {
    pProblem = PhysicsCases::InstantiateProblem(
        caseName, physicsConfigFromDict(config), *grid);
  } catch (std::invalid_argument const &e) {
    // configure() has raised RuntimeError for a bad configuration since it
    // existed, and a case rejecting its own table -- "there should be a
    // [DiffusionProblem] section" -- is exactly that. Translated for the same
    // reason the loadSolverConfig call below it is.
    throw std::runtime_error(e.what());
  }
}

void PyRunner::configure(const py::dict &config) {
  if (!pProblem && caseName.empty())
    throw std::runtime_error("Transport system not set. Please set transport "
                             "system before configuring solver.");
  // Reconfiguring abandons a sliced solve that is still running. ~SystemSolver
  // does not free the SUNDIALS objects -- only destroySundials() does -- so
  // dropping the solver without this leaks every one of them, and silently.
  abandonSlices();

  // Set stored problem to null to allow reconfiguration after object creation
  system = nullptr;
  grid = nullptr;
  // ...and with the grid goes anything holding it. The restart DGSolns took a
  // reference to *grid until they started copying it, and `restarting` is
  // sticky, so a configuration that does not ask for a restart has to say so
  // rather than inherit the last one. Cleared here, before the config is even
  // parsed, so it holds on the throwing paths too.
  //
  // A C++ case is rebuilt below rather than cleared, so there may be nothing
  // here to clear on the first call.
  if (pProblem)
    pProblem->clearRestart();

  // Every key this accepts is declared in ConfigSchema.cpp, the same table
  // runManta reads. This function used to carry its own `params` list and its
  // own reader, which is how the dict came to want `tZero` where a config file
  // wanted `t_initial`, and to default Absolute_tolerance to a different
  // number.
  DictConfigSource source(config);
  cfg = [&] {
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

  k = 1;
  grid = makeGrid(cfg, cfg.restart ? &restart_file : nullptr, k);

  if (!caseName.empty())
    instantiatePhysicsCase(config);

  if (cfg.solveAdjoint)
    adjoint = pProblem->createAdjointProblem();

  if (cfg.restart) {
    std::vector<double> Y, dYdt;
    Index nDOF_file = LoadFromFile(restart_file, Y, dYdt);

    // Make sure degrees of freedom are consistent with restart file
    const Index nCells = grid->getNCells();
    const Index nDOF = pProblem->getNumVars() * 3 * nCells * (k + 1) +
                       pProblem->getNumVars() * (nCells + 1) +
                       pProblem->getNumScalars() +
                       pProblem->getNumAux() * nCells * (k + 1);

    if (nDOF_file != nDOF)
      throw std::invalid_argument(
          "nVars/nAux/nScalars in restart file inconsistent with physics case");

    // The file's own degree, which is what its DOF are laid out at.
    pProblem->setRestartValues(Y, dYdt, *grid, k);

    // The run's degree, which may differ. setInitialConditions projects across
    // the difference; equal degrees keep the copy path.
    k = restartRunOrder(cfg, k);
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

// Replace `system` with the one an adaptive run settles on.
//
// The solver being destroyed is the argument's own, so it goes before the next
// is built -- runAdaptiveDegree does that internally, and this only has to not
// keep the old one alive alongside the new. `grid`, `adjoint` and `pProblem`
// are untouched; the driver re-attaches the adjoint problem and replays the
// configuration against every level it builds.
void PyRunner::adaptDegree(double tFinal) {
  system.reset();
  system = runAdaptiveDegree(cfg, *pProblem, adjoint.get(), *grid, k, tFinal);
}

void PyRunner::run(double tFinal) {
  if (!configured) {
    throw std::runtime_error(
        "Error: Runner must be configured before running solver.");
  }
  if (cfg.DegreeAdaptation) {
    // run() means "integrate the transient", and degree adaptation is a
    // steady-only feature -- so this is refused rather than quietly turned into
    // a steady solve. Note what run() does two lines below when the config
    // carries SteadyStateTolerance: it *clears* the flag and warns, because the
    // caller asked for the path rather than the endpoint. Silently doing the
    // opposite here would contradict it.
    throw std::runtime_error(
        "DegreeAdaptation is for steady solves; call run_ss() rather than "
        "run(). Adapting the degree across a transient would restart each "
        "level from the previous one's final state.");
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
  if (cfg.DegreeAdaptation) {
    // run_ss() arms steady-state termination whether or not the key was
    // present, so the tolerance has to reach every level rather than the one
    // configure() built. Writing it into the config the driver replays is what
    // does that -- setting it on `system` here would be lost with that solver.
    cfg.SteadyStateTolerance = steady_state_tolerance;
    adaptDegree(0);
    std::println("Done.");
    return;
  }
  system->setSteadyStateTolerance(steady_state_tolerance);
  system->runSolver(0);

  std::println("Done.");
}

PyRunner::~PyRunner() {
  // A destructor cannot let an exception out, and destroySundials() is not
  // expected to throw -- but "not expected to" is not a guarantee worth
  // terminating the interpreter over.
  try {
    abandonSlices();
  } catch (...) {
  }
}

SystemSolver::SteadyOutcome PyRunner::runSlice(bool resume, bool estimate) {
  system->setEstimateObjectiveOnFinish(estimate);
  try {
    if (resume)
      system->continueSteadyState();
    else
      system->solveSteadyState();
  } catch (...) {
    // OutOfSteps is not a failure: the step budget is spent, the last accepted
    // iterate is in Y and the pseudo-time step SER climbed to is still on the
    // solver, so continue_steady() picks up both. Returning it is what lets a
    // driver tell it apart from a dead solve without reading the message.
    if (system->lastSteadyOutcome() != SystemSolver::SteadyOutcome::OutOfSteps) {
      // The same bargain integrate() makes for a failed steady solve: write the
      // last state reached -- which is exactly the run whose state is worth
      // looking at -- close the files, then let the caller hear about it. No
      // adjoint solve, because there is no converged state to define it at.
      try {
        system->writeSteadyState();
        system->closeOutputFiles();
      } catch (...) {
        // A failure while reporting a failure. The original is the one worth
        // propagating, so this one is dropped rather than replacing it.
      }
      abandonSlices();
      throw;
    }
  }

  // Leave the state where getSolution() reads it. finishRun() does this at the
  // end of a run; without it here, a driver looking between slices would be
  // handed the initial condition -- silently, since yJac is always a valid
  // state, just not this one.
  system->captureState();
  return system->lastSteadyOutcome();
}

SystemSolver::SteadyOutcome PyRunner::start_steady(bool estimate) {
  if (!configured)
    throw std::runtime_error(
        "Error: Runner must be configured before running solver.");
  if (slicing)
    throw std::runtime_error(
        "start_steady() called while a sliced solve is already running. Use "
        "continue_steady() to carry on, or finish_steady() to end it.");
  if (cfg.DegreeAdaptation)
    throw std::runtime_error(
        "DegreeAdaptation cannot be combined with a sliced steady solve: "
        "adapting the degree replaces the solver, and a slice loop holds the "
        "state of the one it started on. Run one or the other.");

  system->setSteadyStateTolerance(steady_state_tolerance);
  system->initialize();
  slicing = true;
  return runSlice(false, estimate);
}

SystemSolver::SteadyOutcome PyRunner::continue_steady(bool estimate) {
  if (!slicing)
    throw std::runtime_error(
        "continue_steady() called with no sliced solve running. Call "
        "start_steady() first.");
  return runSlice(true, estimate);
}

void PyRunner::finish_steady(void) {
  if (!slicing)
    throw std::runtime_error(
        "finish_steady() called with no sliced solve running.");
  slicing = false;
  system->writeSteadyState();
  system->finishRun();
  system->destroySundials();
  std::println("Done.");
}

void PyRunner::abandonSlices(void) {
  if (!slicing)
    return;
  slicing = false;
  if (system != nullptr)
    system->destroySundials();
}

py::dict PyRunner::steadyStats(void) const {
  using namespace pybind11::literals;
  const auto s = system->lastSteadyStats();
  return py::dict(
      "outcome"_a = system->lastSteadyOutcome(), "steps"_a = s.steps,
      "rejected"_a = s.rejected, "residual_norm"_a = s.residualNorm,
      "newton_iterations"_a = s.newtonIters, "residual_evaluations"_a = s.residualEvals,
      "jacobian_builds"_a = s.jacBuilds, "jacobian_solves"_a = s.jacSolves,
      "pseudo_transient_step"_a = system->getPseudoTransientStep());
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

py::dict PyRunner::objectiveEstimate(void) const {
  using namespace pybind11::literals;
  const auto e = system->lastObjectiveEstimate();
  if (!e.valid)
    return py::dict();
  return py::dict("value"_a = e.value, "corrected"_a = e.corrected,
                  "uncertainty"_a = e.uncertainty);
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
PyRunner::getDerivative(Index var,
                        std::optional<std::vector<Position>> const &points) {
  // Same shape as getSolution below, reading yJac.q rather than yJac.u -- and
  // for the same reason: q is a DGApprox over the solver's own basis, so it
  // evaluates at any x directly. The alternative a caller had before this
  // existed was to fit something to getSolution's samples and differentiate
  // that, which is a different function.
  const std::vector<Position> xs =
      points ? points.value() : system->yJac.getPoints();

  Vector sol(xs.size());
  for (size_t i = 0; i < xs.size(); i++) {
    const auto &p = xs[i];
    if (p < grid->lowerBoundary() || p > grid->upperBoundary())
      throw std::out_of_range("Requested point outside of grid boundaries");
    sol(i) = system->yJac.q(var)(p);
  }
  return sol;
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
