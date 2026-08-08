#include "PyRunner.hpp"
#include "Logging.hpp"
#include <pybind11/eigen.h>
#include <string>
#include <print>
// Load restart data into vectors
int LoadFromFile(netCDF::NcFile &restart_file, std::vector<double> &Y,
                 std::vector<double> &dYdt);

// Parameters required by "configure" function to be passed to SystemSolver
static const map_t params = {
    {"restart", Parameter<bool>{.required = false, ._default = false}},
    //
    {"RestartFile", Parameter<std::string>{.required = false, ._default = ""}},
    //
    {"High_Grid_Boundary",
     Parameter<bool>{.required = false, ._default = false}},
    //
    {"Lower_Boundary_Fraction",
     Parameter<double>{.required = false, ._default = 0.2}},
    //
    {"Upper_Boundary_Fraction",
     Parameter<double>{.required = false, ._default = 0.2}},
    //
    {"Polynomial_degree", Parameter<unsigned int>{.required = true}},
    //
    {"Grid_size", Parameter<int>{.required = true}},
    //
    {"Grid_points",
     Parameter<std::vector<double>>{.required = false, ._default = {}}},
    //
    // Needed whenever Grid_points is not supplied. They cannot be marked
    // required here because the Grid_points path legitimately omits them, so
    // configure() checks for them explicitly on the branch that uses them.
    // Without these entries getValueWithDefault fell through to params.at(),
    // which threw out_of_range and was reported as the thoroughly misleading
    // "Failed to retrieve default value for key: Lower_boundary; possible type
    // mismatch."
    {"Lower_boundary", Parameter<double>{.required = false, ._default = 0.0}},
    //
    {"Upper_boundary", Parameter<double>{.required = false, ._default = 1.0}},
    //
    {"tau", Parameter<double>{.required = false, ._default = 1.0}},
    //
    {"delta_t", Parameter<double>{.required = true}},
    //
    {"tZero", Parameter<double>{.required = false, ._default = 0.0}},
    //
    {"Relative_tolerance",
     Parameter<double>{.required = false, ._default = 1e-3}},
    //
    {"Absolute_tolerance",
     Parameter<std::vector<double>>{.required = false, ._default = {1e-3}}},
    //
    {"MinStepSize", Parameter<double>{.required = false, ._default = 1e-7}},
    //
    {"OutputPoints", Parameter<int>{.required = false, ._default = 301}},
    //
    {"solveAdjoint", Parameter<bool>{.required = false, ._default = false}},
    //
    {"OutputFilename", Parameter<std::string>{.required = true}},
    //
    {"SteadyStateTolerance",
     Parameter<double>{.required = false, ._default = 1e-3}},
    //
    {"WriteOutput", Parameter<bool>{.required = false, ._default = true}},
    //
    // netCDF is the default output; the plain-text .dat files are opt-in.
    // WriteDatFile controls <stem>.dat, WriteDebugDatFiles the .dydt.dat and
    // .res.dat pair (which additionally need a PHYSICS_DEBUG build).
    {"WriteDatFile", Parameter<bool>{.required = false, ._default = false}},
    //
    {"WriteDebugDatFiles",
     Parameter<bool>{.required = false, ._default = false}},
    //
    {"zeroFlux", Parameter<bool>{.required = false, ._default = false}},
    //
    {"initialTimestep", Parameter<double>{.required = false, ._default = 0.0}},
    //
    // Let IDA grow the step by up to 10x rather than 2x between steps. Useful
    // for an optimisation driver calling run_ss() in a loop, where the transient
    // is not the interesting part; off by default because it makes IDA more
    // likely to overshoot and retry.
    {"aggressiveTimesteps",
     Parameter<bool>{.required = false, ._default = false}}};

template <typename T>
T getValueWithDefault(std::string_view key, const py::dict &d) {
  if (d.contains(key)) {
    try {
      auto val = d[key.data()].cast<T>();
      logmsg<LOG_LEVEL::INFO>("Using value {} for parameter {}", val, key);
      return val;
    } catch (const std::exception &e) {
      throw std::runtime_error(
          "The following error occured while trying to get the value of key: " +
          std::string(key) + " from config:\n" + e.what() + "\n");
    }
  } else {
    try {
      auto val = std::get<Parameter<T>>(params.at(key))._default;
      logmsg<LOG_LEVEL::INFO>("Using default value {} for parameter {}", val,
                              key);
      return val;
    } catch (...) {
      throw std::runtime_error("Failed to retrieve default value for key: " +
                               std::string(key) + "; possible type mismatch.");
    }
  }
};

void PyRunner::configure(const py::dict &config) {
  if (!pProblem)
    throw std::runtime_error("Transport system not set. Please set transport "
                             "system before configuring solver.");
  // Set stored problem to null to allow reconfiguration after object creation
  system = nullptr;
  grid = nullptr;

  // Check if config contains required params
  std::string requiredParams = "";
  for (auto &[key, val] : params) {
    std::visit(
        [&](const auto &v) {
          if (v.required && !config.contains(key.data())) {
            requiredParams +=
                std::string(key) +
                ", "; // throw std::runtime_error("Required parameter: " + key +
                      // " not contained in config.");
          }
        },
        val);
  }
  if (!requiredParams.empty())
    throw std::runtime_error("Required parameter(s): " + requiredParams +
                             " not contained in config.");

  // Configure MaNTA
  bool isRestarting = getValueWithDefault<bool>("restart", config);
  netCDF::NcFile restart_file;
  std::string fname =
      getValueWithDefault<std::string>("OutputFilename", config);
  if (isRestarting) {
    std::string fbase = std::filesystem::path(fname).stem();
    std::string fileName =
        getValueWithDefault<std::string>("RestartFile", config);
    fileName =
        !fileName.empty() ? std::string(fileName) : fbase + ".restart.nc";
    try {
      restart_file.open(fileName, netCDF::NcFile::FileMode::read);
    } catch (...) {
      std::string msg = "Failed to open restart netCDF file at: " +
                        std::string(std::filesystem::absolute(
                            std::filesystem::path(fileName)));
      throw std::runtime_error(msg);
    }
  }

  unsigned int k = 1;
  if (!isRestarting) {

    k = getValueWithDefault<unsigned int>("Polynomial_degree", config);

    auto CellBoundaries =
        getValueWithDefault<std::vector<double>>("Grid_points", config);

    if (CellBoundaries.size() > 0) {

      grid = std::make_unique<Grid>(CellBoundaries);
    } else {
      // Solver parameters
      double lBound, uBound, lowerBoundaryFraction, upperBoundaryFraction;
      bool highGridBoundary;
      int nCells;
      highGridBoundary =
          getValueWithDefault<bool>("High_Grid_Boundary", config);

      // Required on this branch, but not listed as required in `params`
      // because the Grid_points branch above does not need them -- so the
      // up-front required-parameter check cannot catch them. Say so clearly
      // rather than silently defaulting the domain to [0, 1].
      if (!config.contains("Lower_boundary") ||
          !config.contains("Upper_boundary"))
        throw std::runtime_error(
            "Required parameter(s): Lower_boundary, Upper_boundary must be "
            "given unless Grid_points is supplied.");

      lBound = getValueWithDefault<double>("Lower_boundary", config);
      uBound = getValueWithDefault<double>("Upper_boundary", config);

      lowerBoundaryFraction =
          getValueWithDefault<double>("Lower_Boundary_Fraction", config);
      upperBoundaryFraction =
          getValueWithDefault<double>("Upper_Boundary_Fraction", config);

      nCells = getValueWithDefault<int>("Grid_size", config);
      grid =
          std::make_unique<Grid>(lBound, uBound, nCells, highGridBoundary,
                                 lowerBoundaryFraction, upperBoundaryFraction);
    }
  } else {
    // Load grid from restart file
    netCDF::NcGroup GridGroup = restart_file.getGroup("Grid");
    auto nPoints = GridGroup.getDim("Index").getSize();
    std::vector<Position> CellBoundaries(nPoints);

    GridGroup.getVar("CellBoundaries").getVar(CellBoundaries.data());

    grid = std::make_unique<Grid>(CellBoundaries);

    GridGroup.getVar("PolyOrder").getVar(&k);
  }

  bool solveAdjoint = getValueWithDefault<bool>("solveAdjoint", config);
  if (solveAdjoint)
    adjoint = pProblem->createAdjointProblem();

  if (isRestarting) {
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

    pProblem->setRestartValues(Y, dYdt, *grid, k);
  }

  system = std::make_unique<SystemSolver>(*grid, k, pProblem.get());

  double dt = getValueWithDefault<double>("delta_t", config);
  std::vector<double> atol =
      getValueWithDefault<std::vector<double>>("Absolute_tolerance", config);
  double rtol = getValueWithDefault<double>("Relative_tolerance", config);
  double tau = getValueWithDefault<double>("tau", config);
  double tZero = getValueWithDefault<double>("tZero", config);
  double dt_min = getValueWithDefault<double>("MinStepSize", config);
  double dt0 = getValueWithDefault<double>("initialTimestep", config);
  int nOutput = getValueWithDefault<int>("OutputPoints", config);

  steady_state_tolerance =
      getValueWithDefault<double>("SteadyStateTolerance", config);

  system->setOutputCadence(dt);
  system->setTolerances(atol, rtol);
  system->setTau(tau);
  system->setInitialTime(tZero);
  system->setInitialTimestep(dt0);
  system->setInputFile(fname);
  system->setSolveAdjoint(solveAdjoint);
  if (solveAdjoint)
    system->setAdjointProblem(adjoint.get());

  system->setNOutput(nOutput);
  system->setMinStepSize(dt_min);

  system->setNOutput(nOutput);
  system->setMinStepSize(dt_min);
  system->setZeroFlux(getValueWithDefault<bool>("zeroFlux", config));
  system->setWriteDatFile(getValueWithDefault<bool>("WriteDatFile", config));
  system->setWriteDebugDatFiles(
      getValueWithDefault<bool>("WriteDebugDatFiles", config));
  system->setAggressiveTimesteps(
      getValueWithDefault<bool>("aggressiveTimesteps", config));

  bool writeOutput = getValueWithDefault<bool>("WriteOutput", config);

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

void PyRunner::run_ss() {
  if (!configured) {
    throw std::runtime_error(
        "Error: Runner must be configured before running solver.");
  }
  system->setSteadyStateTolerance(steady_state_tolerance);
  system->runSolver(0);

  std::println("Done.");
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

  Matrix G_p = system->G_p(Eigen::all, Eigen::seq(0, np_internal - 1));

  // Create output to pass back to Python
  using namespace pybind11::literals;
  py::dict gp("G_p"_a = G_p);
  if (adjoint->getNpBoundary() > 0) {
    Matrix G_p_boundary =
        system->G_p(Eigen::all, Eigen::seq(np_internal, adjoint->getNp() - 1));
    gp["G_p_boundary"] = G_p_boundary;
  }

  Vector G(adjoint->getNg());
  for (Index i = 0; i < adjoint->getNg(); i++)
    G(i) = adjoint->GFn(i, system->yJac);

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
