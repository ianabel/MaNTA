// #include <ida/ida.h>                  /* prototypes for IDA fcts., consts.    */
// #include <nvector/nvector_serial.h>   /* access to serial N_Vector            */
// #include <sunmatrix/sunmatrix_band.h> /* access to band SUNMatrix             */
// #include <sunlinsol/sunlinsol_band.h> /* access to band SUNLinearSolver       */
// #include <sundials/sundials_types.h>  /* definition of type sunrealtype          */
#include <pybind11/eigen.h>

// #include "SunLinSolWrapper.hpp"
// #include "SunMatrixWrapper.hpp"
#include "PyRunner.hpp"
#include "ErrorChecker.hpp"

#include <string>
#include <type_traits>

// Load restart data into vectors
int LoadFromFile(netCDF::NcFile &restart_file, std::vector<double> &Y, std::vector<double> &dYdt);

// Parameters required by "configure" function to be passed to SystemSolver
static const map_t params = {{"restart", Parameter<bool>{.required = false, ._default = false}},
                             //
                             {"RestartFile", Parameter<std::string>{.required = false, ._default = ""}},
                             //
                             {"High_Grid_Boundary", Parameter<bool>{.required = false, ._default = false}},
                             //
                             {"Lower_Boundary_Fraction", Parameter<double>{.required = false, ._default = 0.2}},
                             //
                             {"Upper_Boundary_Fraction", Parameter<double>{.required = false, ._default = 0.2}},
                             //
                             {"Polynomial_degree", Parameter<unsigned int>{.required = true}},
                             //
                             {"Grid_size", Parameter<int>{.required = true}},
                             //
                             {"Grid_points", Parameter<std::vector<double>>{.required = false, ._default = {}}},
                             //
                             {"tau", Parameter<double>{.required = false, ._default = 1.0}},
                             //
                             {"delta_t", Parameter<double>{.required = true}},
                             //
                             {"tZero", Parameter<double>{.required = false, ._default = 0.0}},
                             //
                             {"Relative_tolerance", Parameter<double>{.required = false, ._default = 1e-3}},
                             //
                             {"Absolute_tolerance", Parameter<std::vector<double>>{.required = false, ._default = {1e-3}}},
                             //
                             {"MinStepSize", Parameter<double>{.required = false, ._default = 1e-7}},
                             //
                             {"OutputPoints", Parameter<int>{.required = false, ._default = 301}},
                             //
                             {"solveAdjoint", Parameter<bool>{.required = false, ._default = false}},
                             //
                             {"OutputFilename", Parameter<std::string>{.required = true}},
                             //
                             {"SteadyStateTolerance", Parameter<double>{.required = false, ._default = 1e-3}},
                             //
                             {"WriteOutput", Parameter<bool>{.required = false, ._default = true}},
                             //
                             {"useCalcIC", Parameter<bool>{.required = false, ._default = true}}};

template <typename T>
T getValueWithDefault(std::string key, const py::dict &d)
{
    if (d.contains(key))
    {
        try
        {
            return d[key.c_str()].cast<T>();
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("The following error occured while trying to get the value of key: " + key + " from config:\n" + e.what() + "\n");
        }
    }
    else
    {
        try
        {
            std::cerr << "INFO: Using default value for configuration option " << key << std::endl;
            return std::get<Parameter<T>>(params.at(key))._default;
        }
        catch (...)
        {
            throw std::runtime_error("Failed to retrieve default value for key: " + key + "; possible type mismatch.");
        }
    }
};

void PyRunner::configure(const py::dict &config)
{
    if (!pProblem)
        throw std::runtime_error("Transport system not set. Please set transport system before configuring solver.");
    // Set stored problem to null to allow reconfiguration after object creation
    system = nullptr;
    grid = nullptr;

    // Check if config contains required params
    std::string requiredParams = "";
    for (auto &[key, val] : params)
    {
        std::visit(
            [&](const auto &v)
            {
                if (v.required && !config.contains(key.c_str()))
                {
                    requiredParams += key + ", "; // throw std::runtime_error("Required parameter: " + key + " not contained in config.");
                }
            },
            val);
    }
    if (!requiredParams.empty())
        throw std::runtime_error("Required parameter(s): " + requiredParams + " not contained in config.");

    // Configure MaNTA
    bool isRestarting = getValueWithDefault<bool>("restart", config);
    netCDF::NcFile restart_file;
    std::string fname = getValueWithDefault<std::string>("OutputFilename", config);
    if (isRestarting)
    {
        std::string fbase = std::filesystem::path(fname).stem();
        std::string fileName = getValueWithDefault<std::string>("RestartFile", config);
        fileName = !fileName.empty() ? std::string(fileName) : fbase + ".restart.nc";
        try
        {
            restart_file.open(fileName, netCDF::NcFile::FileMode::read);
        }
        catch (...)
        {
            std::string msg = "Failed to open restart netCDF file at: " + std::string(std::filesystem::absolute(std::filesystem::path(fileName)));
            throw std::runtime_error(msg);
        }
    }

    unsigned int k = 1;
    if (!isRestarting)
    {

        k = getValueWithDefault<unsigned int>("Polynomial_degree", config);

        auto CellBoundaries = getValueWithDefault<std::vector<double>>("Grid_points", config);

        if (CellBoundaries.size() > 0)
        {
            std::cerr << "INFO: Creating grid with cell boundaries at x = [";
            for (const auto &p : CellBoundaries)
                std::cerr << p << ", ";
            std::cerr << "]" << std::endl;

            grid = std::make_unique<Grid>(CellBoundaries);
        }
        else
        {
            // Solver parameters
            double lBound, uBound, lowerBoundaryFraction, upperBoundaryFraction;
            bool highGridBoundary;
            int nCells;
            highGridBoundary = getValueWithDefault<bool>("High_Grid_Boundary", config);
            lBound = getValueWithDefault<double>("Lower_boundary", config);
            uBound = getValueWithDefault<double>("Upper_boundary", config);

            lowerBoundaryFraction = getValueWithDefault<double>("Lower_Boundary_Fraction", config);
            upperBoundaryFraction = getValueWithDefault<double>("Upper_Boundary_Fraction", config);

            nCells = getValueWithDefault<int>("Grid_size", config);

            std::cerr << "INFO: Creating grid with " << nCells << " cells from x = " << lBound << " to x = " << uBound << std::endl;

            grid = std::make_unique<Grid>(lBound, uBound, nCells, highGridBoundary, lowerBoundaryFraction, upperBoundaryFraction);
        }
    }
    else
    {
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

    if (isRestarting)
    {
        std::vector<double> Y, dYdt;
        Index nDOF_file = LoadFromFile(restart_file, Y, dYdt);

        // Make sure degrees of freedom are consistent with restart file
        const Index nCells = grid->getNCells();
        const Index nDOF = pProblem->getNumVars() * 3 * nCells * (k + 1) + pProblem->getNumVars() * (nCells + 1) + pProblem->getNumScalars() + pProblem->getNumAux() * nCells * (k + 1);

        if (nDOF_file != nDOF)
            throw std::invalid_argument("nVars/nAux/nScalars in restart file inconsistent with physics case");

        pProblem->setRestartValues(Y, dYdt, *grid, k);
    }

    system = std::make_unique<SystemSolver>(*grid, k, pProblem.get());

    double dt = getValueWithDefault<double>("delta_t", config);
    std::vector<double> atol = getValueWithDefault<std::vector<double>>("Absolute_tolerance", config);
    double rtol = getValueWithDefault<double>("Relative_tolerance", config);
    double tau = getValueWithDefault<double>("tau", config);
    double tZero = getValueWithDefault<double>("tZero", config);
    double dt_min = getValueWithDefault<double>("MinStepSize", config);
    int nOutput = getValueWithDefault<int>("OutputPoints", config);

    steady_state_tolerance = getValueWithDefault<double>("SteadyStateTolerance", config);

    system->setOutputCadence(dt);
    system->setTolerances(atol, rtol);
    system->setTau(tau);
    system->setInitialTime(tZero);
    system->setInputFile(fname);
    system->setSolveAdjoint(solveAdjoint);
    if (solveAdjoint)
        system->setAdjointProblem(adjoint.get());

    system->setNOutput(nOutput);
    system->setMinStepSize(dt_min);

    system->setNOutput(nOutput);
    system->setMinStepSize(dt_min);
    system->setUseCalcIC(getValueWithDefault<bool>("useCalcIC", config));

    bool writeOutput = getValueWithDefault<bool>("WriteOutput", config);

    configured = true;
    std::cerr << "Configuration done." << std::endl;
}

void PyRunner::run(double tFinal)
{
    if (!configured)
    {
        throw std::runtime_error("Error: Runner must be configured before running solver.");
    }
    if (system->TerminateOnSteadyState)
    {
        std::cerr << "INFO: \"run\" called but TerminateOnSteadyState is set to true. If you intended to run to steady state please call \"run_ss\". Running to passed tFinal" << std::endl;
        system->TerminateOnSteadyState = false;
    }
    system->runSolver(tFinal);

    std::cout << "Done." << std::endl;
}

void PyRunner::run_ss()
{
    if (!configured)
    {
        throw std::runtime_error("Error: Runner must be configured before running solver.");
    }
    system->setSteadyStateTolerance(steady_state_tolerance);
    system->runSolver(0);

    std::cout << "Done." << std::endl;
}

py::tuple PyRunner::getAdjointGradients(void)
{
    if (adjoint == nullptr)
        throw std::runtime_error("\"getAdjointGradients\" called but adjoint problem not set");

    // system->runAdjointSolve();

    auto np_internal = adjoint->getNpInternal();

    Matrix G_p = system->G_p(Eigen::all, Eigen::seq(0, np_internal - 1));

    // Create output to pass back to Python
    using namespace pybind11::literals;
    py::dict gp("G_p"_a = G_p);
    if (adjoint->getNpBoundary() > 0)
    {
        Matrix G_p_boundary = system->G_p(Eigen::all, Eigen::seq(np_internal, adjoint->getNp() - 1));
        gp["G_p_boundary"] = G_p_boundary;
    }

    Vector G(adjoint->getNg());
    for (Index i = 0; i < adjoint->getNg(); i++)
        G(i) = adjoint->GFn(i, system->yJac);

    return py::make_tuple(G, gp);
}

Vector PyRunner::getSolution(Index var, std::optional<std::vector<Position>> const &points)
{
    if (points)
    {
        Vector sol(points.value().size());

        for (size_t i = 0; i < points.value().size(); i++)
        {
            const auto &p = points.value()[i];

            if (p < grid->lowerBoundary() || p > grid->upperBoundary())
                throw std::out_of_range("Requested point outside of grid boundaries");

            sol(i) = system->yJac.u(var)(p);
        }
        return sol;
    }
    else
    {
        const auto points = system->y.getPoints();
        Vector sol(points.size());
        for (size_t i = 0; i < points.size(); i++)
        {
            const auto &p = points[i];

            if (p < grid->lowerBoundary() || p > grid->upperBoundary())
                throw std::out_of_range("Requested point outside of grid boundaries");

            sol(i) = system->y.u(var)(p);
        }
        return sol;
    }
}