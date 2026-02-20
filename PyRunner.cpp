#include "PyRunner.hpp"
#include <string>

// Load restart data into vectors
int LoadFromFile(netCDF::NcFile &restart_file, std::vector<double> &Y, std::vector<double> &dYdt);

template <typename T>
T visitAndReturnValue(std::string key, const py::dict &d)
{
    if (d.contains(key))
    {
        try
        {
            return d[key.c_str()].cast<T>();
        }
        catch (const std::exception &e)
        {
            std::cerr << "The following error occured while trying to get the value of key: " << key << " from config:" << std::endl
                      << e.what() << std::endl;

            throw;
        }
    }
    else
    {
        try
        {
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
    // Set stored problem to null to allow reconfiguration after object creation
    system = nullptr;
    grid = nullptr;
    adjoint = nullptr;

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

    bool isRestarting = visitAndReturnValue<bool>("restart", config);
    netCDF::NcFile restart_file;
    std::string fname = visitAndReturnValue<std::string>("OutputFilename", config);
    if (isRestarting)
    {
        std::string fbase = std::filesystem::path(fname).stem();
        std::string fileName = visitAndReturnValue<std::string>("RestartFile", config);
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

    // Grid *grid;
    unsigned int k = 1;
    if (!isRestarting)
    {
        // Solver parameters
        double lBound, uBound, lowerBoundaryFraction, upperBoundaryFraction;
        bool highGridBoundary;
        int nCells;

        k = visitAndReturnValue<unsigned int>("Polynomial_degree", config);

        highGridBoundary = visitAndReturnValue<bool>("High_Grid_Boundary", config);
        lBound = visitAndReturnValue<double>("Lower_boundary", config);
        uBound = visitAndReturnValue<double>("Upper_boundary", config);

        lowerBoundaryFraction = visitAndReturnValue<double>("Lower_Boundary_Fraction", config);
        upperBoundaryFraction = visitAndReturnValue<double>("Upper_Boundary_Fraction", config);

        nCells = visitAndReturnValue<int>("Grid_size", config);

        grid = std::make_unique<Grid>(lBound, uBound, nCells, highGridBoundary, lowerBoundaryFraction, upperBoundaryFraction);
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

    bool solveAdjoint = visitAndReturnValue<bool>("solveAdjoint", config);
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

    system = std::make_unique<SystemSolver>(*grid, k, pProblem.get(), adjoint.get());

    double dt = visitAndReturnValue<double>("delta_t", config);
    std::vector<double> atol = visitAndReturnValue<std::vector<double>>("Absolute_tolerance", config);
    double rtol = visitAndReturnValue<double>("Relative_tolerance", config);
    double tau = visitAndReturnValue<double>("tau", config);
    double tZero = visitAndReturnValue<double>("tZero", config);
    double dt_min = visitAndReturnValue<double>("MinStepSize", config);
    int nOutput = visitAndReturnValue<int>("OutputPoints", config);

    tFinal = visitAndReturnValue<double>("tFinal", config);

    system->setOutputCadence(dt);
    system->setTolerances(atol, rtol);
    system->setTau(tau);
    system->setInitialTime(tZero);
    system->setInputFile(fname);
    system->setSolveAdjoint(solveAdjoint);

    system->setNOutput(nOutput);
    system->setMinStepSize(dt_min);
}

int PyRunner::run(void)
{

    system->runSolver(tFinal);

    std::cout << "Done." << std::endl;
    return 0;
}
