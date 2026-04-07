#include <ida/ida.h>                  /* prototypes for IDA fcts., consts.    */
#include <nvector/nvector_serial.h>   /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_band.h> /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h> /* access to band SUNLinearSolver       */
#include <sundials/sundials_types.h>  /* definition of type sunrealtype          */
#include <pybind11/eigen.h>

#include "SunLinSolWrapper.hpp"
#include "SunMatrixWrapper.hpp"
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
                             {"tau", Parameter<double>{.required = false, ._default = 1.0}},
                             //
                             {"delta_t", Parameter<double>{.required = true}},
                             //
                             {"tZero", Parameter<double>{.required = false, ._default = 0.0}},
                             //
                             {"Relative_tolerance", Parameter<double>{.required = false, ._default = 1e-3}},
                             //
                             {"Absolute_tolerance", Parameter<std::vector<double>>{.required = false, ._default = {1e-2}}},
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
                             {"WriteOutput", Parameter<bool>{.required = false, ._default = true}}};

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
        // Solver parameters
        double lBound, uBound, lowerBoundaryFraction, upperBoundaryFraction;
        bool highGridBoundary;
        int nCells;

        k = getValueWithDefault<unsigned int>("Polynomial_degree", config);

        highGridBoundary = getValueWithDefault<bool>("High_Grid_Boundary", config);
        lBound = getValueWithDefault<double>("Lower_boundary", config);
        uBound = getValueWithDefault<double>("Upper_boundary", config);

        lowerBoundaryFraction = getValueWithDefault<double>("Lower_Boundary_Fraction", config);
        upperBoundaryFraction = getValueWithDefault<double>("Upper_Boundary_Fraction", config);

        nCells = getValueWithDefault<int>("Grid_size", config);

        std::cerr << "INFO: Creating grid with " << nCells << " cells from x = " << lBound << " to x = " << uBound << std::endl;

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

    system->setNOutput(nOutput);
    system->setMinStepSize(dt_min);

    bool writeOutput = getValueWithDefault<bool>("WriteOutput", config);
    // Creation of solver function
    runner = system->makeSolver(LS, sunMat, IDA_mem, retval, Y, dYdt, constraints, id, res, absTolVec, tout, tret, writeOutput);

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
        std::cerr << "\"run\" called but TerminateOnStateState is set to true. If you intended to run to steady state please call \"run_ss\". Running to passed tFinal" << std::endl;
        system->TerminateOnSteadyState = false;
    }
    runner(tFinal);

    std::cout << "Done." << std::endl;
}

void PyRunner::run_ss()
{
    if (!configured)
    {
        throw std::runtime_error("Error: Runner must be configured before running solver.");
    }
    system->setSteadyStateTolerance(steady_state_tolerance);
    runner(0); // Final time doesn't matter so just pass 0

    std::cout << "Done." << std::endl;
}

py::tuple PyRunner::runAdjointSolve(void)
{
    if (adjoint == nullptr)
        throw std::runtime_error("\"runAdjointSolve\" but adjoint problem not set");
    system->runAdjointSolve();

    auto np_internal = adjoint->getNpInternal();
    std::cout << "np_interal " << np_internal << std::endl;
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
        G(i) = adjoint->GFn(i, system->y);

    return py::make_tuple(G, gp);
}

// Returns all points the fluxes and sources will be evaluated at
std::vector<double> PyRunner::getPoints(void)
{
    return system->y.getPoints();
}

extern "C"
{
    int IDAEwtSet(N_Vector, N_Vector, void *);
}

int static_residual(sunrealtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data);
int JacSetup(sunrealtype tt, sunrealtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

std::function<void(double)> SystemSolver::makeSolver(SUNLinearSolver &LS, // linear solver memory structure
                                                     SUNMatrix &sunMat,
                                                     void *IDA_mem, // IDA memory structure
                                                     int &retval,
                                                     N_Vector &Y,           // vector for storing solution
                                                     N_Vector &dYdt,        // vector for storing time derivative of solution
                                                     N_Vector &constraints, // vector for storing constraints
                                                     N_Vector &id,          // vector for storing id (which elements are algebraic or differentiable)
                                                     N_Vector &res,         // vector for storing residual
                                                     N_Vector &absTolVec,   // vector for storing absolute tolerances
                                                     sunrealtype &tout, sunrealtype &tret, bool writeOutput)
{

    if (!initialised)
        initialiseMatrices();

    //-------------------------------------System Design----------------------------------------------

    IDA_mem = IDACreate(ctx);
    if (ErrorChecker::check_retval((void *)IDA_mem, "IDACreate", 0))
        throw std::runtime_error("Sundials Initialization Error");

    retval = IDASetUserData(IDA_mem, static_cast<void *>(this));
    if (ErrorChecker::check_retval(&retval, "IDASetUserData", 1))
        throw std::runtime_error("Sundials Initialization Error");

    //-----------------------------Initial conditions-------------------------------

    // Set original vector lengths
    Y = N_VNew_Serial(nVars * 3 * nCells * (k + 1) + nVars * (nCells + 1) + nScalars + nAux * nCells * (k + 1), ctx);
    if (ErrorChecker::check_retval((void *)Y, "N_VNew_Serial", 0))
        throw std::runtime_error("Sundials Initialization Error");

    dYdt = N_VClone(Y);
    if (ErrorChecker::check_retval((void *)dYdt, "N_VClone", 0))
        throw std::runtime_error("Sundials Initialization Error");

    // Initialise Y and dYdt
    setInitialConditions(Y, dYdt);

    // ----------------- Allocate and initialize all other sun-vectors. -------------

    res = N_VClone(Y);
    if (ErrorChecker::check_retval((void *)res, "N_VClone", 0))
        std::runtime_error("Sundials initialization Error, run in debug to find");
    // sunrealtype tRes;

    // No constraints are imposed as negative coefficients may allow for a better fit across a cell
    constraints = N_VClone(Y);
    if (ErrorChecker::check_retval((void *)constraints, "N_VClone", 0))
        std::runtime_error("Sundials initialization Error, run in debug to find");

    // Specify only u as differential
    id = N_VClone(Y);
    if (ErrorChecker::check_retval((void *)id, "N_VClone", 0))
        std::runtime_error("Sundials initialization Error, run in debug to find");

    DGSoln isDifferential(nVars, grid, k, nScalars, nAux);
    isDifferential.Map(N_VGetArrayPointer(id));
    isDifferential.zeroCoeffs();
    for (Index v = 0; v < nVars; ++v)
        for (Index i = 0; i < nCells; ++i)
            isDifferential.u(v).getCoeff(i).second.Constant(k + 1, 1.0);

    for (Index s = 0; s < nScalars; ++s)
    {
        if (problem->isScalarDifferential(s))
        {
            isDifferential.Scalar(s) = 1.0;
        }
    }

    retval = IDASetId(IDA_mem, id);
    if (ErrorChecker::check_retval(&retval, "IDASetId", 1))
        std::runtime_error("Sundials initialization Error, run in debug to find");

    wgt = N_VClone(res);
    // Initialise IDA
    retval = IDAInit(IDA_mem, static_residual, t0, Y, dYdt);
    if (ErrorChecker::check_retval(&retval, "IDAInit", 1))
        std::runtime_error("Sundials initialization Error, run in debug to find");

    // Set tolerances
    absTolVec = N_VClone(Y);
    if (ErrorChecker::check_retval((void *)absTolVec, "N_VClone", 0))
        std::runtime_error("Sundials initialization Error, run in debug to find");
    VectorWrapper absTolVals(N_VGetArrayPointer(absTolVec), N_VGetLength(absTolVec));
    absTolVals.setZero();

    DGSoln tolerances(nVars, grid, k, nScalars, nAux);
    tolerances.Map(N_VGetArrayPointer(absTolVec));
    for (Index i = 0; i < nCells; ++i)
    {
        for (Index v = 0; v < nVars; ++v)
        {
            if (atol.size() == 1)
            {
                double absTol = atol[0];
                tolerances.u(v).getCoeff(i).second.setConstant(absTol);
                tolerances.q(v).getCoeff(i).second.setConstant(absTol);
                tolerances.sigma(v).getCoeff(i).second.setConstant(absTol);
                tolerances.lambda(v).setConstant(absTol);
            }
            else if (atol.size() == nVars)
            {
                double absTolU, absTolQ, absTolSigma;
                absTolU = atol[v];
                absTolQ = atol[v];
                absTolSigma = atol[v];
                tolerances.u(v).getCoeff(i).second.setConstant(absTolU);
                tolerances.q(v).getCoeff(i).second.setConstant(absTolQ);
                tolerances.sigma(v).getCoeff(i).second.setConstant(absTolSigma);
                tolerances.lambda(v).setConstant(absTolU);
            }
        }

        for (Index a = 0; a < nAux; ++a)
        {
            tolerances.Aux(a).getCoeff(i).second.setConstant(atol[0]);
        }
    }

    for (Index i = 0; i < nScalars; ++i)
        tolerances.Scalar(i) = atol[0];

    retval = IDAWFtolerances(IDA_mem, SystemSolver::getErrorWeights_static);
    if (ErrorChecker::check_retval(&retval, "IDAWFtolerances", 1))
        std::runtime_error("Sundials initialization Error, run in debug to find");

    //--------------set up user-built objects------------------

    // Use empty SunMatrix Object
    sunMat = SunMatrixNew(ctx);

    // The only linear solver wrapper ever constructed from this object so we can give it a pointer to 'this' and
    // it won't hold it beyond the lifetime of this function call.
    LS = SunLinSolWrapper::SunLinSol(this, IDA_mem, ctx);

    if (IDASetLinearSolver(IDA_mem, LS, sunMat) != SUN_SUCCESS)
        std::runtime_error("Error in IDASetLinearSolver");

    IDASetJacFn(IDA_mem, JacSetup);

    IDASetMaxNonlinIters(IDA_mem, 10);

    // Initialise text output and write out initial condition massaged by CalcIC
    //------------------------------Solve------------------------------
    // Update initial solution to be within tolerance of the residual equation

    // retval = IDACalcIC(IDA_mem, IDA_YA_YDP_INIT, dt);
    // retval = 0;
    // if (ErrorChecker::check_retval(&retval, "IDASolve", 1))
    // {
    //     throw std::runtime_error("IDACalcIC could not complete");
    // }

    long int nresevals = 0;
    // IDAGetNumResEvals(IDA_mem, &nresevals);
    // std::cout << "Number of Residual Evaluations due to IDACalcIC " << nresevals << std::endl;

    // if (nresevals > 10)
    //     std::cerr << " IDACalcIC required " << nresevals << " residual evaluations. Check config." << std::endl;

    // This also writes the t0 timeslice

    IDASetMaxNumSteps(IDA_mem, 50000);

    IDASetMinStep(IDA_mem, min_step_size);

    t = t0;
    tout = t0;
    tret = t0;

    auto fsolve = [&, IDA_mem, writeOutput](double tFinal)
    {
        // Steady-state stopping conditions
        sunrealtype dydt_rel_tol = steady_state_tol;
        sunrealtype dydt_abs_tol = 1e-2;
        if (t0 > tFinal && !TerminateOnSteadyState)
        {
            std::cout << "Initial time t = " << t0 << " is after the end of the simulation at t = " << tFinal << std::endl;
            throw std::runtime_error("Simulation ends before it begins.");
        }

        std::string baseName = inputFilePath.stem();
        if (writeOutput)
        {
            initialiseNetCDF(baseName + ".nc", nOut);
        }

        // Solving Loop
        while (tFinal - tret > min_step_size || TerminateOnSteadyState)
        {
            tout += dt;
            if (tout > tFinal && !TerminateOnSteadyState)
                tout = tFinal; // Never ask for results beyond tFinal
            retval = IDASolve(IDA_mem, tout, &tret, Y, dYdt, IDA_NORMAL);
            if (ErrorChecker::check_retval(&retval, "IDASolve", 1))
            {
                // try to emit final data
                if (writeOutput)
                {
                    WriteTimeslice(tret);
                    nc_output.Close();
                }
                throw std::runtime_error("IDASolve could not complete");
            }

            long int nstep_tmp;
            IDAGetNumSteps(IDA_mem, &nstep_tmp);
            std::cout << "Writing output at " << tret << " ( " << nstep_tmp << " timesteps )" << std::endl;

            if (writeOutput)
            {
                WriteTimeslice(tret);
            }
            // Check if steady-state is achieved (test the lambda points)
            if (TerminateOnSteadyState)
            {
                sunrealtype dydt_norm = 0.0;
                for (Index i = 0; i < nCells; i++)
                    for (Index v = 0; v < nVars; v++)
                    {
                        sunrealtype xi = dydt.lambda(v)[i] * dt;
                        sunrealtype wi = 1.0 / (y.lambda(v)[i] * dydt_rel_tol + dydt_abs_tol);
                        dydt_norm += xi * xi * wi * wi;
                    }
                dydt_norm = sqrt(dydt_norm);
                if (dydt_norm < 1.0)
                {
                    std::cout << "Steady State achieved at time t = " << tret << std::endl;
                    break;
                }
            }

            // Diagnostics go here
        }

        long int nsteps, njacevals;
        IDAGetNumSteps(IDA_mem, &nsteps);
        IDAGetNumResEvals(IDA_mem, &nresevals);
        IDAGetNumLinSolvSetups(IDA_mem, &njacevals);

        std::cout << "Total Number of Timesteps             :" << nsteps << std::endl;
        std::cout << "Total Number of Residual Evaluations  :" << nresevals << std::endl;
        std::cout << "Total Number of Jacobian Computations :" << njacevals << std::endl;

        problem->finaliseDiagnostics(nc_output);

        WriteRestartFile(baseName + ".restart.nc", Y, dYdt, nOut);
        if (writeOutput)
            nc_output.Close();
    };
    return fsolve;
}

PyRunner::~PyRunner()
{
    system->nc_output.Close(); // Make sure netCDF file is closed

    // No SunLinSol wrapper classes exist beyond this point, so we are safe in using raw pointers to construct them.
    SUNLinSolFree(LS);

    MatDestroy(sunMat);

    IDAFree(&IDA_mem);

    // Free the raw data buffers allocated by SUNDIALS

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    N_VDestroy(constraints);
    N_VDestroy(id);
    N_VDestroy(res);
    N_VDestroy(absTolVec);

    SUNContext_Free(&system->ctx);
}
