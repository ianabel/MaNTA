#include <ida/ida.h>				  /* prototypes for IDA fcts., consts.    */
#include <nvector/nvector_serial.h>	  /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_band.h> /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h> /* access to band SUNLinearSolver       */
#include <sundials/sundials_types.h>  /* definition of type sunrealtype          */
#include <toml.hpp>
#include <iostream>
#include <fstream>
#include <memory>

#include "Types.hpp"
#include "SystemSolver.hpp"
#include "gridStructures.hpp"
#include "SunLinSolWrapper.hpp"
#include "SunMatrixWrapper.hpp"
#include "ErrorChecker.hpp"

// Unadvertised, but in the library
extern "C"
{
	int IDAEwtSet(N_Vector, N_Vector, void *);
}

int static_residual(sunrealtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data);
int JacSetup(sunrealtype tt, sunrealtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

ISTATUS SystemSolver::initialize()
{

	if (!initialised)
		initialiseMatrices();

	//-------------------------------------System Design----------------------------------------------

	IDA_mem = IDACreate(ctx);
	if (ErrorChecker::check_retval((void *)IDA_mem, "IDACreate", 0))
	{
		logmsg<LOG_LEVEL::ERROR>("Sundials Initialization Error on line {}", __LINE__ - 2);
		return ISTATUS::FAILURE;
	}

	retval = IDASetUserData(IDA_mem, static_cast<void *>(this));
	if (ErrorChecker::check_retval(&retval, "IDASetUserData", 1))
	{
		logmsg<LOG_LEVEL::ERROR>("Sundials Initialization Error on line {}", __LINE__ - 2);
		return ISTATUS::FAILURE;
	}

	//-----------------------------Initial conditions-------------------------------

	// Set original vector lengths
	Y = N_VNew_Serial(nVars * 3 * nCells * (k + 1) + nVars * (nCells + 1) + nScalars + nAux * nCells * (k + 1), ctx);
	if (ErrorChecker::check_retval((void *)Y, "N_VNew_Serial", 0))
	{
		logmsg<LOG_LEVEL::ERROR>("Sundials Initialization Error on line {}", __LINE__ - 2);
		return ISTATUS::FAILURE;
	}

	dYdt = N_VClone(Y);
	if (ErrorChecker::check_retval((void *)dYdt, "N_VClone", 0))
	{
		logmsg<LOG_LEVEL::ERROR>("Sundials Initialization Error on line {}", __LINE__ - 2);
		return ISTATUS::FAILURE;
	}

	// Initialise Y and dYdt
	setInitialConditions(Y, dYdt);

    // If we are optimizing, we can determine if G is decreasing and avoid solving
	if (optimizeMode)
	{
		if (!adjointProblem)
		{
			logmsg<LOG_LEVEL::ERROR>("Optimize mode on but adjointProblem not set");
			return ISTATUS::FAILURE;
		}
		Vector dGdt(adjointProblem->getNg());
		for (Index gIndex = 0; gIndex < adjointProblem->getNg(); gIndex++)
		{
			dGdt(gIndex) = adjointProblem->GFn(gIndex, dydt);
		}
		logmsg<LOG_LEVEL::PDEBUG>("Initial dGdt = {}", dGdt(0));
		const double stoptol = 0.1; // acceptable decrease in G
		for (const auto& dG : dGdt) 
		{
			if (dt * dG < -stoptol)
			{
				logmsg<LOG_LEVEL::WARNING>("Negative value of dGdt = {} encountered; Optimize mode assumes that we are maximizing G, so a negative dGdt indicates a bad step.", dG);
				return ISTATUS::NEGATIVE_DGDT;
			}
		}
	}

	// ----------------- Allocate and initialize all other sun-vectors. -------------

	res = N_VClone(Y);
	if (ErrorChecker::check_retval((void *)res, "N_VClone", 0))
	{
		logmsg<LOG_LEVEL::ERROR>("Sundials Initialization Error on line {}", __LINE__ - 2);
		return ISTATUS::FAILURE;
	}
		// sunrealtype tRes;

	// No constraints are imposed as negative coefficients may allow for a better fit across a cell
	constraints = N_VClone(Y);
	if (ErrorChecker::check_retval((void *)constraints, "N_VClone", 0))
	{
		logmsg<LOG_LEVEL::ERROR>("Sundials Initialization Error on line {}", __LINE__ - 2);
		return ISTATUS::FAILURE;
	}

	// Specify only u as differential
	id = N_VClone(Y);
	if (ErrorChecker::check_retval((void *)id, "N_VClone", 0))
	{
		logmsg<LOG_LEVEL::ERROR>("Sundials Initialization Error on line {}", __LINE__ - 2);
		return ISTATUS::FAILURE;
	}

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
	{
		logmsg<LOG_LEVEL::ERROR>("Sundials Initialization Error on line {}", __LINE__ - 2);
		return ISTATUS::FAILURE;
	}

	// Initialise IDA
	retval = IDAInit(IDA_mem, static_residual, t0, Y, dYdt);
	if (ErrorChecker::check_retval(&retval, "IDAInit", 1))
	{
		logmsg<LOG_LEVEL::ERROR>("Sundials Initialization Error on line {}", __LINE__ - 2);
		return ISTATUS::FAILURE;
	}

	// Set tolerances
	absTolVec = N_VClone(Y);
	if (ErrorChecker::check_retval((void *)absTolVec, "N_VClone", 0))
	{
		logmsg<LOG_LEVEL::ERROR>("Sundials Initialization Error on line {}", __LINE__ - 2);
		return ISTATUS::FAILURE;
	}
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
	std::string baseName = inputFilePath.stem();
	out0.open(baseName + ".dat");

	out0 << "# Time indexes blocks. " << std::endl;
	out0 << "# Columns Headings: " << std::endl;
	out0 << "# x";
	for (Index v = 0; v < nVars; ++v)
		out0 << "\t"
			 << "var" << v << " u"
			 << "\t"
			 << "var" << v << " q"
			 << "\t"
			 << "var" << v << " sigma"
			 << "\t"
			 << "var" << v << " source";
	out0 << std::endl;

	if (physics_debug)
	{
		wgt = N_VClone(res);
		dydt_out.open(baseName + ".dydt.dat");
		dydt_out << "# dydt before CalcIC" << std::endl;
		printOnNodes(dydt_out, t0, dYdt);
		res_out.open(baseName + ".res.dat");
		residual(t0, Y, dYdt, res);
		getErrorWeights(Y, wgt);
		double residual_val = N_VWrmsNorm(res, wgt);
		res_out << "# Residual norm at t = " << t0 << " (pre-calcIC) is " << residual_val << std::endl;
		printOnNodes(res_out, t0, res);
		out0 << "# t = " << t0 << " (pre-calcIC) " << std::endl;
		print(out0, t0, nOut, true);
	}

	//------------------------------Solve------------------------------
	// Update initial solution to be within tolerance of the residual equation

	IDASetNonlinConvCoefIC(IDA_mem, 0.01);
	retval = IDACalcIC(IDA_mem, IDA_YA_YDP_INIT, dt0 > 0.0 ? dt0 : dt);
	retval = 0;
	if (ErrorChecker::check_retval(&retval, "IDASolve", 1))
	{
		throw std::runtime_error("IDACalcIC could not complete");
	}

	IDAGetNumResEvals(IDA_mem, &nresevals);
	logmsg<LOG_LEVEL::INFO>("Number of Residual Evaluations due to IDACalcIC: {}", nresevals);

	if (nresevals > 10)
    logmsg<LOG_LEVEL::WARNING>("IDACalcIC required {} residual evaluations. Check settings in {}", nresevals, std::string(inputFilePath));

	print(out0, t0, nOut, true);
	if (physics_debug)
	{
		IDAGetConsistentIC(IDA_mem, Y, dYdt);
		residual(t0, Y, dYdt, res);
		dydt_out << "# After CalcIC " << std::endl;
		printOnNodes(dydt_out, t0, dYdt);

	

		IDAEwtSet(Y, wgt, IDA_mem);



		res_out << "# Residual norm at t = " << t0 << " (post-CalcIC) is " << N_VWrmsNorm(res, wgt) << std::endl;
		printOnNodes(res_out, t0, res);
	}

	// This also writes the t0 timeslice
	initialiseNetCDF(baseName + ".nc", nOut);

	IDASetMaxNumSteps(IDA_mem, 50000);

	IDASetMinStep(IDA_mem, min_step_size);

	t = t0;
	tout = t0;
	tret = t0;

	if (dt0 > 0.0)
		IDASetInitStep(IDA_mem, dt0);
	if (aggressiveTimesteps)
		IDASetEtaMax(IDA_mem, 4.0); // Default is 2.0
	return ISTATUS::SUCCESS;
}

void SystemSolver::runSolver(double tFinal)
{
	// Steady-state stopping conditions
	sunrealtype dydt_rel_tol = steady_state_tol;
	sunrealtype dydt_abs_tol = 1e-3;
	if (t0 > tFinal)
	{
    logmsg<LOG_LEVEL::ERROR>("Initial time t = {} is after the end of the simulation at t = {}", t0, tFinal);
		throw std::runtime_error("Simulation ends before it begins.");
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
			print(out0, tret, nOut, true);
			if (physics_debug)
				printOnNodes(dydt_out, tret, dYdt);
			WriteTimeslice(tret);
			out0.close();
			nc_output.Close();

			throw std::runtime_error("IDASolve could not complete");
		}

		long int nstep_tmp;
		IDAGetNumSteps(IDA_mem, &nstep_tmp);
		std::cout << "Writing output at " << tret << " ( " << nstep_tmp << " timesteps )" << std::endl;
		print(out0, tret, nOut, Y, true);
		if (physics_debug)
		{
			printOnNodes(dydt_out, tret, dYdt);
			residual(tret, Y, dYdt, res);
			IDAEwtSet(Y, wgt, IDA_mem);
			res_out << "# Residual norm at t = " << tret << " is " << N_VWrmsNorm(res, wgt) << std::endl;
			printOnNodes(res_out, tret, res);
		}
		WriteTimeslice(tret);

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
			if (physics_debug)
				std::cout << " dy/dt norm inferred from lambdas is " << dydt_norm << std::endl;
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

	if (solveAdjoint)
	{
		runAdjointSolve();
		// WriteAdjoints();
	}

	problem->finaliseDiagnostics(nc_output);
	out0.close();
	if (physics_debug)
	{
		dydt_out.close();
		res_out.close();
	}
	nc_output.Close();
	std::string baseName = inputFilePath.stem();
	WriteRestartFile(baseName + ".restart.nc", Y, dYdt, nOut);
}

void SystemSolver::runAdjointSolve()
{
	if (solveAdjoint)
	{
    logmsg<LOG_LEVEL::INFO>("Computing adjoints");
		initializeMatricesForAdjointSolve();
		solveAdjointState(0);
		computeAdjointGradients();
	}
	else
	{
    logmsg<LOG_LEVEL::ERROR>("Error: runAdjointSolve called but \"solveAdjoint\" was set to false");
	}
}

void SystemSolver::destroySundials()
{
	// No SunLinSol wrapper classes exist beyond this point, so we are safe in using raw pointers to construct them.
	SUNLinSolFree(LS);
	if (sunMat)
		MatDestroy(sunMat);

	IDAFree(&IDA_mem);

	// Free the raw data buffers allocated by SUNDIALS

	if (physics_debug)
		N_VDestroy(wgt);

	N_VDestroy(Y);
	N_VDestroy(dYdt);
	N_VDestroy(constraints);
	N_VDestroy(id);
	N_VDestroy(res);
	N_VDestroy(absTolVec);

	SUNContext_Free(&ctx);
}

/*
 * SUNDIALS Calls this function to recompute the local Jacobian
 * This is the function that should set the point at which the sub-matrices for the Jacobian solve are evaluated
 */
int JacSetup(sunrealtype tt, sunrealtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
	// Sundials looks for a Jacobian, but our Jacobian equation is solved without computing the jacobian.
	// We use this function to capture t and cj for the solve.
	auto System = reinterpret_cast<SystemSolver *>(user_data);
	System->setJacTime(tt);
	System->setAlpha(cj);
	System->setJacEvalY(yy, yp);
	System->updateBoundaryConditions(tt);
	System->updateMatricesForJacSolve();
	return 0;
}
