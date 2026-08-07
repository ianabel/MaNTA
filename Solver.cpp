#include <ida/ida.h>				  /* prototypes for IDA fcts., consts.    */
#include <nvector/nvector_serial.h>	  /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_band.h> /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h> /* access to band SUNLinearSolver       */
#include <sundials/sundials_types.h>  /* definition of type sunrealtype          */
#include <toml.hpp>
#include <fstream>
#include <print>
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

void SystemSolver::runSolver(double tFinal)
{
	//---------------------------Variable assiments-------------------------------
	SUNLinearSolver LS = NULL; // linear solver memory structure
	void *IDA_mem = NULL;	   // IDA memory structure
	int retval;

	N_Vector Y = NULL;			 // vector for storing solution
	N_Vector dYdt = NULL;		 // vector for storing time derivative of solution
	N_Vector constraints = NULL; // vector for storing constraints
	N_Vector id = NULL;			 // vector for storing id (which elements are algebraic or differentiable)
	N_Vector res = NULL;		 // vector for storing residual
	N_Vector absTolVec = NULL;	 // vector for storing absolute tolerances
	double delta_t = dt;
	sunrealtype tout, tret;

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

	// Steady-state stopping conditions
	sunrealtype dydt_rel_tol = steady_state_tol;
	sunrealtype dydt_abs_tol = 1e-3;

	retval = IDAWFtolerances(IDA_mem, SystemSolver::getErrorWeights_static);
	if (ErrorChecker::check_retval(&retval, "IDAWFtolerances", 1))
		std::runtime_error("Sundials initialization Error, run in debug to find");

	//--------------set up user-built objects------------------

	// Use empty SunMatrix Object
	SUNMatrix sunMat = SunMatrixNew(ctx);

	// The only linear solver wrapper ever constructed from this object so we can give it a pointer to 'this' and
	// it won't hold it beyond the lifetime of this function call.
	LS = SunLinSolWrapper::SunLinSol(this, IDA_mem, ctx);

	if (IDASetLinearSolver(IDA_mem, LS, sunMat) != SUN_SUCCESS)
		std::runtime_error("Error in IDASetLinearSolver");

	IDASetJacFn(IDA_mem, JacSetup);

	IDASetMaxNonlinIters(IDA_mem, 10);

	std::string baseName = inputFilePath.stem();

	// The .dat files are opt-in; netCDF below is what a run produces by
	// default. Nothing is opened unless asked for, so a plain run leaves no
	// text output behind at all.
	std::ofstream out0;
	if (writeDatFile)
	{
		out0.open(baseName + ".dat");
		std::println(out0, "# Time indexes blocks. ");
		std::println(out0, "# Columns Headings: ");
		std::print(out0, "# x");
		for (Index v = 0; v < nVars; ++v)
			std::print(out0, "\tvar{0} u\tvar{0} q\tvar{0} sigma\tvar{0} u_star\tvar{0} source", v);
		std::println(out0, "");
	}

	std::ofstream dydt_out, res_out;

	// The diagnostic .dat files need both the option and a PHYSICS_DEBUG build:
	// the residual norms and error weights they report are only computed on
	// that path, and `wgt` is only allocated there. One flag for both because
	// they are always written together.
	const bool debugDat = physics_debug && writeDebugDatFiles;

	if (debugDat)
	{
		wgt = N_VClone(res);
		dydt_out.open(baseName + ".dydt.dat");
		std::println(dydt_out, "# dydt before CalcIC");
		printOnNodes(dydt_out, t0, dYdt);
		res_out.open(baseName + ".res.dat");
		residual(t0, Y, dYdt, res);
		getErrorWeights(Y, wgt);
		double residual_val = N_VWrmsNorm(res, wgt);
		std::println(res_out, "# Residual norm at t = {:g} (pre-calcIC) is {:g}", t0, residual_val);
		printOnNodes(res_out, t0, res);
		if (writeDatFile)
		{
			std::println(out0, "# t = {:g} (pre-calcIC) ", t0);
			print(out0, t0, nOut, true);
		}
	}

	//------------------------------Solve------------------------------
	// Update initial solution to be within tolerance of the residual equation

	retval = IDACalcIC(IDA_mem, IDA_YA_YDP_INIT, dt0 > 0.0 ? dt0 : delta_t);
	retval = 0;
	if (ErrorChecker::check_retval(&retval, "IDASolve", 1))
	{
		throw std::runtime_error("IDACalcIC could not complete");
	}

	long int nresevals = 0;
	IDAGetNumResEvals(IDA_mem, &nresevals);
  logmsg<LOG_LEVEL::INFO>("Number of Residual Evaluations due to IDACalcIC: {}", nresevals);

	if (nresevals > 10)
    logmsg<LOG_LEVEL::WARNING>("IDACalcIC required {} residual evaluations. Check settings in {}", nresevals, std::string(inputFilePath));

	if (writeDatFile)
		print(out0, t0, nOut, true);
	if (debugDat)
	{
		IDAGetConsistentIC(IDA_mem, Y, dYdt);
		residual(t0, Y, dYdt, res);
		std::println(dydt_out, "# After CalcIC ");
		printOnNodes(dydt_out, t0, dYdt);

	

		IDAEwtSet(Y, wgt, IDA_mem);



		std::println(res_out, "# Residual norm at t = {:g} (post-CalcIC) is {:g}", t0,
					 N_VWrmsNorm(res, wgt));
		printOnNodes(res_out, t0, res);
	}

	// This also writes the t0 timeslice
	initialiseNetCDF(baseName + ".nc", nOut);

	IDASetMaxNumSteps(IDA_mem, 50000);

	IDASetMinStep(IDA_mem, min_step_size);

	t = t0;
	tout = t0;
	tret = t0;
	delta_t = dt;

	if (problem->isRestarting()) // If restarting, try to continue at same delta t
	{
		IDASetInitStep(IDA_mem, delta_t);
	}
	if (dt0 > 0.0)
		IDASetInitStep(IDA_mem, dt0);

	if (t0 > tFinal)
	{
    logmsg<LOG_LEVEL::ERROR>("Initial time t = {} is after the end of the simulation at t = {}", t0, tFinal);
		throw std::runtime_error("Simulation ends before it begins.");
	}

	// Solving Loop
	while (tFinal - tret > min_step_size || TerminateOnSteadyState)
	{
		tout += delta_t;
		if (tout > tFinal && !TerminateOnSteadyState)
			tout = tFinal; // Never ask for results beyond tFinal
		retval = IDASolve(IDA_mem, tout, &tret, Y, dYdt, IDA_NORMAL);
		if (ErrorChecker::check_retval(&retval, "IDASolve", 1))
		{
			// try to emit final data
			if (writeDatFile)
				print(out0, tret, nOut, true);
			if (debugDat)
      {
	      residual(tret, Y, dYdt, res);
        IDAEwtSet(Y, wgt, IDA_mem);
        std::println(res_out, "# Residual norm at t = {:g} is {:g}", tret,
                     N_VWrmsNorm(res, wgt));
        printOnNodes(res_out, tret, res);
        printOnNodes(dydt_out, tret, dYdt);
      }
			WriteTimeslice(tret);
			if (writeDatFile)
				out0.close();
			nc_output.Close();

			throw std::runtime_error("IDASolve could not complete");
		}

		long int nstep_tmp;
		IDAGetNumSteps(IDA_mem, &nstep_tmp);
		std::println("Writing output at {:g} ( {} timesteps )", tret, nstep_tmp);
		if (writeDatFile)
			print(out0, tret, nOut, Y, true);
		if (debugDat)
		{
			printOnNodes(dydt_out, tret, dYdt);
			residual(tret, Y, dYdt, res);
			IDAEwtSet(Y, wgt, IDA_mem);
			std::println(res_out, "# Residual norm at t = {:g} is {:g}", tret,
						 N_VWrmsNorm(res, wgt));
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
					sunrealtype xi = dydt.lambda(v)[i] * delta_t;
					sunrealtype wi = 1.0 / (y.lambda(v)[i] * dydt_rel_tol + dydt_abs_tol);
					dydt_norm += xi * xi * wi * wi;
				}
			dydt_norm = sqrt(dydt_norm);
			if (physics_debug)
				std::println(" dy/dt norm inferred from lambdas is {:g}", dydt_norm);
			if (dydt_norm < 1.0)
			{
				std::println("Steady State achieved at time t = {:g}", tret);
				break;
			}
		}

		// Diagnostics go here
	}

	long int nsteps, njacevals;
	IDAGetNumSteps(IDA_mem, &nsteps);
	IDAGetNumResEvals(IDA_mem, &nresevals);
	IDAGetNumLinSolvSetups(IDA_mem, &njacevals);

	std::println("Total Number of Timesteps             :{}", nsteps);
	std::println("Total Number of Residual Evaluations  :{}", nresevals);
	std::println("Total Number of Jacobian Computations :{}", njacevals);

	if (solveAdjoint)
	{
		runAdjointSolve();
		// WriteAdjoints();
	}

	problem->finaliseDiagnostics(nc_output);
	if (writeDatFile)
		out0.close();
	if (debugDat)
	{
		dydt_out.close();
		res_out.close();
	}
	nc_output.Close();

	WriteRestartFile(baseName + ".restart.nc", Y, dYdt, nOut);

	// Leave yJac holding the *final* solution. It is the only copy that
	// outlives this function -- `y` is a non-owning view over Y, which is
	// destroyed a few lines below -- and it is what PyRunner::getSolution and
	// getAdjointGradients read. Until now it held whatever state IDA last
	// evaluated a Jacobian at, which can be several steps stale, so a caller
	// asking for "the solution" got a slightly earlier one.
	//
	// Deliberately after runAdjointSolve(): the adjoint solve above is defined
	// at the state its matrices were built from, so moving this earlier would
	// change the gradients.
	setJacEvalY(Y, dYdt);

	// No SunLinSol wrapper classes exist beyond this point, so we are safe in using raw pointers to construct them.
	SUNLinSolFree(LS);

	MatDestroy(sunMat);

	IDAFree(&IDA_mem);

	// Free the raw data buffers allocated by SUNDIALS

	if (debugDat)
		N_VDestroy(wgt);

	N_VDestroy(Y);
	N_VDestroy(dYdt);
	N_VDestroy(constraints);
	N_VDestroy(id);
	N_VDestroy(res);
	N_VDestroy(absTolVec);

	// `ctx` belongs to the SystemSolver: it is created in the constructor
	// (SystemSolver.cpp:18) and freed in the destructor (:65). Freeing it here
	// as well left the member NULL, so a *second* runSolver call on the same
	// object failed at IDACreate with "Sundials Initialization Error" -- which
	// is why PyRunner::run works only once per configure(), even though it goes
	// to the trouble of clearing TerminateOnSteadyState for a repeat call.
	// The standalone binary never noticed because it runs once and exits.

	nc_output.Close();
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
