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

    for (Index s = 0; s < nScalars; ++s) {
      if( problem->isScalarDifferential( s ) ) {
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
	sunrealtype dydt_abs_tol = 1e-2;

	retval = IDAWFtolerances(IDA_mem, SystemSolver::getErrorWeights_static );
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

	// Initialise text output and write out initial condition massaged by CalcIC
	std::string baseName = inputFilePath.stem();
	std::ofstream out0(baseName + ".dat");

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

	std::ofstream dydt_out, res_out;

	N_Vector wgt;

	if (physics_debug)
    {
        wgt = N_VClone(res);
        dydt_out.open(baseName + ".dydt.dat");
        dydt_out << "# dydt before CalcIC" << std::endl;
        print(dydt_out, t0, nOut, dYdt);
        res_out.open(baseName + ".res.dat");
        residual(t0, Y, dYdt, res);
        getErrorWeights(Y, wgt);
        double residual_val = N_VWrmsNorm(res, wgt);
        res_out << "# Residual norm at t = " << t0 << " (pre-calcIC) is " << residual_val << std::endl;
        print(res_out, t0, nOut, res);
        out0 << "# t = " << t0 << " (pre-calcIC) " << std::endl;
        print(out0, t0, nOut, true);
    }

    //------------------------------Solve------------------------------
	// Update initial solution to be within tolerance of the residual equation
	retval = IDACalcIC(IDA_mem, IDA_YA_YDP_INIT, min_step_size);
	if (ErrorChecker::check_retval(&retval, "IDASolve", 1))
	{
		throw std::runtime_error("IDACalcIC could not complete");
	}

	long int nresevals = 0;
	IDAGetNumResEvals(IDA_mem, &nresevals);
	std::cout << "Number of Residual Evaluations due to IDACalcIC " << nresevals << std::endl;

	if (nresevals > 10)
		std::cerr << " IDACalcIC required " << nresevals << " residual evaluations. Check settings in " << inputFilePath << std::endl;

	print(out0, t0, nOut, true);
	if (physics_debug)
	{

		dydt_out << "# After CalcIC " << std::endl;
		print(dydt_out, t0, nOut, dYdt);

		residual(t0, Y, dYdt, res);

		IDAEwtSet(Y, wgt, IDA_mem);

		res_out << "# Residual norm at t = " << t0 << " (post-CalcIC) is " << N_VWrmsNorm(res, wgt) << std::endl;
		print(res_out, t0, nOut, res);
	}

	// This also writes the t0 timeslice
	initialiseNetCDF(baseName + ".nc", nOut);

	IDASetMaxNumSteps(IDA_mem, 50000);

	IDASetMinStep(IDA_mem, min_step_size);

	t = t0;
	tout = t0;
	tret = t0;
	delta_t = dt;

	if (t0 > tFinal)
	{
		std::cout << "Initial time t = " << t0 << " is after the end of the simulation at t = " << tFinal << std::endl;
		throw std::runtime_error("Simulation ends before it begins.");
	}

	// Solving Loop
	while (tFinal - tret > min_step_size)
	{
		tout += delta_t;
		if (tout > tFinal)
			tout = tFinal; // Never ask for results beyond tFinal
		retval = IDASolve(IDA_mem, tout, &tret, Y, dYdt, IDA_NORMAL);
		if (ErrorChecker::check_retval(&retval, "IDASolve", 1))
		{
			// try to emit final data
			print(out0, tret, nOut, true);
			if (physics_debug)
				print(dydt_out, tret, nOut, dYdt);
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
			print(dydt_out, tret, nOut, dYdt);
			residual(tret, Y, dYdt, res);
			IDAEwtSet(Y, wgt, IDA_mem);
			res_out << "# Residual norm at t = " << tret << " is " << N_VWrmsNorm(res, wgt) << std::endl;
			print(res_out, tret, nOut, res);
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

	problem->finaliseDiagnostics(nc_output);
	out0.close();
	if (physics_debug)
	{
		dydt_out.close();
		res_out.close();
	}
	nc_output.Close();

	// No SunLinSol wrapper classes exist beyond this point, so we are safe in using raw pointers to construct them.
	SUNLinSolFree(LS);

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

	nc_output.Close();
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
	System->setJacEvalY(yy);
	System->updateBoundaryConditions(tt);
	System->updateMatricesForJacSolve();
	return 0;
}
