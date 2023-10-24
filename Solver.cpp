#include <ida/ida.h>				  /* prototypes for IDA fcts., consts.    */
#include <nvector/nvector_serial.h>	  /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_band.h> /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h> /* access to band SUNLinearSolver       */
#include <sundials/sundials_types.h>  /* definition of type realtype          */
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

int static_residual(realtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data);
int JacSetup(realtype tt, realtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

void SystemSolver::runSolver(std::string inputFile)
{
	//---------------------------Variable assiments-------------------------------
	SUNLinearSolver LS = NULL; // linear solver memory structure
	void *IDA_mem = NULL;	   // IDA memory structure
	int retval, iout;

	N_Vector Y = NULL;			 // vector for storing solution
	N_Vector dYdt = NULL;		 // vector for storing time derivative of solution
	N_Vector constraints = NULL; // vector for storing constraints
	N_Vector id = NULL;			 // vector for storing id (which elements are algebraic or differentiable)
	N_Vector res = NULL;		 // vector for storing residual
	N_Vector absTolVec = NULL;	 // vector for storing absolute tolerances
	int nOut = 301;
	double tFinal;
	realtype rtol;
	realtype t0 = 0.0, t1, tout, tret;

	// TODO: Move all config parsing into a separate function and just make some of these member functions
	//--------------------------------------Read File---------------------------------------
	const auto configFile = toml::parse(inputFile);
	const auto config = toml::find<toml::value>(configFile, "configuration");

	realtype deltatPrint;
	auto printdt = toml::find(config, "delta_t");
	if (config.count("delta_t") != 1)
		throw std::invalid_argument("delta_t unspecified or specified more than once");
	else if (printdt.is_integer())
		deltatPrint = static_cast<double>(printdt.as_floating());
	else if (printdt.is_floating())
		deltatPrint = static_cast<double>(printdt.as_floating());
	else
		throw std::invalid_argument("delta_t specified incorrrectly");

	double delta_t = deltatPrint * 0.5;
	t1 = delta_t;

	auto tEnd = toml::find(config, "t_final");
	if (config.count("t_final") != 1)
		throw std::invalid_argument("tEnd unspecified or specified more than once");
	else if (tEnd.is_integer())
		tFinal = static_cast<double>(tEnd.as_floating());
	else if (tEnd.is_floating())
		tFinal = static_cast<double>(tEnd.as_floating());
	else
		throw std::invalid_argument("tEnd specified incorrrectly");
	double totalSteps = tFinal / delta_t;
	int stepsPerPrint = floor(totalSteps * (deltatPrint / tFinal));

	// If dt is set to greater than tf, just print tf
	if (tFinal < deltatPrint)
	{
		t1 = tFinal;
		stepsPerPrint = 1;
		totalSteps = 1;
	}

	auto relTol = toml::find(config, "Relative_tolerance");
	if (config.count("Relative_tolerance") != 1)
		rtol = 1.0e-5;
	else if (relTol.is_integer())
		rtol = static_cast<double>(relTol.as_floating());
	else if (relTol.is_floating())
		rtol = static_cast<double>(relTol.as_floating());
	else
		throw std::invalid_argument("relative_tolerance specified incorrrectly");

	realtype atol;
	auto absTol = toml::find(config, "Absolute_tolerance");
	if (config.count("Absolute_tolerance") != 1)
		atol = 1.0e-2;
	else if (absTol.is_integer())
		atol = static_cast<double>(absTol.as_floating());
	else if (absTol.is_floating())
		atol = static_cast<double>(absTol.as_floating());
	else
		throw std::invalid_argument("Absolute_tolerance specified incorrrectly");

	//-------------------------------------System Design----------------------------------------------
	SUNContext ctx;
	retval = SUNContext_Create(nullptr, &ctx);

	IDA_mem = IDACreate(ctx);
	if (ErrorChecker::check_retval((void *)IDA_mem, "IDACreate", 0))
		throw std::runtime_error("Sundials Initialization Error");

	retval = IDASetUserData(IDA_mem, static_cast<void *>(this));
	if (ErrorChecker::check_retval(&retval, "IDASetUserData", 1))
		throw std::runtime_error("Sundials Initialization Error");

	//-----------------------------Initial conditions-------------------------------

	initialiseMatrices();

	// Set original vector lengths
	Y = N_VNew_Serial(nVars * 3 * nCells * (k + 1) + nVars * (nCells + 1), ctx);
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
	// realtype tRes;

	// No constraints are imposed as negative coefficients may allow for a better fit across a cell
	constraints = N_VClone(Y);
	if (ErrorChecker::check_retval((void *)constraints, "N_VClone", 0))
		std::runtime_error("Sundials initialization Error, run in debug to find");

	// Specify only u as differential
	id = N_VClone(Y);
	if (ErrorChecker::check_retval((void *)id, "N_VClone", 0))
		std::runtime_error("Sundials initialization Error, run in debug to find");

	DGSoln isDifferential( nVars, grid, k );
	isDifferential.Map( N_VGetArrayPointer( id ) );
	isDifferential.zeroCoeffs();
	for ( Index v = 0; v < nVars; ++v )
		for ( Index i = 0; i < nCells; ++i )
			isDifferential.u( v ).getCoeff( i ).second.Constant( k + 1, 1.0 );
	
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

	DGSoln tolerances(nVars, grid, k);
	tolerances.Map(N_VGetArrayPointer(absTolVec));
	double dx = (grid.upperBoundary() - grid.lowerBoundary()) / nCells;
	// TODO: re-add user tolerance interface
	for (Index i = 0; i < nCells; ++i)
	{
		for (Index v = 0; v < nVars; ++v)
		{
			tolerances.u(v).getCoeff(i).second.setConstant(atol);
			tolerances.q(v).getCoeff(i).second.setConstant(atol / dx);
			tolerances.sigma(v).getCoeff(i).second.setConstant(atol * dx / delta_t);
			tolerances.lambda(v).setConstant(atol);
		}
	}

	retval = IDASVtolerances(IDA_mem, rtol, absTolVec);
	if (ErrorChecker::check_retval(&retval, "IDASVtolerances", 1))
		std::runtime_error("Sundials initialization Error, run in debug to find");

	//--------------set up user-built objects------------------

	// Use empty SunMatrix Object
	SUNMatrix sunMat = SunMatrixNew(ctx);

	// The only linear solver wrapper ever constructed from this object so we can give it a pointer to 'this' and
	// it won't hold it beyond the lifetime of this function call.
	LS = SunLinSolWrapper::SunLinSol(this, IDA_mem, ctx);

	if (IDASetLinearSolver(IDA_mem, LS, sunMat) != SUNLS_SUCCESS)
		std::runtime_error("Error in IDASetLinearSolver");

	IDASetJacFn(IDA_mem, JacSetup);

	IDASetMaxNonlinIters(IDA_mem, 10);

	/*
	//Attach Diagnostics
	Diagnostic diagnostic(system, system->plasma);
	*/

	//------------------------------Solve------------------------------
	// Update initial solution to be within tolerance of the residual equation
	retval = IDACalcIC(IDA_mem, IDA_YA_YDP_INIT, delta_t);
	if (ErrorChecker::check_retval(&retval, "IDASolve", 1))
	{
		throw std::runtime_error("IDACalcIC could not complete");
	}

	// Initialise text output and write out initial condition massaged by CalcIC
	std::string baseName = inputFile.substr(0, inputFile.rfind("."));
	std::ofstream out0(baseName + ".dat");

	out0 << "# Time indexes blocks. " << std::endl;
	out0 << "# Columns Headings: " << std::endl;
	out0 << "# x";
	for (Index v = 0; v < nVars; ++v)
		out0 << "\t"
			 << "var" << v << " u" << "\t"
			 << "var" << v << " q" << "\t"
			 << "var" << v << " sigma" << "\t"
			 << "var" << v << " source" ;
	out0 << std::endl;

	print(out0, t0, nOut);

	// std::ofstream dydt_out(baseName + ".dydt.dat");
	// std::ofstream res_out(baseName + ".res.dat");

	// print( dydt_out, t0, nOut, dYdt );

	// residual( t0, Y, dYdt, res );

	// res_out << "Residual l_inf norm at t = " << t0 << " is " << N_VMaxNorm( res ) << std::endl;

	initialiseNetCDF(baseName + ".nc", nOut);
	WriteTimeslice(t0);

	//

	IDASetMaxNumSteps(IDA_mem, 50000);

	IDASetMinStep(IDA_mem, 1e-7);

	t = t0;

	// Solving Loop
	for (tout = t1, iout = 1; iout <= totalSteps; iout++, tout += delta_t)
	{
		retval = IDASolve(IDA_mem, tout, &tret, Y, dYdt, IDA_NORMAL);
		if (ErrorChecker::check_retval(&retval, "IDASolve", 1))
		{
			print(out0, tret, nOut);
			// print( dydt_out, t0, nOut, dYdt );
			WriteTimeslice(tret);
			nc_output.Close();

			throw std::runtime_error("IDASolve could not complete");
		}

		if (iout % stepsPerPrint == 0)
		{
			std::cout << "Writing output at " << tret << std::endl;
			print( out0, tret, nOut, Y );
			// print( dydt_out, tret, nOut, dYdt );
			// residual( tret, Y, dYdt, res );
			// res_out << "Residual l_inf norm at t = " << tret << " is " << N_VMaxNorm( res ) << " ; L1 Norm is " << N_VL1Norm( res ) << std::endl;
			WriteTimeslice( tret );

			// Diagnostics go here
		}
	}

	std::cerr << "Total number of steps taken = " << total_steps << std::endl;

	out0.close();
	// dydt_out.close();
	nc_output.Close();

	IDAFree(&IDA_mem);

	// No SunLinSol wrapper classes exist beyond this point, so we are safe in using raw pointers to construct them.
	SUNLinSolFree(LS);

	// Free the raw data buffers allocated by SUNDIALS
	N_VDestroy(Y);
	N_VDestroy(dYdt);
	N_VDestroy(constraints);
	N_VDestroy(id);
	N_VDestroy(res);
	N_VDestroy(absTolVec);
}

/* 
 * SUNDIALS Calls this function to recompute the local Jacobian
 * This is the function that should set the point at which the sub-matrices for the Jacobian solve are evaluated
 */
int JacSetup(realtype tt, realtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
	// Sundials looks for a Jacobian, but our Jacobian equation is solved without computing the jacobian.
	// We use this function to capture t and cj for the solve.
	auto System = reinterpret_cast<SystemSolver *>(user_data);
	System->SetTime( tt );
	System->setAlpha( cj );
	System->setJacEvalY( yy );
	System->updateBoundaryConditions( tt );
	System->updateMatricesForJacSolve();
	return 0;
}
