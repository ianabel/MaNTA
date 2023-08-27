#include <ida/ida.h>                   /* prototypes for IDA fcts., consts.    */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_band.h>  /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h>  /* access to band SUNLinearSolver       */
#include <sundials/sundials_types.h>   /* definition of type realtype          */
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

int residual(realtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data);
int EmptyJac(realtype tt, realtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

void SystemSolver::runSolver( std::string inputFile )
{
	//---------------------------Variable assiments-------------------------------
	SUNLinearSolver LS = NULL;				//linear solver memory structure
	void *IDA_mem   = NULL;					//IDA memory structure
	int retval, iout;

	N_Vector Y = NULL;				//vector for storing solution
	N_Vector dYdt = NULL;			//vector for storing time derivative of solution
	N_Vector constraints = NULL;	//vector for storing constraints
	N_Vector id = NULL;				//vector for storing id (which elements are algebraic or differentiable)
	N_Vector res = NULL;			//vector for storing residual
	N_Vector absTolVec = NULL;		//vector for storing absolute tolerances
	int nOut = 301;
	double tFinal;
	realtype rtol;
	bool printToFile = true;
	realtype t0 = 0.0, t1, tout, tret;


	// TODO: Move all config parsing into a separate function and just make some of these member functions
	//--------------------------------------Read File---------------------------------------
	const auto configFile = toml::parse( inputFile );
	const auto config = toml::find<toml::value>( configFile, "configuration" );

	realtype deltatPrint;
	auto printdt = toml::find(config, "delta_t");
	if( config.count("delta_t") != 1 ) throw std::invalid_argument( "delta_t unspecified or specified more than once" );
	else if( printdt.is_integer() ) deltatPrint = static_cast<double>(printdt.as_floating());
	else if( printdt.is_floating() ) deltatPrint = static_cast<double>(printdt.as_floating());
	else throw std::invalid_argument( "delta_t specified incorrrectly" );

	double delta_t = deltatPrint*0.5;
	t1 = delta_t;

	auto tEnd = toml::find(config, "t_final");
	if( config.count("t_final") != 1 ) throw std::invalid_argument( "tEnd unspecified or specified more than once" );
	else if( tEnd.is_integer() ) tFinal = static_cast<double>(tEnd.as_floating());
	else if( tEnd.is_floating() ) tFinal = static_cast<double>(tEnd.as_floating());
	else throw std::invalid_argument( "tEnd specified incorrrectly" );
	double totalSteps = tFinal/delta_t;
	int stepsPerPrint = floor(totalSteps*(deltatPrint/tFinal));

	//If dt is set to greater than tf, just print tf
	if(tFinal<deltatPrint)
	{
		t1=tFinal;
		stepsPerPrint = 1;
		totalSteps = 1;
	}

	auto relTol = toml::find(config, "Relative_tolerance");
	if( config.count("Relative_tolerance") != 1 ) rtol = 1.0e-5;
	else if( relTol.is_integer() ) rtol = static_cast<double>(relTol.as_floating());
	else if( relTol.is_floating() ) rtol = static_cast<double>(relTol.as_floating());
	else throw std::invalid_argument( "relative_tolerance specified incorrrectly" );

	/*
	realtype atol;
	auto absTol = toml::find(config, "Absolute_tolerance");
	if( config.count("Absolute_tolerance") != 1 ) atol = 1.0e-5;
	else if( absTol.is_integer() ) atol = static_cast<double>(absTol.as_floating());
	else if( absTol.is_floating() ) atol = static_cast<double>(absTol.as_floating());
	else throw std::invalid_argument( "Absolute_tolerance specified incorrrectly" );
	*/
	//-------------------------------------System Design----------------------------------------------
	SUNContext ctx;
    retval = SUNContext_Create(nullptr, &ctx);

	IDA_mem = IDACreate(ctx);
	if(ErrorChecker::check_retval((void *)IDA_mem, "IDACreate", 0)) 
		throw std::runtime_error("Sundials Initialization Error");

	retval = IDASetUserData(IDA_mem, static_cast<void*>( this ) );
	if(ErrorChecker::check_retval(&retval, "IDASetUserData", 1))
		throw std::runtime_error("Sundials Initialization Error");

	//-----------------------------Initial conditions-------------------------------

	initialiseMatrices();

	//Set original vector lengths
	Y = N_VNew_Serial(nVars*3*nCells*(k+1) + nVars*(nCells+1), ctx);
	if(ErrorChecker::check_retval((void *)Y, "N_VNew_Serial", 0))
		throw std::runtime_error("Sundials Initialization Error");
	
	dYdt = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)dYdt, "N_VClone", 0))
		throw std::runtime_error("Sundials Initialization Error");
	VectorWrapper yVec( N_VGetArrayPointer( Y ), N_VGetLength( Y ) ); 
	VectorWrapper dydtVec( N_VGetArrayPointer( dYdt ), N_VGetLength( dYdt ) ); 
	yVec.setZero();
	dydtVec.setZero();

	//Initialise Y and dYdt
	setInitialConditions(Y, dYdt);
	
	// ----------------- Allocate and initialize all other sun-vectors. -------------

	res = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)res, "N_VClone", 0))
		std::runtime_error("Sundials initialization Error, run in debug to find");
	// realtype tRes;

	//No constraints are imposed as negative coefficients may allow for a better fit across a cell
	constraints = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)constraints, "N_VClone", 0))
		std::runtime_error("Sundials initialization Error, run in debug to find");

	//Specify only u as differential
	id = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)id, "N_VClone", 0))
		std::runtime_error("Sundials initialization Error, run in debug to find");
	VectorWrapper idVals( N_VGetArrayPointer( id ), nVars*3*nCells*(k+1) + nVars*(nCells+1) );
	idVals.setZero();
	for(Index i=0; i < nCells; i++)
	{
		for(Index j=0; j< nVars*(k+1); j++)
		{
			idVals[ i*3*nVars*(k+1) + 2*nVars*(k+1) + j ] = 1.0;//U vals
		}
	}
	retval = IDASetId(IDA_mem, id);
	if(ErrorChecker::check_retval(&retval, "IDASetId", 1)) 
		std::runtime_error("Sundials initialization Error, run in debug to find");

	//Initialise IDA 
	retval = IDAInit(IDA_mem, residual, t0, Y, dYdt); 
	if(ErrorChecker::check_retval(&retval, "IDAInit", 1)) 
		std::runtime_error("Sundials initialization Error, run in debug to find");

	//Set tolerances
	absTolVec = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)absTolVec, "N_VClone", 0))
		std::runtime_error("Sundials initialization Error, run in debug to find");
	VectorWrapper absTolVals( N_VGetArrayPointer( absTolVec ), N_VGetLength( absTolVec ) );
	absTolVals.setZero();

	DGSoln tolerances( nVars, grid, k );
	tolerances.Map( N_VGetArrayPointer( absTolVec ) );
	double dx = (grid.upperBoundary() - grid.lowerBoundary())/nCells;

	// TODO: re-add user tolerance interface
	for( Index i = 0; i < nCells; ++i )
	{
		for ( Index v = 0; v < nVars; ++v ) {
			tolerances.u( v ).getCoeff( i ).second.setConstant(1.0);
			tolerances.q( v ).getCoeff( i ).second.setConstant(1.0/dx);
			tolerances.sigma( v ).getCoeff( i ).second.setConstant(1.0 * dx/delta_t );
			tolerances.lambda( v ).setConstant( 1.0 );
		}
	}

	retval = IDASVtolerances(IDA_mem, rtol, absTolVec);
	if(ErrorChecker::check_retval(&retval, "IDASVtolerances", 1)) 
		std::runtime_error("Sundials initialization Error, run in debug to find");

	//--------------set up user-built objects------------------

	// Use empty SunMatrix Object
	SUNMatrix sunMat = SunMatrixNew(ctx);

	// The only linear solver wrapper ever constructed from this object so we can give it a pointer to 'this' and
	// it won't hold it beyond the lifetime of this function call.
	LS = SunLinSolWrapper::SunLinSol(this, IDA_mem, ctx);
 
	if ( IDASetLinearSolver(IDA_mem, LS, sunMat) != SUNLS_SUCCESS )
		std::runtime_error("Error in IDASetLinearSolver");

	IDASetJacFn(IDA_mem, EmptyJac);

	IDASetMaxNonlinIters(IDA_mem, 10);

	/*
	//Attach Diagnostics
	Diagnostic diagnostic(system, system->plasma);
	*/
	
	//------------------------------Solve------------------------------
	std::ofstream out0( inputFile.substr(0, inputFile.rfind(".")) + ".dat" );

	out0 << "# Time indexes blocks. " << std::endl;
	out0 << "# Columns Headings: " << std::endl;
	out0 << "# x";
	for ( Index v = 0; v < nVars; ++v )
		out0 << "\t" << "var" << v << " u" << "\t" <<  "var" << v << " q" << "\t" <<  "var" << v << " sigma";
	out0 << std::endl;

	if(printToFile)
		print(out0, t0, nOut, 0);
	
	IDASetMaxNumSteps(IDA_mem, 50000);

	//Update initial solution to be within tolerance of the residual equation
	retval = IDACalcIC(IDA_mem, IDA_YA_YDP_INIT, delta_t);
	if(ErrorChecker::check_retval(&retval, "IDASolve", 1)) 
	{
		print(out0, t0, nOut, 0);
		throw std::runtime_error("IDACalcIC could not complete");
	}

	N_Vector dydtWRMS;
	N_Vector Weights;
	dydtWRMS = N_VClone(Y);
	Weights = N_VClone(Y);
	VectorWrapper dydtWRMSvec( N_VGetArrayPointer( dydtWRMS ), N_VGetLength( dydtWRMS ) );
	VectorWrapper Weightsvec( N_VGetArrayPointer( Weights ), N_VGetLength( Weights ) );
	Weightsvec.setZero();

	for(int i = 0; i<dydtWRMSvec.size(); i++)
	{
		Weightsvec[i] = 1.0/(rtol*yVec[i] + absTolVals[i]);
		dydtWRMSvec[i] = Weightsvec[i]*dydtVec[i];
	}

	IDASetMinStep(IDA_mem, 1.0e-6);

	//Solving Loop
	for (tout = t1, iout = 1; iout <= totalSteps; iout++, tout += delta_t) 
	{
		if(iout%stepsPerPrint)std::cout << tout - delta_t << std::endl;
		retval = IDASolve(IDA_mem, tout, &tret, Y, dYdt, IDA_NORMAL);
		if(ErrorChecker::check_retval(&retval, "IDASolve", 1)) 
		{
			print(out0, tret, nOut, 0);
			throw std::runtime_error("IDASolve could not complete");
		}

		if(iout%stepsPerPrint == 0)
		{
			print(out0, tout, nOut, Y );

			// Diagnostics go here
		}
	}

	std::cerr << "Total number of steps taken = " << total_steps << std::endl;

	IDAFree( &IDA_mem );

	// No SunLinSol wrapper classes exist beyond this point, so we are safe in using raw pointers to construct them.
	SUNLinSolFree( LS );
	
	// Free the raw data buffers allocated by SUNDIALS
	N_VDestroy( Y );
	N_VDestroy( dYdt );
	N_VDestroy( constraints );
	N_VDestroy( id );
	N_VDestroy( res );
	N_VDestroy( absTolVec );

}

int EmptyJac(realtype tt, realtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
	//This function is purely superficial
	//Sundials looks for a Jacobian, but our Jacobian equation is solved without computing the jacobian. 
	//So we pass a fake one to sundials to prevent an error
	return 0;
}


