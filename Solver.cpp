#include <ida/ida.h>                   /* prototypes for IDA fcts., consts.    */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_band.h>  /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h>  /* access to band SUNLinearSolver       */
#include <sundials/sundials_types.h>   /* definition of type realtype          */
#include <toml.hpp>
#include <iostream>
#include <fstream>
#include <memory>

#include "SystemSolver.hpp"
#include "gridStructures.hpp"
#include "SunLinSolWrapper.hpp"
#include "SunMatrixWrapper.hpp"
#include "ErrorChecker.hpp"
#include "Diagnostic.hpp"

int residual(realtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data);
int EmptyJac(realtype tt, realtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

void runSolver( std::shared_ptr<SystemSolver> system, std::string const& inputFile )
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
	realtype rtol, atol;
	int nCells = system->nCells, nVar = system->nVar, k = system->k;
	bool printToFile = true;
	realtype t0 = 0.0, t1, tout, tret;


	//--------------------------------------Read File---------------------------------------
	const auto configFile = toml::parse( inputFile );
	const auto config = toml::find<toml::value>( configFile, "configuration" );

	realtype deltatPrint;
	auto printdt = toml::find(config, "delta_t");
	if( !config.count("delta_t") == 1 ) throw std::invalid_argument( "delta_t unspecified or specified more than once" );
	else if( printdt.is_integer() ) deltatPrint = static_cast<double>(printdt.as_floating());
	else if( printdt.is_floating() ) deltatPrint = static_cast<double>(printdt.as_floating());
	else throw std::invalid_argument( "delta_t specified incorrrectly" );

	double delta_t = deltatPrint*0.5;
	t1 = delta_t;

	auto tEnd = toml::find(config, "t_final");
	if( !config.count("t_final") == 1 ) throw std::invalid_argument( "tEnd unspecified or specified more than once" );
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
	if( !config.count("Relative_tolerance") == 1 ) rtol = 1.0e-5;
	else if( relTol.is_integer() ) rtol = static_cast<double>(relTol.as_floating());
	else if( relTol.is_floating() ) rtol = static_cast<double>(relTol.as_floating());
	else throw std::invalid_argument( "relative_tolerance specified incorrrectly" );

	auto absTol = toml::find(config, "Absolute_tolerance");
	if( !config.count("Absolute_tolerance") == 1 ) atol = 1.0e-5;
	else if( absTol.is_integer() ) atol = static_cast<double>(absTol.as_floating());
	else if( absTol.is_floating() ) atol = static_cast<double>(absTol.as_floating());
	else throw std::invalid_argument( "Absolute_tolerance specified incorrrectly" );
	//-------------------------------------System Design----------------------------------------------
	SUNContext ctx;
    retval = SUNContext_Create(nullptr, &ctx);

	IDA_mem = IDACreate(ctx);
	if(ErrorChecker::check_retval((void *)IDA_mem, "IDACreate", 0)) 
		throw std::runtime_error("Sundials Initialization Error");

	UserData *data = new UserData();
	if(ErrorChecker::check_retval((void *)data, "malloc", 2))
		throw std::runtime_error("Sundials Initialization Error");


	data = (UserData*) malloc(sizeof *data);
	data->system = system;
	retval = IDASetUserData(IDA_mem, data);
	if(ErrorChecker::check_retval(&retval, "IDASetUserData", 1))
		throw std::runtime_error("Sundials Initialization Error");

	//-----------------------------Initial conditions-------------------------------

	system->initialiseMatrices();

	//Set original vector lengths
	Y = N_VNew_Serial(nVar*3*nCells*(k+1) + nVar*(nCells+1), ctx);
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
	system->setInitialConditions(Y, dYdt);
	
	// ----------------- Allocate and initialize all other sun-vectors. -------------

	res = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)res, "N_VClone", 0))
		std::runtime_error("Sundials initialization Error, run in debug to find");
	realtype tRes;

	//No constraints are imposed as negative coefficients may allow for a better fit across a cell
	constraints = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)constraints, "N_VClone", 0))
		std::runtime_error("Sundials initialization Error, run in debug to find");

	//Specify only u as differential
	id = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)id, "N_VClone", 0))
		std::runtime_error("Sundials initialization Error, run in debug to find");
	VectorWrapper idVals( N_VGetArrayPointer( id ), nVar*3*nCells*(k+1) + nVar*(nCells+1) );
	idVals.setZero();
	for(int i=0; i < nCells; i++)
	{
		for(int j=0; j< nVar*(k+1); j++)
		{
			idVals[ i*3*nVar*(k+1) + 2*nVar*(k+1) + j ] = 1.0;//U vals
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
	VectorWrapper absTolVals( N_VGetArrayPointer( absTolVec ), nVar*3*nCells*(k+1) + nVar*(nCells+1) );
	absTolVals.setConstant(atol);

	retval = IDASVtolerances(IDA_mem, rtol, absTolVec);
	if(ErrorChecker::check_retval(&retval, "IDASVtolerances", 1)) 
		std::runtime_error("Sundials initialization Error, run in debug to find");

	//--------------set up user-built objects------------------

	//Use empty SunMatrix Object
	SUNMatrix sunMat = SunMatrixNew(ctx);
	LS = SunLinSolWrapper::SunLinSol(system, IDA_mem, ctx);
 
	int err = IDASetLinearSolver(IDA_mem, LS, sunMat); 
	IDASetJacFn(IDA_mem, EmptyJac);

	IDASetMaxNonlinIters(IDA_mem, 10);

	//Attach Diagnostics
	Diagnostic diagnostic(system, system->plasma);
	
	//------------------------------Solve------------------------------
	std::ofstream out0( inputFile.substr(0, inputFile.rfind(".")) + ".plot" );
	std::ofstream out1( "Te.plot" );
	std::ofstream out2( "omega.plot" );

	if(printToFile)
	{
		system->print(out0, t0, nOut, 0);
		if(nVar > 1) system->print(out1, t0, nOut, 1);
		if(nVar > 2) system->print(out2, t0, nOut, 2);
	}
	
	IDASetMaxNumSteps(IDA_mem, 50);
	//Update initial solution to be within tolerance of the residual equation
	retval = IDACalcIC(IDA_mem, IDA_YA_YDP_INIT, delta_t);
	if(ErrorChecker::check_retval(&retval, "IDASolve", 1)) 
	{
		system->print(out0, t0, nOut, 0);
		if(nVar > 1) system->print(out1, t0, nOut, 1);
		if(nVar > 2) system->print(out2, t0, nOut, 2);
		throw std::runtime_error("IDACalcIC could not complete");
	}

	//??To Do: remove these when done with them
	std::ofstream file;
	file.open("time.txt");

	std::ofstream resfile0;
	std::ofstream resfile1;
	resfile0.open("res0.txt");
	resfile1.open("res1.txt");

	std::ofstream diagnostics;
	diagnostics.open("diagnostics.txt");

	//Solving Loop
	for (tout = t1, iout = 1; iout <= totalSteps; iout++, tout += delta_t) 
	{
		if(iout%stepsPerPrint)std::cout << tout - delta_t << std::endl;
		retval = IDASolve(IDA_mem, tout, &tret, Y, dYdt, IDA_NORMAL);
		if(ErrorChecker::check_retval(&retval, "IDASolve", 1)) 
		{
			system->print(out0, tout, nOut, 0);
			if(nVar > 1) system->print(out1, t0, nOut, 1);
			if(nVar > 2) system->print(out2, t0, nOut, 2);
			throw std::runtime_error("IDASolve could not complete");
		}

		if(iout%stepsPerPrint == 0)
		{
			system->print(out0, tout, nOut, 0);
			if(nVar > 1) system->print(out1, tout, nOut, 1);
			if(nVar > 2) system->print(out2, t0, nOut, 2);

			//Print Diagnostics
			diagnostics << tout << "	" << diagnostic.Voltage() << std::endl;
		}
	}
	std::cerr << "Total number of steps taken = " << system->total_steps << std::endl;
	delete data, IDA_mem;
}

int EmptyJac(realtype tt, realtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
	//This function is purely superficial
	//Sundials looks for a Jacobian, but our Jacobian equation is solved without computing the jacobian. 
	//So we pass a fake one to sundials to prevent an error
	return 0;
}

