#include <ida/ida.h>                   /* prototypes for IDA fcts., consts.    */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_band.h>  /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h>  /* access to band SUNLinearSolver       */
#include <sundials/sundials_types.h>   /* definition of type realtype          */
#include <iostream>
#include <fstream>

#include "SystemSolver.hpp"
#include "gridStructures.hpp"
#include "SunLinSolWrapper.hpp"
#include "SunMatrixWrapper.hpp"
#include "ErrorChecker.hpp"

int residual(realtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data);
int EmptyJac(realtype tt, realtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

void runSolver( SystemSolver& system, const sunindextype k, const sunindextype nCells, const sunindextype nVar, int nOut, double tFinal, realtype rtol, realtype atol, Fn gradu_0, Fn u_0, Fn sigma_0, double lBound, double uBound, bool printToFile = true)
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
	double delta_t = system.getdt();
	realtype t0 = 0.0, t1 = delta_t, deltatPrint = 0.1, tout, tret;; 
	double totalSteps = tFinal/delta_t;
	int stepsPerPrint = floor(totalSteps*(deltatPrint/tFinal));

	//-------------------------------------System Design----------------------------------------------

	std::function<double( double, double )> g_D = [ = ]( double x, double t ) {
		if ( x == lBound ) {
			// u(0.0) == a
			return 0.0;
		} else if ( x == uBound ) {
			// u(1.0) == a
			return 0.0;
		}
		throw std::logic_error( "Boundary condition function being eval'd not on boundary ?!" );
	};

	std::function<double( double, double )> g_N = [ = ]( double x, double t ) {
		if ( x == lBound ) {
			// ( q + c u ) . n  a @ x = 0.0
			return 0.0;
		} else if ( x == uBound ) {
			// ( q + c u ) . n  b @ x = 1.0
			return ( double )std::nan( "" );
		}
		throw std::logic_error( "Boundary condition function being eval'd not on boundary ?!" );
	};

	auto DirichletBCs = std::make_shared<BoundaryConditions>();
	DirichletBCs->UpperBound = uBound;
	DirichletBCs->isLBoundDirichlet = true;
	DirichletBCs->isUBoundDirichlet = true;
	DirichletBCs->g_D = g_D;
	DirichletBCs->g_N = g_N;
	system.setBoundaryConditions(DirichletBCs.get());

	SUNContext ctx;
    retval = SUNContext_Create(nullptr, &ctx);

	IDA_mem = IDACreate(ctx);
	if(ErrorChecker::check_retval((void *)IDA_mem, "IDACreate", 0)) 
		throw std::runtime_error("Sundials Initialization Error");

	UserData *data = new UserData();
	if(ErrorChecker::check_retval((void *)data, "malloc", 2))
		throw std::runtime_error("Sundials Initialization Error");


	data = (UserData*) malloc(sizeof *data);
	data->system = &system;
	retval = IDASetUserData(IDA_mem, data);
	if(ErrorChecker::check_retval(&retval, "IDASetUserData", 1))
		throw std::runtime_error("Sundials Initialization Error");

	//-----------------------------Initial conditions-------------------------------

	system.initialiseMatrices();

	//Set original vector lengths
	Y = N_VNew_Serial(nVar*3*nCells*(k+1) + nVar*(nCells+1), ctx);
	if(ErrorChecker::check_retval((void *)Y, "N_VNew_Serial", 0))
		throw std::runtime_error("Sundials Initialization Error");
	
	dYdt = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)dYdt, "N_VClone", 0))
		throw std::runtime_error("Sundials Initialization Error");
	VectorWrapper yVec( N_VGetArrayPointer( Y ), N_VGetLength( Y ) ); 
	VectorWrapper dydtVec( N_VGetArrayPointer( Y ), N_VGetLength( Y ) ); 
	yVec.setZero();
	dydtVec.setZero();

	//Initialise Y and dYdt
	system.setInitialConditions(u_0, gradu_0, sigma_0, Y, dYdt);


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
	retval = IDASStolerances(IDA_mem, rtol, atol);
	if(ErrorChecker::check_retval(&retval, "IDASStolerances", 1)) 
		std::runtime_error("Sundials initialization Error, run in debug to find");

	//--------------set up user-built objects------------------

	//Use empty SunMatrix Object
	SUNMatrix sunMat = SunMatrixNew(ctx);
	LS = SunLinSolWrapper::SunLinSol(system, IDA_mem, ctx);
 
	int err = IDASetLinearSolver(IDA_mem, LS, sunMat); 
	IDASetJacFn(IDA_mem, EmptyJac);
	
	//------------------------------Solve------------------------------
	std::ofstream out0( "u_t_0.plot" );
	std::ofstream out1( "u_t_1.plot" );


	
	if(printToFile)
	{
		system.print(out0, t0, nOut, 0);
		if(nVar > 1) system.print(out1, t0, nOut, 1);
	}
	//Update initial solution to be within tolerance of the residual equation
	retval = IDACalcIC(IDA_mem, IDA_YA_YDP_INIT, delta_t);
	if(ErrorChecker::check_retval(&retval, "IDASolve", 1)) 
	{
		system.print(out0, t0, nOut, 0);
		throw std::runtime_error("IDACalcIC could not complete");
	}

	for (tout = t1, iout = 1; iout <= totalSteps; iout++, tout += delta_t) 
	{
		std::cout << tout << std::endl;
		retval = IDASolve(IDA_mem, tout, &tret, Y, dYdt, IDA_NORMAL);
		if(ErrorChecker::check_retval(&retval, "IDASolve", 1)) 
		{
			system.print(out0, t0, nOut, 0);
			throw std::runtime_error("IDASolve could not complete");
		}

		if( printToFile && iout%stepsPerPrint == 0)
		{
			system.print(out0, tout, nOut, 0);
			if(nVar > 1) system.print(out1, tout, nOut, 1);
		}
	}

	delete data, IDA_mem;
}

int EmptyJac(realtype tt, realtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
	//This function is purely superficial
	//Sundials looks for a Jacobian, but our Jacobian equation is solved without computing the jacobian. 
	//So we pass a fake one to sundials to prevent an error
	return 0;
}

