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

int main()
{
	//---------------------------Variable assiments-------------------------------
	const sunindextype k = 3;		//Polynomial degree of each cell
	const sunindextype nCells = 60;			//Total number of cells
	SUNLinearSolver LS = NULL;				//linear solver memory structure
	void *IDA_mem   = NULL;					//IDA memory structure
	const double lBound = 0.0, uBound = 10;	//Spacial bounds
	int retval, iout;
	int nOut = 100;

	N_Vector Y = NULL;				//vector for storing solution
	N_Vector dYdt = NULL;			//vector for storing time derivative of solution
	N_Vector constraints = NULL;	//vector for storing constraints
	N_Vector id = NULL;				//vector for storing id (which elements are algebraic or differentiable)
	N_Vector res = NULL;			//vector for storing residual
	const double delta_t = 0.001;
	realtype rtol = 1.0e-5, atol = 1.0e-5, t0 = 0.0, t1 = delta_t, tFinal = 0.5, deltatPrint = 0.1, tout, tret;; 
	double totalSteps = tFinal/delta_t;
	int stepsPerPrint = floor(totalSteps*(deltatPrint/tFinal));

	//-------------------------------------System Design----------------------------------------------

	std::function<double( double )> g_D = [ = ]( double x ) {
		if ( x == lBound ) {
			// u(0.0) == a
			return 0.0;
		} else if ( x == uBound ) {
			// u(1.0) == a
			return 0.0;
		}
		throw std::logic_error( "Boundary condition function being eval'd not on boundary ?!" );
	};

	std::function<double( double )> g_N = [ = ]( double x ) {
		if ( x == lBound ) {
			// ( q + c u ) . n  a @ x = 0.0
			return 0.0;
		} else if ( x == uBound ) {
			// ( q + c u ) . n  b @ x = 1.0
			return ( double )std::nan( "" );
		}
		throw std::logic_error( "Boundary condition function being eval'd not on boundary ?!" );
	};

	const double c_const = 0.0;
	const double kappa_const = 1.0;
	std::function<double( double )> f = [ = ]( double x ){ 
		return 0.0;
		// return 2.0 * ( 2.0 * x*x*x*x - 7.0 * x*x + 2.0 ) * ::exp( -x*x ); 
	};
	std::function<double( double )> c = [ = ]( double x ){ return c_const;};
	std::function<double( double )> kappa = [ = ]( double x ){ return kappa_const;};
	std::function<double( double )> tau = [ & ]( double x ){ return ( ::fabs( c( x ) ) + kappa( x )/2.0 );};

	BoundaryConditions DirichletBCs;
	DirichletBCs.LowerBound = lBound;
	DirichletBCs.UpperBound = uBound;
	DirichletBCs.isLBoundDirichlet = true;
	DirichletBCs.isUBoundDirichlet = true;
	DirichletBCs.g_D = g_D;
	DirichletBCs.g_N = g_N;


	const Grid grid(lBound, uBound, nCells);
	SystemSolver system(grid, k, nCells, delta_t, f, tau, c, kappa, DirichletBCs);

	IDA_mem = IDACreate();
	if(ErrorChecker::check_retval((void *)IDA_mem, "IDACreate", 0)) return(1);

	UserData *data = new UserData();
	if(ErrorChecker::check_retval((void *)data, "malloc", 2))
		return -1;

	data = (UserData*) malloc(sizeof *data);
	data->system = &system;
	retval = IDASetUserData(IDA_mem, data);
	if(ErrorChecker::check_retval(&retval, "IDASetUserData", 1)) return(1);

	//-----------------------------Initial conditions-------------------------------

	system.initialiseMatrices();
	double a = 5.0;
	double b = 4.0; 
	std::function<double( double )> u_0 = [=]( double y ){ return ::exp( -b*( y - a )*( y - a ) ); };

	//Set original vector lengths
	Y = N_VNew_Serial(2*nCells*(k+1));
	if(ErrorChecker::check_retval((void *)Y, "N_VNew_Serial", 0)) return(1);
	dYdt = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)dYdt, "N_VClone", 0)) return(1);

	//Initialise Y and dYdt
	system.setInitialConditions(u_0, Y, dYdt);

	VectorWrapper yVec( N_VGetArrayPointer( Y ), N_VGetLength( Y ) ); 

	// ----------------- Allocate and initialize all other sun-vectors. -------------

	res = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)res, "N_VClone", 0))
		return -1;
	realtype tRes;

	//No constraints are imposed as negative coefficients may allow for a better fit across a cell
	constraints = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)constraints, "N_VClone", 0))
		return -1;

	//Specify only u as differential
	id = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)id, "N_VClone", 0))
		return -1;
	realtype* idVal = N_VGetArrayPointer(id);
	for(int i=0; i < nCells; i++)
	{
		for(int j=0; j<k+1; j++)
		{
			idVal[i*2*(k+1)+k+1+j] = 1.0;//U vals
			idVal[i*2*(k+1)+j] = 0.0;//Q vals
		}
	}
	retval = IDASetId(IDA_mem, id);
	if(ErrorChecker::check_retval(&retval, "IDASetId", 1)) return(1);

	//Initialise IDA 
	retval = IDAInit(IDA_mem, residual, t0, Y, dYdt); 
	if(ErrorChecker::check_retval(&retval, "IDAInit", 1)) return(1);

	//Set tolerances
	retval = IDASStolerances(IDA_mem, rtol, atol);
	if(ErrorChecker::check_retval(&retval, "IDASStolerances", 1)) return(1);

	//--------------set up user-built objects------------------

	//Use empty SunMatrix Object
	SUNMatrix sunMat = SunMatrixNew();
	LS = SunLinSolWrapper::SunLinSol(system, IDA_mem);
 
	int err = IDASetLinearSolver(IDA_mem, LS, sunMat); 
	IDASetJacFn(IDA_mem, EmptyJac);
	
	//------------------------------Solve------------------------------

	//Update initial solution to be within tolerance of the residual equation
	IDACalcIC(IDA_mem, IDA_YA_YDP_INIT, delta_t);
	
	std::ofstream out( "u_t.plot" );
	system.print(out, t0, nOut);

	for (tout = t1, iout = 1; iout <= totalSteps; iout++, tout += delta_t) 
	{
		retval = IDASolve(IDA_mem, tout, &tret, Y, dYdt, IDA_NORMAL);
		if(ErrorChecker::check_retval(&retval, "IDASolve", 1)) 
			return -1;
		if(iout%stepsPerPrint == 0)
		{
			system.print(out, tout, nOut);
		}
	}

	delete data;
}

int EmptyJac(realtype tt, realtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
	//This function is purely superficial
	//Sundials looks for a Jacobian, but our Jacobian equation is solved without computing the jacobian. 
	//So we pass a fake one to sundials to prevent an error
	return 0;
}

double EvalCoeffs( LegendreBasis & B, Coeff_t cs, double x )
{
	for ( auto const & pair : cs )
	{
		if ( pair.first.contains( x ) )
			return B.Evaluate( pair.first, pair.second, x );
	}
	return std::nan( "" );
}
