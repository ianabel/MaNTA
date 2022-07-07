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
	const sunindextype k = 3;		//Polynomial degree of each cell
	const sunindextype nCells = 4;			//Total number of cells
	SUNLinearSolver LS = NULL;				//linear solver memory structure
	void *IDA_mem   = NULL;					//IDA memory structure
	const double lBound = 0.0, uBound = 10;	//Spacial bounds
	int retval, iout;
	int nOut = 10;

	N_Vector Y = NULL;				//vector for storing solution
	N_Vector dYdt = NULL;			//vector for storing time derivative of solution
	N_Vector constraints = NULL;	//vector for storing constraints
	N_Vector id = NULL;				//vector for storing id (which elements are algebraic or differentiable)
	N_Vector res = NULL;			//vector for storing residual
	realtype rtol = 1.0e-5, atol = 1.0e-3, t0 = 0.0, t1 = 0.01, tout, tret;; //?rtol is 0.0 in examples?

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

	const double c_const = 0;
	const double kappa_const = 1;
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

	const double delta_t = 0.009;

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

	//To Do:
	//Check initial condition satisfies residual equation
	system.initialiseMatrices();
	double a = 5.0;
	double b = 4.0; 
	std::function<double( double )> u_0 = [=]( double y ){ return ::exp( -b*( y - a )*( y - a ) ); };

	Y = N_VNew_Serial(2*nCells*(k+1) + nCells + 1);
	if(ErrorChecker::check_retval((void *)Y, "N_VNew_Serial", 0)) return(1);
	dYdt = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)dYdt, "N_VClone", 0)) return(1);

	system.setInitialConditions(u_0, Y, dYdt);

	// Set global solution vector lengths. 
	//const sunindextype global_length = nCells*(k+1) + nCells+1; ?delete?

	// Choose zero-based (C-style) indexing.
	// const sunindextype index_base = 0; ?delete?

	/* Allocate and initialize N-vectors. */
	//Already initilised Y and dYdt

	res = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)res, "N_VClone", 0))
		return -1;
	realtype tRes;
	residual(tRes,Y, dYdt, res, data);

	//impose constraints for nonnegative u solution values
	constraints = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)constraints, "N_VClone", 0))
		return -1;
	VectorWrapper constraintsVec( N_VGetArrayPointer( constraints ), N_VGetLength( constraints ) );
	for(int i=nCells*(k+1); i < 2*nCells*(k+1) + 2; i++) constraintsVec[i] = 1.0; //u and lambda variables are positive

	id = N_VClone(Y);
	if(ErrorChecker::check_retval((void *)id, "N_VClone", 0))
		return -1;
	realtype* idVal = N_VGetArrayPointer(id);
	for(int i=nCells*(k+1); i < 2*nCells*(k+1); i++) idVal[i] = 1.0; // u valuables are the only differential variables

	/* Set which components are algebraic or differential */
	retval = IDASetId(IDA_mem, id);
	if(ErrorChecker::check_retval(&retval, "IDASetId", 1)) return(1);

	retval = IDAInit(IDA_mem, residual, t0, Y, dYdt); 
	if(ErrorChecker::check_retval(&retval, "IDAInit", 1)) return(1);

	retval = IDASStolerances(IDA_mem, rtol, atol);
	if(ErrorChecker::check_retval(&retval, "IDASStolerances", 1)) return(1);

	SUNMatrix sunMat = SunMatrixNew();
	LS = SunLinSolWrapper::SunLinSol(system, IDA_mem);
 
	int err = IDASetLinearSolver(IDA_mem, LS, sunMat); 
	IDASetJacFn(IDA_mem, EmptyJac);

	//?Currently don't call IDACalcIC but might come in useful. Defo for transport calculations?
	std::ofstream out( "u_t.plot" );
	system.print(out, t0, nOut);

	for (tout = t1, iout = 1; iout <= 11; iout++, tout *= 2.0) 
	{
		
		retval = IDASolve(IDA_mem, tout, &tret, Y, dYdt, IDA_NORMAL);
		if(ErrorChecker::check_retval(&retval, "IDASolve", 1))
			return -1;

		system.print(out, tout, nOut);
	}

	delete data;
}

int EmptyJac(realtype tt, realtype cj, N_Vector yy, N_Vector yp, N_Vector rr, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
	static_cast<UserData*>(user_data)->system->setAlpha(cj);
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
