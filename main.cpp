#include <ida/ida.h>                   /* prototypes for IDA fcts., consts.    */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_band.h>  /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h>  /* access to band SUNLinearSolver       */
#include <sundials/sundials_types.h>   /* definition of type realtype          */

#include "SystemSolver.hpp"
#include "gridStructures.hpp"
#include "SunLinSolWrapper.hpp"
#include "SunMatrixWrapper.hpp"

int main()
{
	const sunindextype polyCount = 3;		//Polynomial degree of each cell
	const sunindextype nCells = 50;			//Total number of cells
	SUNLinearSolver LS = NULL;				//linear solver memory structure
	void *IDA_mem   = NULL;					//IDA memory structure
	const double lBound = 0, uBound = 10;	//Spacial bounds

	N_Vector Y = NULL;				//vector for storing solution
	N_Vector dYdt = NULL;			//vector for storing time derivative of solution
	N_Vector constraints = NULL;	//vector for storing constraints
	N_Vector id = NULL;				//vector for storing id (which elements are algebraic or differentiable)
	N_Vector res = NULL;			//vector for storing residual

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
	SystemSolver system(grid, polyCount, nCells, delta_t, f, tau, c, kappa, DirichletBCs);

	//To Do:
	//Check initial condition satisfies residual equation
	system.initialiseMatrices();
	double a = 5.0;
	double b = 4.0; 
	std::function<double( double )> u_0 = [=]( double y ){ return ::exp( -b*( y - a )*( y - a ) ); };
	system.setInitialConditions(u_0);

	//To Do: 
	//set up sundials enviornment LS+mat
	//Build residual equaiton system

	UserData *data = new UserData();
	//?figure out what to put in this and initialise?

	// Set global solution vector lengths. 
	const sunindextype global_length = nCells*(k+1) + nCells+1;

	// Choose zero-based (C-style) indexing.
	const sunindextype index_base = 0;

	SUNMatrix sunMat = SunMatrixNew();
	LS = SunLinSolWrapper::SunLinSol(system);

	IDASetLinearSolver(IDA_mem, LS, sunMat); 

	delete data;
}