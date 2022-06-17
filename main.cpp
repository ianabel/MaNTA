#include <ida/ida.h>                   /* prototypes for IDA fcts., consts.    */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_band.h>  /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h>  /* access to band SUNLinearSolver       */
#include <sundials/sundials_types.h>   /* definition of type realtype          */

#include "SystemSolver.hpp"
#include "gridStructures.hpp"
#include "SunLinSolWrapper.hpp"

int main()
{
	//Defining constants
	const sunindextype polyCount = 3;		//Polynomial degree of each cell
	const sunindextype nCells = 50;		//Total number of cells
	N_Vector u = NULL;				//vector for storing solution
	SUNLinearSolver LS = NULL;		//linear solver memory structure
	void *arkode_mem   = NULL;		//ARKODE memory structure
	const double lBound = 0, uBound = 10;	//Spacial bounds

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
	

}