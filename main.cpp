#include <ida/ida.h>                   /* prototypes for IDA fcts., consts.    */
#include <functional>

#include "SystemSolver.hpp"

void runSolver( SystemSolver& system, const sunindextype k, const sunindextype nCells, int nOut, double tFinal, realtype rtol, realtype atol, Fn u_0, double lBound, double uBound, bool printToFile = true);

int main()
{
	//---------------------------Variable assiments-------------------------------
	const sunindextype k = 3;		//Polynomial degree of each cell
	const sunindextype nCells = 100;			//Total number of cells
	const double lBound = 0.0, uBound = 10;	//Spacial bounds
	int nOut = 20;
	double tFinal = 1.0, delta_t = 0.001;
	realtype rtol = 1.0e-5, atol = 1.0e-5;

	const double c_const = 0.0;
	const double kappa_const = 1.0;
	std::function<double( double )> f = [ = ]( double x ){ 
		return 0.0;
		// return 2.0 * ( 2.0 * x*x*x*x - 7.0 * x*x + 2.0 ) * ::exp( -x*x ); 
	};
	std::function<double( double )> c = [ = ]( double x ){ return c_const;};
	std::function<double( double )> kappa = [ = ]( double x ){ return kappa_const;};
	std::function<double( double )> tau = [ & ]( double x ){ return ( ::fabs( c( x ) ) + kappa( x )/2.0 );};


	double a = 5.0;
	double b = 4.0; 
	std::function<double( double )> u_0 = [=]( double y ){ return ::exp( -b*( y - a )*( y - a ) ); };

	const Grid grid(lBound, uBound, nCells);
	SystemSolver system(grid, k, nCells, delta_t, f, tau, c, kappa);

	runSolver(system, k, nCells, nOut, tFinal, rtol, atol, u_0, lBound, uBound);
}