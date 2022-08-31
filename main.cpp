#include <ida/ida.h>                   /* prototypes for IDA fcts., consts.    */
#include <functional>

#include "SystemSolver.hpp"

void runSolver( SystemSolver& system, const sunindextype k, const sunindextype nCells, const sunindextype nVar, int nOut, double tFinal, realtype rtol, realtype atol, Fn u_0, double lBound, double uBound, bool printToFile = true);

int main()
{
	//---------------------------Variable assiments-------------------------------
	const sunindextype k = 3;		//Polynomial degree of each cell
	const sunindextype nCells = 50;			//Total number of cells
	const sunindextype nVar = 3;
	const double lBound = 0.0, uBound = 10;	//Spacial bounds
	int nOut = 100;
	double tFinal = 0.5, delta_t = 0.01;
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

	Eigen::MatrixXd kappaMat = Eigen::MatrixXd::Identity(nVar,nVar)*kappa_const;
	kappaMat(0,0) = 1.0;
	kappaMat(0,1) = 0.0;
	kappaMat(0,2) = 0.0;
	kappaMat(1,0) = 0.0;
	kappaMat(1,1) = 1.0;
	kappaMat(1,2) = 0.0;
	kappaMat(2,0) = 0.0;
	kappaMat(2,1) = 0.0;
	kappaMat(2,2) = 1.0;


	double a = 5.0;
	double b = 4.0; 
	std::function<double( double )> u_0 = [=]( double y ){ return ::exp( -b*( y - a )*( y - a ) ); };

	const Grid grid(lBound, uBound, nCells);
	SystemSolver system(grid, k, nCells, nVar, delta_t, f, tau, c, kappaMat);

	runSolver(system, k, nCells, nVar, nOut, tFinal, rtol, atol, u_0, lBound, uBound);
}