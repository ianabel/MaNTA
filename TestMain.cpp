#include <ida/ida.h>
#include <functional>
#include <iostream>
#include <fstream>

#include "SystemSolver.hpp"
#include "ErrorTester.hpp"
#include "SystemSolver.hpp"
#include "math.h"

void runSolver( SystemSolver& system, const sunindextype k, const sunindextype nCells, int nOut, double tFinal, realtype rtol, realtype atol, Fn u_0, double lBound, double uBound, bool printToFile = true);

int main()
{
	//---------------------------Variable assiments-------------------------------
	const sunindextype k = 3;		//Polynomial degree of each cell
	const double lBound = 0.0, uBound = 10;	//Spacial bounds
	const double L = uBound - lBound;
	int nOut = 20;
	double tFinal = 0.1, delta_t = 0.001;
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


	//double a = 5.0;
	//double b = 4.0; 
	//std::function<double( double )> u_0 = [=]( double y ){ return ::exp( -b*( y - a )*( y - a ) ); };

	//std::function<double( double, double )> uExact = [=]( double y, double t ){ return ::exp( -b*( y - a )*( y - a )/(1+4*b*t) )/( ::sqrt(4*b + 1/t)*::sqrt(t) ); };
	//std::function<double( double, double )> qExact = [=]( double y, double t ){ return ::exp( -b*( y - a )*( y - a )/(1 + 4*b*t) )*(a - y)/( ::sqrt(4*b + 1/t)*::sqrt(t)*(1 + 4*b*t) ); };

	//std::function<double( double )> u_0 = [=]( double y ){ return ::sin(M_PI*y/L); };
	//std::function<double( double, double )> uExact = [=]( double y, double t ){ return ::exp(-M_PI*M_PI/(L*L)*kappa(y)*t)*::sin(M_PI*y/L); };
	//std::function<double( double, double )> qExact = [=]( double y, double t ){ return (M_PI/L)*::exp(-M_PI*M_PI/(L*L)*kappa(y)*t)*::cos(M_PI*y/L); };

	std::function<double( double )> u_0 = [=]( double y )
		{ return ::sin(M_PI*y/L)
		+5*::sin(2*M_PI*y/L)
		+13*::sin(5*M_PI*y/L); };
	std::function<double( double, double )> uExact = [=]( double y, double t )
		{ return ::exp(-M_PI*M_PI/(L*L)*kappa(y)*t)*::sin(M_PI*y/L)
		+5*::exp(-4*M_PI*M_PI/(L*L)*kappa(y)*t)*::sin(2*M_PI*y/L)
		+13*::exp(-25*M_PI*M_PI/(L*L)*kappa(y)*t)*::sin(5*M_PI*y/L); };
	std::function<double( double, double )> qExact = [=]( double y, double t )
		{ return (M_PI/L)*::exp(-M_PI*M_PI/(L*L)*kappa(y)*t)*::cos(M_PI*y/L)
		+(10*M_PI/L)*::exp(-4*M_PI*M_PI/(L*L)*kappa(y)*t)*::cos(2*M_PI*y/L)
		+(65*M_PI/L)*::exp(-25*M_PI*M_PI/(L*L)*kappa(y)*t)*::cos(5*M_PI*y/L); };

	ErrorTester errorTest(uExact, qExact, tFinal);
	errorTest.setBounds(lBound, uBound);

	for(int nCells = 30; nCells<=200; nCells+=10)
	{
		const Grid grid(lBound, uBound, nCells);
		SystemSolver system(grid, k, nCells, delta_t, f, tau, c, kappa);
		runSolver(system, k, nCells, nOut, tFinal, rtol, atol, u_0, lBound, uBound, false);

		errorTest.L2Norm(k, nCells, system);
		errorTest.H1SemiNorm(k, nCells, system);
		std::cout << nCells << std::endl;
	}

	std::cout.precision(17);
	std::cout << "L2 Norm" << std::endl;
	for (auto& val : errorTest.L2vals)
		std::cout << val.first << " " << val.second << std::endl;

	std::cout << std::endl << "H1 Semi-Norm" << std::endl;
	for (auto& val : errorTest.H1vals)
		std::cout << val.first << " " << val.second << std::endl;
}

