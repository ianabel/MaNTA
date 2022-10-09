#include <ida/ida.h>
#include <functional>
#include <iostream>
#include <fstream>

#include "SystemSolver.hpp"
#include "ErrorTester.hpp"
#include "SystemSolver.hpp"
#include "math.h"


void runSolver( SystemSolver& system, const sunindextype k, const sunindextype nCells, const sunindextype nVar, int nOut, double tFinal, realtype rtol, realtype atol, Fn gradu_0, Fn u_0, Fn sigma_0, double lBound, double uBound, bool printToFile = true);
void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj);
void buildSourceObj(std::shared_ptr<SourceObj> sourceobj);

int main()
{
	//---------------------------Variable assiments-------------------------------
	const sunindextype k = 3;		//Polynomial degree of each cell
	const sunindextype nVar = 2;
	const double lBound = 0.0, uBound = 1.0;	//Spacial bounds
	const double L = uBound - lBound;
	int nOut = 300;
	double tFinal = 3.0, delta_t = 1.0;
	realtype rtol = 1.0e-6, atol = 1.0e-6;

	const double c_const = 0.0;
	const double kappa_const = 1.0;
	std::function<double( double )> f = [ = ]( double x ){ 
		return 0.0;
		// return 2.0 * ( 2.0 * x*x*x*x - 7.0 * x*x + 2.0 ) * ::exp( -x*x ); 
	};
	std::function<double( double )> c = [ = ]( double x ){ return c_const;};
	std::function<double( double )> kappa = [ = ]( double x ){ return kappa_const;};
	std::function<double( double )> tau = [ & ]( double x ){ return ( 0.5 );};


	double a = 5.0;
	double b = 4.0; 
	double beta = 1.0;
	std::function<double( double )> u_0 = [=]( double y ){ return ::exp( -b*( y - a )*( y - a ) ); };
	std::function<double( double )> gradu_0 = [=]( double y ){ return -2*a*(y - 5)*::exp( -b*( y - a )*( y - a ) ); };
	//std::function<double( double, double )> uExact = [=]( double y, double t ){ return ::exp( -b*( y - a )*( y - a )/(1+4*b*t) )/( ::sqrt(4*b + 1/t)*::sqrt(t) ); };
	//std::function<double( double, double )> qExact = [=]( double y, double t ){ return ::exp( -b*( y - a )*( y - a )/(1 + 4*b*t) )*(a - y)/( ::sqrt(4*b + 1/t)*::sqrt(t)*(1 + 4*b*t) ); };
	
	//std::function<double( double )> u_0 = [=]( double y ){ return ::exp( -b*( y - a )*( y - a ) ); }; //gaussian
	//std::function<double( double )> gradu_0 = [=]( double y ){ return (1.0+::tanh(::sqrt(1.0/48.0)*y))/((::pow(::cosh(::sqrt(1.0/48.0)*y),2.0))*16.0*::sqrt(3.0)); }; //Fisher Example - exact Sol
	//std::function<double( double )> sigma_0 = [=]( double y ){ return -1.0*gradu_0(y); }; //Fisher case

	//std::function<double( double )> u_0 = [=]( double y ){ return -0.125*(::pow(::cosh(::sqrt(1.0/48.0)*y),-2.0) - 2.0*::tanh(::sqrt(1.0/48.0)*y) - 2.0); }; //Fisher example - exact sol
	//std::function<double( double )> gradu_0 = [=]( double y ){ return -2*b*(y - a)*::exp( -b*( y - a )*( y - a ) ); }; //gaussian
	//std::function<double( double )> sigma_0 = [=]( double y ){ return -1.0*(1.0 + u_0(y)*u_0(y))*gradu_0(y) - u_0(y)*u_0(y)*u_0(y)*gradu_0(y); };
	//std::function<double( double, double )> uExact = [=]( double y, double t ){ return -0.125*(::pow(::cosh(::sqrt(1.0/48.0)*y + 5.0*t/24.0),-2.0) - 2.0*::tanh(::sqrt(1.0/48.0)*y + 5.0*t/24) - 2.0); };
	//std::function<double( double, double )> qExact = [=]( double y, double t ){ return (1.0+::tanh(::sqrt(1.0/48.0)*y + 5.0*t/24.0))/((::pow(::cosh(::sqrt(1.0/48.0)*y + 5.0*t/24.0),2.0))*16.0*::sqrt(3.0)); };
	
	//std::function<double( double )> u_0 = [=]( double y ){ return ::sin(M_PI*y/L); };
	//std::function<double( double, double )> uExact = [=]( double y, double t ){ return ::exp(-M_PI*M_PI/(L*L)*kappa(y)*t)*::sin(M_PI*y/L); };
	//std::function<double( double, double )> qExact = [=]( double y, double t ){ return (M_PI/L)*::exp(-M_PI*M_PI/(L*L)*kappa(y)*t)*::cos(M_PI*y/L); };

	/*
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
	*/

	ErrorTester errorTest(uExact, qExact, tFinal);
	errorTest.setBounds(lBound, uBound);

	for(int nCells = 10; nCells<=20; nCells+=10)
	{
		const Grid grid(lBound, uBound, nCells);
		SystemSolver system(grid, k, nCells, nVar, delta_t, f, tau, c);
		
		//set the diffusion object
		auto diffobj = std::make_shared< DiffusionObj >(k, nVar);
		buildDiffusionObj(diffobj);
		system.setDiffobj(diffobj);

		auto sourceobj = std::make_shared< SourceObj >(k, nVar);
		buildSourceObj(sourceobj);
		system.setSourceobj(sourceobj);

		runSolver(system, k, nCells, nVar, nOut, tFinal, rtol, atol, gradu_0, u_0, sigma_0, lBound, uBound);

		errorTest.L2Norm(k, nCells, nVar, system);
		errorTest.H1SemiNorm(k, nCells, nVar, system);
		std::cout << nCells << std::endl;

		std::ofstream out0( "u_t_0_" + std::to_string(nCells) + ".plot" );
		std::ofstream out1( "u_t_1_" + std::to_string(nCells) + ".plot" );
		system.print(out0, tFinal, nOut, 0);
	}

	std::cout.precision(17);
	for(int var = 0; var < nVar; var++)
	{
		for (auto& val : errorTest.L2vals[var])
			std::cout << val.first << std::endl;

		std::cout << "variable: " << var << std::endl << "L2 Norm:" << std::endl;
		for (auto& val : errorTest.L2vals[var])
			std::cout << val.second << std::endl;

		std::cout << std::endl << "H1 Semi-Norm" << std::endl;
		for (auto& val : errorTest.H1vals[var])
			std::cout << val.second << std::endl;
	}
}
