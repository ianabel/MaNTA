#include <ida/ida.h>
#include <functional>
#include <iostream>
#include <fstream>

#include "SystemSolver.hpp"
#include "ErrorTester.hpp"
#include "SystemSolver.hpp"
#include "math.h"

void runSolver( SystemSolver& system, const sunindextype k, const sunindextype nCells, const sunindextype nVar, int nOut, double tFinal, realtype rtol, realtype atol, Fn gradu_0, Fn u_0, double lBound, double uBound, bool printToFile = true);
void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj);

int main()
{
	//---------------------------Variable assiments-------------------------------
	const sunindextype k = 3;		//Polynomial degree of each cell
	const sunindextype nVar = 2;
	const double lBound = 0.0, uBound = 10;	//Spacial bounds
	const double L = uBound - lBound;
	int nOut = 300;
	double tFinal = 0.5, delta_t = 0.1;
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
	std::function<double( double )> gradu_0 = [=]( double y ){ return -2*a*(y - 5)*::exp( -b*( y - a )*( y - a ) ); };


	std::function<double( double, double )> uExact = [=]( double y, double t ){ return ::exp( -b*( y - a )*( y - a )/(1+4*b*t) )/( ::sqrt(4*b + 1/t)*::sqrt(t) ); };
	std::function<double( double, double )> qExact = [=]( double y, double t ){ return ::exp( -b*( y - a )*( y - a )/(1 + 4*b*t) )*(a - y)/( ::sqrt(4*b + 1/t)*::sqrt(t)*(1 + 4*b*t) ); };

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

	for(int nCells = 10; nCells<=11; nCells+=10)
	{
		const Grid grid(lBound, uBound, nCells);
		SystemSolver system(grid, k, nCells, nVar, delta_t, f, tau, c);
		
		//set the diffusion object
		auto diffobj = std::make_shared< DiffusionObj >(k, nVar);
		buildDiffusionObj(diffobj);
		system.setDiffobj(diffobj);

		runSolver(system, k, nCells, nVar, nOut, tFinal, rtol, atol, gradu_0, u_0, lBound, uBound);

		errorTest.L2Norm(k, nCells, nVar, system);
		errorTest.H1SemiNorm(k, nCells, nVar, system);
		std::cout << nCells << std::endl;

		std::ofstream out0( "u_t_0_" + std::to_string(nCells) + ".plot" );
		std::ofstream out1( "u_t_1_" + std::to_string(nCells) + ".plot" );
		system.print(out0, tFinal, nOut, 0);
		system.print(out1, tFinal, nOut, 1);
	}

	std::cout.precision(17);
	for(int var = 0; var < nVar; var++)
	{
		std::cout << "variable: " << var << std::endl << "L2 Norm:" << std::endl;
		for (auto& val : errorTest.L2vals[var])
			std::cout << val.first << " " << val.second << std::endl;

		std::cout << std::endl << "H1 Semi-Norm" << std::endl;
		for (auto& val : errorTest.H1vals[var])
			std::cout << val.first << " " << val.second << std::endl;
	}
}

void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj)
{
	auto nVar = diffobj->nVar;
	if(nVar != 2) throw std::runtime_error("check your kappa build, you did it wrong.");

	diffobj->clear();
	double kappa_const = 1.0;

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 1.5*q(x,0) - 0.5*q(x,1);};
	std::function<double( double, DGApprox, DGApprox )> kappa1 = [ = ]( double x, DGApprox q, DGApprox u ){ return -0.5*q(x,0) + 1.5*q(x,1);};
	diffobj->kappaFuncs.push_back(kappa0);
	diffobj->kappaFuncs.push_back(kappa1);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 1.5;};
	std::function<double( double, DGApprox, DGApprox )> dkappa0dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return -0.5;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return -0.5;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 1.5;};

	std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa0du1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1du1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	diffobj->delqKappaFuncs.resize(nVar);
	diffobj->deluKappaFuncs.resize(nVar);

	diffobj->delqKappaFuncs[0].push_back(dkappa0dq0);
	diffobj->delqKappaFuncs[0].push_back(dkappa0dq1);
	diffobj->delqKappaFuncs[1].push_back(dkappa1dq0);
	diffobj->delqKappaFuncs[1].push_back(dkappa1dq1);

	diffobj->deluKappaFuncs[0].push_back(dkappa0du0);
	diffobj->deluKappaFuncs[0].push_back(dkappa0du1);
	diffobj->deluKappaFuncs[1].push_back(dkappa1du0);
	diffobj->deluKappaFuncs[1].push_back(dkappa1du1);
}