#include <ida/ida.h>                   /* prototypes for IDA fcts., consts.    */
#include <functional>

#include "SystemSolver.hpp"

void runSolver( SystemSolver& system, const sunindextype k, const sunindextype nCells, const sunindextype nVar, int nOut, double tFinal, realtype rtol, realtype atol, Fn gradu_0, Fn u_0, Fn sigma_0, double lBound, double uBound, bool printToFile = true);
void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj);
void buildSourceObj(std::shared_ptr<SourceObj> sourceobj);

/*
int main()
{
	//---------------------------Variable assiments-------------------------------
	std::cerr.precision(17);
	const sunindextype k = 2;		//Polynomial degree of each cell
	const sunindextype nCells = 30;			//Total number of cells
	const sunindextype nVar = 2;
	const double lBound = -10.0, uBound = 10.0;	//Spacial bounds
	double L = uBound - lBound;
	int nOut = 201;
	double tFinal = 5.0, delta_t = 1.0;
	realtype rtol = 1.0e-10, atol = 1.0e-10;

	const double c_const = 0.0;
	std::function<double( double )> f = [ = ]( double x ){ 
		return 0.0;
		// return 2.0 * ( 2.0 * x*x*x*x - 7.0 * x*x + 2.0 ) * ::exp( -x*x ); 
		//return (-1*kappa0/4)*(::exp(-0.75*(x-5)*(x-5)) * (73.0 + 3.0*x*(x-10)) + ::exp(-0.25 * (x-5)*(x-5)) * (23 + x*(x-10))); // For non-linear manufactured solution case, some error from diricelet BCs
	};
	std::function<double( double )> c = [ = ]( double x ){ return c_const;};
	std::function<double( double )> tau = [ & ]( double x ){ return ( 1.0 );};

	double a = 5.0;
	double b = 4.0; 
	double beta = 1.0;
	auto xi = [ ]( double q_ )
	{
		if( ::abs(q_) > 0.5) return 10*::pow( ::abs(q_) - 0.5, 0.5) + 1.0;
		else return 1.0;
	};
	
	std::function<double( double )> u_0 = [=]( double y ){ return ::exp( -b*( y - a )*( y - a ) ); }; //gaussian
	//std::function<double( double )> u_0 = [=]( double y ){ return 1/(::cosh(10*y)*::cosh(10*y)); }; //Fisher example
	//std::function<double( double )> u_0 = [=]( double y ){ return -0.125*(::pow(::cosh(::sqrt(1.0/48.0)*y),-2.0) - 2.0*::tanh(::sqrt(1.0/48.0)*y) - 2.0); }; //Fisher example - exact sol
	//std::function<double( double )> u_0 = [=]( double y ){ return 0.5*(1.0 + ::cos(M_PI*y)); }; //constant u_0
	//std::function<double( double )> u_0 = [=]( double y ){ return 1.0 - 2.0/(1 + ::exp(100*(1-y))); }; //shestakov
	//std::function<double( double )> u_0 = [=]( double y ){ return 1.0-y;}; //PTransp u_0
	//std::function<double( double )> u_0 = [=]( double y ){ return 1.0;};
	//std::function<double( double )> u_0 = [=]( double y ){ return 1; }; //0 u_0

	std::function<double( double )> gradu_0 = [=]( double y ){ return -2.0*b*(y - a)*::exp( -b*( y - a )*( y - a ) ); }; //gaussian
	//std::function<double( double )> gradu_0 = [=]( double y ){ return -20*::tanh(10*y)/(::cosh(10*y)*::cosh(10*y)); }; //Fisher Example
	//std::function<double( double )> gradu_0 = [=]( double y ){ return (1.0+::tanh(::sqrt(1.0/48.0)*y))/((::pow(::cosh(::sqrt(1.0/48.0)*y),2.0))*16.0*::sqrt(3.0)); }; //Fisher Example - exact Sol
	//std::function<double( double )> gradu_0 = [=]( double y ){ return -200*::exp(100*(1-y))/((1 + ::exp(100*(1-y)))*(1 + ::exp(100*(1-y)))); }; //constant u_0
	//std::function<double( double )> gradu_0 = [=]( double y ){ return -0.5*M_PI*::sin(M_PI*y); }; //constant u_0
	//std::function<double( double )> gradu_0 = [=]( double y ){ return -1.0; }; //PTransp u_0
	//std::function<double( double )> gradu_0 = [=]( double y ){ return 0; }; //0 u_0
	
	std::function<double( double )> sigma_0 = [=]( double y ){ return -1.0*gradu_0(y); }; //linear case
	//std::function<double( double )> sigma_0 = [=]( double y ){ return -1.0*y*xi(gradu_0(y))*gradu_0(y); }; //PTransp case
	//std::function<double( double )> sigma_0 = [=]( double y ){ return -0.1*gradu_0(y); }; //Fisher case
	//std::function<double( double )> sigma_0 = [=]( double y ){ return -1.0*(1.0+::pow(u_0(y),2))*gradu_0(y) ; };
	std::function<double( double )> sigma_0 = [=]( double y )
	{
		//if(::pow(::abs(gradu_0(y) / u_0(y)),2)*gradu_0(y) > 100.0) return -100.0;
		//if(::pow(::abs(gradu_0(y) / u_0(y)),2)*gradu_0(y) < -100.0) return 100.0;
		double u_ = std::max(0.1, u_0(y));
		return -1*::pow(::abs(gradu_0(y) / u_),2)*gradu_0(y); 
	}; //Shestakov case

	const Grid grid(lBound, uBound, nCells);
	SystemSolver system(grid, k, nCells, nVar, delta_t, f, tau, c);

	auto diffobj = std::make_shared< DiffusionObj >(k, nVar);
	buildDiffusionObj(diffobj);
	system.setDiffobj(diffobj);

	auto sourceobj = std::make_shared< SourceObj >(k, nVar);
	buildSourceObj(sourceobj);
	system.setSourceobj(sourceobj);

	runSolver(system, k, nCells, nVar, nOut, tFinal, rtol, atol, gradu_0, u_0, sigma_0, lBound, uBound);
}
*/