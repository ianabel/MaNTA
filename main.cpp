#include <ida/ida.h>                   /* prototypes for IDA fcts., consts.    */
#include <functional>

#include "SystemSolver.hpp"

void runSolver( SystemSolver& system, const sunindextype k, const sunindextype nCells, const sunindextype nVar, int nOut, double tFinal, realtype rtol, realtype atol, Fn gradu_0, Fn u_0, Fn sigma_0, double lBound, double uBound, bool printToFile = true);
void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj);

int main()
{
	//---------------------------Variable assiments-------------------------------
	const sunindextype k = 3;		//Polynomial degree of each cell
	const sunindextype nCells = 37;			//Total number of cells
	const sunindextype nVar = 1;
	const double lBound = 0.0, uBound = 10;	//Spacial bounds
	int nOut = 300;
	double tFinal = 30.0, delta_t = 5.0;
	realtype rtol = 1.0e-5, atol = 1.0e-5;

	const double c_const = 0.0;
	const double kappa0 = 1.0;
	std::function<double( double )> f = [ = ]( double x ){ 
		//return 0.0;
		// return 2.0 * ( 2.0 * x*x*x*x - 7.0 * x*x + 2.0 ) * ::exp( -x*x ); 
		return (-1*kappa0/4)*(::exp(-0.75*(x-5)*(x-5)) * (73.0 + 3.0*x*(x-10)) + ::exp(-0.25 * (x-5)*(x-5)) * (23 + x*(x-10))); // For non-linear manufactured solution case, some error from diricelet BCs
	};
	std::function<double( double )> c = [ = ]( double x ){ return c_const;};
	std::function<double( double )> tau = [ & ]( double x ){ return ( ::fabs( c( x ) ) + kappa0/2.0 );};

	double a = 5.0;
	double b = 1.0; 
	double beta = 1.0;
	std::function<double( double )> u_0 = [=]( double y ){ return (a/::sqrt(M_PI))*::exp( -b*( y - a )*( y - a ) ); };
	std::function<double( double )> gradu_0 = [=]( double y ){ return -2*b*(a/::sqrt(M_PI))*(y - 5)*::exp( -b*( y - a )*( y - a ) ); };
	//std::function<double( double )> sigma_0 = [=]( double y ){ return gradu_0(y); };
	std::function<double( double )> sigma_0 = [=]( double y ){ return -1*kappa0*(1.0 + u_0(y)*u_0(y))*gradu_0(y); };

	const Grid grid(lBound, uBound, nCells);
	SystemSolver system(grid, k, nCells, nVar, delta_t, f, tau, c);

	auto diffobj = std::make_shared< DiffusionObj >(k, nVar);
	buildDiffusionObj(diffobj);
	system.setDiffobj(diffobj);

	runSolver(system, k, nCells, nVar, nOut, tFinal, rtol, atol, gradu_0, u_0, sigma_0, lBound, uBound);
}

//This is user input. Eventually this will be put in an input file format, but for now its hard coded.
/*
void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj)
{
	auto nVar = diffobj->nVar;
	if(nVar != 2) throw std::runtime_error("check your kappa build, you did it wrong.");

	diffobj->clear();
	double kappa_const = 1.0;

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return (7/4)*q(x,0) - (::sqrt(3)/4)*q(x,1);};
	std::function<double( double, DGApprox, DGApprox )> kappa1 = [ = ]( double x, DGApprox q, DGApprox u ){ return -(::sqrt(3)/4)*q(x,0) + (7/4)*q(x,1);};
	diffobj->kappaFuncs.push_back(kappa0);
	diffobj->kappaFuncs.push_back(kappa1);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return (7/4);};
	std::function<double( double, DGApprox, DGApprox )> dkappa0dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return -(::sqrt(3)/4);};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return -(::sqrt(3)/4);};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return (7/4);};

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

void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj)
{
	auto nVar = diffobj->nVar;
	if(nVar != 2) throw std::runtime_error("check your kappa build, you did it wrong.");

	diffobj->clear();
	double kappa_const = 1.0;

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return (5-::abs(x-5))*q(x,0);};
	std::function<double( double, DGApprox, DGApprox )> kappa1 = [ = ]( double x, DGApprox q, DGApprox u ){ return (5-::abs(x-5))*q(x,1);};
	diffobj->kappaFuncs.push_back(kappa0);
	diffobj->kappaFuncs.push_back(kappa1);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return (5-::abs(x-5));};
	std::function<double( double, DGApprox, DGApprox )> dkappa0dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	std::function<double( double, DGApprox, DGApprox )> dkappa1dq1 = [ = ]( double x, DGApprox q, DGApprox u ){ return (5-::abs(x-5));};

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
*/

//single variable non-linear case
void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj)
{
	auto nVar = diffobj->nVar;
	if(nVar != 1) throw std::runtime_error("check your kappa build, you did it wrong.");

	diffobj->clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return  beta*(1.0 + u(x,0)*u(x,0))*q(x,0);};
	diffobj->kappaFuncs.push_back(kappa0);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*(1 + u(x,0)*u(x,0));};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta*u(x,0)*q(x,0);};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	diffobj->delqKappaFuncs.resize(nVar);
	diffobj->deluKappaFuncs.resize(nVar);

	diffobj->delqKappaFuncs[0].push_back(dkappa0dq0);

	diffobj->deluKappaFuncs[0].push_back(dkappa0du0);
}


/*
//Single variable linear case
void buildDiffusionObj(std::shared_ptr<DiffusionObj> diffobj)
{
	auto nVar = diffobj->nVar;
	if(nVar != 1) throw std::runtime_error("check your kappa build, you did it wrong.");

	diffobj->clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return  1.0*q(x,0);};
	diffobj->kappaFuncs.push_back(kappa0);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 1.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	diffobj->delqKappaFuncs.resize(nVar);
	diffobj->deluKappaFuncs.resize(nVar);

	diffobj->delqKappaFuncs[0].push_back(dkappa0dq0);

	diffobj->deluKappaFuncs[0].push_back(dkappa0du0);
}
*/
