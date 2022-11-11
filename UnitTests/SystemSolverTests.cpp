#define BOOST_TEST_MODULE MTS
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_band.h>  /* access to band SUNMatrix             */
#include <sunlinsol/sunlinsol_band.h>  /* access to band SUNLinearSolver       */
#include <sundials/sundials_types.h>   /* definition of type realtype          */

#include <boost/test/included/unit_test.hpp>

#include "../SystemSolver.hpp"
#include "../DiffusionObj.hpp"

int residual(realtype tres, N_Vector Y, N_Vector dydt, N_Vector resval, void *user_data);

void buildTestDiffObj(std::shared_ptr<DiffusionObj> diffobj)
{
	auto nVar = diffobj->nVar;
	if(nVar != 1) throw std::runtime_error("check your kappa build, you did it wrong.");

	diffobj->clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> kappa0 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*(1.0 + ::pow(u(x,0),2))*q(x,0);};
	diffobj->kappaFuncs.push_back(kappa0);

	std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return beta*(1.0 + ::pow(u(x,0),2));};

	std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta*::pow(u(x,0),1)*q(x,0);};

	diffobj->delqKappaFuncs.resize(nVar);
	diffobj->deluKappaFuncs.resize(nVar);

	diffobj->delqKappaFuncs[0].push_back(dkappa0dq0);

	diffobj->deluKappaFuncs[0].push_back(dkappa0du0);
}

void buildTestSourceObj(std::shared_ptr<SourceObj> sourceobj)
{
	auto nVar = sourceobj->nVar;
	if(nVar != 1) throw std::runtime_error("check your source build, you did it wrong.");

	sourceobj->clear();
	double beta = 1.0;

	std::function<double( double, DGApprox, DGApprox )> F_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return  0.0;};
	sourceobj->sourceFuncs.push_back(F_0);

	std::function<double( double, DGApprox, DGApprox )> dF_0dq_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0dq0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 2*beta;};

	std::function<double( double, DGApprox, DGApprox )> dF_0du_0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};
	//std::function<double( double, DGApprox, DGApprox )> dkappa0du0 = [ = ]( double x, DGApprox q, DGApprox u ){ return 0.0;};

	sourceobj->dFdqFuncs.resize(nVar);
	sourceobj->dFduFuncs.resize(nVar);

	sourceobj->dFdqFuncs[0].push_back(dF_0dq_0);

	sourceobj->dFduFuncs[0].push_back(dF_0du_0);
}

BOOST_AUTO_TEST_SUITE( functiontal_test_suite, * boost::unit_test::tolerance( 1e-6 ) )

BOOST_AUTO_TEST_CASE( SysSolver_Core_functional_test)
{
	const sunindextype k = 3;		//Polynomial degree of each cell
	const sunindextype nCells = 7;			//Total number of cells
	const sunindextype nVar = 1;
	const double lBound = 0.0, uBound = 10.0;	//Spacial bounds
	double delta_t = 0.001;
	realtype tres = 0.0;
	std::function<double( double )> c = [ = ]( double x ){ return 0.0;};
	std::function<double( double )> tau = [ & ]( double x ){ return ( 1.0 );};
	std::function<double( double )> f = [ = ]( double x ){ return 0.0; };

	const Grid grid(lBound, uBound, nCells);
	SystemSolver system(grid, k, nCells, nVar, delta_t, f, tau, c);
	system.setTesting(true);

	auto diffobj = std::make_shared< DiffusionObj >(k, nVar);
	buildTestDiffObj(diffobj);
	system.setDiffobj(diffobj);

	auto sourceobj = std::make_shared< SourceObj >(k, nVar);
	buildTestSourceObj(sourceobj);
	system.setSourceobj(sourceobj);

	double a = 5.0;
	double b = 4.0;
	std::function<double( double )> u_0 = [=]( double y ){ return ::exp( -b*( y - a )*( y - a ) ); }; //gaussian
	std::function<double( double )> gradu_0 = [=]( double y ){ return -2.0*b*(y - a)*::exp( -b*( y - a )*( y - a ) ); }; //gaussian
	std::function<double( double )> sigma_0 = [=]( double y ){ return -1.0*(1.0+u_0(y)*u_0(y))*gradu_0(y) ; };

	N_Vector Y = NULL, dYdt = NULL, resval = NULL;
	SUNContext ctx;
	Y = N_VNew_Serial(nVar*3*nCells*(k+1) + nVar*(nCells+1), ctx);
	dYdt = N_VClone(Y);
	resval = N_VClone(Y);
	VectorWrapper yVec( N_VGetArrayPointer( Y ), N_VGetLength( Y ) ); 
	VectorWrapper dydtVec( N_VGetArrayPointer( Y ), N_VGetLength( Y ) ); 
	VectorWrapper resVec( N_VGetArrayPointer( Y ), N_VGetLength( Y ) ); 
	yVec.setZero();
	dydtVec.setZero();

	std::function<double( double, double )> g_D_ = [ = ]( double x, double t ) { return 0.0; };
	auto DirichletBCs = std::make_shared<BoundaryConditions>();
	DirichletBCs->LowerBound = lBound;
	DirichletBCs->UpperBound = uBound;
	DirichletBCs->isLBoundDirichlet = true;
	DirichletBCs->isUBoundDirichlet = true;
	DirichletBCs->g_D = g_D_;
	DirichletBCs->g_N = g_D_;
	system.setBoundaryConditions(DirichletBCs);

	system.setInitialConditions(u_0, gradu_0, sigma_0, Y, dYdt);

	BOOST_TEST( yVec.norm() == 2.7441072546887524 );

	UserData *data = new UserData();
	data->system = &system;

	residual(tres, Y, dYdt, resval, data);
	BOOST_TEST( system.resNorm == 4.9841996051198381 );

	yVec.setZero();
	system.solveJacEq(resval, Y);
	BOOST_TEST( yVec.norm() == 1.2420106057992257 );

	
	delete data;
}

BOOST_AUTO_TEST_SUITE_END()
