
#include <boost/test/unit_test.hpp>

#include "Types.hpp"
#include <toml.hpp>
#include "SystemSolver.hpp"
#include "TestDiffusion.hpp"

using namespace toml::literals::toml_literals;

// raw string literal (`R"(...)"` is useful for this purpose)
const toml::value config_snippet = u8R"(
    [DiffusionProblem]
	 Kappa = 1.0
)"_toml;

BOOST_TEST_DONT_PRINT_LOG_VALUE( Grid );
BOOST_TEST_DONT_PRINT_LOG_VALUE( Matrix );

BOOST_AUTO_TEST_SUITE( system_solver_test_suite )

BOOST_AUTO_TEST_CASE( systemsolver_init_tests )
{
	Grid testGrid( 0.0, 1.0, 4 );
	Index k = 1; // Start piecewise linear
	SystemSolver *system = nullptr;
	double dt = 0.1;

	TestDiffusion problem( config_snippet );

	BOOST_CHECK_NO_THROW( system = new SystemSolver( testGrid, k, dt, &problem ) );

	system->resetCoeffs();

	BOOST_TEST( system->k == k );
	BOOST_TEST( system->grid == testGrid );
	BOOST_TEST( system->nVars == 1 );

	BOOST_TEST(  ( system->A_cellwise[ 0 ] - Matrix::Identity( k + 1, k + 1 ) ).norm() < 1e-9 );
	BOOST_TEST(  ( system->A_cellwise[ 1 ] - Matrix::Identity( k + 1, k + 1 ) ).norm() < 1e-9 );
	BOOST_TEST(  ( system->A_cellwise[ 2 ] - Matrix::Identity( k + 1, k + 1 ) ).norm() < 1e-9 );
	BOOST_TEST(  ( system->A_cellwise[ 3 ] - Matrix::Identity( k + 1, k + 1 ) ).norm() < 1e-9 );

	Matrix ref( k + 1, k + 1 );
	// Derivative matrix
	ref << 0.0, 13.85640646055103,
	       0.0, 0.0;
	BOOST_TEST(  ( system->B_cellwise[ 0 ] - ref ).norm() < 1e-9 );
	BOOST_TEST(  ( system->B_cellwise[ 1 ] - ref ).norm() < 1e-9 );
	BOOST_TEST(  ( system->B_cellwise[ 2 ] - ref ).norm() < 1e-9 );
	BOOST_TEST(  ( system->B_cellwise[ 3 ] - ref ).norm() < 1e-9 );

	ref << 4.0, 0.0,
	       0.0, 12.0;

	BOOST_TEST(  ( system->D_cellwise[ 0 ] - ref ).norm() < 1e-9 );
	BOOST_TEST(  ( system->D_cellwise[ 1 ] - ref ).norm() < 1e-9 );
	BOOST_TEST(  ( system->D_cellwise[ 2 ] - ref ).norm() < 1e-9 );
	BOOST_TEST(  ( system->D_cellwise[ 3 ] - ref ).norm() < 1e-9 );

	double TwoRootThree = 2.0*::sqrt( 3.0 );

	ref <<  0.0, 0.0,
	        2.0, TwoRootThree;
	BOOST_TEST(  ( system->C_cellwise[ 0 ] - ref ).norm() < 1e-9 );

	ref << -2.0, TwoRootThree,
	        2.0, TwoRootThree;
	BOOST_TEST(  ( system->C_cellwise[ 1 ] - ref ).norm() < 1e-9 );
	BOOST_TEST(  ( system->C_cellwise[ 2 ] - ref ).norm() < 1e-9 );

	ref <<  -2.0, TwoRootThree,
	         0.0, 0.0;
	BOOST_TEST(  ( system->C_cellwise[ 3 ] - ref ).norm() < 1e-9 );

	double RootThree = ::sqrt( 3.0 );
	ref <<  0.0,       -1.0,
	        0.0, -RootThree;
	BOOST_TEST(  ( system->E_cellwise[ 0 ] - ref ).norm() < 1e-9 );

	ref << -1.0,       -1.0,
	        RootThree, -RootThree;
	BOOST_TEST(  ( system->E_cellwise[ 1 ] - ref ).norm() < 1e-9 );
	BOOST_TEST(  ( system->E_cellwise[ 2 ] - ref ).norm() < 1e-9 );

	ref << -1.0,       0.0,
	        RootThree, 0.0;
	BOOST_TEST(  ( system->E_cellwise[ 3 ] - ref ).norm() < 1e-9 );

}

BOOST_AUTO_TEST_SUITE_END()
