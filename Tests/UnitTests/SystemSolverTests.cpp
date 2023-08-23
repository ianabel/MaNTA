
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

BOOST_AUTO_TEST_SUITE( system_solver_test_suite, * boost::unit_test::tolerance( 1e-6 ) )

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


}

BOOST_AUTO_TEST_SUITE_END()
