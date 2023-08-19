
#include <boost/test/unit_test.hpp>

#include <toml.hpp>
#include "TestDiffusion.hpp"

using namespace toml::literals::toml_literals;

// raw string literal (`R"(...)"` is useful for this purpose)
const toml::value config_snipper = u8R"(
    title = "this is TOML literal"
    [table]
    key = "value"
)"_toml;

BOOST_AUTO_TEST_SUITE( diffusionproblem_suite, * boost::unit_test::tolerance( 1e-6 ) )

BOOST_AUTO_TEST_CASE( construct_diffusion_problem )
{
	std::map<std::string, double> parameters{
		{ "CentralCellField", 1.0 }
	};
	
	MirrorPlasma *pSamplePlasma = nullptr;

	BOOST_REQUIRE_NO_THROW( pSamplePlasma = new MirrorPlasma( parameters, "Hydrogen", false, false, false, false, false, false, "", "", "" ) );

	BOOST_TEST( pSamplePlasma->CentralCellFieldStrength == 1.0 );

}

BOOST_AUTO_TEST_SUITE_END()
