#include <boost/test/unit_test.hpp>
#include <PhysicsCases/MirrorPlasma/PlasmaConstants.hpp>
#include <PhysicsCases/MagneticFields.hpp>

constexpr static double L = 0.6;
constexpr static double Rm = 10.0;
constexpr static double Bz = 0.34;
constexpr static double PlasmaWidth = 0.2;

const auto B = createMagneticField<StraightMagneticField>(L, Bz, Rm);

BOOST_AUTO_TEST_SUITE(mirror_plasma_test_suite, *boost::unit_test::tolerance(1e-3))

BOOST_AUTO_TEST_CASE(plasma_init_tests)
{
    PlasmaConstants *plasma = nullptr;

    BOOST_CHECK_NO_THROW(plasma = new PlasmaConstants("Hydrogen", B, PlasmaWidth));

    delete plasma;
}

BOOST_AUTO_TEST_CASE(neutral_model_tests)
{

    PlasmaConstants plasma("Hydrogen", B, PlasmaWidth);
    const double ni = 0.1;
    const double nneutrals = 4.5e+13 / 1e20;
    const double Ti = 0.974;
    const double Te = 0.758;
    const double M = 5.46;
    const double vtheta = M * sqrt(Te);

    const double CXLossRate_ref = 4.4642e19;

    double CXLossRate = plasma.ChargeExchangeLossRate(ni, nneutrals, vtheta, Ti).val;

    BOOST_TEST(CXLossRate == CXLossRate_ref);
    // double IonizationRate = plasma.IonizationRate(ni, nneutrals, vtheta, Te, Ti).val;
}

BOOST_AUTO_TEST_SUITE_END()
