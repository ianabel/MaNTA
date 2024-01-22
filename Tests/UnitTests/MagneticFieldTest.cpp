#include <boost/test/unit_test.hpp>
#include "../../PhysicsCases/MagneticFields.hpp"
#include "Types.hpp"
#include <toml.hpp>
#include <filesystem>

const std::filesystem::path file = std::filesystem::current_path().string() + "/Tests/UnitTests/Bfield.ref.nc";

BOOST_AUTO_TEST_SUITE(magnetic_fields_test_suite, *boost::unit_test::tolerance(1e-8))

BOOST_AUTO_TEST_CASE(magnetic_fields_init_tests)
{
    BOOST_CHECK_NO_THROW(CylindricalMagneticField B(file));
}

BOOST_AUTO_TEST_CASE(magnetic_fields_values)
{
    const double B0 = 1.0;
    const double Rm0 = 3.3;

    const double Bmid = 0.75 * B0;
    const double Rm_mid = 0.75 * B0;

    const double Rmid = 0.5 * (0.7 + 0.2);

    CylindricalMagneticField B(file);

    BOOST_TEST(B.Bz_R(Rmid) == Bmid);

    const double Vmid = M_PI * Rmid * Rmid;
    const double Psi_mid = B.Psi_V(Vmid);

    BOOST_TEST(B.R(Psi_mid) == Rmid);

    const double dRdV_mid = 1. / (2.0 * M_PI * Rmid);

    BOOST_TEST(B.dRdV(Vmid) == dRdV_mid);
}

BOOST_AUTO_TEST_SUITE_END()
