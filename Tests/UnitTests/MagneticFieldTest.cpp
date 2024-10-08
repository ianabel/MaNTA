#include <boost/test/unit_test.hpp>
#include "../../PhysicsCases/MagneticFields.hpp"
#include "Types.hpp"
#include <toml.hpp>
#include <filesystem>

const std::filesystem::path file("./Tests/UnitTests/Bfield.ref.nc");

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
    const double Rm_mid = 0.75 * Rm0;

    const double Rl = 0.2;
    const double Rr = 0.7;

    const double Rmid = 0.5 * (Rr + Rl);

    CylindricalMagneticField B(file);

    BOOST_TEST(B.Bz_R(Rmid) == Bmid);

    const double Vmid = M_PI * Rmid * Rmid;
    const double Psi_mid = B.Psi_V(Vmid);

    BOOST_TEST(B.MirrorRatio(Vmid) == Rm_mid);
    BOOST_TEST(B.R(Psi_mid) == Rmid);

    const double dRdV_mid = 1. / (2.0 * M_PI * Rmid);

    const double Vl = M_PI*Rl*Rl;
    const double Vr = M_PI*Rr*Rr;
    const double dRdV_left = 1./(2.0*M_PI*Rl);
    const double dRdV_right = 1./(2.0*M_PI*Rr);

    BOOST_TEST(B.dRdV(Vmid) == dRdV_mid);
    BOOST_TEST(B.dRdV(Vl) == dRdV_left);
    BOOST_TEST(B.dRdV(Vr) == dRdV_right);
}

BOOST_AUTO_TEST_SUITE_END()
