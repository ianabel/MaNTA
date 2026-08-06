// Tests for the magnetic-geometry classes (PhysicsCases/MagneticFields.hpp).
//
// This file was committed alongside a Bfield.ref.nc fixture but was never
// listed in TEST_SOURCES, so it never compiled and had bit-rotted: it called a
// Bz_R() member declared nowhere, and R() with one argument where the override
// takes two.
//
// It also cannot be repaired as written. It targeted CylindricalMagneticField,
// whose constructor now requires a *V-indexed* netCDF file supplying
// V/Bz/VPrime/Psi/Rm/R/dRdV/L, but Bfield.ref.nc is the older *R-indexed*
// format holding only R/Bz/Rm/Psi -- both the test and its fixture predate the
// refactor from radius- to volume-indexing. Re-enabling those cases means
// regenerating the fixture in the new format; see Tests/README.md.
//
// What is covered here instead is StraightMagneticField, which is fully
// analytic, needs no fixture, and is the implementation MirrorPlasma actually
// uses. The assertions are the internal identities the class must satisfy, so
// they stay meaningful if the formulas are refactored.

#include <boost/test/unit_test.hpp>

#include "../../PhysicsCases/MagneticFields.hpp"
#include "Types.hpp"

#include <cmath>

BOOST_AUTO_TEST_SUITE(magnetic_fields_test_suite, *boost::unit_test::tolerance(1e-10))

namespace
{
constexpr double L_z = 0.6;
constexpr double B_z = 0.3;
constexpr double Rm = 10.0;

// The volume coordinate of a flux surface at radius R in a straight cylinder.
double volumeAt(double R) { return M_PI * R * R * L_z; }
} // namespace

BOOST_AUTO_TEST_CASE(straight_field_is_uniform_without_a_gradient)
{
    StraightMagneticField Bconcrete(L_z, B_z, Rm);
    // Access through the base class: the derived Real overrides hide the base's
    // double overloads, and production code holds a shared_ptr<MagneticField>.
    const MagneticField &B = Bconcrete;

    // With m = 0 the field is uniform in V, and L_V is constant by construction.
    for (double V : {0.1, 0.5, 1.0, 2.0})
    {
        BOOST_TEST(B.B(V, 0.0) == B_z);
        BOOST_TEST(B.L_V(V) == L_z);
    }
}

BOOST_AUTO_TEST_CASE(straight_field_applies_a_linear_gradient)
{
    // The five-argument constructor adds B = B_z - m (V - Vmin).
    const double Vmin = 0.2, m = 0.05;
    StraightMagneticField Bconcrete(L_z, B_z, Rm, Vmin, m);
    const MagneticField &B = Bconcrete;

    for (double V : {0.2, 0.5, 1.0})
        BOOST_TEST(B.B(V, 0.0) == B_z - m * (V - Vmin));

    // Mirror ratio scales inversely with the local field.
    for (double V : {0.2, 0.5, 1.0})
        BOOST_TEST(B.MirrorRatio(V, 0.0) == Rm * B_z / (B_z - m * (V - Vmin)));
}

BOOST_AUTO_TEST_CASE(straight_field_radius_and_volume_are_inverse)
{
    StraightMagneticField Bconcrete(L_z, B_z, Rm);
    const MagneticField &B = Bconcrete;

    // V = pi R^2 L_z, so R_V must invert it.
    for (double R : {0.05, 0.2, 0.5, 1.0})
        BOOST_TEST(B.R_V(volumeAt(R), 0.0) == R);
}

BOOST_AUTO_TEST_CASE(straight_field_dRdV_matches_derivative_of_R_V)
{
    StraightMagneticField Bconcrete(L_z, B_z, Rm);
    const MagneticField &B = Bconcrete;

    for (double R : {0.1, 0.35, 0.8})
    {
        const double V = volumeAt(R);

        // Closed form: dR/dV = 1 / (2 pi L_z R)
        BOOST_TEST(B.dRdV(V, 0.0) == 1.0 / (2.0 * M_PI * L_z * R));

        // ...and it really is the derivative of R_V.
        const double h = 1e-6 * V;
        const double fd = (B.R_V(V + h, 0.0) - B.R_V(V - h, 0.0)) / (2.0 * h);
        BOOST_TEST(B.dRdV(V, 0.0) == fd, boost::test_tools::tolerance(1e-6));
    }
}

BOOST_AUTO_TEST_CASE(straight_field_VPrime_is_dV_dPsi)
{
    StraightMagneticField Bconcrete(L_z, B_z, Rm);
    const MagneticField &B = Bconcrete;

    for (double V : {0.1, 0.5, 1.5})
    {
        // Closed form: V' = 2 pi L_z / B
        BOOST_TEST(B.VPrime(V) == 2.0 * M_PI * L_z / B_z);

        // V' is dV/dPsi, so V' * dPsi/dV == 1.
        const double h = 1e-6 * V;
        const double dPsidV = (B.Psi_V(V + h) - B.Psi_V(V - h)) / (2.0 * h);
        BOOST_TEST(B.VPrime(V) * dPsidV == 1.0, boost::test_tools::tolerance(1e-6));
    }
}

BOOST_AUTO_TEST_CASE(straight_field_R_of_Psi_round_trips)
{
    StraightMagneticField Bconcrete(L_z, B_z, Rm);
    const MagneticField &B = Bconcrete;

    // R(Psi_V(V)) must return the radius of that flux surface.
    for (double R : {0.1, 0.4, 0.9})
    {
        const double V = volumeAt(R);
        BOOST_TEST(B.R(B.Psi_V(V), 0.0) == B.R_V(V, 0.0));
    }
}

BOOST_AUTO_TEST_CASE(straight_field_Rmin_Rmax_relation)
{
    StraightMagneticField Bconcrete(L_z, B_z, Rm);
    const MagneticField &B = Bconcrete;

    // Rmax is the radius at the midplane; Rmin is that compressed by the
    // square root of the mirror ratio.
    for (double R : {0.1, 0.4, 0.9})
    {
        const double V = volumeAt(R);
        BOOST_TEST(B.Rmax(V) == R);
        BOOST_TEST(B.Rmin(V) == R / std::sqrt(Rm));
        BOOST_TEST(B.Rmin(V) < B.Rmax(V));
    }
}

BOOST_AUTO_TEST_CASE(straight_field_autodiff_gradients_match_finite_differences)
{
    // MirrorPlasma differentiates through these via autodiff, so the dual-number
    // overloads must carry correct gradients, not just correct values.
    StraightMagneticField B(L_z, B_z, Rm, 0.0, 0.05);
    const MagneticField &Bbase = B;

    for (double V0 : {0.3, 0.7, 1.2})
    {
        Real V = V0;
        V.grad = 1.0; // seed d/dV

        const double h = 1e-6 * V0;

        Real Rv = B.R_V(V, Real(0.0));
        const double fd_R = (Bbase.R_V(V0 + h, 0.0) - Bbase.R_V(V0 - h, 0.0)) / (2.0 * h);
        BOOST_TEST(Rv.grad == fd_R, boost::test_tools::tolerance(1e-6));

        Real Bv = B.B(V, Real(0.0));
        const double fd_B = (Bbase.B(V0 + h, 0.0) - Bbase.B(V0 - h, 0.0)) / (2.0 * h);
        BOOST_TEST(Bv.grad == fd_B, boost::test_tools::tolerance(1e-6));

        Real Rmv = B.MirrorRatio(V, Real(0.0));
        const double fd_Rm = (Bbase.MirrorRatio(V0 + h, 0.0) - Bbase.MirrorRatio(V0 - h, 0.0)) / (2.0 * h);
        BOOST_TEST(Rmv.grad == fd_Rm, boost::test_tools::tolerance(1e-6));
    }
}

BOOST_AUTO_TEST_SUITE_END()
