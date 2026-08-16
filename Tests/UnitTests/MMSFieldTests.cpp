// Order of accuracy with the field coupled. This is the test that catches a
// sign error in the *equations* -- a wrong A1 or A2 in the residual converges
// at the right rate to the wrong function, so only a closed-form comparison
// sees it. A Jacobian error is invisible here by construction.
//
// Read LOCAL orders, not the least-squares slope: a fit averages a changing
// rate away, which is how the nonlinear-flux superconvergence breakdown stayed
// invisible to n <= 32.
//
// The problem, the manufactured source and the sweep live in MMSHarness.hpp;
// the field models in ManufacturedFields.hpp. What is here is the pair of
// algebra checks on the source, and the four studies. Tests/README.md carries
// the measured tables, the fallback counts and the two mutations that show the
// studies are not vacuous.
#include <boost/test/unit_test.hpp>

#include "MMSHarness.hpp"
#include "ManufacturedFields.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace
{

using namespace mms;

/// The flux the manufactured source was derived from, kappa g(x, t) u_x,
/// evaluated on the *exact* solution. Independent of the solver and of the
/// physics case -- the case is checked against it below.
template <ManufacturedFieldModel M>
double exactFlux(double x, double t)
{
    return coupledKappa * M::geometryExact(x, t) * exactDerivative(x, t);
}

/// d(exactFlux)/dx by the six-point central stencil, which is O(h^6).
///
/// A plain central difference is not accurate enough for the 1e-10 the check
/// below wants: at h = 1e-5 its truncation error is O(h^2 f''') ~ 1e-8, so the
/// check would have to be loosened to a tolerance that no longer distinguishes a
/// correct source from one with a missing term. This stencil at h = 2e-3 has a
/// truncation error of h^6 f^(7)/140 ~ 2e-13 against a round-off of ~1e-12, and
/// measures both models' sources at ~1e-12.
///
/// The flux carries cos^2(pi x), so f^(7) goes as (2 pi)^7 rather than pi^7 --
/// which is why h = 5e-3, the obvious first choice, lands at 4e-11 rather than
/// the 3e-13 a single-frequency estimate predicts. Close enough to 1e-10 to be
/// worth not doing.
///
/// The stencil reaches 3h either side, so a sample point must be at least that
/// far from a kink in the vector model's piecewise linear geometry -- see
/// kVectorSampleX.
template <ManufacturedFieldModel M>
double dFluxdx(double x, double t)
{
    constexpr double h = 2e-3;
    return (-1.0 * exactFlux<M>(x - 3.0 * h, t) + 9.0 * exactFlux<M>(x - 2.0 * h, t) -
            45.0 * exactFlux<M>(x - h, t) + 45.0 * exactFlux<M>(x + h, t) -
            9.0 * exactFlux<M>(x + 2.0 * h, t) + 1.0 * exactFlux<M>(x + 3.0 * h, t)) /
           (60.0 * h);
}

/// u_t - d_x[ kappa g u_x ] - S at the exact solution: zero if the algebra is
/// right. u = sin(pi x)(1 + t) is linear in t, so the central difference in time
/// is exact to round-off and contributes nothing to the tolerance.
template <ManufacturedFieldModel M>
double sourceResidual(double x, double t)
{
    constexpr double ht = 1e-3;
    const double dudt = (exactSolution(x, t + ht) - exactSolution(x, t - ht)) / (2.0 * ht);
    return dudt - dFluxdx<M>(x, t) - coupledSource<M>(x, t);
}

/// The scatter of x the single-DOF model is checked at, and the t values both
/// are, spanning the interval the studies integrate over.
const std::vector<double> kSampleX = {0.07, 0.13, 0.37, 0.5, 0.62, 0.81, 0.94};
const std::vector<double> kSampleT = {0.0, 0.1, 0.25};

/// The multi-DOF model's scatter. Its geometry is the *hat interpolant* of psi,
/// so g is only piecewise linear: g' jumps at 0.25/0.5/0.75 and the manufactured
/// source jumps with it. A finite-difference stencil straddling one of those
/// measures the average of two different one-sided derivatives and reports a
/// residual of 1.5e-2 -- as x = 0.5 above does -- which is a statement about the
/// stencil, not about the source.
///
/// So the kinks are excluded, and 0.45/0.55 and 0.2/0.3 are in instead: the
/// source is checked on *both* sides of two of them, which is what makes the
/// exclusion a matter of where the classical form holds rather than a way of
/// not looking. The order study is unaffected -- every grid it refines over has
/// cell boundaries at the kinks, so the jumps never fall inside a cell.
const std::vector<double> kVectorSampleX = {0.07, 0.13, 0.2,  0.3,  0.37,
                                            0.45, 0.55, 0.62, 0.81, 0.94};

/// The grids every sweep refines over. Multiples of 4, so the kinks in
/// ManufacturedFieldVector's geometry -- and therefore the jumps in its
/// manufactured source -- always fall on cell boundaries.
const std::vector<Index> kCells = {4, 8, 16, 32};

std::string sweepStats(Rates const &r)
{
    std::string s;
    for (size_t i = 0; i < r.cells.size(); ++i)
        s += std::format("{}n={} {} fallbacks in {} solves", i ? ", " : "", r.cells[i],
                         r.fallbacks[i], r.fieldSolves[i]);
    return s;
}

} // namespace

BOOST_AUTO_TEST_SUITE(mms_field_tests)

BOOST_AUTO_TEST_CASE(the_manufactured_source_is_consistent_with_the_exact_solution)
{
    // Check the algebra before checking the solver. If S were wrong the studies
    // below would still converge -- to a different function -- and the rate
    // would look fine, so this is the half of the split that an order study
    // cannot do. It also costs nothing: no solver is involved.
    double worstScalar = 0.0, worstVector = 0.0;
    for (double t : kSampleT)
    {
        for (double x : kSampleX)
            worstScalar =
                std::max(worstScalar, std::abs(sourceResidual<ManufacturedField>(x, t)));
        for (double x : kVectorSampleX)
            worstVector = std::max(
                worstVector, std::abs(sourceResidual<ManufacturedFieldVector>(x, t)));
    }

    BOOST_TEST_MESSAGE(std::format(
        "worst |u_t - d_x[kappa g u_x] - S|:  single-DOF {:.3e}  multi-DOF {:.3e}",
        worstScalar, worstVector));

    BOOST_CHECK_LT(worstScalar, 1e-10);
    BOOST_CHECK_LT(worstVector, 1e-10);
}

BOOST_AUTO_TEST_CASE(the_physics_case_computes_the_flux_the_source_was_derived_for)
{
    // The other half of the algebra check: the source above is only the right
    // source if SigmaFn really is kappa g q. A case whose flux disagreed with it
    // would converge to the wrong function at the right rate, exactly as a sign
    // error would.
    ManufacturedGeometricDiffusion<ManufacturedField> problem;
    const double t = 0.17;

    for (double x : kSampleX)
    {
        State s(1, 0, 0, 1);
        s.u(0) = exactSolution(x, t);
        s.q(0) = exactDerivative(x, t);
        s.geom(0) = ManufacturedField::geometryExact(x, t);

        BOOST_TEST(problem.SigmaFn(0, s, x, t) == exactFlux<ManufacturedField>(x, t),
                   boost::test_tools::tolerance(1e-13));

        // ...and the geometry derivative that A1 is built from is the derivative
        // of that flux, which is the one place the coupling's first factor can
        // be checked without a solver.
        Vector dg(1);
        dg.setZero();
        problem.dSigmaFn_dGeometry(0, dg, s, x, t);
        BOOST_TEST(dg[0] == coupledKappa * s.q(0), boost::test_tools::tolerance(1e-13));
    }
}

BOOST_AUTO_TEST_CASE(the_coupled_problem_converges_at_k_plus_one_in_u)
{
    // The headline study, run on *both* solve modes.
    //
    // solveCoupledJacIterative escalates to the exact Schur solve when it
    // exhausts FieldSolveMaxSweeps, so a sweep that never converged would yield
    // exactly the exact path's numbers with nothing in the result to say so.
    // Running both and requiring them to agree is what makes the default mode's
    // rate a measurement of the default mode. The fallback counts are reported
    // either way.
    //
    // The two modes are not expected to agree bit for bit -- they are different
    // Jacobian solves, so IDA's Newton lands on different points inside the same
    // tolerance and can take different steps -- but they solve the same
    // equations to the same accuracy, so the rates must agree far more closely
    // than the 0.25 of headroom the rate assertion itself carries.
    for (Index k = 1; k <= 3; ++k)
    {
        const Rates iterative = solveCoupledAndMeasure<ManufacturedField>(k, kCells);
        const Rates exact = solveCoupledAndMeasure<ManufacturedField>(
            k, kCells, false, SystemSolver::FieldSolveMode::Exact);

        double worstGap = 0.0;
        for (size_t i = 0; i < iterative.localU.size(); ++i)
            worstGap = std::max(worstGap, std::abs(iterative.localU[i] - exact.localU[i]));

        BOOST_TEST_MESSAGE("k = " << k << " local orders in u: iterative "
                                  << format(iterative.localU) << "  exact "
                                  << format(exact.localU)
                                  << std::format("  (worst gap {:.2e})", worstGap)
                                  << "\n    iterative sweep: " << sweepStats(iterative)
                                  << iterative.detail);

        for (double o : iterative.localU)
            BOOST_CHECK_GT(o, k + 1 - 0.25);
        for (double o : exact.localU)
            BOOST_CHECK_GT(o, k + 1 - 0.25);

        for (size_t i = 0; i < iterative.localU.size(); ++i)
            BOOST_TEST(std::abs(iterative.localU[i] - exact.localU[i]) < 0.01,
                       "k = " << k << ": the two solve modes disagree between n="
                              << iterative.cells[i] << " and n=" << iterative.cells[i + 1]
                              << " (iterative " << iterative.localU[i] << ", exact "
                              << exact.localU[i] << ")");
    }
}

BOOST_AUTO_TEST_CASE(psi_converges_too)
{
    // psi = Int u dx, so its error is the quadrature error of an O(h^{k+1})
    // function and should fall at least as fast.
    //
    // Measured, it falls *faster*: 4.82, 3.93, 3.87 at k = 2, i.e. k+2 rather
    // than the k+1 asserted. That is not luck. The field quadrature is exact on
    // a degree-k field, so psi_h is exactly Int u_h dx and its error is
    // Int (u_h - u) dx -- a linear functional of the error, which superconverges
    // by the usual duality argument rather than tracking the L2 norm. The
    // assertion stays at k+1 because that is what the discretisation guarantees;
    // the extra order is recorded in Tests/README.md, not pinned here.
    const Rates r = solveCoupledAndMeasure<ManufacturedField>(2, kCells);
    BOOST_TEST_MESSAGE("local orders in psi: " << format(r.localExtra) << r.detail);
    for (double o : r.localExtra)
        BOOST_CHECK_GT(o, 2.75);
}

BOOST_AUTO_TEST_CASE(the_multi_dof_field_converges_the_same_way)
{
    // Five field DOFs, a tridiagonal B and a dense dGeometry/dpsi, so the whole
    // coupling is exercised rather than its one-dimensional shadow.
    const Rates r = solveCoupledAndMeasure<ManufacturedFieldVector>(2, kCells);
    BOOST_TEST_MESSAGE("multi-DOF local orders in u: " << format(r.localU)
                                                       << "  psi: " << format(r.localExtra)
                                                       << "\n    sweep: " << sweepStats(r)
                                                       << r.detail);
    for (double o : r.localU)
        BOOST_CHECK_GT(o, 2.75);

    // psi too: five DOFs constrained by L psi = f(u_h) rather than one, which is
    // the case where a row or column confusion in the field block would show.
    // Measured 4.03, 4.01, 4.00 -- k+2, for the same reason the single-DOF psi
    // superconverges -- but asserted at k+1.
    for (double o : r.localExtra)
        BOOST_CHECK_GT(o, 2.75);
}

BOOST_AUTO_TEST_CASE(superconvergent_coupling_reaches_k_plus_two)
{
    // Geometry is a function of (psi, x) and star nodes are just more x, so
    // this should work through ComputePhysics's states.size() loop with no
    // special case.
    //
    // It does: u* reaches 4.24, 3.99, 3.97 at k = 2, so this asserts k+2 rather
    // than asserting that the flag throws with a field attached -- which was the
    // alternative, following the precedent that spatial adjoint parameters with
    // Superconvergent = true throw rather than silently redefining the answer.
    //
    // Note what this does *not* show. At k = 2 the flag-off column already
    // reaches k+2 here (4.47, 4.08, 3.96, from psi_converges_too's run), exactly
    // as it does for the uncoupled linear problem, so the flag is preserving the
    // extra order rather than restoring it. The k = 1 case, where the uncoupled
    // study shows a genuine restoration, is not measured with a field attached.
    const Rates r = solveCoupledAndMeasure<ManufacturedField>(2, kCells, /*superconvergent=*/true);
    BOOST_TEST_MESSAGE("local orders in u*: " << format(r.localStar)
                                              << "  u: " << format(r.localU) << r.detail);
    for (double o : r.localStar)
        BOOST_CHECK_GT(o, 3.75);
}

BOOST_AUTO_TEST_SUITE_END()
