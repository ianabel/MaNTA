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
#include <map>
#include <string>
#include <vector>

namespace
{

using namespace mms;

/// The flux the manufactured source was derived from, kappa g(x, t) u_x,
/// evaluated on the *exact* solution. Independent of the solver and of the
/// physics case -- the case is checked against it below.
template <ManufacturedFieldModel M>
double exactFlux(M const &model, double x, double t)
{
    return coupledKappa * model.geometryExact(x, t) * exactDerivative(x, t);
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
double dFluxdx(M const &m, double x, double t)
{
    constexpr double h = 2e-3;
    return (-1.0 * exactFlux(m, x - 3.0 * h, t) + 9.0 * exactFlux(m, x - 2.0 * h, t) -
            45.0 * exactFlux(m, x - h, t) + 45.0 * exactFlux(m, x + h, t) -
            9.0 * exactFlux(m, x + 2.0 * h, t) + 1.0 * exactFlux(m, x + 3.0 * h, t)) /
           (60.0 * h);
}

/// u_t - d_x[ kappa g u_x ] - S at the exact solution: zero if the algebra is
/// right. u = sin(pi x)(1 + t) is linear in t, so the central difference in time
/// is exact to round-off and contributes nothing to the tolerance.
template <ManufacturedFieldModel M>
double sourceResidual(M const &m, double x, double t)
{
    constexpr double ht = 1e-3;
    const double dudt = (exactSolution(x, t + ht) - exactSolution(x, t - ht)) / (2.0 * ht);
    return dudt - dFluxdx(m, x, t) - coupledSource(m, x, t);
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

/// Every refinement of an iterative-mode sweep solved its Jacobian by sweeping.
///
/// **This has to be asserted, not merely printed.** solveCoupledJacIterative
/// escalates to the exact Schur solve on exhausting FieldSolveMaxSweeps, and the
/// escalation returns the exact path's answer bit for bit -- so a change that
/// made the sweep stop converging would leave every order in this file
/// unchanged, make the two-mode agreement check pass *more* strongly, and
/// silently stop measuring the iterative mode at all. The fallback count is the
/// only thing that can tell the difference, and a count that is looked at but
/// not checked is the same vacuity in a different place.
void checkNoFallbacks(Rates const &r)
{
    for (size_t i = 0; i < r.cells.size(); ++i)
        BOOST_TEST(r.fallbacks[i] == 0L,
                   "the sweep escalated to the exact solve at n="
                       << r.cells[i] << " (" << r.fallbacks[i] << " fallbacks in "
                       << r.fieldSolves[i]
                       << " solves), so this refinement measures the exact path");
}

/// The flag-off, iterative-mode sweep at degree k, computed once.
///
/// the_coupled_problem_converges_at_k_plus_one_in_u and psi_converges_too both
/// want the k = 2 column and each sweep is four full integrations, so it is
/// computed on first use and kept. A std::map rather than a vector because the
/// reference must stay valid as later degrees are added: map nodes are stable,
/// vector elements are not.
Rates const &iterativeSweep(Index k)
{
    static std::map<Index, Rates> cache;
    auto it = cache.find(k);
    if (it == cache.end())
        it = cache.emplace(k, solveCoupledAndMeasure<ManufacturedField>(k, kCells)).first;
    return it->second;
}

} // namespace

BOOST_AUTO_TEST_SUITE(mms_field_tests)

BOOST_AUTO_TEST_CASE(the_manufactured_source_is_consistent_with_the_exact_solution)
{
    // Check the algebra before checking the solver. If S were wrong the studies
    // below would still converge -- to a different function -- and the rate
    // would look fine, so this is the half of the split that an order study
    // cannot do. It also costs nothing: no solver is involved.
    const ManufacturedField scalarModel;
    const ManufacturedFieldVector vectorModel;

    double worstScalar = 0.0, worstVector = 0.0;
    for (double t : kSampleT)
    {
        for (double x : kSampleX)
            worstScalar =
                std::max(worstScalar, std::abs(sourceResidual(scalarModel, x, t)));
        for (double x : kVectorSampleX)
            worstVector =
                std::max(worstVector, std::abs(sourceResidual(vectorModel, x, t)));
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
    const ManufacturedField model;
    ManufacturedGeometricDiffusion<ManufacturedField> problem(model);
    const double t = 0.17;

    for (double x : kSampleX)
    {
        State s(1, 0, 0, 1);
        s.u(0) = exactSolution(x, t);
        s.q(0) = exactDerivative(x, t);
        s.geom(0) = model.geometryExact(x, t);

        BOOST_TEST(problem.SigmaFn(0, s, x, t) == exactFlux(model, x, t),
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
        const Rates &iterative = iterativeSweep(k);
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

        // ...and that the iterative column really was measured on the iterative
        // path. Without this the agreement check above is satisfied *best* by
        // the sweep failing completely.
        checkNoFallbacks(iterative);

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
    // Measured, it starts faster and slows down: 4.82, 3.93, 3.87 at k = 2 and
    // 5.12, 4.37, 4.10 at k = 3. There is a mechanism for an extra order -- the
    // field quadrature is exact on a degree-k field, so psi_h is exactly
    // Int u_h dx and its error is Int (u_h - u) dx, a linear functional of the
    // error rather than its L2 norm -- but that is not what is asserted here,
    // because the sweep does not show a *settled* k+2.
    //
    // A rate that is still falling at n = 32 is the pattern the nonlinear-flux
    // postprocessing already showed in this codebase (Tests/README.md, "the two
    // italicised flag-off entries"): u* fell by 6.9, 11.7, 9.1 and then 2.3, so
    // a sweep ending at n = 32 reported 3.21 and looked perfectly healthy. Until
    // this one is refined far enough to distinguish a genuine k+2 from a
    // pre-asymptotic transient, k+1 is the claim the evidence supports.
    const Rates &r = iterativeSweep(2);
    BOOST_TEST_MESSAGE("local orders in psi: " << format(r.localExtra) << r.detail);
    for (double o : r.localExtra)
        BOOST_CHECK_GT(o, 2.75);
    checkNoFallbacks(r);
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
    // Measured 4.03, 4.01, 4.00 -- and this one does *not* decay over the sweep,
    // unlike the single-DOF psi above. Still asserted at k+1: three refinements
    // is not enough to call it settled either.
    for (double o : r.localExtra)
        BOOST_CHECK_GT(o, 2.75);

    checkNoFallbacks(r);
}

BOOST_AUTO_TEST_CASE(superconvergent_coupling_reaches_k_plus_two)
{
    // Geometry is a function of (psi, x) and star nodes are just more x, so
    // this should work through ComputePhysics's states.size() loop with no
    // special case.
    //
    // It does, at every degree, so this asserts k+2 rather than asserting that
    // the flag throws with a field attached -- which was the alternative,
    // following the precedent that spatial adjoint parameters with
    // Superconvergent = true throw rather than silently redefining the answer.
    //
    // **k = 1 is the row that earns the test.** It is the only configuration in
    // this file where the flag-on assertion is not also satisfied flag-off, i.e.
    // the only one showing the flag *doing* something rather than failing to
    // break something:
    //
    //     k = 1:  flag off  u* 2.21, 2.09, 2.05   |   flag on  u* 3.12, 3.05, 3.01
    //     k = 2:  flag off  u* 4.47, 4.08, 3.96   |   flag on  u* 4.24, 3.99, 3.97
    //     k = 3:  flag off  u* 4.99, 4.89, 4.71   |   flag on  u* 5.27, 4.98, 4.94
    //
    // That is not a new phenomenon: it reproduces, under coupling, the split
    // MMSConvergenceTests measures without one and Tests/README.md records as
    // unexplained -- flag off, the interpolatory scheme's postprocessing
    // superconverges at k = 2 but not at k = 1, and the flag restores it at
    // k = 1 and preserves it at k = 2. The coupling neither causes nor cures it.
    //
    // The k = 3 flag-off row is the one genuinely new number, and it decays:
    // 4.99, 4.89, 4.71, the same shape as the nonlinear flux's transient
    // superconvergence. The flag-on column does not (5.27, 4.98, 4.94).
    const std::vector<double> floorFor = {0.0, 2.75, 3.75, 4.65};

    for (Index k = 1; k <= 3; ++k)
    {
        const Rates on = solveCoupledAndMeasure<ManufacturedField>(k, kCells,
                                                                   /*superconvergent=*/true);
        const Rates &off = iterativeSweep(k);

        BOOST_TEST_MESSAGE("k = " << k << " local orders in u*: flag off "
                                  << format(off.localStar) << "  flag on "
                                  << format(on.localStar) << "   (u, flag on: "
                                  << format(on.localU) << ")" << on.detail);

        // k+2, at every step of the sweep. k = 3 gets 0.35 of headroom rather
        // than 0.25 because its finest u* is ~2e-9, within about an order of the
        // 1e-9 relative integration tolerance, so that point is the closest in
        // the file to measuring the integrator rather than the mesh.
        for (double o : on.localStar)
            BOOST_CHECK_GT(o, floorFor[static_cast<size_t>(k)]);

        // u* must be worth having: better than u_h itself, which is the
        // assertion MMSConvergenceTests makes for the uncoupled problems.
        for (size_t i = 0; i < on.localStar.size(); ++i)
            BOOST_TEST(on.localStar[i] > on.localU[i] + 0.5,
                       "k = " << k << ": u* is no better than u between n="
                              << on.cells[i] << " and n=" << on.cells[i + 1] << " (u "
                              << on.localU[i] << ", u* " << on.localStar[i] << ")");

        checkNoFallbacks(on);

        // ...and the demonstration itself, at the one degree where there is one
        // to make. Asserted rather than left to the message above, because a
        // change that quietly cost the flag its k = 1 restoration would
        // otherwise show up nowhere: every other assertion in this file is
        // satisfied flag-off as well as flag-on.
        if (k == 1)
            for (size_t i = 0; i < on.localStar.size(); ++i)
                BOOST_TEST(on.localStar[i] > off.localStar[i] + 0.5,
                           "k = 1: the flag no longer restores the postprocessing's extra "
                           "order between n="
                               << on.cells[i] << " and n=" << on.cells[i + 1]
                               << " (flag off " << off.localStar[i] << ", flag on "
                               << on.localStar[i] << ")");
    }
}

BOOST_AUTO_TEST_SUITE_END()
