// Tests for the Mixed (Robin) boundary condition's assembly.
//
// A Mixed end contributes one row of the condensed trace system,
//
//     (b q + d sigma).n + tau (u - lambda) + n a lambda = n c
//
// which divided through by the outward normal is the case author's
// `a u + b q + d sigma = c` with no normals in it. That is the convention today's
// Neumann already follows: its +-phi in Cq and its +-c in L_global cancel, so a
// Neumann value means `q = g` at *both* ends.
//
// Two consequences are what this file exists to pin.
//
//  * `Mixed(a=0, b=1, d=0)` must reproduce Neumann **exactly**, and
//    `Mixed(0, 0, 1)` must reproduce Neumann under `zeroFlux`. Not to a
//    tolerance: the same numbers, because the coefficients multiply the same
//    basis evaluations.
//
//    Read what those two now prove carefully. They were written before the
//    Neumann path was expressed in terms of this one, when they compared two
//    independent pieces of assembly; that is what licensed the reimplementation.
//    Since it landed, both kinds go through the same two lines, so what survives
//    is a check on the *coefficient mapping* -- that a Neumann end becomes b = 1,
//    and d = 1 rather than b = 1 when `zeroFlux` is set. Still worth having, and
//    still load bearing: reversing the mapping fails both. But the independent
//    check that the flag's behaviour did not move is the end-to-end one, in
//    python-examples/shestakov-nonlinear (ANALYSIS.md section 8), whose numbers
//    were measured before any of this existed.
//
//  * The `a` coefficient goes on the *lambda* column (H), not on the interior u
//    (G), and carries the normal. Both halves are easy to get wrong and neither
//    shows up as a crash: the HDG literature relates the numerical flux to the
//    trace unknown (Cui & Zhang, refs/HDG-Helmholtz-Robin.pdf), and a sign error
//    on the normal converges to the wrong function at the right rate -- the
//    failure mode CLAUDE.md records for the sigma convention, where an order
//    study passes and only a closed-form comparison catches it.
//
// The matrices compared here are the ones the trace system is built from:
// SystemSolver.cpp:1027 forms K_cell from CG_cellwise (Csigma, Cq, G) and
// H_cellwise, and the adjoint transposes the same six containers, so an
// equivalence at this level is an equivalence of both operators.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "FiniteDifferenceJacobian.hpp"
#include "SystemSolver.hpp"
#include "Types.hpp"

#include <sundials/sundials_context.h>

#include <cmath>
#include <filesystem>
#include <numbers>
#include <toml.hpp>
#include <vector>

using namespace toml::literals::toml_literals;

namespace
{

// Linear diffusion with sigma_hat = kappa q, kappa != 1 deliberately: it is what
// makes q, sigma_hat and the stored sigma three distinguishable numbers, so a
// `d sigma` term cannot be confused with a `b q` one.
class MixedDiffusion : public TransportSystem
{
public:
    explicit MixedDiffusion(BoundaryCondition lower, BoundaryCondition upper,
                            double kappa_ = 2.0)
        : TransportSystem({.variables = {{"u", "the diffused quantity", "", lower, upper}}}),
          kappa(kappa_)
    {
    }

    Value SigmaFn(Index, const State &s, Position, Time) override { return kappa * s.q(0); }
    Value Sources(Index, const State &, Position, Time) override { return 1.0; }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = kappa; }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    Value InitialValue(Index, Position x) const override { return x * (1.0 - x); }
    Value InitialDerivative(Index, Position x) const override { return 1.0 - 2.0 * x; }

    Value LowerBoundary(Index, Time) const override { return cLower; }
    Value UpperBoundary(Index, Time) const override { return cUpper; }

    double kappa;
    double cLower = 0.0, cUpper = 0.0;
};

// Just enough of a solver to have initialiseMatrices run: tau has to be set
// before it (SystemSolver.cpp says so), and nothing here goes near IDA.
struct Assembled
{
    Grid grid;
    SystemSolver sys;

    Assembled(TransportSystem &problem, Index k, Index cells, bool zeroFlux = false,
              double tau = 1.5)
        : grid(0.0, 1.0, cells), sys(grid, k, &problem)
    {
        sys.setTau(tau);
        sys.setZeroFlux(zeroFlux);
        sys.resetCoeffs();
        sys.initialiseMatrices();
    }
};

// Exact equality, element by element, over the four face matrices the trace
// system is assembled from. Returns the largest absolute difference so a failure
// message can say how far off it is rather than just that it is.
double faceMatrixDifference(SystemSolver const &a, SystemSolver const &b, Index cells)
{
    double worst = 0.0;
    auto cmp = [&worst](Eigen::MatrixXd const &l, Eigen::MatrixXd const &r)
    {
        BOOST_REQUIRE(l.rows() == r.rows());
        BOOST_REQUIRE(l.cols() == r.cols());
        worst = std::max(worst, (l - r).cwiseAbs().maxCoeff());
    };
    for (Index i = 0; i < cells; ++i)
    {
        // All six containers initializeMatricesForAdjointSolve reuses and
        // transposes (SystemSolver.cpp:1513-1534). Comparing the whole set is
        // what makes this an equivalence of the *adjoint* operator as well as the
        // forward one, since the adjoint is built from these and nothing else
        // that knows about a boundary.
        cmp(a.Csigma_cellwise[i], b.Csigma_cellwise[i]);
        cmp(a.Cq_cellwise[i], b.Cq_cellwise[i]);
        cmp(a.G_cellwise[i], b.G_cellwise[i]);
        cmp(a.H_cellwise[i], b.H_cellwise[i]);
        cmp(a.C_cellwise[i], b.C_cellwise[i]);
        cmp(a.E_cellwise[i], b.E_cellwise[i]);
    }
    return worst;
}

const Index k = 2, cells = 4;

// ------------------------------------------------- the closed-form fixtures --

// Steady linear diffusion with no source: d/dx (kappa q) = 0, so u is *linear*.
// A linear function lies in P_k for every k >= 1, and the exact solution
// satisfies the discrete mixed row exactly -- at the exact state lambda is the u
// trace, so the tau (u - lambda) term vanishes and the row reduces to
// `a u + b q + d sigma = c` -- so HDG reproduces it to round-off rather than to
// an order. That is what makes these assertions machine precision and not a
// convergence rate, and it is the only kind of test that can catch the sign of
// `a`: get it wrong and the method still converges, at the right rate, to a
// different straight line.
//
// The initial condition adds sin(pi x), which respects the Dirichlet end and
// decays, so the run has something to relax *from* -- starting at the answer
// would test nothing about the boundary driving it there.
//
// The coefficients below are not free. The energy identity for this problem is
// d/dt Int u^2/2 = [u q] - Int q^2, so the boundary term at the *lower* end is
// -u(0) q(0), and a homogeneous `a u + b q = 0` there gives q = -(a/b) u and a
// contribution +(a/b) u(0)^2. Dissipation therefore wants **a and b of opposite
// signs at the lower end**, and of the same sign at the upper one, where the
// bracket enters with the other sign. Choosing them the other way round is not a
// discretisation problem but an anti-dissipative boundary condition, and the
// symptom is a run that diverges -- 1e15 by t = 10 when this test was first
// written with a = 2, b = +1 below.
class LinearRelax : public TransportSystem
{
public:
    LinearRelax(BoundaryCondition lower, BoundaryCondition upper, double cL, double cU,
                double slopeGuess)
        : TransportSystem({.variables = {{"u", "the diffused quantity", "", lower, upper}}}),
          cLower(cL), cUpper(cU), slope(slopeGuess)
    {
    }

    Value SigmaFn(Index, const State &s, Position, Time) override { return s.q(0); }
    Value Sources(Index, const State &, Position, Time) override { return 0.0; }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 1.0; }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    // Any smooth start that respects whichever end is Dirichlet.
    Value InitialValue(Index, Position x) const override
    {
        return intercept + slope * x + std::sin(std::numbers::pi * x);
    }
    Value InitialDerivative(Index, Position x) const override
    {
        return slope + std::numbers::pi * std::cos(std::numbers::pi * x);
    }

    Value LowerBoundary(Index, Time) const override { return cLower; }
    Value UpperBoundary(Index, Time) const override { return cUpper; }

    double cLower, cUpper, slope, intercept = 1.0;
};

// Relax to steady state and return u at a set of points.
Vector relaxed(TransportSystem &problem, std::string const &stem, std::vector<double> const &xs,
               Index order = 3, Index nCells = 6)
{
    Grid grid(0.0, 1.0, nCells);
    SystemSolver sys(grid, order, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile(stem);
    sys.setOutputCadence(10.0);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-12);
    sys.setTolerances({1e-9}, 1e-8);
    // The algebraic rows are in IDA's local error test otherwise, and at these
    // tolerances that is the wall docs/running.rst describes rather than anything
    // about the boundary condition under test.
    sys.setSuppressAlgebraicError(true);
    sys.setWriteOutput(false);
    sys.setWriteDatFile(false);

    {
        CapturedOutput quiet;
        sys.initialize();
        sys.integrate(10.0); // ~100 diffusion times on [0, 1]; fully relaxed
        sys.destroySundials();
    }
    for (const char *ext : {".nc", ".restart.nc", ".dat"})
        std::filesystem::remove(stem + ext);

    Vector out(static_cast<Eigen::Index>(xs.size()));
    for (size_t i = 0; i < xs.size(); ++i)
        out(static_cast<Eigen::Index>(i)) = sys.yJac.u(0)(xs[i]);
    return out;
}

const std::vector<double> samplePoints{0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0};

Vector straightLine(double intercept, double slope)
{
    Vector out(static_cast<Eigen::Index>(samplePoints.size()));
    for (size_t i = 0; i < samplePoints.size(); ++i)
        out(static_cast<Eigen::Index>(i)) = intercept + slope * samplePoints[i];
    return out;
}

} // namespace

BOOST_AUTO_TEST_SUITE(mixed_boundary_tests)

BOOST_AUTO_TEST_CASE(mixed_b_equals_one_is_neumann_exactly)
{
    // The equivalence that licenses everything else. Both ends, so a sign that is
    // right at one end only cannot pass.
    MixedDiffusion neumann(BoundaryKind::Neumann, BoundaryKind::Neumann);
    MixedDiffusion mixed(BoundaryCondition::mixed(0.0, 1.0, 0.0),
                         BoundaryCondition::mixed(0.0, 1.0, 0.0));

    Assembled n(neumann, k, cells);
    Assembled m(mixed, k, cells);

    BOOST_TEST(faceMatrixDifference(n.sys, m.sys, cells) == 0.0);
}

BOOST_AUTO_TEST_CASE(mixed_d_equals_one_is_neumann_under_zero_flux_exactly)
{
    // The other half: `zeroFlux` reinterprets a Neumann end as a condition on
    // sigma, which is `d = 1`. Proving this is what lets the flag be
    // reimplemented as a preset instead of remaining a second mechanism.
    MixedDiffusion neumann(BoundaryKind::Neumann, BoundaryKind::Neumann);
    MixedDiffusion mixed(BoundaryCondition::mixed(0.0, 0.0, 1.0),
                         BoundaryCondition::mixed(0.0, 0.0, 1.0));

    Assembled n(neumann, k, cells, /* zeroFlux */ true);
    Assembled m(mixed, k, cells, /* zeroFlux */ false);

    BOOST_TEST(faceMatrixDifference(n.sys, m.sys, cells) == 0.0);
}

BOOST_AUTO_TEST_CASE(a_mixed_end_does_not_disturb_the_other_one)
{
    // Mixed below, Dirichlet above. The Dirichlet end must keep its identically
    // zero row -- that is what makes a finite-differenced Jacobian rank-deficient
    // by exactly the Dirichlet count, which SolveJacTests asserts by index.
    MixedDiffusion sys(BoundaryCondition::mixed(1.0, 1.0, 0.0), BoundaryKind::Dirichlet);
    Assembled a(sys, k, cells);

    const Index last = cells - 1;
    // Upper end, Dirichlet: Csigma, Cq, G and H all zero in that row.
    BOOST_TEST(a.sys.Csigma_cellwise[last](1, 0) == 0.0);
    BOOST_TEST(a.sys.Cq_cellwise[last](1, 0) == 0.0);
    BOOST_TEST(a.sys.G_cellwise[last](1, 0) == 0.0);
    BOOST_TEST(a.sys.H_cellwise[last](1, 1) == 0.0);

    // Lower end, Mixed: none of them is.
    BOOST_TEST(a.sys.Cq_cellwise[0](0, 0) != 0.0);
    BOOST_TEST(a.sys.H_cellwise[0](0, 0) != 0.0);
}

BOOST_AUTO_TEST_CASE(the_coefficients_scale_the_entries_they_should)
{
    // b scales Cq, d scales Csigma, and neither touches G, which stays the tau
    // stabilisation. Compared against b = d = 1 so the factor is visible rather
    // than inferred.
    MixedDiffusion unit(BoundaryCondition::mixed(0.0, 1.0, 1.0), BoundaryKind::Dirichlet);
    MixedDiffusion scaled(BoundaryCondition::mixed(0.0, 3.0, 7.0), BoundaryKind::Dirichlet);

    Assembled u(unit, k, cells);
    Assembled s(scaled, k, cells);

    for (Index i = 0; i < k + 1; ++i)
    {
        BOOST_TEST(s.sys.Cq_cellwise[0](0, i) == 3.0 * u.sys.Cq_cellwise[0](0, i));
        BOOST_TEST(s.sys.Csigma_cellwise[0](0, i) == 7.0 * u.sys.Csigma_cellwise[0](0, i));
        BOOST_TEST(s.sys.G_cellwise[0](0, i) == u.sys.G_cellwise[0](0, i));
    }
}

BOOST_AUTO_TEST_CASE(the_a_coefficient_lands_on_the_lambda_column_with_the_normal)
{
    // a goes into H, not G, and carries the outward normal: -a below, +a above,
    // so that dividing the row by n leaves a plain `a u` for the case author.
    //
    // Reading it off H directly rather than through a solve is deliberate. This
    // is the one sign in the change that a convergence study cannot see -- get it
    // backwards and the method converges, at the right rate, to the wrong
    // function.
    const double tau = 1.5, a = 0.25;
    MixedDiffusion sys(BoundaryCondition::mixed(a, 1.0, 0.0),
                       BoundaryCondition::mixed(a, 1.0, 0.0));
    Assembled asm_(sys, k, cells, false, tau);

    BOOST_TEST(asm_.sys.H_cellwise[0](0, 0) == -tau - a);
    BOOST_TEST(asm_.sys.H_cellwise[cells - 1](1, 1) == -tau + a);

    // And with a = 0 it is exactly the Neumann diagonal, which the equivalence
    // tests above depend on.
    MixedDiffusion noA(BoundaryCondition::mixed(0.0, 1.0, 0.0),
                       BoundaryCondition::mixed(0.0, 1.0, 0.0));
    Assembled zero(noA, k, cells, false, tau);
    BOOST_TEST(zero.sys.H_cellwise[0](0, 0) == -tau);
    BOOST_TEST(zero.sys.H_cellwise[cells - 1](1, 1) == -tau);
}

BOOST_AUTO_TEST_CASE(a_mixed_end_receives_its_datum)
{
    // The matrices above are only half of a boundary condition; the other half is
    // c, which reaches the row through L_global. `updateBoundaryConditions` fills
    // that for every end that is *not* Dirichlet, so Mixed already gets it -- but
    // nothing above would notice if it did not, because the coefficients would
    // still be right and the row would just be missing its right-hand side.
    // Checked at a nonzero t as well, since only L_global carries time dependence.
    MixedDiffusion sys(BoundaryCondition::mixed(0.25, 1.0, 0.0),
                       BoundaryCondition::mixed(0.25, 1.0, 0.0));
    sys.cLower = 0.75;
    sys.cUpper = -0.5;
    Assembled a(sys, k, cells);
    a.sys.updateBoundaryConditions(0.0);

    // Lower gets -c, upper gets +c: the same convention Neumann uses, which is
    // what makes the equivalence above hold for the datum too.
    BOOST_TEST(a.sys.L_global(0) == -0.75);
    BOOST_TEST(a.sys.L_global(cells) == -0.5);
}

BOOST_AUTO_TEST_CASE(the_datum_matches_neumanns_exactly)
{
    // Same statement as the matrix equivalence, for the right-hand side: with
    // b = 1 a Mixed end and a Neumann end given the same value must put the same
    // numbers in L_global.
    MixedDiffusion neumann(BoundaryKind::Neumann, BoundaryKind::Neumann);
    neumann.cLower = 0.75;
    neumann.cUpper = -0.5;
    MixedDiffusion mixed(BoundaryCondition::mixed(0.0, 1.0, 0.0),
                         BoundaryCondition::mixed(0.0, 1.0, 0.0));
    mixed.cLower = 0.75;
    mixed.cUpper = -0.5;

    Assembled n(neumann, k, cells);
    Assembled m(mixed, k, cells);
    n.sys.updateBoundaryConditions(0.3);
    m.sys.updateBoundaryConditions(0.3);

    BOOST_TEST((n.sys.L_global - m.sys.L_global).cwiseAbs().maxCoeff() == 0.0);
}

BOOST_AUTO_TEST_CASE(the_a_coefficient_reaches_the_global_h_matrix_too)
{
    // The local HGlobalMat is accumulated from Hvar and copied into the member
    // H_global_mat (SystemSolver.cpp:552). It feeds only H_global, which is dead
    // code today -- factorised and never read -- but a half-updated matrix left
    // for whoever revives it is worse than either state.
    const double tau = 1.5, a = 0.25;
    MixedDiffusion sys(BoundaryCondition::mixed(a, 1.0, 0.0), BoundaryKind::Dirichlet);
    Assembled asm_(sys, k, cells, false, tau);

    BOOST_TEST(asm_.sys.H_global_mat(0, 0) == -tau - a);
}

// ------------------------------------------------ the closed-form solutions --

BOOST_AUTO_TEST_CASE(a_mixed_lower_end_reaches_its_closed_form)
{
    // Mixed below with a = 2, b = -1, c = -1 -- opposite signs, so dissipative --
    // and Dirichlet above with u(1) = 1. Steady u = A + Bx with A + B = 1 and
    // 2A - B = -1, so B = 1, A = 0:
    //
    //     u = x
    //
    // Check by hand at the boundary: u(0) = 0, q = 1, and 2(0) - 1(1) = -1.
    LinearRelax problem(BoundaryCondition::mixed(2.0, -1.0, 0.0), BoundaryKind::Dirichlet,
                        -1.0, 1.0, /* slope guess */ 0.0);
    problem.intercept = 1.0; // IC is 1 + sin(pi x): u(1) = 1, the Dirichlet end

    const Vector got = relaxed(problem, "mixed_lower_closed_form", samplePoints);
    const Vector want = straightLine(0.0, 1.0);

    BOOST_TEST((got - want).cwiseAbs().maxCoeff() < 1e-8);
}

BOOST_AUTO_TEST_CASE(a_mixed_upper_end_reaches_its_closed_form)
{
    // The mirror, and the point of having both: the `a` coefficient carries the
    // outward normal, so an implementation that used one sign at both ends passes
    // the test above and fails this one.
    //
    // Dirichlet below with u(0) = 1; Mixed above with a = 2, b = 1, c = 3.
    // A = 1 and 2(A + B) + B = 3 give B = 1/3:
    //
    //     u = 1 + x/3
    //
    // At the boundary: u(1) = 4/3, q = 1/3, and 2(4/3) + 1(1/3) = 3.
    LinearRelax problem(BoundaryKind::Dirichlet, BoundaryCondition::mixed(2.0, 1.0, 0.0),
                        1.0, 3.0, /* slope guess */ 0.0);
    problem.intercept = 1.0; // IC is 1 + sin(pi x): u(0) = 1, the Dirichlet end

    const Vector got = relaxed(problem, "mixed_upper_closed_form", samplePoints);
    const Vector want = straightLine(1.0, 1.0 / 3.0);

    BOOST_TEST((got - want).cwiseAbs().maxCoeff() < 1e-8);
}

BOOST_AUTO_TEST_CASE(the_d_coefficient_multiplies_the_stored_sigma)
{
    // d = 1 alone, with a *nonzero* c, which is the only way to tell the stored
    // sigma from the physical flux apart: the stored sigma is -sigma_hat, and
    // here sigma_hat = q, so sigma = -q.
    //
    // Mixed below with d = 1, c = 0.5; Dirichlet above with u(1) = 1.
    // sigma(0) = -B = 0.5 gives B = -0.5, and A + B = 1 gives A = 1.5:
    //
    //     u = 1.5 - 0.5 x
    //
    // Read against sigma_hat instead and the sign flips, giving u = 0.5 + 0.5x --
    // a different straight line, so this discriminates rather than merely passes.
    LinearRelax problem(BoundaryCondition::mixed(0.0, 0.0, 1.0), BoundaryKind::Dirichlet,
                        0.5, 1.0, /* slope guess */ 0.0);
    problem.intercept = 1.0;

    const Vector got = relaxed(problem, "mixed_d_closed_form", samplePoints);
    const Vector want = straightLine(1.5, -0.5);

    BOOST_TEST((got - want).cwiseAbs().maxCoeff() < 1e-8);
}

BOOST_AUTO_TEST_CASE(a_mixed_end_with_b_equal_one_matches_a_neumann_run)
{
    // The equivalence at the level of an answer rather than a matrix, which is
    // what the unit-level one stopped being once both kinds shared a code path.
    // Neumann with value g and Mixed(0, 1, 0) with c = g must relax to the same
    // solution; a nonzero g, so the two are distinguishable from zero.
    LinearRelax neumann(BoundaryKind::Neumann, BoundaryKind::Dirichlet, -0.4, 1.0, 0.0);
    LinearRelax mixed(BoundaryCondition::mixed(0.0, 1.0, 0.0), BoundaryKind::Dirichlet,
                      -0.4, 1.0, 0.0);

    const Vector n = relaxed(neumann, "mixed_vs_neumann_n", samplePoints);
    const Vector m = relaxed(mixed, "mixed_vs_neumann_m", samplePoints);

    BOOST_TEST((n - m).cwiseAbs().maxCoeff() == 0.0);

    // And it is the line the condition asks for: q = -0.4 with u(1) = 1.
    BOOST_TEST((m - straightLine(1.4, -0.4)).cwiseAbs().maxCoeff() < 1e-8);
}

// ----------------------------------------------------- order of accuracy --

namespace
{

// Steady manufactured problem with a solution that is *not* a polynomial, so
// there is a rate to observe rather than exactness. u_e = exp(x), so
// sigma_hat = q = exp(x) and the steady equation -d/dx sigma_hat = S needs
// S = -exp(x). Robin below with a = 2, b = -1 -- dissipative, per the note above
// -- so c = 2 u_e(0) - q_e(0) = 2 - 1 = 1; Dirichlet above with u(1) = e.
//
// This is the test that would catch a boundary condition which is *consistent*
// but imposed on the wrong quantity. That failure does not reduce the order to
// zero, it pins it at one and makes it independent of k -- which is exactly the
// signature commit 30ef962 found in the Jardin benchmark, where k = 5 bought no
// rate over k = 3 and a wrong Neumann value was the cause.
class ExpSteady : public TransportSystem
{
public:
    ExpSteady()
        : TransportSystem({.variables = {{"u", "", "",
                                          BoundaryCondition::mixed(2.0, -1.0, 0.0),
                                          BoundaryKind::Dirichlet}}})
    {
    }

    Value SigmaFn(Index, const State &s, Position, Time) override { return s.q(0); }
    Value Sources(Index, const State &, Position x, Time) override { return -std::exp(x); }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 1.0; }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    // Start away from the answer, but respecting the Dirichlet end.
    Value InitialValue(Index, Position x) const override { return 1.0 + (std::numbers::e - 1.0) * x; }
    Value InitialDerivative(Index, Position) const override { return std::numbers::e - 1.0; }

    Value LowerBoundary(Index, Time) const override { return 1.0; }         // c = 2 - 1
    Value UpperBoundary(Index, Time) const override { return std::numbers::e; }

    static double exact(double x) { return std::exp(x); }
};

// Max error in u over a fine sample, after relaxing to steady state.
double expError(Index order, Index nCells)
{
    ExpSteady problem;
    std::vector<double> xs;
    for (int i = 0; i <= 40; ++i)
        xs.push_back(static_cast<double>(i) / 40.0);
    const Vector got = relaxed(problem, "mixed_order_" + std::to_string(order) + "_" +
                                            std::to_string(nCells),
                               xs, order, nCells);
    double worst = 0.0;
    for (size_t i = 0; i < xs.size(); ++i)
        worst = std::max(worst,
                         std::abs(got(static_cast<Eigen::Index>(i)) - ExpSteady::exact(xs[i])));
    return worst;
}

} // namespace

BOOST_AUTO_TEST_CASE(a_mixed_boundary_keeps_the_order_of_accuracy)
{
    // Two orders, so the observed rate has to *change* with k. A condition
    // imposed on the wrong quantity gives first order at every k, which is what
    // makes the k-dependence the discriminating part rather than either rate on
    // its own.
    for (Index order : {1, 2})
    {
        const double e1 = expError(order, 4);
        const double e2 = expError(order, 8);
        const double rate = std::log2(e1 / e2);
        BOOST_TEST_MESSAGE("k = " << order << ": " << e1 << " -> " << e2
                                  << ", observed order " << rate);
        // k+1 expected; allow a wide margin, since the point is to separate
        // "converges at the right order" from "pinned at one".
        BOOST_TEST(rate > static_cast<double>(order) + 0.5);
    }
}

// ------------------------------------------------- the Jacobian of that row --

BOOST_AUTO_TEST_CASE(the_mixed_row_appears_in_the_finite_differenced_jacobian)
{
    // Two things at once, and the second is the reason for the first.
    //
    // A Mixed row is *not* a zero row, unlike a Dirichlet one, so a problem with
    // Mixed at both ends has no undefined rows at all -- where SolveJacTests
    // asserts a set of exactly two, by index, for its Dirichlet fixture. If the
    // mixed row were missing from residual() the count here would be two again,
    // silently, and the linear solve would still "work" while solving something
    // else.
    //
    // And the solve has to be consistent with it: J dy = g over every row.
    MixedDiffusion problem(BoundaryCondition::mixed(2.0, -1.0, 0.0),
                           BoundaryCondition::mixed(2.0, 1.0, 0.0));
    const Index order = 2, nCells = 3;
    Grid grid(0.0, 1.0, nCells);
    SystemSolver sys(grid, order, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile("mixed_fdjac");
    sys.setOutputCadence(1.0);
    sys.setNOutput(5);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-12);
    sys.setTolerances({1e-8}, 1e-6);
    sys.setWriteOutput(false);
    sys.setWriteDatFile(false);

    SUNContext ctx = nullptr;
    SUNContext_Create(SUN_COMM_NULL, &ctx);
    {
        CapturedOutput quiet;
        sys.initialize();
    }

    const Index n = N_VGetLength(sys.Y);
    const double t = 0.0, cj = 2.0;
    sys.setJacTime(t);
    sys.setAlpha(cj);
    sys.setJacEvalY(sys.Y, sys.dYdt);
    sys.updateBoundaryConditions(t);
    sys.updateMatricesForJacSolve();

    const Matrix J = fdjac::jacobian(sys, sys.Y, sys.dYdt, t, cj);
    const std::vector<Index> empty = fdjac::undefinedRows(J);

    // Mixed at both ends: nothing undefined.
    BOOST_TEST(empty.empty());

    // And the solve satisfies it.
    N_Vector g = N_VNew_Serial(n, ctx);
    N_Vector dy = N_VClone(g);
    double *ga = N_VGetArrayPointer(g);
    for (Index i = 0; i < n; ++i)
        ga[i] = 0.1 + 0.01 * static_cast<double>((i * 7) % 13);
    {
        CapturedOutput quiet;
        sys.solveJacEq(g, dy);
    }
    Vector dyv(n), gv(n);
    const double *dya = N_VGetArrayPointer(dy);
    for (Index i = 0; i < n; ++i)
    {
        dyv(i) = dya[i];
        gv(i) = ga[i];
    }
    BOOST_TEST(fdjac::relativeResidual(J, dyv, gv, empty) < 1e-6);

    N_VDestroy(g);
    N_VDestroy(dy);
    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
    SUNContext_Free(&ctx);
}

// ------------------------------------------------------------------ restart --

BOOST_AUTO_TEST_CASE(a_restart_recovers_c_from_the_mixed_row)
{
    // setRestartValues recovers a boundary value from the restarted profile so a
    // case need not know it on resume. For a Mixed end the one number to recover
    // is c, which means evaluating the row's own left-hand side -- and `d` against
    // the *stored* sigma, since that is what the assembly multiplies. Reading the
    // other one is the defect commit 9018d0c fixed, one level up.
    // Deliberately *not* derived from the fixtures above: they override
    // LowerBoundary/UpperBoundary, and an overriding case never reads uL/uR at
    // all, so the recovery under test would be invisible. That is the third of
    // the three conditions commit 9018d0c records as needed to reach this code.
    struct RestartMixed : public TransportSystem
    {
        RestartMixed()
            : TransportSystem({.variables = {{"u", "", "",
                                              BoundaryCondition::mixed(2.0, 3.0, 5.0),
                                              BoundaryKind::Neumann}}})
        {
        }
        Value SigmaFn(Index, const State &s, Position, Time) override { return s.q(0); }
        Value Sources(Index, const State &, Position, Time) override { return 0.0; }
        void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
        void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 1.0; }
        void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
        void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
        void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
        Value InitialValue(Index, Position x) const override { return x; }
        Value InitialDerivative(Index, Position) const override { return 1.0; }
    } sys;

    const Index nCells = 3, order = 2, nVars = 1;
    Grid grid(0.0, 1.0, nCells);
    DGSoln shape(nVars, grid, order);
    std::vector<double> Ydata(shape.getDoF(), 0.0), dYdata(shape.getDoF(), 0.0);

    // u = 2 + 3x, q = 7, sigma = -11: three numbers no two of which could be
    // confused, so the assertion below says which of them each coefficient got.
    {
        DGSoln tmp(nVars, grid, order, Ydata.data());
        tmp.AssignU([](Index, double x) { return 2.0 + 3.0 * x; });
        tmp.AssignQ([](Index, double) { return 7.0; });
        tmp.AssignSigma([](Index, const State &, Position, Time) -> Value { return -11.0; });
        tmp.EvaluateLambda();
    }

    sys.setRestartValues(Ydata, dYdata, grid, order);

    // Lower is Mixed(a=2, b=3, d=5) at x = 0, where u = 2:
    //     2(2) + 3(7) + 5(-11) = 4 + 21 - 55 = -30
    BOOST_TEST(sys.LowerBoundary(0, 0.0) == -30.0, boost::test_tools::tolerance(1e-12));
    // Upper is Neumann, so still q and only q.
    BOOST_TEST(sys.UpperBoundary(0, 0.0) == 7.0, boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_SUITE_END()
