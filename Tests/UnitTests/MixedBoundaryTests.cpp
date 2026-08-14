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
//    basis evaluations. That equivalence is the whole safety argument for the
//    change, and it is what licenses reimplementing `zeroFlux` in terms of this
//    path rather than hoping the two agree.
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

#include "SystemSolver.hpp"
#include "Types.hpp"

#include <toml.hpp>

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

BOOST_AUTO_TEST_SUITE_END()
