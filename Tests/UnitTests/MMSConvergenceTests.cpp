// Order-of-accuracy tests by the method of manufactured solutions.
//
// This is the strongest correctness signal available for the solver as a whole:
// it exercises the residual, the block assembly, the static-condensation solve
// and IDA's time integration together, and it fails if any of them is even
// slightly wrong in a way the unit tests cannot express. A wrong sign in a
// single block still converges -- to the wrong answer -- but it does not
// converge at the right *rate* to the right limit.
//
// The manufactured solution here is
//
//     u(x, t) = sin(pi x) * (1 + t)      on [0, 1]
//
// which vanishes at both ends for every t, so it is consistent with the
// homogeneous Dirichlet boundary conditions. That matters: an MMS whose exact
// solution does not satisfy the boundary conditions imposed by the physics case
// converges at the wrong rate, or not at all. (The `UseMMS` option on
// `LinearDiffusion` has exactly that problem -- see the note at the end of this
// file.)
//
// Substituting into d_t u = d_x( kappa d_x u ) + S gives
//
//     S(x, t) = sin(pi x) * ( 1 + kappa pi^2 (1 + t) )
//
// The expected L2 rate for HDG with a degree-k basis is k+1.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "SystemSolver.hpp"
#include "Types.hpp"

#include <boost/math/quadrature/gauss.hpp>

#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <typeinfo>
#include <vector>

namespace
{

using std::numbers::pi;

constexpr double KAPPA = 0.7;

double exactSolution(double x, double t) { return std::sin(pi * x) * (1.0 + t); }
double exactDerivative(double x, double t) { return pi * std::cos(pi * x) * (1.0 + t); }

class ManufacturedDiffusion : public TransportSystem
{
public:
    ManufacturedDiffusion() { nVars = 1; }

    // u = 0 at both ends for all t, matching the manufactured solution.
    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
    bool isLowerBoundaryDirichlet(Index) const override { return true; }
    bool isUpperBoundaryDirichlet(Index) const override { return true; }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return KAPPA * s.Derivative[0];
    }

    // The manufactured forcing: d_t u - kappa u_xx evaluated on the exact
    // solution. Written out rather than differentiated numerically so the test
    // does not depend on any of the machinery it is testing.
    Value Sources(Index, const State &, Position x, Time t) override
    {
        return std::sin(pi * x) * (1.0 + KAPPA * pi * pi * (1.0 + t));
    }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = KAPPA;
    }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }

    Value InitialValue(Index, Position x) const override { return exactSolution(x, 0.0); }
    Value InitialDerivative(Index, Position x) const override
    {
        return exactDerivative(x, 0.0);
    }
};

// The same manufactured solution with a nonlinear reaction term added --
// F(u) = u^3 - u, the Chaffee-Infante / Allen-Cahn nonlinearity of Example 4.1
// in Chen, Cockburn, Singler & Zhang.
//
// This, and not a non-polynomial source, is where the interpolatory method loses
// superconvergence: I_h F(u_h) evaluates the nonlinearity at u_h, which is only
// O(h^(k+1)) accurate pointwise, so the consistency error enters at that order.
// Interpolating a *known* smooth source at the Chebyshev nodes does not have the
// same effect -- see the note in the superconvergence test case below.
//
// Solving u_t - u_xx + F(u) = f for the same u = sin(pi x)(1 + t) gives
//
//     f = sin(pi x) ( 1 + pi^2 (1 + t) ) + F(u_exact)
//
// and MaNTA's source is S = f - F(u), which reduces to the right forcing at the
// exact solution while retaining a genuine dependence on the state -- the
// dependence that exercises the chain rule through the postprocessing.
class ManufacturedReaction : public TransportSystem
{
public:
    ManufacturedReaction() { nVars = 1; }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
    bool isLowerBoundaryDirichlet(Index) const override { return true; }
    bool isUpperBoundaryDirichlet(Index) const override { return true; }

    static double F(double u) { return u * u * u - u; }
    static double dF(double u) { return 3.0 * u * u - 1.0; }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return s.Derivative[0];
    }

    Value Sources(Index, const State &s, Position x, Time t) override
    {
        const double ue = exactSolution(x, t);
        const double f = std::sin(pi * x) * (1.0 + pi * pi * (1.0 + t)) + F(ue);
        return f - F(s.Variable[0]);
    }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 1.0;
    }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = -dF(s.Variable[0]);
    }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }

    Value InitialValue(Index, Position x) const override { return exactSolution(x, 0.0); }
    Value InitialDerivative(Index, Position x) const override
    {
        return exactDerivative(x, 0.0);
    }
};

// Cell-by-cell Gauss-30 quadrature of (f - u_exact)^2. Independent of the
// basis's own integration weights, which are part of what is under test.
double l2ErrorOf(std::function<double(double)> f, Grid const &grid, double t)
{
    boost::math::quadrature::gauss<double, 30> gauss;
    double total = 0.0;
    for (size_t cell = 0; cell < grid.getNCells(); ++cell)
    {
        Interval const &I = grid[cell];
        auto integrand = [&](double x)
        {
            const double d = f(x) - exactSolution(x, t);
            return d * d;
        };
        total += gauss.integrate(integrand, I.x_l, I.x_u);
    }
    return std::sqrt(total);
}

double l2Error(SystemSolver &sys, Grid const &grid, double t)
{
    return l2ErrorOf([&](double x) { return sys.yJac.u(0)(x); }, grid, t);
}

// The two errors every run reports: the solution's own and the postprocessed
// one. HDG gives k+1 for the first; the second is k+2 when the method is
// superconvergent, and that difference is the whole point of the feature.
struct Errors
{
    double u;
    double uStar;
};

// Run to tFinal on a uniform grid of nCells cells at degree k, and return the L2
// errors of the final solution and of its postprocessing.
//
// runSolver writes <stem>.nc / .dat / .restart.nc into the working directory,
// so the output name is unique per case and the files are removed afterwards.
template <class Problem = ManufacturedDiffusion>
Errors solveAndMeasureBoth(Index k, Index nCells, double tFinal,
                           bool superconvergent = false)
{
    Grid grid(0.0, 1.0, nCells);
    Problem problem;

    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.setSuperconvergent(superconvergent);
    sys.resetCoeffs();

    const std::string stem = "mms_" + std::string(typeid(Problem).name()) + "_k" +
                             std::to_string(k) + "_n" + std::to_string(nCells) +
                             (superconvergent ? "_sc" : "");
    sys.setInputFile(stem);

    sys.setOutputCadence(tFinal);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-14);
    // Tight enough that the temporal error is far below the spatial error even
    // on the finest grid here (the smallest spatial error in the sweep is about
    // 1e-6, at k = 3 with 16 cells), so the measured rate is the spatial one.
    //
    // Not tighter: at 1e-12 IDA cannot get off the ground for k >= 2, failing
    // at t = 0 with "the error test failed repeatedly or with |h| = hmin". That
    // is a real limit of this solver, not of the manufactured problem.
    sys.setTolerances({1e-11}, 1e-9);

    {
        // runSolver reports its step counts and IDACalcIC warnings; sixteen
        // integrations of that is a hundred lines of noise around a passing
        // test. The measured orders are reported by BOOST_TEST_MESSAGE instead.
        CapturedOutput quiet;
        sys.runSolver(tFinal);
    }

    // u* was last reconstructed from `y`, whose N_Vector runSolver has since
    // destroyed. Rebuild it from yJac, which the solver owns.
    sys.postprocessor->computeUStar(sys.yJac);

    const Errors err{
        l2Error(sys, grid, tFinal),
        l2ErrorOf([&](double x) { return sys.getPostprocessor()->uStar(0)(x); },
                  grid, tFinal)};

    for (const char *suffix : {".nc", ".dat", ".restart.nc"})
        std::remove((stem + suffix).c_str());

    return err;
}

// The u-only form the existing cases below are written against.
double solveAndMeasure(Index k, Index nCells, double tFinal)
{
    return solveAndMeasureBoth(k, nCells, tFinal).u;
}

// Least-squares slope of log(error) against log(1/nCells) -- the observed order.
double observedOrder(std::vector<Index> const &cellCounts,
                     std::vector<double> const &errors)
{
    const size_t n = cellCounts.size();
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double x = std::log(1.0 / static_cast<double>(cellCounts[i]));
        const double y = std::log(errors[i]);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

} // namespace

BOOST_AUTO_TEST_SUITE(mms_convergence_tests)

BOOST_AUTO_TEST_CASE(the_manufactured_source_is_consistent_with_the_exact_solution)
{
    // Check the algebra before checking the solver. If S were wrong, the
    // convergence tests below would still converge -- to a different function --
    // and the rate would look fine.
    ManufacturedDiffusion problem;
    const double t = 0.37, h = 1e-5;

    for (double x : {0.13, 0.5, 0.81})
    {
        const double d2udx2 = (exactSolution(x + h, t) - 2.0 * exactSolution(x, t) +
                               exactSolution(x - h, t)) /
                              (h * h);
        const double dudt = (exactSolution(x, t + h) - exactSolution(x, t - h)) / (2.0 * h);

        State s(1);
        s.Variable[0] = exactSolution(x, t);
        s.Derivative[0] = exactDerivative(x, t);

        const double S = problem.Sources(0, s, x, t);
        BOOST_TEST(dudt - KAPPA * d2udx2 == S, boost::test_tools::tolerance(1e-5));
    }

    // And the exact solution really does satisfy the boundary conditions, for
    // every time -- the property the LinearDiffusion MMS lacks.
    for (double t2 : {0.0, 0.5, 5.0})
    {
        BOOST_TEST(exactSolution(0.0, t2) == 0.0, boost::test_tools::tolerance(1e-15));
        BOOST_TEST(exactSolution(1.0, t2) == 0.0, boost::test_tools::tolerance(1e-15));
    }
}

BOOST_AUTO_TEST_CASE(h_refinement_converges_at_the_expected_order,
                     *boost::unit_test::tolerance(1e-9))
{
    // The headline test. For each polynomial degree, refine the grid and fit
    // the observed L2 order; HDG gives k+1.
    const double tFinal = 0.25;

    struct Case
    {
        Index k;
        std::vector<Index> cells;
        double lowerBound; // the rate must be at least this
    };

    // Grids are kept modest: each entry is a full time integration at 1e-12
    // tolerances, and the asymptotic rate is already clear over these.
    const std::vector<Case> cases = {
        {1, {4, 8, 16, 32}, 1.8},
        {2, {4, 8, 16}, 2.8},
        {3, {4, 8, 16}, 3.7},
    };

    for (auto const &c : cases)
    {
        std::vector<double> errors;
        for (Index n : c.cells)
            errors.push_back(solveAndMeasure(c.k, n, tFinal));

        const double rate = observedOrder(c.cells, errors);

        std::string detail;
        for (size_t i = 0; i < c.cells.size(); ++i)
            detail += " n=" + std::to_string(c.cells[i]) + " err=" +
                      std::to_string(errors[i]);
        BOOST_TEST_MESSAGE("k = " << c.k << ": order " << rate << detail);

        BOOST_TEST(rate > c.lowerBound,
                   "k = " << c.k << ": observed order " << rate << " (expected "
                          << c.k + 1 << ")" << detail);

        // Errors must actually decrease -- a flat-but-noisy sequence can fit a
        // steep slope if one point is an outlier.
        for (size_t i = 1; i < errors.size(); ++i)
            BOOST_TEST(errors[i] < errors[i - 1],
                       "error did not decrease from n=" << c.cells[i - 1] << " to n="
                                                        << c.cells[i]);
    }
}

BOOST_AUTO_TEST_CASE(p_refinement_converges_faster_than_any_fixed_order)
{
    // Raising the polynomial degree on a fixed grid must reduce the error
    // sharply, since the exact solution is analytic. This is the complementary
    // sweep to the one above, and it catches an error that depends on k in the
    // wrong way -- a basis-order mistake that h-refinement alone would hide.
    const double tFinal = 0.25;
    const Index nCells = 4;

    std::vector<double> errors;
    for (Index k : {1, 2, 3, 4})
        errors.push_back(solveAndMeasure(k, nCells, tFinal));

    std::string detail;
    for (size_t i = 0; i < errors.size(); ++i)
        detail += " k=" + std::to_string(i + 1) + " err=" + std::to_string(errors[i]);
    BOOST_TEST_MESSAGE("p-refinement:" << detail);

    for (size_t i = 1; i < errors.size(); ++i)
        BOOST_TEST(errors[i] < errors[i - 1] / 3.0,
                   "raising k from " << i << " to " << i + 1
                                     << " did not reduce the error enough:" << detail);
}

BOOST_AUTO_TEST_CASE(the_solution_is_accurate_at_a_later_time_too)
{
    // The manufactured solution grows linearly in t, so running four times as
    // long is a real test of the time integration rather than a repeat of the
    // spatial one: a source term evaluated at the wrong time, or a dudt block
    // with the wrong coefficient, shows up here and not at t = 0.25.
    const Index k = 3, nCells = 8;

    const double early = solveAndMeasure(k, nCells, 0.25);
    const double late = solveAndMeasure(k, nCells, 1.0);

    BOOST_TEST_MESSAGE("t=0.25 err=" << early << "   t=1.0 err=" << late);

    // The exact solution is (1+t) times its t=0 shape, so the error may grow
    // with it -- but only in proportion, not by orders of magnitude.
    BOOST_TEST(late < 10.0 * early,
               "error at t=1 (" << late << ") is far worse than at t=0.25 (" << early
                                << ")");
    BOOST_TEST(late > 0.0);
}

// --------------------------------------------------------- superconvergence --

namespace
{
struct Rates
{
    double uOff, starOff, uOn, starOn;
};

// Refine, fit both orders, flag off and flag on, and report all four.
template <class Problem>
Rates measureRates(Index k, std::vector<Index> const &cells, double tFinal)
{
    std::vector<double> uOff, starOff, uOn, starOn;
    for (Index n : cells)
    {
        const Errors off = solveAndMeasureBoth<Problem>(k, n, tFinal, false);
        const Errors on = solveAndMeasureBoth<Problem>(k, n, tFinal, true);
        uOff.push_back(off.u);
        starOff.push_back(off.uStar);
        uOn.push_back(on.u);
        starOn.push_back(on.uStar);
    }
    return {observedOrder(cells, uOff), observedOrder(cells, starOff),
            observedOrder(cells, uOn), observedOrder(cells, starOn)};
}

std::string report(Index k, Rates const &r)
{
    std::string s = "k = " + std::to_string(k) + ":  flag off  u " +
                    std::to_string(r.uOff) + "  u* " + std::to_string(r.starOff) +
                    "   |   flag on  u " + std::to_string(r.uOn) + "  u* " +
                    std::to_string(r.starOn) + "   (u should be " +
                    std::to_string(k + 1) + ", u* with the flag on " +
                    std::to_string(k + 2) + ")";
    return s;
}
} // namespace

BOOST_AUTO_TEST_CASE(the_postprocessing_superconverges_with_the_flag_on)
{
    // The headline result: with the flag on, u* converges at k+2 while u_h keeps
    // its optimal k+1.
    //
    // Measured here with a linear, constant-coefficient flux and a source that
    // does not depend on u. Recording what this actually shows, because it is not
    // what a first reading of the papers predicts:
    //
    //     k = 1:  flag off  u 1.96  u* 2.19   |   flag on  u 1.96  u* 3.05
    //     k = 2:  flag off  u 2.97  u* 4.08   |   flag on  u 2.97  u* 4.03
    //
    // So at k = 1 the interpolatory scheme really has lost the superconvergence
    // (u* is no better than u_h) and the flag restores it. At k = 2 it had not
    // lost it, and the flag preserves it. The reason the loss is not universal
    // here is that interpolating a *known* smooth source at the Chebyshev nodes
    // leaves an error that is very nearly L2-orthogonal to P_k, so it does not
    // pollute the duality argument the way the papers' I_h F(u_h) does -- there
    // the nonlinearity is evaluated at u_h, which itself carries O(h^(k+1))
    // error. The reaction-term case below is the one that isolates that
    // mechanism.
    //
    // Hence the assertions: u* must reach k+2 with the flag on, and u_h must not
    // regress. Nothing is asserted about the flag improving on the flag-off rate,
    // because for this problem there is not always anything to improve.
    const double tFinal = 0.25;

    for (auto const &c : std::vector<std::pair<Index, std::vector<Index>>>{
             {1, {4, 8, 16, 32}}, {2, {4, 8, 16}}})
    {
        const Rates r = measureRates<ManufacturedDiffusion>(c.first, c.second, tFinal);
        BOOST_TEST_MESSAGE(report(c.first, r));

        BOOST_TEST(r.uOff > c.first + 1 - 0.2,
                   "k = " << c.first << ": u lost its rate with the flag off ("
                          << r.uOff << ")");
        BOOST_TEST(r.uOn > c.first + 1 - 0.2,
                   "k = " << c.first << ": u lost its rate with the flag on ("
                          << r.uOn << ")");
        BOOST_TEST(r.starOn > c.first + 2 - 0.35,
                   "k = " << c.first << ": u* did not reach k+2 = " << c.first + 2
                          << " with the flag on (observed " << r.starOn << ")");
        BOOST_TEST(r.starOn > r.uOn + 0.5,
                   "k = " << c.first
                          << ": u* is no better than u with the flag on (u "
                          << r.uOn << ", u* " << r.starOn << ")");
    }
}

BOOST_AUTO_TEST_CASE(the_flag_restores_superconvergence_for_a_nonlinear_reaction)
{
    // The case the papers are actually about. F(u) = u^3 - u is evaluated at u_h
    // by the existing scheme and at u* by the new one, and that is the difference
    // between losing and keeping the extra order.
    //
    // This is also the strongest available check on the Jacobian chain rule: the
    // source's u-dependence makes B12 contribute to the u column and B11 to the q
    // column, and a wrong chain shows up as IDA failing to converge or as a rate
    // that never reaches k+2.
    const double tFinal = 0.25;

    for (auto const &c : std::vector<std::pair<Index, std::vector<Index>>>{
             {1, {4, 8, 16, 32}}, {2, {4, 8, 16}}})
    {
        const Rates r = measureRates<ManufacturedReaction>(c.first, c.second, tFinal);
        BOOST_TEST_MESSAGE("nonlinear reaction, " + report(c.first, r));

        BOOST_TEST(r.uOn > c.first + 1 - 0.2,
                   "k = " << c.first << ": u lost its rate with the flag on ("
                          << r.uOn << ")");
        BOOST_TEST(r.starOn > c.first + 2 - 0.35,
                   "k = " << c.first << ": u* did not reach k+2 = " << c.first + 2
                          << " with the flag on (observed " << r.starOn << ")");
        BOOST_TEST(r.starOn > r.uOn + 0.5,
                   "k = " << c.first
                          << ": u* is no better than u with the flag on (u "
                          << r.uOn << ", u* " << r.starOn << ")");
    }
}

BOOST_AUTO_TEST_CASE(the_flag_is_rejected_at_degree_zero)
{
    // The reconstruction needs a degree-0 NodalBasis it can evaluate off-node,
    // and there is not one (Basis.hpp:369-377). Better to say so than to produce
    // a silently non-superconvergent run.
    Grid grid(0.0, 1.0, 4);
    ManufacturedDiffusion problem;
    SystemSolver sys(grid, 0, &problem);
    sys.setTau(1.0);
    sys.setSuperconvergent(true);
    BOOST_CHECK_THROW(sys.initialiseMatrices(), std::invalid_argument);
}

// ------------------------------------------------- the built-in MMS option --

BOOST_AUTO_TEST_CASE(the_linear_diffusion_mms_is_inconsistent_with_its_boundaries)
{
    // `LinearDiffusion` and `LinearDiffSourceTest` both accept a `UseMMS`
    // option, and neither is exercised by any regression case (the two configs
    // that mention it set it to false). Recording what is actually there:
    //
    //  * `LinearDiffusion::MMS_Solution` is (1 + growth tanh(rate t)) times the
    //    initial Gaussian, while LowerBoundary/UpperBoundary return 0. The
    //    manufactured solution therefore does not satisfy the boundary
    //    conditions unless the Gaussian is negligible at the domain edges --
    //    with the defaults (Centre = 0.5, InitialWidth = 0.2 on [0,1]) it is
    //    about 0.29 there, so an order-of-accuracy study against it would not
    //    show k+1.
    //
    //  * `LinearDiffSourceTest` reads `useMMS` from its config but never adds
    //    `MMS_Source` to `Source()` -- only `LinearDiffusion` does. Setting
    //    useMMS on that case is silently a no-op beyond adding a netCDF group.
    //
    // Neither is changed here: they are someone's physics options, and the
    // convergence testing above does not need them. This case exists so the
    // situation is written down and checked rather than rediscovered.
    const double centre = 0.5, initialWidth = 0.2, height = 1.0;
    const double alpha = 1.0 / initialWidth;

    auto gaussian = [&](double x)
    { return height * std::exp(-alpha * (x - centre) * (x - centre)); };

    BOOST_TEST(gaussian(0.0) > 0.25,
               "the default LinearDiffusion MMS profile is not small at x = 0 ("
                   << gaussian(0.0) << "), so it cannot match a zero Dirichlet "
                                       "boundary condition");
    BOOST_TEST(gaussian(1.0) > 0.25);

    // Narrowing it does make the mismatch negligible, which is how the option
    // could be used for a convergence study if anyone wants to.
    const double narrow = 0.01;
    BOOST_TEST(height * std::exp(-(0.25) / narrow) < 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()
