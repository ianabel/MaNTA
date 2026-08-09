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
// converges at the wrong rate, or not at all. `LinearDiffusion` used to carry a
// built-in `UseMMS` option with exactly that problem -- its manufactured solution
// was the initial Gaussian, about 0.29 at the domain edge against a boundary
// condition of 0 -- and it has been removed rather than fixed. The manufactured
// problems in this file are self-contained and never used it.
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

// Cell-by-cell Gauss-30 quadrature of (u_h - u_exact)^2. Independent of the
// basis's own integration weights, which are part of what is under test.
double l2Error(SystemSolver &sys, Grid const &grid, double t)
{
    boost::math::quadrature::gauss<double, 30> gauss;
    double total = 0.0;
    for (size_t cell = 0; cell < grid.getNCells(); ++cell)
    {
        Interval const &I = grid[cell];
        auto integrand = [&](double x)
        {
            const double d = sys.yJac.u(0)(x) - exactSolution(x, t);
            return d * d;
        };
        total += gauss.integrate(integrand, I.x_l, I.x_u);
    }
    return std::sqrt(total);
}

// Run to tFinal on a uniform grid of nCells cells at degree k, and return the
// L2 error of the final solution.
//
// runSolver writes <stem>.nc / .dat / .restart.nc into the working directory,
// so the output name is unique per case and the files are removed afterwards.
double solveAndMeasure(Index k, Index nCells, double tFinal)
{
    Grid grid(0.0, 1.0, nCells);
    ManufacturedDiffusion problem;

    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();

    const std::string stem =
        "mms_k" + std::to_string(k) + "_n" + std::to_string(nCells);
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

    const double err = l2Error(sys, grid, tFinal);

    for (const char *suffix : {".nc", ".dat", ".restart.nc"})
        std::remove((stem + suffix).c_str());

    return err;
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

BOOST_AUTO_TEST_SUITE_END()
