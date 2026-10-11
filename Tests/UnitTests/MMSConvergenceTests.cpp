// Order-of-accuracy tests by the method of manufactured solutions.
//
// This is the strongest correctness signal available for the solver as a whole:
// it exercises the residual, the block assembly, the static-condensation solve
// and IDA's time integration together, and it fails if any of them is even
// slightly wrong in a way the unit tests cannot express. A wrong sign in a
// single block still converges -- to the wrong answer -- but it does not
// converge at the right *rate* to the right limit.
//
// The manufactured solution, the sweep and the least-squares fit are shared with
// the aux/scalar studies and live in MMSHarness.hpp. This file holds the
// problems with no aux variables and no global scalars; MMSAuxScalarTests.cpp
// holds the coupled ones.
//
// Substituting u = sin(pi x)(1 + t) into d_t u = d_x( kappa d_x u ) + S gives
//
//     S(x, t) = sin(pi x) * ( 1 + kappa pi^2 (1 + t) )
//
// The expected L2 rate for HDG with a degree-k basis is k+1.

#include <boost/test/unit_test.hpp>

#include "MMSHarness.hpp"

#include <cmath>
#include <format>
#include <string>
#include <utility>
#include <vector>

namespace
{

using namespace mms;

constexpr double KAPPA = 0.7;

class ManufacturedDiffusion : public TransportSystem
{
public:
    // u = 0 at both ends for all t, matching the manufactured solution.
    ManufacturedDiffusion() : TransportSystem({.variables = numberedFields(1)}) {}

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return KAPPA * s.q(0);
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
    ManufacturedReaction() : TransportSystem({.variables = numberedFields(1)}) {}

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    static double F(double u) { return u * u * u - u; }
    static double dF(double u) { return 3.0 * u * u - 1.0; }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return s.q(0);
    }

    Value Sources(Index, const State &s, Position x, Time t) override
    {
        const double ue = exactSolution(x, t);
        const double f = std::sin(pi * x) * (1.0 + pi * pi * (1.0 + t)) + F(ue);
        return f - F(s.u(0));
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
        v[0] = -dF(s.u(0));
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

// A genuinely nonlinear *flux*, sigma_hat = (1 + u^2) q.
//
// This is the case the theory does not cover. Paper I treats -Laplacian u + F(u)
// and names F(grad u, u) as open in its conclusion, so the k+2 rate here is a
// measurement rather than a confirmation. It is also the configuration that puts
// *all* of the u-dependence in the flux: the source below is a function of x and
// t alone, so every dSources_* is zero and the only route from u to the residual
// is through sigma_hat.
//
// With A = 1 + t, s = sin(pi x), c = cos(pi x), the exact solution u = A s has
//
//     d_x[ (1 + u^2) u_x ] = -A pi^2 s (1 + A^2 s^2) + 2 A^3 pi^2 s c^2
//
// and MaNTA integrates u_t - d_x[sigma_hat] = S, so
//
//     S = s + A pi^2 s (1 + A^2 s^2) - 2 A^3 pi^2 s c^2
//
// Note the sign: the stored sigma is -sigma_hat, and a source derived from
// u_t + d_x[sigma_hat] would give an anti-diffusion equation that still
// converges, at the right rate, to the wrong function.
class ManufacturedNonlinearFlux : public TransportSystem
{
public:
    ManufacturedNonlinearFlux() : TransportSystem({.variables = numberedFields(1)}) {}

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index, const State &st, Position, Time) override
    {
        return (1.0 + st.u(0) * st.u(0)) * st.q(0);
    }

    Value Sources(Index, const State &, Position x, Time t) override
    {
        return nonlinearFluxSource(x, t);
    }

    void dSigmaFn_dq(Index, VectorRef v, const State &st, Position, Time) override
    {
        v[0] = 1.0 + st.u(0) * st.u(0);
    }
    void dSigmaFn_du(Index, VectorRef v, const State &st, Position, Time) override
    {
        v[0] = 2.0 * st.u(0) * st.q(0);
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

/// The u-only form the existing cases below are written against.
double solveAndMeasure(Index k, Index nCells, double tFinal)
{
    return solveAndMeasureBoth<ManufacturedDiffusion>(k, nCells, tFinal).u;
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
        s.u(0) = exactSolution(x, t);
        s.q(0) = exactDerivative(x, t);

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

// ------------------------------------------------------- nonlinear flux --

BOOST_AUTO_TEST_CASE(the_nonlinear_flux_source_is_consistent_with_the_exact_solution)
{
    // Same guard as for the diffusion case, against the harder algebra:
    // u_t - d_x[ (1 + u^2) u_x ] must equal S at the exact solution. Getting the
    // sign of the divergence term wrong here produces an anti-diffusion problem
    // that still converges at k+1, to the wrong function.
    ManufacturedNonlinearFlux problem;
    const double t = 0.37, h = 1e-5;

    auto flux = [&](double x)
    {
        const double u = exactSolution(x, t);
        return (1.0 + u * u) * exactDerivative(x, t);
    };

    for (double x : {0.13, 0.5, 0.81})
    {
        const double dFluxdx = (flux(x + h) - flux(x - h)) / (2.0 * h);
        const double dudt = (exactSolution(x, t + h) - exactSolution(x, t - h)) / (2.0 * h);

        State s(1);
        s.u(0) = exactSolution(x, t);
        s.q(0) = exactDerivative(x, t);

        const double S = problem.Sources(0, s, x, t);
        BOOST_TEST(dudt - dFluxdx == S, boost::test_tools::tolerance(1e-5));

        // ...and that the flux hook really is the function the source was
        // derived for, rather than the source matching a different sigma_hat.
        BOOST_TEST(problem.SigmaFn(0, s, x, t) == flux(x),
                   boost::test_tools::tolerance(1e-12));
    }
}

BOOST_AUTO_TEST_CASE(the_flux_derivatives_agree_with_a_finite_difference)
{
    // dSigmaFn_du = 2 u q and dSigmaFn_dq = 1 + u^2 are the only route from u to
    // the residual in this problem, so a mistake in either is a mistake in the
    // whole Jacobian. Central-difference the case's own SigmaFn against them --
    // no solver involved, so this fails before the order study does and says
    // something much more specific when it does.
    ManufacturedNonlinearFlux problem;
    const double t = 0.37, h = 1e-6;

    for (double x : {0.13, 0.5, 0.81})
    {
        const double u = exactSolution(x, t), q = exactDerivative(x, t);

        auto sigmaAt = [&](double uu, double qq)
        {
            State s(1);
            s.u(0) = uu;
            s.q(0) = qq;
            return problem.SigmaFn(0, s, x, t);
        };

        State s(1);
        s.u(0) = u;
        s.q(0) = q;

        Vector du(1), dq(1);
        du.setZero();
        dq.setZero();
        problem.dSigmaFn_du(0, du, s, x, t);
        problem.dSigmaFn_dq(0, dq, s, x, t);

        BOOST_TEST(du[0] == (sigmaAt(u + h, q) - sigmaAt(u - h, q)) / (2.0 * h),
                   boost::test_tools::tolerance(1e-6));
        BOOST_TEST(dq[0] == (sigmaAt(u, q + h) - sigmaAt(u, q - h)) / (2.0 * h),
                   boost::test_tools::tolerance(1e-6));
    }
}

BOOST_AUTO_TEST_CASE(the_order_survives_a_nonlinear_flux)
{
    // sigma_hat(u, q) is outside the papers' theory -- their conclusion names
    // F(grad u, u) as open -- so this study is a measurement, not a check of a
    // predicted rate. SolveJacTests already shows the flag-on Jacobian is right
    // for such a flux; what was missing was whether u* still gains an order.
    //
    // Measured, on the sweep below:
    //
    //     k = 1:  flag off  u 1.96  u* 2.81   |   flag on  u 1.92  u* 3.08
    //     k = 2:  flag off  u 2.94  u* 4.20   |   flag on  u 2.92  u* 4.42
    //
    // and this is the one case in the file where the flag is doing something the
    // flag-off path does not eventually do by itself. Read the per-n errors, not
    // the fitted slope: at k = 1 with the flag *off*, u* falls
    //
    //     4.55e-2  6.58e-3  5.60e-4  6.18e-5  2.74e-5
    //
    // -- ratios 6.9, 11.7, 9.1, and then 2.3. It superconverges over the coarse
    // grids and then stops, so the 2.81 above is a fit through a rate that is
    // still falling; a sweep stopping at n = 32 reports 3.21 and looks
    // superconvergent. With the flag *on* the same column falls by 8.5, 8.7, 8.4,
    // 8.2 -- 2^3 every time, k+2 and asymptotic.
    //
    // That is the result. For a flux the theory does not cover, the interpolatory
    // scheme's postprocessing gain is real but transient without the flag, and
    // durable with it. It is also why this sweep runs to n = 64 rather than
    // stopping at 32 like its neighbours: at 32 the two columns are
    // indistinguishable and the case would assert nothing the linear one does not.
    const double tFinal = 0.25;

    for (auto const &c : std::vector<std::pair<Index, std::vector<Index>>>{
             {1, {4, 8, 16, 32, 64}}, {2, {4, 8, 16, 32}}})
    {
        const Rates r = measureRates<ManufacturedNonlinearFlux>(c.first, c.second, tFinal);
        BOOST_TEST_MESSAGE("nonlinear flux, " + report(c.first, r));

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

BOOST_AUTO_TEST_CASE(the_flag_off_superconvergence_at_k2_is_genuine_not_pre_asymptotic)
{
    // Settles a specific doubt about the k = 1 / k = 2 split recorded above.
    //
    // The nonlinear-flux case shows that flag-off postprocessing can superconverge
    // over coarse grids and then stop, so the obvious suspicion is that the linear
    // k = 2 flag-off entry (u* = 4.08 over n = 4, 8, 16) is the same transient
    // caught before it breaks, and that the anomaly is an artefact of where each
    // sweep happens to end. Refining to n = 64 says otherwise:
    //
    //     n=4    u* 2.101e-04
    //     n=8    u* 1.228e-05    local order 4.10
    //     n=16   u* 7.387e-07    local order 4.05
    //     n=32   u* 4.516e-08    local order 4.03
    //     n=64   u* 2.739e-09    local order 4.04
    //
    // Four consecutive refinements at k+2, with no sign of the decay the nonlinear
    // flux shows by its third. So the linear k = 2 flag-off superconvergence is
    // real and durable, the anomaly is not a pre-asymptotic artefact, and the two
    // phenomena are distinct: the interpolatory scheme keeps the extra order here
    // and loses it for a nonlinear flux.
    //
    // The last point is worth 2.7e-9, close enough to the 1e-9 relative tolerance
    // to ask whether it is measuring space at all. The control below says it is:
    // loosening the tolerance tenfold moves u* by about 1%, where a
    // tolerance-limited error would move by ten.
    const double tFinal = 0.25;
    const std::vector<Index> cells = {4, 8, 16, 32, 64};

    const Rates r = measureRates<ManufacturedDiffusion>(2, cells, tFinal);
    BOOST_TEST_MESSAGE("refined linear k=2, " + report(2, r));

    // The substantive claim: every step of the sweep holds k+2, flag off. A
    // single fitted slope would average a breakdown away, which is exactly the
    // failure this test exists to detect.
    for (size_t i = 1; i < cells.size(); ++i)
    {
        BOOST_TEST(r.localStarOff(i) > 4.0 - 0.35,
                   "flag-off u* lost k+2 between n=" << cells[i - 1] << " and n="
                                                     << cells[i] << " (local order "
                                                     << r.localStarOff(i) << ")");
        BOOST_TEST(r.localStarOn(i) > 4.0 - 0.35,
                   "flag-on u* lost k+2 between n=" << cells[i - 1] << " and n="
                                                    << cells[i] << " (local order "
                                                    << r.localStarOn(i) << ")");
    }

    // The control that makes the above mean anything: if the finest error were
    // set by the time integration rather than by h, changing the tolerance would
    // move it in proportion.
    const Errors dflt =
        solveAndMeasureBoth<ManufacturedDiffusion>(2, 64, tFinal, false, {});
    const Errors loose =
        solveAndMeasureBoth<ManufacturedDiffusion>(2, 64, tFinal, false, {1e-10, 1e-8});
    const Errors tight =
        solveAndMeasureBoth<ManufacturedDiffusion>(2, 64, tFinal, false, {3e-12, 1e-10});

    BOOST_TEST_MESSAGE(std::format(
        "  n=64 flag off u*:  default {:.6e}  looser(x10) {:.6e}  tighter {:.6e}",
        dflt.uStar, loose.uStar, tight.uStar));

    BOOST_TEST(std::abs(loose.uStar - tight.uStar) / dflt.uStar < 0.1,
               "the finest u* moves with the time tolerance ("
                   << loose.uStar << " vs " << tight.uStar
                   << "), so that point measures the integrator, not the mesh");
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

BOOST_AUTO_TEST_SUITE_END()
