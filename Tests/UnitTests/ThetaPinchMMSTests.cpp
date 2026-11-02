// Order of accuracy of the coupled theta pinch, against a closed form.
//
// The registered pair -- ThetaPinchDiffusion's flux and aux row, read through
// ThetaPinchEquilibrium's geometry -- with only the source, the wall datum and
// the initial state replaced by manufactured ones. A wrong coupled *residual*
// (a sign, a factor, a slot read in the wrong place) converges at the right rate
// to the wrong function, so no Jacobian check sees it; this does.
//
// The manufactured density n(x, t) fixes the pressure p = T n without reference
// to the geometry, so the equilibrium is closed form: force balance
// pi^2 / V'^2 + p = C gives V' = pi / sqrt(C - T n), and the wall,
// Int_0^1 V' dx = pi, fixes C(t). N = V' n is then the transport solution, and
// the source is d_t N - d_x sigma_hat on it.
//
// Two studies: the transport mesh h at one spline interval per cell, the
// model's default, so the field refines with it; and the spline's intervals per
// cell at a fixed mesh and a degree high enough that the spline is what limits
// the answer. Read LOCAL orders, as MMSFieldTests.cpp says why. A third case
// runs one refinement under both FieldSolve modes and compares their step
// counts, which is the only place a sweep that stalls IDA shows up.
#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "MMSHarness.hpp"
#include "../../PhysicsCases/ThetaPinchDiffusion.hpp"
#include "../../PhysicsCases/ThetaPinchEquilibrium.hpp"

#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/tools/roots.hpp>

#include <cmath>
#include <ida/ida.h>
#include <nvector/nvector_serial.h>
#include <algorithm>
#include <format>
#include <memory>
#include <vector>

namespace
{

using namespace toml::literals::toml_literals;

// The run tests' parameters: T = 0.3 puts the on-axis pressure at 0.56 by
// t = 0.25, so V' is far from its vacuum value, and Gamma = 2 makes the
// diffusivity read B.
const toml::value pinchConfig = u8R"(
    [ThetaPinchDiffusion]
    Temperature = 0.3
    D0 = 0.5
    Gamma = 2.0
)"_toml;
constexpr double kT = 0.3, kD0 = 0.5, kGamma = 2.0;

constexpr double kPi = M_PI;

/// The manufactured solution and everything derived from it, in normalised
/// units (a = B0 = 1, so Phi_a = pi and the wall area is pi).
class PinchExact
{
public:
    static double n(double x, double t) { return (1.0 + 0.5 * std::cos(kPi * x)) * (1.0 + t); }
    static double n_x(double x, double t) { return -0.5 * kPi * std::sin(kPi * x) * (1.0 + t); }
    static double n_xx(double x, double t) { return -0.5 * kPi * kPi * std::cos(kPi * x) * (1.0 + t); }
    static double n_t(double x, double) { return 1.0 + 0.5 * std::cos(kPi * x); }

    /// C(t), from Int_0^1 1 / sqrt(C - T n) dx = 1.
    double C(double t) const
    {
        refresh(t);
        return cachedC;
    }

    /// dC/dt, from differentiating the wall constraint: Int V'_t dx = 0.
    double Cdot(double t) const
    {
        refresh(t);
        return cachedCdot;
    }

    double Vp(double x, double t) const { return kPi / std::sqrt(C(t) - kT * n(x, t)); }
    double Vp_x(double x, double t) const
    {
        return 0.5 * kPi * kT * n_x(x, t) * std::pow(C(t) - kT * n(x, t), -1.5);
    }
    double Vp_t(double x, double t) const
    {
        return -0.5 * kPi * (Cdot(t) - kT * n_t(x, t)) * std::pow(C(t) - kT * n(x, t), -1.5);
    }
    /// The area enclosed by surface x, Int_0^x V'.
    double A(double x, double t) const
    {
        if (x <= 0.0)
            return 0.0;
        return GK::integrate([&](double s) { return Vp(s, t); }, 0.0, x, 5, 1e-15);
    }

    double N(double x, double t) const { return Vp(x, t) * n(x, t); }
    double N_x(double x, double t) const { return Vp_x(x, t) * n(x, t) + Vp(x, t) * n_x(x, t); }
    double N_t(double x, double t) const { return Vp_t(x, t) * n(x, t) + Vp(x, t) * n_t(x, t); }

    /// sigma_hat = V' D <|grad x|^2> d_x n with D = D0 (V' / pi)^Gamma and
    /// <|grad x|^2> = 4 pi A / V'^2, i.e. 4 D0 pi^(1 - Gamma) A n_x V'^(Gamma - 1).
    double flux(double x, double t) const
    {
        return 4.0 * kD0 * std::pow(kPi, 1.0 - kGamma) * A(x, t) * n_x(x, t) *
               std::pow(Vp(x, t), kGamma - 1.0);
    }
    double dflux_dx(double x, double t) const
    {
        const double V = Vp(x, t), a = A(x, t);
        return 4.0 * kD0 * std::pow(kPi, 1.0 - kGamma) *
               (std::pow(V, kGamma) * n_x(x, t) + a * n_xx(x, t) * std::pow(V, kGamma - 1.0) +
                (kGamma - 1.0) * a * n_x(x, t) * std::pow(V, kGamma - 2.0) * Vp_x(x, t));
    }

    /// d_t N - d_x sigma_hat: MaNTA integrates a u_t - d_x sigma_hat = S.
    double source(double x, double t) const { return N_t(x, t) - dflux_dx(x, t); }

    /// The four geometry slots at the exact solution, in the model's order.
    Vector geometry(double x, double t) const
    {
        Vector g(4);
        g(ThetaPinchEquilibrium::Vprime) = Vp(x, t);
        g(ThetaPinchEquilibrium::dVprime_dx) = Vp_x(x, t);
        g(ThetaPinchEquilibrium::gradx2) = 4.0 * kPi * A(x, t) / (Vp(x, t) * Vp(x, t));
        g(ThetaPinchEquilibrium::Bfield) = kPi / Vp(x, t);
        return g;
    }

private:
    using GK = boost::math::quadrature::gauss_kronrod<double, 61>;

    void refresh(double t) const
    {
        if (haveCache && t == cachedT)
            return;
        auto constraint = [&](double C)
        {
            return GK::integrate([&](double x) { return 1.0 / std::sqrt(C - kT * n(x, t)); }, 0.0, 1.0, 5,
                                 1e-15) -
                   1.0;
        };
        // C - T n > 0 everywhere, and n peaks on axis at 1.5 (1 + t).
        const double lo = kT * n(0.0, t) * (1.0 + 1e-12);
        boost::math::tools::eps_tolerance<double> tol(52);
        std::uintmax_t iters = 200;
        const auto bracket = boost::math::tools::toms748_solve(constraint, lo, 10.0, tol, iters);
        cachedC = 0.5 * (bracket.first + bracket.second);

        const double C = cachedC;
        auto w = [&](double x) { return std::pow(C - kT * n(x, t), -1.5); };
        const double num = GK::integrate([&](double x) { return w(x) * n_t(x, t); }, 0.0, 1.0, 5, 1e-15);
        const double den = GK::integrate(w, 0.0, 1.0, 5, 1e-15);
        cachedCdot = kT * num / den;
        cachedT = t;
        haveCache = true;
    }

    mutable bool haveCache = false;
    mutable double cachedT = 0.0, cachedC = 0.0, cachedCdot = 0.0;
};

/// The registered transport case with a manufactured source, wall datum and
/// initial state. Its flux, aux row and parameters are ThetaPinchDiffusion's own.
class ManufacturedThetaPinch : public ThetaPinchDiffusion
{
public:
    ManufacturedThetaPinch(PinchExact const &e, Grid const &grid)
        : ThetaPinchDiffusion(pinchConfig, grid), exact(e)
    {
    }

    Value UpperBoundary(Index, Time t) const override { return exact.N(1.0, t); }

    Real Source(Index, RealVector, RealVector, RealVector, RealVector, RealVector, RealVector, Real x,
                Time t) override
    {
        return exact.source(x.val, t);
    }

    autodiff::dual2nd InitialFunction(Index, autodiff::dual2nd x, autodiff::dual2nd t) const override
    {
        const double C = exact.C(t.val.val);
        const autodiff::dual2nd n = (1.0 + 0.5 * cos(kPi * x)) * (1.0 + t);
        return kPi / sqrt(C - kT * n) * n;
    }

    Value InitialAuxValue(Index, Position x, Time t) const override { return kT * PinchExact::n(x, t); }

private:
    PinchExact const &exact;
};

struct Errors
{
    double N, Vp, C;
    long steps = 0, solves = 0, fallbacks = 0;
};

constexpr double tFinal = 0.25;

/// One coupled run to tFinal, and its errors in N, in V' and in C.
Errors solveOnce(Index k, Index nCells, Index intervalsPerCell = 1,
                 SystemSolver::FieldSolveMode mode = SystemSolver::FieldSolveMode::Iterative)
{
    const PinchExact exact;
    Grid grid(0.0, 1.0, nCells);
    ManufacturedThetaPinch problem(exact, grid);
    auto field = std::make_shared<ThetaPinchEquilibrium>(
        ThetaPinchEquilibrium::Parameters{.intervalsPerCell = intervalsPerCell}, grid);

    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.setFieldModel(field);
    sys.setFieldSolveMode(mode);
    sys.resetCoeffs();
    sys.setInputFile("theta_pinch_mms");
    sys.setOutputCadence(tFinal);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-14);
    const mms::Tolerances tol;
    sys.setTolerances({tol.absolute}, tol.relative);
    sys.setWriteOutput(false);
    sys.setWriteDatFile(false);
    long steps = 0;
    {
        CapturedOutput quiet;
        sys.initialize();
        sys.integrate(tFinal);
        IDAGetNumSteps(sys.IDA_mem, &steps);
        sys.destroySundials();
    }

    const Vector psi = sys.getSolution().getField();
    const double Nerr = mms::l2ErrorAgainst([&](double x) { return sys.yJac.u(0)(x); },
                                            [&](double x, double t) { return exact.N(x, t); }, grid, tFinal);
    Vector g(4);
    const double Vperr = mms::l2ErrorAgainst(
        [&](double x)
        {
            g.setZero();
            field->Geometry(g, psi, x, tFinal);
            return g(ThetaPinchEquilibrium::Vprime);
        },
        [&](double x, double t) { return exact.Vp(x, t); }, grid, tFinal);
    const auto stats = sys.getFieldSweepStats();
    return {Nerr, Vperr, std::abs(psi(psi.size() - 1) - exact.C(tFinal)), steps, stats.solves, stats.fallbacks};
}

std::string table(std::vector<Index> const &res, std::vector<Errors> const &e, const char *name)
{
    std::string s;
    for (size_t i = 0; i < res.size(); ++i)
    {
        s += std::format("\n      {}={:<4} N {:.3e}  V' {:.3e}  C {:.3e}   ({} steps, {} field solves, {} fallbacks)",
                         name, res[i], e[i].N, e[i].Vp, e[i].C, e[i].steps, e[i].solves, e[i].fallbacks);
        if (i > 0)
            s += std::format("\n            local order: N {:.2f}  V' {:.2f}  C {:.2f}",
                             mms::localOrder(res[i - 1], res[i], e[i - 1].N, e[i].N),
                             mms::localOrder(res[i - 1], res[i], e[i - 1].Vp, e[i].Vp),
                             mms::localOrder(res[i - 1], res[i], e[i - 1].C, e[i].C));
    }
    return s;
}

} // namespace

BOOST_AUTO_TEST_SUITE(theta_pinch_mms_tests)

// The algebra before the solver: the case's own SigmaFn, evaluated on the exact
// state and the exact geometry, is the flux the source was derived from, and
// N_t - d_x sigma_hat - S vanishes with its x derivative taken by a six-point
// stencil of that SigmaFn. If either failed the studies would still converge,
// to a different function.
BOOST_AUTO_TEST_CASE(the_manufactured_source_is_the_cases_own_pde)
{
    const PinchExact exact;
    Grid grid(0.0, 1.0, 4);
    ManufacturedThetaPinch problem(exact, grid);
    TransportSystem &ts = problem;

    auto caseFlux = [&](double x, double t)
    {
        State s(1, 0, 1, 4);
        s.u(0) = exact.N(x, t);
        s.q(0) = exact.N_x(x, t);
        s.phi(0) = kT * PinchExact::n(x, t);
        const Vector g = exact.geometry(x, t);
        for (Index i = 0; i < 4; ++i)
            s.geom(i) = g(i);
        return ts.SigmaFn(0, s, x, t);
    };

    double worstFlux = 0.0, worstPDE = 0.0, worstAux = 0.0, scale = 0.0;
    for (double t : {0.0, 0.1, 0.25})
        for (double x : {0.07, 0.2, 0.37, 0.5, 0.63, 0.81, 0.94})
        {
            worstFlux = std::max(worstFlux, std::abs(caseFlux(x, t) - exact.flux(x, t)));

            constexpr double h = 2e-3;
            const double dFdx = (-caseFlux(x - 3 * h, t) + 9 * caseFlux(x - 2 * h, t) - 45 * caseFlux(x - h, t) +
                                 45 * caseFlux(x + h, t) - 9 * caseFlux(x + 2 * h, t) + caseFlux(x + 3 * h, t)) /
                                (60 * h);
            constexpr double ht = 1e-4;
            const double dNdt = (exact.N(x, t + ht) - exact.N(x, t - ht)) / (2 * ht);
            worstPDE = std::max(worstPDE, std::abs(dNdt - dFdx - exact.source(x, t)));
            scale = std::max(scale, std::abs(exact.source(x, t)));

            // ...and the aux row the field reads is p = T n at the exact state.
            State s(1, 0, 1, 4);
            s.u(0) = exact.N(x, t);
            s.phi(0) = kT * PinchExact::n(x, t);
            const Vector g = exact.geometry(x, t);
            for (Index i = 0; i < 4; ++i)
                s.geom(i) = g(i);
            worstAux = std::max(worstAux, std::abs(ts.AuxG(0, s, x, t)));
        }

    BOOST_TEST_MESSAGE(std::format("worst |sigma_case - sigma_exact| {:.2e}, |N_t - d_x sigma - S| {:.2e} "
                                   "against |S| {:.2e}, |AuxG| {:.2e}",
                                   worstFlux, worstPDE, scale, worstAux));
    BOOST_TEST(scale > 0.1);
    BOOST_TEST(worstFlux < 1e-12);
    BOOST_TEST(worstPDE < 1e-8 * scale);
    BOOST_TEST(worstAux < 1e-13);

    // The equilibrium: force balance and the wall, at the exact solution.
    using GK = boost::math::quadrature::gauss_kronrod<double, 61>;
    for (double t : {0.0, 0.25})
    {
        const double wall = GK::integrate([&](double x) { return exact.Vp(x, t); }, 0.0, 1.0, 5, 1e-15);
        BOOST_TEST(std::abs(wall - kPi) < 1e-13);
        BOOST_TEST(std::abs(exact.A(1.0, t) - kPi) < 1e-13);
    }
}

// The run that found the sweep's stopping test wanting. Under a relative
// 2-norm test on psi alone, this case at 1e-9 / 1e-11 held IDA at order one
// with h ~ 2e-5 for more than 20000 steps, against the exact solve's 213: the
// test left an error in sigma and q near the axis that was negligible in the
// 2-norm and O(1) in IDA's error weights. The stop is now relative in those
// weights (solveCoupledJacIterative), and the two modes take comparable steps.
BOOST_AUTO_TEST_CASE(the_iterative_solve_does_not_stall_ida)
{
    const Errors iterative = solveOnce(3, 8, 1, SystemSolver::FieldSolveMode::Iterative);
    const Errors exact = solveOnce(3, 8, 1, SystemSolver::FieldSolveMode::Exact);
    BOOST_TEST_MESSAGE(std::format("k = 3, 8 cells: iterative {} steps, N error {:.6e}; exact {} steps, "
                                   "N error {:.6e}",
                                   iterative.steps, iterative.N, exact.steps, exact.N));
    BOOST_TEST(iterative.fallbacks == 0);
    // Step sequences at these tolerances move by tens of percent under any
    // perturbation of the corrections -- 201 to 426 over the variants of the
    // test that were measured -- while the stall was a factor of a hundred.
    BOOST_TEST(iterative.steps < 2 * exact.steps);
    BOOST_TEST(std::abs(iterative.N - exact.N) <= 1e-6 * exact.N);
}

// Every local order, not a fitted slope, with the measured table in the
// comment. One interval per cell, the model's default, on 8, 16 and 32 cells:
//
//     k = 1   N 1.99, 1.99    V' 2.55, 2.49
//     k = 2   N 3.01, 3.00    V' 3.59, 3.53
//     k = 3   N 3.95, 3.99    V' 3.97, 3.98
//
// and on 64 cells as well, N 2.00, 3.00, 4.03 -- left out for its cost. N goes
// at k + 1, which is the transport's own rate: the coupling costs nothing in
// order. V' is a smooth function of p_h smoothed by a C^2 spline, and goes at
// k + 1 or better until the spline's own 4 caps it. C, a single integral
// constraint, converges too but not at a rate clean enough to assert (1.4 to
// 4.1 over the same refinements); it is printed.
BOOST_AUTO_TEST_CASE(the_coupled_theta_pinch_converges_at_k_plus_one)
{
    const std::vector<Index> cells = {8, 16, 32};
    for (Index k : {1, 2, 3})
    {
        std::vector<Errors> e;
        for (Index n : cells)
            e.push_back(solveOnce(k, n));
        BOOST_TEST_MESSAGE("k = " << k << ", one interval per cell" << table(cells, e, "n"));

        const double wantN = static_cast<double>(k + 1) - 0.1;
        const double wantVp = std::min<double>(k + 1, 4) - 0.1;
        for (size_t i = 0; i < cells.size(); ++i)
        {
            BOOST_TEST(e[i].fallbacks == 0, "k = " << k << ", n = " << cells[i] << ": " << e[i].fallbacks
                                                   << " fallbacks, so this measures the exact solve");
            if (i == 0)
                continue;
            const double oN = mms::localOrder(cells[i - 1], cells[i], e[i - 1].N, e[i].N);
            const double oVp = mms::localOrder(cells[i - 1], cells[i], e[i - 1].Vp, e[i].Vp);
            BOOST_TEST(oN >= wantN, "k = " << k << ", n = " << cells[i] << ": N at " << oN);
            BOOST_TEST(oVp >= wantVp, "k = " << k << ", n = " << cells[i] << ": V' at " << oVp);
        }
    }
}

// The field's own resolution, where it is what limits the answer: at k = 6 on
// 8 cells the transport is good to ~1e-8, and one spline interval per cell is
// not. Halving the spline's knot spacing takes V' and N down at the spline's 4
// (measured 4.00 and 4.03); further halvings are where the transport's own
// error comes back in, and are not asserted.
BOOST_AUTO_TEST_CASE(intervals_per_cell_refine_the_field_at_fourth_order)
{
    const std::vector<Index> ms = {1, 2};
    std::vector<Errors> e;
    for (Index m : ms)
        e.push_back(solveOnce(6, 8, m));
    BOOST_TEST_MESSAGE("k = 6, n = 8" << table(ms, e, "m"));
    BOOST_TEST(e[0].fallbacks == 0);
    BOOST_TEST(e[1].fallbacks == 0);
    BOOST_TEST(mms::localOrder(1, 2, e[0].Vp, e[1].Vp) >= 3.8);
    BOOST_TEST(mms::localOrder(1, 2, e[0].N, e[1].N) >= 3.8);
}

BOOST_AUTO_TEST_SUITE_END()
