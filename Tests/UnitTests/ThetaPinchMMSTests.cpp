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
//
// The mesh study runs at every field-coupling tier built (`Tier`), transient
// and -- with the solution frozen at tFinal -- steady, and holds each lower tier
// to tier 2's answer as well as to its orders.
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>

#include "CapturedOutput.hpp"
#include "MMSHarness.hpp"
#include "../../PhysicsCases/ThetaPinchDiffusion.hpp"
#include "../../PhysicsCases/ThetaPinchEquilibrium.hpp"
#include "../../FieldModelEquilibrium.hpp"
#include "../../PhysicsCases/ThetaPinchPressureBalance.hpp"

#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/tools/roots.hpp>

#include <cmath>
#include <ida/ida.h>
#include <nvector/nvector_serial.h>
#include <algorithm>
#include <map>
#include <tuple>
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

// The steady solves' stopping tolerance: far below any spatial error measured.
constexpr double steadyTolerance = 1e-8;

/// The manufactured solution and everything derived from it, in normalised
/// units (a = B0 = 1, so Phi_a = pi and the wall area is pi).
///
/// Frozen at a time, it is a steady solution instead: n is the transient one at
/// that time, n_t = 0, so C is constant and the source is -d_x sigma_hat alone.
class PinchExact
{
public:
    PinchExact() = default;
    explicit PinchExact(double frozenAt) : frozen(frozenAt) {}

    bool isSteady() const { return frozen >= 0.0; }

    double n(double x, double t) const { return (1.0 + 0.5 * std::cos(kPi * x)) * (1.0 + at(t)); }
    double n_x(double x, double t) const { return -0.5 * kPi * std::sin(kPi * x) * (1.0 + at(t)); }
    double n_xx(double x, double t) const { return -0.5 * kPi * kPi * std::cos(kPi * x) * (1.0 + at(t)); }
    double n_t(double x, double) const { return isSteady() ? 0.0 : 1.0 + 0.5 * std::cos(kPi * x); }

    /// C(t), from Int_0^1 1 / sqrt(C - T n) dx = 1.
    double C(double t) const
    {
        refresh(at(t));
        return cachedC;
    }

    /// dC/dt, from differentiating the wall constraint: Int V'_t dx = 0.
    double Cdot(double t) const
    {
        refresh(at(t));
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

    double at(double t) const { return isSteady() ? frozen : t; }

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

    double frozen = -1.0;
    mutable bool haveCache = false;
    mutable double cachedT = 0.0, cachedC = 0.0, cachedCdot = 0.0;
};

/// The registered transport case with a manufactured source, wall datum and
/// initial state. Its flux, aux row and parameters are ThetaPinchDiffusion's own.
///
/// The initial state is always the transient solution at t = 0, so a steady
/// solve towards a frozen `e` starts away from its answer and has to get there.
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
        const double C = start.C(t.val.val);
        const autodiff::dual2nd n = (1.0 + 0.5 * cos(kPi * x)) * (1.0 + t);
        return kPi / sqrt(C - kT * n) * n;
    }

    Value InitialAuxValue(Index, Position x, Time t) const override { return kT * start.n(x, t); }

private:
    PinchExact const &exact;
    const PinchExact start; // transient, whatever exact is
};

struct Errors
{
    double N, Vp, C;
    // Time steps for a transient, Newton iterations for a steady solve.
    long steps = 0, solves = 0, fallbacks = 0;
    // Tiers 0 and 1: equilibrium solves, one per distinct transport state.
    long equilibria = 0;
    // N and V' at sampleRule's points, so two runs on one grid can be compared
    // with each other and not only with the exact solution.
    Vector Nat, Vpat;
};

/// A 16-point Gauss rule in every cell: points and weights.
struct SampleRule
{
    std::vector<double> x, w;
    explicit SampleRule(Grid const &grid)
    {
        using G = boost::math::quadrature::gauss<double, 16>;
        for (size_t c = 0; c < grid.getNCells(); ++c)
        {
            const double mid = 0.5 * (grid[c].x_l + grid[c].x_u), half = 0.5 * (grid[c].x_u - grid[c].x_l);
            for (size_t q = 0; q < G::abscissa().size(); ++q)
                for (double sgn : {-1.0, 1.0})
                {
                    x.push_back(mid + sgn * half * G::abscissa()[q]);
                    w.push_back(half * G::weights()[q]);
                }
        }
    }
    /// The L2 norm of a - b over the grid.
    double distance(Vector const &a, Vector const &b) const
    {
        double total = 0.0;
        for (size_t j = 0; j < w.size(); ++j)
            total += w[j] * (a(j) - b(j)) * (a(j) - b(j));
        return std::sqrt(total);
    }
};

constexpr double tFinal = 0.25;

/// Which field-coupling tier carries the field
/// (docs/superpowers/specs/2026-10-23-field-coupling-tiers-design.md).
enum class Tier
{
    Coupled,    // tier 2: psi in the DAE, the coupled Jacobian (FieldCoupling = "coupled")
    Consistent, // tier 0: the same model as a black box, FieldModelEquilibrium ("consistent")
    Reduced,    // tier 1: the black box with its exact G' in the Jacobian ("reduced")
    ReducedDifferenced, // tier 1, G' a forward difference of the equilibrium solve
    Native,     // tier 1 on ThetaPinchPressureBalance: the balance in closed form, no spline
};

const char *tierName(Tier tier)
{
    switch (tier)
    {
    case Tier::Coupled:
        return "coupled (tier 2)";
    case Tier::Consistent:
        return "consistent (tier 0)";
    case Tier::Reduced:
        return "reduced (tier 1)";
    case Tier::ReducedDifferenced:
        return "reduced, differenced G' (tier 1)";
    case Tier::Native:
        return "native pressure balance (tier 1)";
    }
    return "?";
}

/// Attach the registered field model to `sys` the way the tier does.
void attachField(SystemSolver &sys, std::shared_ptr<ThetaPinchEquilibrium> const &field, Grid const &grid,
                 Tier tier, SystemSolver::FieldSolveMode mode)
{
    using C = SystemSolver::EquilibriumCoupling;
    switch (tier)
    {
    case Tier::Coupled:
        sys.setFieldModel(field);
        break;
    case Tier::Consistent:
        sys.setEquilibriumModel(std::make_shared<FieldModelEquilibrium>(field));
        return; // no coupling to solve, so no FieldSolve
    case Tier::Reduced:
        sys.setEquilibriumModel(std::make_shared<FieldModelEquilibrium>(field), C::Reduced);
        break;
    case Tier::ReducedDifferenced:
        sys.setEquilibriumModel(std::make_shared<FieldModelEquilibrium>(field, true), C::Reduced);
        break;
    case Tier::Native:
        sys.setEquilibriumModel(std::make_shared<ThetaPinchPressureBalance>(
                                    ThetaPinchPressureBalance::Parameters{}, grid),
                                C::Reduced);
        break;
    }
    sys.setFieldSolveMode(mode);
}

struct Run
{
    Index k = 1, nCells = 8, intervalsPerCell = 1;
    SystemSolver::FieldSolveMode mode = SystemSolver::FieldSolveMode::Iterative;
    Tier tier = Tier::Coupled;
    // A transient to tFinal against the transient solution, or a steady Newton
    // solve against that solution frozen at tFinal, from the transient's t = 0.
    bool steady = false;
};

/// One run, and its errors in N, in V' and in C.
Errors solveOnce(Run const &run)
{
    const PinchExact exact = run.steady ? PinchExact(tFinal) : PinchExact();
    Grid grid(0.0, 1.0, run.nCells);
    ManufacturedThetaPinch problem(exact, grid);
    auto field = std::make_shared<ThetaPinchEquilibrium>(
        ThetaPinchEquilibrium::Parameters{.intervalsPerCell = run.intervalsPerCell}, grid);

    SystemSolver sys(grid, run.k, &problem);
    sys.setTau(1.0);
    attachField(sys, field, grid, run.tier, run.mode);
    sys.resetCoeffs();
    sys.setInputFile("theta_pinch_mms");
    sys.setOutputCadence(tFinal);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-14);
    const mms::Tolerances tol;
    sys.setTolerances({tol.absolute}, tol.relative);
    if (run.steady)
    {
        sys.setSteadyStateTolerance(steadyTolerance);
        sys.setSteadyMode(SystemSolver::SteadyMode::Newton);
    }
    sys.setWriteOutput(false);
    sys.setWriteDatFile(false);
    long steps = 0;
    {
        CapturedOutput quiet;
        sys.initialize();
        sys.integrate(tFinal);
        if (run.steady)
            steps = sys.lastSteadyStats().newtonIters;
        else
            IDAGetNumSteps(sys.IDA_mem, &steps);
        sys.destroySundials();
    }

    // The geometry and C through the solver's own provider, so every tier is
    // measured the same way: from psi in the state at tier 2, from the
    // equilibrium the final state maps to at tier 0.
    const double C =
        run.tier == Tier::Coupled  ? sys.getSolution().getField().tail(1)(0)
        : run.tier == Tier::Native ? ThetaPinchPressureBalance::totalPressure(sys.equilibriumAt(sys.yJac, tFinal))
                                   : FieldModelEquilibrium::psi(sys.equilibriumAt(sys.yJac, tFinal)).tail(1)(0);
    const double Nerr = mms::l2ErrorAgainst([&](double x) { return sys.yJac.u(0)(x); },
                                            [&](double x, double t) { return exact.N(x, t); }, grid, tFinal);
    Vector g(4);
    const double Vperr = mms::l2ErrorAgainst(
        [&](double x)
        {
            g.setZero();
            sys.geometryAt(g, sys.yJac, x, tFinal);
            return g(ThetaPinchEquilibrium::Vprime);
        },
        [&](double x, double t) { return exact.Vp(x, t); }, grid, tFinal);
    const auto stats = sys.getFieldSweepStats();
    Errors out{Nerr, Vperr, std::abs(C - exact.C(tFinal)), steps, stats.solves, stats.fallbacks,
               sys.getEquilibriumSolves()};
    const SampleRule rule(grid);
    out.Nat.resize(static_cast<Eigen::Index>(rule.x.size()));
    out.Vpat.resize(out.Nat.size());
    for (size_t j = 0; j < rule.x.size(); ++j)
    {
        out.Nat(static_cast<Eigen::Index>(j)) = sys.yJac.u(0)(rule.x[j]);
        g.setZero();
        sys.geometryAt(g, sys.yJac, rule.x[j], tFinal);
        out.Vpat(static_cast<Eigen::Index>(j)) = g(ThetaPinchEquilibrium::Vprime);
    }
    return out;
}

Errors solveOnce(Index k, Index nCells, Index intervalsPerCell = 1,
                 SystemSolver::FieldSolveMode mode = SystemSolver::FieldSolveMode::Iterative)
{
    return solveOnce(Run{.k = k, .nCells = nCells, .intervalsPerCell = intervalsPerCell, .mode = mode});
}

std::string table(std::vector<Index> const &res, std::vector<Errors> const &e, const char *name)
{
    std::string s;
    for (size_t i = 0; i < res.size(); ++i)
    {
        s += std::format("\n      {}={:<4} N {:.3e}  V' {:.3e}  C {:.3e}   ({} steps, {} field solves, {} fallbacks, "
                         "{} equilibria)",
                         name, res[i], e[i].N, e[i].Vp, e[i].C, e[i].steps, e[i].solves, e[i].fallbacks,
                         e[i].equilibria);
        if (i > 0)
            s += std::format("\n            local order: N {:.2f}  V' {:.2f}  C {:.2f}",
                             mms::localOrder(res[i - 1], res[i], e[i - 1].N, e[i].N),
                             mms::localOrder(res[i - 1], res[i], e[i - 1].Vp, e[i].Vp),
                             mms::localOrder(res[i - 1], res[i], e[i - 1].C, e[i].C));
    }
    return s;
}

/// Every tier, tier 2 first: the steady study runs them all and compares each
/// with tier 2.
constexpr Tier builtTiers[] = {Tier::Coupled, Tier::Consistent, Tier::Reduced, Tier::ReducedDifferenced};

/// The grids each study refines over. The steady study starts at 16: at k = 3 the
/// step from 8 is pre-asymptotic in N (3.89), and a steady solve is cheap enough
/// to go one further instead.
std::vector<Index> const &studyCells(bool steady, Tier tier = Tier::Coupled)
{
    static const std::vector<Index> transient = {8, 16, 32}, steadyCells = {16, 32, 64};
    // The native model one level finer: its gradient recovery reaches five cells
    // at k = 3, so the coarsest step is still pre-asymptotic there (3.89 in N,
    // 3.80 in V' from 16 to 32 cells, transient).
    static const std::vector<Index> nativeTransient = {16, 32, 64}, nativeSteady = {32, 64, 128};
    if (tier == Tier::Native)
        return steady ? nativeSteady : nativeTransient;
    return steady ? steadyCells : transient;
}

/// One study -- a tier, transient or steady, at degree k, over studyCells -- run
/// once and kept, so a lower tier is compared with the tier-2 sweep its own test
/// already ran rather than paying for it again.
std::vector<Errors> const &study(Tier tier, bool steady, Index k,
                                 std::vector<Index> const &cells)
{
    static std::map<std::tuple<Tier, bool, Index, std::vector<Index>>, std::vector<Errors>> cache;
    const auto key = std::make_tuple(tier, steady, k, cells);
    auto it = cache.find(key);
    if (it == cache.end())
    {
        std::vector<Errors> e;
        for (Index n : cells)
            e.push_back(solveOnce(Run{.k = k, .nCells = n, .tier = tier, .steady = steady}));
        it = cache.emplace(key, std::move(e)).first;
    }
    return it->second;
}

std::vector<Errors> const &study(Tier tier, bool steady, Index k)
{
    return study(tier, steady, k, studyCells(steady, tier));
}

/// Every local order of N and V': N at the transport's k + 1, V' at k + 1 or the
/// spline's 4, whichever is lower. And no tier-2 sweep fell back, so the
/// iterative solve is what was measured.
void checkOrders(Tier tier, bool steady, Index k)
{
    const auto &cells = studyCells(steady, tier);
    const auto &e = study(tier, steady, k);
    BOOST_TEST_MESSAGE(tierName(tier) << (steady ? ", steady" : ", transient") << ", k = " << k
                                      << ", one interval per cell" << table(cells, e, "n"));

    const double wantN = static_cast<double>(k + 1) - 0.1;
    const double wantVp = std::min<double>(k + 1, 4) - 0.1;
    for (size_t i = 0; i < cells.size(); ++i)
    {
        BOOST_TEST(e[i].fallbacks == 0, tierName(tier) << ", k = " << k << ", n = " << cells[i] << ": "
                                                       << e[i].fallbacks
                                                       << " fallbacks, so this measures the exact solve");
        if (i == 0)
            continue;
        const double oN = mms::localOrder(cells[i - 1], cells[i], e[i - 1].N, e[i].N);
        const double oVp = mms::localOrder(cells[i - 1], cells[i], e[i - 1].Vp, e[i].Vp);
        BOOST_TEST(oN >= wantN, tierName(tier) << ", k = " << k << ", n = " << cells[i] << ": N at " << oN);
        BOOST_TEST(oVp >= wantVp, tierName(tier) << ", k = " << k << ", n = " << cells[i] << ": V' at " << oVp);
    }
}

/// A tier against tier 2 on the same grids. All solve the same discrete
/// equations -- tiers 0 and 1 differ only in carrying psi as a cache rather than
/// as unknowns, and in what of the coupling their Jacobians carry -- so their errors in N and V' must agree to the solve's own
/// accuracy, not merely converge at one rate: orders alone would pass a tier
/// that converged to a different function.
///
/// The solve's accuracy is rtol ||.|| for a steady solve, where both stop on a
/// Newton test (measured gaps 3e-13 to 2e-11 against an rtol ||N|| of 4.4e-9).
/// A transient's is ten times that. The two tiers take different steps -- 580
/// against 326 at k = 1 on 8 cells -- and IDA holds the *local* error to rtol,
/// the global one only to a small multiple of it: the gap there is 3.4e-9, 0.77
/// of rtol ||N||, and the iterative-against-exact comparison above has seen
/// compilers move such a gap sevenfold. 10 rtol ||N|| is still a tenth of the
/// smallest spatial error compared, 3.1e-7 at k = 3 on 32 cells. Tier 1, taking
/// tier 2's steps, comes within 0.04 of rtol ||N||.
void checkAgainstCoupled(Tier tier, bool steady, Index k)
{
    const auto &cells = studyCells(steady);
    const auto &e = study(tier, steady, k);
    const auto &c = study(Tier::Coupled, steady, k);
    const double t = tFinal; // a steady study's exact solution is frozen there
    const PinchExact exact = steady ? PinchExact(tFinal) : PinchExact();
    for (size_t i = 0; i < cells.size(); ++i)
    {
        const Grid grid(0.0, 1.0, cells[i]);
        const double normN = mms::l2ErrorAgainst([](double) { return 0.0; },
                                                 [&](double x, double tt) { return exact.N(x, tt); }, grid, t);
        const double normVp = mms::l2ErrorAgainst([](double) { return 0.0; },
                                                  [&](double x, double tt) { return exact.Vp(x, tt); }, grid, t);
        const double rtol = mms::Tolerances{}.relative, bound = steady ? 1.0 : 10.0;
        BOOST_TEST_MESSAGE(std::format("{} against coupled, k = {}, n = {}: N {:.2e}, V' {:.2e} (rtol ||.|| {:.2e}, "
                                       "{:.2e}; bound {}x that)",
                                       tierName(tier), k, cells[i], std::abs(e[i].N - c[i].N),
                                       std::abs(e[i].Vp - c[i].Vp), rtol * normN, rtol * normVp, bound));
        BOOST_TEST(std::abs(e[i].N - c[i].N) <= bound * rtol * normN,
                   tierName(tier) << ", k = " << k << ", n = " << cells[i] << ": N error " << e[i].N
                                  << " against coupled " << c[i].N);
        BOOST_TEST(std::abs(e[i].Vp - c[i].Vp) <= bound * rtol * normVp,
                   tierName(tier) << ", k = " << k << ", n = " << cells[i] << ": V' error " << e[i].Vp
                                  << " against coupled " << c[i].Vp);
    }
}

/// The native model against tier 2 on the native study's grids. It is a
/// different discretisation of the same balance -- the pressure's own
/// polynomial, not a spline projection -- so the two agree to their
/// discretisation errors and no better, and what is asserted is that they do:
/// the distance between their N converges at k + 1 like either error, and stays
/// within a few times tier 2's own. Measured, the distance over tier 2's N
/// error runs 0.37 (k = 1), 0.34 to 0.10 (k = 2), 1.9 to 2.1 (k = 3) transient
/// and 0.87, 0.29 to 0.16, 4.1 to 4.2 steady; at k = 3 the native error is the
/// larger, 2.5 to 5 times tier 2's, its recovery stencil being five cells wide.
void checkNativeAgainstCoupled(bool steady, Index k)
{
    const auto &cells = studyCells(steady, Tier::Native);
    const auto &e = study(Tier::Native, steady, k);
    const auto &c = study(Tier::Coupled, steady, k, cells);
    std::vector<double> gap;
    for (size_t i = 0; i < cells.size(); ++i)
    {
        const SampleRule rule(Grid(0.0, 1.0, cells[i]));
        gap.push_back(rule.distance(e[i].Nat, c[i].Nat));
        BOOST_TEST_MESSAGE(std::format("native against coupled, {}, k = {}, n = {}: |N - N_2| {:.3e}, tier 2's N "
                                       "error {:.3e} (ratio {:.2f}), the native one {:.3e}",
                                       steady ? "steady" : "transient", k, cells[i], gap.back(), c[i].N,
                                       gap.back() / c[i].N, e[i].N));
        BOOST_TEST(gap.back() <= 5.0 * c[i].N, "k = " << k << ", n = " << cells[i] << ": the native N is "
                                                       << gap.back() << " from tier 2's, whose own error is "
                                                       << c[i].N);
        if (i > 0)
        {
            // The slack checkNativeOrders allows: the coarser step is
            // pre-asymptotic at k = 3, for the gap as for the error (3.89).
            const double slack = i + 1 == cells.size() ? 0.1 : 0.25;
            const double o = mms::localOrder(cells[i - 1], cells[i], gap[i - 1], gap[i]);
            BOOST_TEST(o >= static_cast<double>(k + 1) - slack,
                       "k = " << k << ", n = " << cells[i] << ": the gap to tier 2 converges at " << o);
        }
    }
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
        s.phi(0) = kT * exact.n(x, t);
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
            s.phi(0) = kT * exact.n(x, t);
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
    const PinchExact pinch;
    const double normN = mms::l2ErrorAgainst([](double) { return 0.0; },
                                             [&](double x, double t) { return pinch.N(x, t); },
                                             Grid(0.0, 1.0, 8), tFinal);
    BOOST_TEST_MESSAGE(std::format("k = 3, 8 cells: iterative {} steps, N error {:.6e}; exact {} steps, "
                                   "N error {:.6e}; they differ by {:.2e} against rtol ||N|| {:.2e}",
                                   iterative.steps, iterative.N, exact.steps, exact.N,
                                   std::abs(iterative.N - exact.N), mms::Tolerances{}.relative * normN));
    BOOST_TEST(iterative.fallbacks == 0);
    // The bound is set by how far either mode's step count moves under
    // round-off, which is a long way, while the stall was a factor of a hundred.
    // IDA falls in and out of an order-2 trap on this case -- 120-200 steps held
    // at order 2 over t in [1e-3, 1e-2], and sometimes again over [0.05, 0.15]
    // -- in both modes, set by round-off. Measured over the tolerances scaled by
    // 0.92-1.08: exact 149 to 278 steps, iterative 142 to 374 at any
    // FieldSolveTolerance from 1e-4 to 1e-7; and on CI's AVX-512 runners
    // iterative 564 against exact 175. The algebraic unknowns in IDA's error
    // test are the cause: with SuppressAlgebraicError both modes take 99-101
    // steps at every one of those tolerances. Five times is clear of that
    // scatter and still forty times inside the stall.
    BOOST_TEST(iterative.steps < 5 * exact.steps);
    // Both runs are IDA solutions at rtol 1e-9, so they agree to the run's
    // integration error and no better, and round-off moves that as much as the
    // field solve does. Measured, against an rtol ||N|| of 4.4e-9: 3e-11 to
    // 6e-11 on g++-15 and g++-16 with three BLAS and on clang++-19 against
    // libstdc++ 16, 2e-10 to 4e-10 on CI's clang++-19 to 21 against libstdc++
    // 14; two *exact* runs on different BLAS already differ by 3e-11. The N
    // error itself is 7.6e-5, spatial, and the same in both.
    BOOST_TEST(std::abs(iterative.N - exact.N) <= mms::Tolerances{}.relative * normN);
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
    for (Index k : {1, 2, 3})
        checkOrders(Tier::Coupled, false, k);
}

// Tier 0, the same model as a black box (FieldModelEquilibrium,
// FieldCoupling = "consistent"), through the same study at the same tolerances:
// the order of the method has to survive taking psi out of the state. Measured,
// on the same grids as tier 2 above:
//
//     k = 1   N 1.99, 1.99    V' 2.55, 2.49    580, 560, 1416 steps (tier 2: 326, 311, 231)
//     k = 2   N 3.01, 3.00    V' 3.59, 3.53    410, 388, 284 steps  (178, 154, 126)
//     k = 3   N 3.95, 3.99    V' 3.97, 3.98    1439, 2234, 521 steps (193, 120, 96)
//
// -- tier 2's orders to the digit, and its errors to 6e-11 to 3.4e-9 in N
// against an rtol ||N|| of 4.4e-9: integration error, which checkAgainstCoupled
// bounds by ten times that. The cost is IDA's steps, 2 to 18 times tier 2's:
// the Jacobian carries no coupling, so Newton converges linearly in it. That is
// affordable only under the default ErrorWeights = "AlgebraicMax": under
// "Pointwise" the k = 1, 8-cell run went 455 s and then failed in IDASolve,
// where it takes 3 s now, while tier 2 runs either way. One case per degree,
// to keep each under a minute.
BOOST_DATA_TEST_CASE(the_consistent_theta_pinch_converges_at_k_plus_one,
                     boost::unit_test::data::make({1, 2, 3}), k)
{
    checkOrders(Tier::Consistent, false, k);
    checkAgainstCoupled(Tier::Consistent, false, k);
}

// Tier 1 (FieldCoupling = "reduced"), with the exact G' FieldModelEquilibrium
// supplies and with a forward difference of its Solve, under the default
// FieldSolve = "iterative". Measured, on the same grids, with IDA's steps
// against tier 2's and tier 0's:
//
//             N            V'           steps, exact G'   differenced G'   tier 2          tier 0
//     k = 1   1.99, 1.99   2.55, 2.49   249, 224, 224     246, 230, 218    326, 311, 231   580, 560, 1416
//     k = 2   3.01, 3.00   3.59, 3.53   172, 154, 122     161, 185, 122    178, 154, 126   410, 388, 284
//     k = 3   3.95, 3.99   3.97, 3.98   157, 121, 97      166, 121, 97     193, 120, 96    1439, 2234, 521
//
// -- tier 2's orders to the digit, at either response, and its N errors to
// 2e-12 - 1.9e-10 against an rtol ||N|| of 4.4e-9, an order closer than tier 0
// gets: the response restores the Newton rate the black box loses, so the
// two take tier 2's steps rather than 2 to 18 times them, and the sweep never
// fell back. FieldSolve = "krylov" gives the same table and errors, but took 45 s
// over the three degrees against the sweep's 16 s (and tier 2's 23 s under
// krylov), so it is not run here.
BOOST_DATA_TEST_CASE(the_reduced_theta_pinch_converges_at_k_plus_one,
                     boost::unit_test::data::make({1, 2, 3}), k)
{
    checkOrders(Tier::Reduced, false, k);
    checkAgainstCoupled(Tier::Reduced, false, k);
}

BOOST_DATA_TEST_CASE(the_reduced_theta_pinch_with_a_differenced_response_converges_at_k_plus_one,
                     boost::unit_test::data::make({1, 2, 3}), k)
{
    checkOrders(Tier::ReducedDifferenced, false, k);
    checkAgainstCoupled(Tier::ReducedDifferenced, false, k);
}

// The registered native model, ThetaPinchPressureBalance at tier 1: the balance
// on the transport's own pressure polynomial, with its gradient recovered from
// cell means (see its header), in place of the field model's spline. Measured,
// transient on 16, 32, 64 cells and steady on 32, 64, 128:
//
//             N transient  V' transient  N steady     V' steady
//     k = 1   2.00, 2.00   2.00, 2.00    1.99, 2.00   2.00, 2.00
//     k = 2   3.08, 3.02   3.09, 3.02    3.05, 3.01   3.05, 3.01
//     k = 3   3.89, 3.97   3.80, 3.94    3.97, 3.99   3.95, 3.99
//
// -- k + 1 everywhere, where the balance read off p_h directly went at k
// (0.99 / 2.24 / 3.11 steady), dV'/dx carrying p_h's derivative. The finest
// step is held to k + 1 - 0.1 and the coarser to k + 1 - 0.25: at k = 3 the
// 16 -> 32 step is still pre-asymptotic. checkNativeAgainstCoupled holds it to
// tier 2's answer.
void checkNativeOrders(bool steady, Index k)
{
    const auto &cells = studyCells(steady, Tier::Native);
    const auto &e = study(Tier::Native, steady, k);
    BOOST_TEST_MESSAGE(tierName(Tier::Native) << (steady ? ", steady" : ", transient") << ", k = " << k
                                              << table(cells, e, "n"));
    for (size_t i = 0; i < cells.size(); ++i)
    {
        BOOST_TEST(e[i].fallbacks == 0);
        if (i == 0)
            continue;
        const double slack = i + 1 == cells.size() ? 0.1 : 0.25;
        const double oN = mms::localOrder(cells[i - 1], cells[i], e[i - 1].N, e[i].N);
        const double oVp = mms::localOrder(cells[i - 1], cells[i], e[i - 1].Vp, e[i].Vp);
        BOOST_TEST(oN >= static_cast<double>(k + 1) - slack, "k = " << k << ", n = " << cells[i] << ": N at " << oN);
        BOOST_TEST(oVp >= std::min<double>(k + 1, 4) - slack,
                   "k = " << k << ", n = " << cells[i] << ": V' at " << oVp);
    }
}

BOOST_DATA_TEST_CASE(the_native_theta_pinch_converges_at_k_plus_one, boost::unit_test::data::make({1, 2, 3}), k)
{
    checkNativeOrders(false, k);
    checkNativeAgainstCoupled(false, k);
}

BOOST_AUTO_TEST_CASE(the_native_steady_solve_converges_at_k_plus_one)
{
    for (Index k : {1, 2, 3})
    {
        checkNativeOrders(true, k);
        checkNativeAgainstCoupled(true, k);
    }
}

// The steady solve at every tier: the source frozen at tFinal, Newton from the
// transient solution at t = 0, on 16, 32 and 64 cells. Measured, alike at every
// tier to every printed digit:
//
//     k = 1   N 2.00, 2.00    V' 2.52, 2.45
//     k = 2   N 3.01, 3.00    V' 3.70, 3.62
//     k = 3   N 3.98, 4.00    V' 3.97, 3.99
//
// with 5 Newton iterations at tier 2, 4 at tier 1 at either response and 8 to
// 14 at tier 0. Tier 1 lands on tier 2's answer to round-off, 3.6e-15 in N
// with the exact G' and 9.0e-15 with the differenced one, where tier 0 stops
// within 2e-11 of it: with G' in the Jacobian the Newton iteration is tier 2's,
// converging quadratically to the same iterate. Cheap -- a second or two per
// tier -- and tier 0 is the path `auto` sends a Differenced case's steady solve
// down.
BOOST_AUTO_TEST_CASE(steady_solves_converge_at_k_plus_one_at_every_tier)
{
    for (Tier tier : builtTiers)
        for (Index k : {1, 2, 3})
        {
            checkOrders(tier, true, k);
            if (tier != Tier::Coupled)
                checkAgainstCoupled(tier, true, k);
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
