// Sources that read du/dt.
//
// A source may depend on State::udot -- the time derivative of an evolved
// variable -- which is what the transport equations of Abel et al., Rep. Prog.
// Phys. 76 (2013) 116201 eq. (201) need and what MaNTA could not previously
// express. The design is
// docs/superpowers/specs/2026-09-21-time-derivative-sources-design.md.
//
// The u row of the residual becomes
//
//     B sigma + D u + E lambda - RF - Pi(S(..., udot)) + X udot
//
// so the Jacobian gains exactly one term, `- alpha * dS/d(udot)`, beside the
// mass term it sits next to. **That term carries alpha**, which is what these
// tests are shaped around:
//
//   * it is invisible in the steady Jacobian (alpha = 0) and therefore in the
//     adjoint, so a finite-difference check has to be run at alpha != 0 -- one
//     here is, and a companion case shows the alpha = 0 check passing with the
//     term deleted, so that nobody later "simplifies" the first into the second;
//   * it multiplies dY/dt, so getting it wrong costs Newton iterations and
//     nothing else, exactly as CLAUDE.md records for the rest of the Jacobian.
//     The order studies below are what catch a wrong *residual*; the
//     finite-difference test is what catches a wrong *Jacobian*; neither sees
//     the other's failure.
//
// The manufactured problem is built so that dropping the term is visible. Two
// variables solve the same equation with the same exact solution, and variable
// 1's source carries `C * udot(0)` with a constant `- C * du_exact/dt`
// subtracted. At the exact solution the two cancel exactly, so the PDE is
// unchanged and the closed form is still the answer; a solver that never filled
// udot would be integrating a source short by `C * du/dt` and would converge to
// something else entirely. `the_coupling_is_reachable` pins that this is a real
// dependence rather than an algebraic identity.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "FiniteDifferenceJacobian.hpp"
#include "MMSHarness.hpp"
#include "PhysicsCases/AutodiffTransportSystem.hpp"
#include "SystemSolver.hpp"
#include "Types.hpp"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>

#include <cmath>
#include <filesystem>
#include <numbers>
#include <string>
#include <toml.hpp>

using namespace mms;
using std::numbers::pi;

namespace
{

/// d(exact)/dt, which is what udot(0) equals at the manufactured solution.
inline double exactTimeDerivative(double x) { return std::sin(pi * x); }

/// u_t - u_xx for u = sin(pi x)(1 + t): the ordinary manufactured forcing.
inline double diffusionSource(double x, double t)
{
    return std::sin(pi * x) * (1.0 + pi * pi * (1.0 + t));
}

/*
    Two variables, each solving u_t - u_xx = S with the shared exact solution.

    Variable 1's source additionally reads du_0/dt:

        S_1 = diffusionSource(x, t) - C * du_exact/dt + C * udot(0)

    The subtraction is what keeps the exact solution exact, and the sign of the
    surviving term is what makes the coupling a real one: off the manifold the
    two do not cancel.

    C is a template parameter so that the same class supplies both the coupled
    problem and its own control at C = 0, rather than two classes that have to be
    kept the same by hand. Only variable 1 declares
    `sourceReadsTimeDerivatives`; variable 0's source reads nothing, and the
    asymmetry is deliberate -- it is what would catch the flag being ignored and
    udot filled for everything, or honoured per system rather than per variable.
*/
template <double C>
class ManufacturedUdot : public TransportSystem
{
public:
    ManufacturedUdot() : TransportSystem(buildSpec()) {}

    static SystemSpec buildSpec()
    {
        SystemSpec spec{.variables = numberedFields(2)};
        spec.variables[1].sourceReadsTimeDerivatives = true;
        return spec;
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index i, const State &s, Position, Time) override { return s.q(i); }

    Value Sources(Index i, const State &s, Position x, Time t) override
    {
        if (i == 0)
            return diffusionSource(x, t);
        return diffusionSource(x, t) - C * exactTimeDerivative(x) + C * s.udot(0);
    }

    void dSigmaFn_dq(Index i, VectorRef v, const State &, Position, Time) override
    {
        v[i] = 1.0;
    }
    void dSigmaFn_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}

    void dSources_dudot(Index i, VectorRef v, const State &, Position, Time) override
    {
        if (i == 1)
            v[0] = C;
    }

    Value InitialValue(Index, Position x) const override { return exactSolution(x, 0.0); }
    Value InitialDerivative(Index, Position x) const override
    {
        return exactDerivative(x, 0.0);
    }

    /// The harness measures u(0) and u*(0); this is what adds u(1), which is the
    /// variable whose source carries the coupling and so the only one whose
    /// error can see it.
    static double extraError(SystemSolver &sys, Grid const &grid, double t)
    {
        return l2ErrorOf([&](double x) { return sys.yJac.u(1)(x); }, grid, t);
    }
};

using Coupled = ManufacturedUdot<0.4>;
using Uncoupled = ManufacturedUdot<0.0>;

/*
    The same coupling written for the autodiff layer.

    Nothing here writes a derivative: `Source` reads `udot(0)` and
    AutodiffTransportSystem::dSources_dudot differentiates the same overload the
    value went through. The point of the case is that the block arrives without
    the case author doing anything beyond declaring the flag.
*/
class AutodiffUdot : public AutodiffTransportSystem
{
public:
    static constexpr double C = 0.4;

    AutodiffUdot()
        : AutodiffTransportSystem(emptyConfig(), theGrid(), buildSpec())
    {
    }

    static toml::value const &emptyConfig()
    {
        static const toml::value cfg = toml::table{};
        return cfg;
    }
    static Grid const &theGrid()
    {
        static const Grid g(0.0, 1.0, 4);
        return g;
    }
    static SystemSpec buildSpec()
    {
        SystemSpec spec{.variables = numberedFields(2)};
        spec.variables[1].sourceReadsTimeDerivatives = true;
        return spec;
    }

    Value InitialValue(Index, Position x) const override { return exactSolution(x, 0.0); }
    Value InitialDerivative(Index, Position x) const override
    {
        return exactDerivative(x, 0.0);
    }

private:
    Real Flux(Index i, RealVector, RealVector q, Real, Time) override { return q(i); }

    Real Source(Index i, RealVector, RealVector, RealVector, RealVector, RealVector,
                RealVector, RealVector udot, Real x, Time t) override
    {
        const double xv = static_cast<double>(x.val);
        if (i == 0)
            return diffusionSource(xv, t);
        return diffusionSource(xv, t) - C * exactTimeDerivative(xv) + C * udot(0);
    }
};

/// A case that cancels its own mass term: S_1 contains 1.0 * udot(1), against
/// aFn = 1. The effective mass matrix X - dS/d(udot) is then singular in the
/// second variable and the row is no longer differential.
class SelfCancellingUdot : public TransportSystem
{
public:
    SelfCancellingUdot() : TransportSystem(buildSpec()) {}

    static SystemSpec buildSpec()
    {
        SystemSpec spec{.variables = numberedFields(2)};
        spec.variables[1].sourceReadsTimeDerivatives = true;
        return spec;
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
    Value SigmaFn(Index i, const State &s, Position, Time) override { return s.q(i); }
    Value Sources(Index i, const State &s, Position x, Time t) override
    {
        if (i == 0)
            return diffusionSource(x, t);
        return diffusionSource(x, t) + s.udot(1);
    }
    void dSigmaFn_dq(Index i, VectorRef v, const State &, Position, Time) override
    {
        v[i] = 1.0;
    }
    void dSigmaFn_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dudot(Index i, VectorRef v, const State &, Position, Time) override
    {
        if (i == 1)
            v[1] = 1.0;
    }
    Value InitialValue(Index, Position x) const override { return exactSolution(x, 0.0); }
    Value InitialDerivative(Index, Position x) const override
    {
        return exactDerivative(x, 0.0);
    }
};

/// A case that declares the flag and then never reads udot. Legal, warned about,
/// and must not be refused.
class DeclaredButUnused : public SelfCancellingUdot
{
public:
    Value Sources(Index, const State &, Position x, Time t) override
    {
        return diffusionSource(x, t);
    }
    void dSources_dudot(Index, VectorRef, const State &, Position, Time) override {}
};

} // namespace

BOOST_AUTO_TEST_SUITE(time_derivative_source_tests)

// ---------------------------------------------------------------------------
// The case itself, before any solver is involved.
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(the_coupling_is_reachable)
{
    // Three claims, and the first two are what make the order study below an
    // honest test rather than a tautology. At the exact solution the added term
    // cancels, so the manufactured source is the ordinary one; away from it it
    // does not, so a solver that never filled udot would be integrating a
    // different equation; and the declared derivative is that dependence.
    Coupled problem;
    const double t = 0.37, x = 0.31;

    State onManifold(2);
    onManifold.udot(0) = exactTimeDerivative(x);
    BOOST_TEST(problem.Sources(1, onManifold, x, t) == diffusionSource(x, t),
               boost::test_tools::tolerance(1e-14));

    State off(2);
    off.udot(0) = exactTimeDerivative(x) + 1.0;
    BOOST_TEST(std::abs(problem.Sources(1, off, x, t) -
                       problem.Sources(1, onManifold, x, t)) > 0.1);

    Vector d = Vector::Zero(2);
    problem.dSources_dudot(1, d, onManifold, x, t);
    BOOST_TEST(d(0) == 0.4, boost::test_tools::tolerance(1e-14));
    BOOST_TEST(d(1) == 0.0);

    // ...and variable 0 is untouched, which is what says the flag is honoured
    // per variable rather than per system.
    Vector d0 = Vector::Zero(2);
    problem.dSources_dudot(0, d0, onManifold, x, t);
    BOOST_TEST(d0.norm() == 0.0);
}

BOOST_AUTO_TEST_CASE(the_autodiff_layer_derives_the_same_block)
{
    // No solver and no hand-written derivative: the autodiff case's
    // dSources_dudot must agree with a central difference of its own Source.
    // This is the check that the new `wrt(udot)` overload is wired to the same
    // function the value goes through -- if it were not, the derivative would be
    // identically zero and every Jacobian on this path one term short.
    AutodiffUdot problem;
    const double t = 0.37, h = 1e-6;

    for (double x : {0.13, 0.5, 0.81})
    {
        auto sourceAt = [&](double udot0)
        {
            State s(2);
            s.udot(0) = udot0;
            return problem.Sources(1, s, x, t);
        };

        const double fd = (sourceAt(h) - sourceAt(-h)) / (2.0 * h);

        State s(2);
        Vector d = Vector::Zero(2);
        problem.dSources_dudot(1, d, s, x, t);

        BOOST_TEST(d(0) == fd, boost::test_tools::tolerance(1e-8));
        BOOST_TEST(d(0) == AutodiffUdot::C, boost::test_tools::tolerance(1e-10));
    }
}

// ---------------------------------------------------------------------------
// The residual: an order study, which is the only thing that sees a wrong one.
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(a_source_reading_udot_converges_at_the_expected_order)
{
    // u(1) is the variable whose source carries the coupling, and it has to
    // converge at k+1 like any other. A solver that dropped the udot term would
    // integrate a source short by C du/dt -- an O(C) error that does not shrink
    // with the mesh -- so the assertion that bites is not the rate on its own
    // but the rate *together with* the error being small.
    const std::vector<Index> cells{4, 8, 16};

    for (Index k : {1, 2, 3})
    {
        const Rates r = measureRates<Coupled>(k, cells, 0.1);
        BOOST_TEST_MESSAGE("k = " << k << " coupled: u " << r.uOff << " extra "
                                  << r.extraOff << r.detail);

        BOOST_TEST(r.uOff > k + 0.8);
        BOOST_TEST(r.extraOff > k + 0.8);
    }
}

BOOST_AUTO_TEST_CASE(the_coupled_and_uncoupled_problems_agree)
{
    // The controlled comparison. Both problems have the same exact solution, so
    // at a given (k, n) their errors should be close -- the coupling changes the
    // path to the answer, not the answer. If the udot term were being dropped,
    // the coupled case's u(1) error would be O(C) and this would fail by orders
    // of magnitude rather than by a factor.
    const Index k = 2, n = 16;

    const Errors coupled = solveAndMeasureBoth<Coupled>(k, n, 0.1);
    const Errors uncoupled = solveAndMeasureBoth<Uncoupled>(k, n, 0.1);

    BOOST_TEST_MESSAGE("coupled u(1) = " << coupled.extra
                                         << ", uncoupled u(1) = " << uncoupled.extra);

    BOOST_TEST(coupled.extra < 10.0 * uncoupled.extra);
    BOOST_TEST(coupled.extra < 1e-3);
}

BOOST_AUTO_TEST_CASE(the_superconvergent_scheme_keeps_its_order)
{
    // The open question the design names. udot is *interpolated* onto the star
    // nodes rather than reconstructed, because d(u*)/dt would run through q_dot,
    // which is an algebraic row's time derivative and has no meaning. If that
    // choice capped the rate this is where it would show, and the answer would
    // be to refuse the combination rather than to ship a scheme that converges
    // an order low.
    const std::vector<Index> cells{4, 8, 16};
    const Index k = 2;

    const Rates r = measureRates<Coupled>(k, cells, 0.1);
    BOOST_TEST_MESSAGE("k = " << k << " superconvergent: u* on " << r.starOn
                              << " (off " << r.starOff << ")" << r.detail);

    BOOST_TEST(r.starOn > k + 1.7);
}

// ---------------------------------------------------------------------------
// The Jacobian: invisible at alpha = 0, which is the trap.
// ---------------------------------------------------------------------------

namespace
{
/// Build a solver on the coupled problem, take it to the point where the
/// Jacobian is factorised at the given alpha, and hand back the pieces the
/// finite-difference comparison needs.
struct JacobianFixture
{
    Grid grid{0.0, 1.0, 4};
    Coupled problem;
    SystemSolver sys{grid, 2, &problem};
    SUNContext ctx{};
    N_Vector Y{}, dYdt{};
    Index n{};

    explicit JacobianFixture(double alpha)
    {
        sys.setTau(1.0);
        sys.resetCoeffs();
        sys.initialiseMatrices();

        SUNContext_Create(SUN_COMM_NULL, &ctx);

        DGSoln shape(problem.getNumVars(), grid, 2);
        n = shape.getDoF();

        Y = N_VNew_Serial(n, ctx);
        dYdt = N_VClone(Y);
        N_VConst(0.0, Y);
        N_VConst(0.0, dYdt);
        sys.setInitialConditions(Y, dYdt);

        // A dY/dt the source can actually see. setInitialConditions solves one
        // out of the residual, but the point of this fixture is to sit at a
        // state where udot is nonzero and arbitrary, since that is where a
        // wrong dS/d(udot) shows up.
        double *dY = N_VGetArrayPointer(dYdt);
        for (Index i = 0; i < n; ++i)
            dY[i] += 0.31 * std::sin(2.0 + 0.9 * i);

        sys.setJacTime(0.0);
        sys.setAlpha(alpha);
        sys.setJacEvalY(Y, dYdt);
        sys.updateBoundaryConditions(0.0);
        sys.updateMatricesForJacSolve();
    }

    ~JacobianFixture()
    {
        N_VDestroy(Y);
        N_VDestroy(dYdt);
        SUNContext_Free(&ctx);
    }
};

/// The worst relative residual of J dy = g over three right-hand sides.
double worstSolveResidual(JacobianFixture &f, double alpha)
{
    const Matrix J = fdjac::jacobian(f.sys, f.Y, f.dYdt, 0.0, alpha);
    const std::vector<Index> skip = fdjac::undefinedRows(J);

    double worst = 0.0;
    for (int trial = 0; trial < 3; ++trial)
    {
        N_Vector g = N_VNew_Serial(f.n, f.ctx);
        N_Vector dy = N_VClone(g);
        double *ga = N_VGetArrayPointer(g);
        for (Index i = 0; i < f.n; ++i)
            ga[i] = std::sin(1.0 + i * (trial + 1) * 0.7);

        const Vector gVec = fdjac::toVector(g);
        f.sys.solveHDGJac(g, dy);
        const Vector dyVec = fdjac::toVector(dy);

        worst = std::max(worst, fdjac::relativeResidual(J, dyVec, gVec, skip));

        N_VDestroy(g);
        N_VDestroy(dy);
    }
    return worst;
}
} // namespace

BOOST_AUTO_TEST_CASE(the_assembled_jacobian_carries_the_udot_block_at_nonzero_alpha)
{
    // The test the design asks for, and the one that fails if
    // `-= alphaValue * Sudot` is deleted from assembleCellMatrix. alpha is IDA's
    // cj, so this is the forward solve's Jacobian rather than the steady one.
    JacobianFixture f(3.7);
    const double worst = worstSolveResidual(f, 3.7);
    BOOST_TEST_MESSAGE("alpha = 3.7: worst ||J dy - g|| / ||g|| = " << worst);
    BOOST_TEST(worst < 1e-6);
}

BOOST_AUTO_TEST_CASE(the_udot_block_is_invisible_at_zero_alpha)
{
    // The companion, and the reason the test above specifies its alpha rather
    // than taking a default. At alpha = 0 the new term is multiplied by zero, so
    // this passes whether or not it is assembled -- which means a check written
    // only at alpha = 0 gates nothing. Recorded as a property so that the two
    // are not later collapsed into one.
    //
    // It also says the steady Jacobian -- and therefore the adjoint, which is
    // built at alpha = 0 -- is unchanged by this feature.
    JacobianFixture f(0.0);
    const double worst = worstSolveResidual(f, 0.0);
    BOOST_TEST_MESSAGE("alpha = 0: worst ||J dy - g|| / ||g|| = " << worst);
    BOOST_TEST(worst < 1e-6);
}

// ---------------------------------------------------------------------------
// The index check.
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(a_cancelled_mass_term_is_refused_by_name)
{
    // S_1 contains 1.0 * udot(1) against aFn = 1, so the u row of variable 1 has
    // no time derivative left and the system's index has risen. Left to IDA that
    // is an IDACalcIC or Newton failure with nothing to say where it came from;
    // here it is a named refusal before the first step.
    Grid grid(0.0, 1.0, 4);
    SelfCancellingUdot problem;
    SystemSolver sys(grid, 2, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile("udot_singular");
    sys.setOutputCadence(0.1);
    sys.setNOutput(2);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-14);
    sys.setTolerances({1e-8}, 1e-6);
    sys.setWriteOutput(false);
    sys.setWriteDatFile(false);

    CapturedOutput quiet;
    BOOST_CHECK_THROW(sys.initialize(), std::invalid_argument);
    sys.destroySundials();

    // The message has to name the variable; that is the whole value of the
    // check over letting IDA report a linesearch failure.
    try
    {
        SystemSolver again(grid, 2, &problem);
        again.setTau(1.0);
        again.resetCoeffs();
        again.setInputFile("udot_singular2");
        again.setOutputCadence(0.1);
        again.setNOutput(2);
        again.setInitialTime(0.0);
        again.setMinStepSize(1e-14);
        again.setTolerances({1e-8}, 1e-6);
        again.setWriteOutput(false);
        again.setWriteDatFile(false);
        again.initialize();
        BOOST_FAIL("expected a refusal");
        again.destroySundials();
    }
    catch (std::invalid_argument const &e)
    {
        const std::string what = e.what();
        BOOST_TEST(what.find("Var1") != std::string::npos);
        BOOST_TEST(what.find("dSources_dudot") != std::string::npos);
    }
}

BOOST_AUTO_TEST_CASE(declaring_the_flag_without_reading_udot_is_allowed)
{
    // A warning, not a refusal. The block being zero costs Newton iterations at
    // worst, and a case may legitimately declare the flag on a variable whose
    // dependence vanishes at the initial state.
    Grid grid(0.0, 1.0, 4);
    DeclaredButUnused problem;
    SystemSolver sys(grid, 2, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile("udot_declared_unused");
    sys.setOutputCadence(0.1);
    sys.setNOutput(2);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-14);
    sys.setTolerances({1e-8}, 1e-6);
    sys.setWriteOutput(false);
    sys.setWriteDatFile(false);

    {
        CapturedOutput quiet;
        BOOST_CHECK_NO_THROW(sys.initialize());
        sys.destroySundials();
    }
}

// ---------------------------------------------------------------------------
// The no-declaration path.
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(a_case_that_declares_nothing_sees_a_zero_udot)
{
    // What keeps every existing case what it was: with no variable declaring the
    // flag, GlobalState carries no VariableDot rows at all, so State::udot is the
    // zero vector its constructor made. A case reading it gets zero rather than
    // stale memory.
    SystemSpec spec{.variables = numberedFields(2)};
    BOOST_TEST(!spec.anySourceReadsTimeDerivatives());

    GlobalState g(4, 2, 2);
    BOOST_TEST(!g.hasVariableDot());
    const State s = g[0];
    BOOST_TEST(s.udot(0) == 0.0);
    BOOST_TEST(s.udot(1) == 0.0);

    Coupled coupled;
    BOOST_TEST(coupled.anySourceReadsTimeDerivatives());
    BOOST_TEST(!coupled.sourceReadsTimeDerivatives(0));
    BOOST_TEST(coupled.sourceReadsTimeDerivatives(1));
}

BOOST_AUTO_TEST_SUITE_END()
