// Bounds on a case's variables: FieldSpec::bound (Positive, NonNegative).
//
// What these pin:
//
//  * the exact cell minimum the whole-cell check is built on, against closed
//    forms;
//  * that the whole-cell check sees an interior dip the nodal check cannot;
//  * that a bound keeps a steady solve physical and converging, on the
//    NewtonStep loop and on the KINSOL Residual path, from a start whose
//    unbounded damped step goes negative with a finite residual -- the
//    rotating-mirror failure;
//  * that a time march honours a bound: the physics never sees a breach, and a
//    solution that must cross the bound stops with the breach named;
//  * that the clamp's Jacobian is the derivative of the residual it clamps;
//  * that a breaching initial condition is refused by name.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "FiniteDifferenceJacobian.hpp"
#include "SystemSolver.hpp"
#include "Types.hpp"
#include "util/PolynomialMinimum.hpp"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <format>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace
{

constexpr Index k = 2, nCells = 6;

void configure(SystemSolver &sys, std::string const &stem)
{
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile(stem);
    sys.setOutputCadence(1.0);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-12);
    sys.setTolerances({1e-8}, 1e-6);
    sys.setWriteOutput(false);
    sys.setEstimateObjectiveOnFinish(false);
}

void removeOutput(std::string const &stem)
{
    for (const char *ext : {".nc", ".restart.nc", ".dat"})
        std::filesystem::remove(stem + ext);
}

FieldSpec bounded(VariableBound bound, BoundaryCondition lower = BoundaryKind::Dirichlet,
                  BoundaryCondition upper = BoundaryKind::Dirichlet)
{
    FieldSpec f{"n", "density", "", lower, upper};
    f.bound = bound;
    return f;
}

// S = S0 (c - tanh u), sigma = q, between Dirichlet ends at u* = atanh(c): the
// steady state is u = u* = 0.1003 everywhere. From a bump of height 2, where
// tanh is flat, the linearised source has almost no restoring term, so a
// damped step at dt ~ 1 throws the middle of the profile far below zero --
// where the residual is perfectly finite. That is the rotating-mirror
// failure: nothing the solver measures says the state is unphysical. Records
// the least u the physics is handed.
class SaturatingSink : public TransportSystem
{
public:
    static constexpr double S0 = 100.0, C = 0.1;
    static double uStar() { return std::atanh(C); }

    explicit SaturatingSink(VariableBound bound) : TransportSystem({.variables = {bounded(bound)}}) {}

    Value LowerBoundary(Index, Time) const override { return uStar(); }
    Value UpperBoundary(Index, Time) const override { return uStar(); }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        seen(s);
        return s.q(0);
    }
    Value Sources(Index, const State &s, Position, Time) override
    {
        seen(s);
        return S0 * (C - std::tanh(s.u(0)));
    }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 1.0; }
    void dSigmaFn_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        const double c = std::cosh(s.u(0));
        v[0] = -S0 / (c * c);
    }
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}

    Value InitialValue(Index, Position x) const override { return uStar() + 8.0 * x * (1.0 - x); }
    Value InitialDerivative(Index, Position x) const override { return 8.0 * (1.0 - 2.0 * x); }

    double minSeen = std::numeric_limits<double>::infinity();

private:
    void seen(const State &s) { minSeen = std::min(minSeen, s.u(0)); }
};

struct SteadyAnswer
{
    bool converged = false;
    std::string message;
    double minSeen = 0.0;
    long rejections = 0;
    double maxError = 0.0;
    int steps = 0;
};

SteadyAnswer solveSaturating(VariableBound bound, SystemSolver::SteadyTest test, bool superconvergent = false,
                             SystemSolver::BoundCheck check = SystemSolver::BoundCheck::Nodal)
{
    const std::string stem = "variable_bound_steady";
    Grid grid(0.0, 1.0, nCells);
    SaturatingSink problem(bound);
    SystemSolver sys(grid, k, &problem);
    configure(sys, stem);
    sys.setSuperconvergent(superconvergent);
    sys.setBoundCheck(check);
    sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
    sys.setPseudoTransientInitialStep(1.0);
    sys.setSteadyStateTolerance(test == SystemSolver::SteadyTest::Residual ? 1e-8 : 1e-3);
    sys.setSteadyTest(test);

    SteadyAnswer a;
    {
        CapturedOutput quiet;
        sys.initialize();
        try
        {
            sys.solveSteadyState();
        }
        catch (std::exception const &e)
        {
            a.message = e.what();
        }
        sys.captureState();
    }
    a.converged = a.message.empty() && sys.lastSteadyOutcome() == SystemSolver::SteadyOutcome::Converged;
    a.minSeen = problem.minSeen;
    a.rejections = sys.boundRejections();
    a.steps = sys.lastSteadyStats().steps;
    for (Index i = 0; i < 9; ++i)
        a.maxError = std::max(a.maxError, std::abs(sys.yJac.u(0)(0.1 * (i + 1)) - SaturatingSink::uStar()));
    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
    removeOutput(stem);
    return a;
}

char const *name(VariableBound b)
{
    return b == VariableBound::Positive ? "Positive" : b == VariableBound::NonNegative ? "NonNegative" : "Uncontrolled";
}

char const *name(SystemSolver::SteadyTest t)
{
    return t == SystemSolver::SteadyTest::Residual ? "Residual" : "NewtonStep";
}

} // namespace

BOOST_AUTO_TEST_SUITE(variable_bound_tests)

BOOST_AUTO_TEST_CASE(the_cell_minimum_is_exact)
{
    // On the NodalBasis nodes of several degrees: polynomials with a known
    // minimum, interior and at an endpoint, and one whose minimum lies
    // between the nodes and well below every nodal value.
    for (Index p : {1, 2, 3, 5, 8})
    {
        const Vector nodes = NodalBasis::getBasis(p).getNodes();
        manta::PolynomialMinimum minimum(nodes);

        // (s - 0.3)^2 - 0.2 for p >= 2: minimum -0.2 at 0.3.
        // 0.5 - s for p = 1: minimum -0.5 at s = 1.
        Vector vals(nodes.size());
        for (Index m = 0; m < nodes.size(); ++m)
        {
            const double s = nodes(m);
            vals(m) = p >= 2 ? (s - 0.3) * (s - 0.3) - 0.2 : 0.5 - s;
        }
        auto r = minimum.minimum(vals);
        BOOST_TEST(r.value == (p >= 2 ? -0.2 : -0.5), boost::test_tools::tolerance(1e-12));
        BOOST_TEST(r.s == (p >= 2 ? 0.3 : 1.0), boost::test_tools::tolerance(1e-6));

        if (p >= 5)
        {
            // T_5(s) = 16 s^5 - 20 s^3 + 5 s, minimum -1, attained at interior
            // points; plus a cubic tilt so the minimum is unique. Compared
            // against a dense sampling, which can only overestimate it.
            auto f = [](double s) { return 16 * std::pow(s, 5) - 20 * std::pow(s, 3) + 5 * s + 0.1 * s * s * s; };
            for (Index m = 0; m < nodes.size(); ++m)
                vals(m) = f(nodes(m));
            r = minimum.minimum(vals);
            double sampled = std::numeric_limits<double>::infinity();
            for (int j = 0; j <= 200000; ++j)
                sampled = std::min(sampled, f(-1.0 + 2.0 * j / 200000.0));
            BOOST_TEST(r.value <= sampled + 1e-12);
            BOOST_TEST(r.value == sampled, boost::test_tools::tolerance(1e-8));
            BOOST_TEST(r.value < vals.minCoeff() - 1e-3, "the minimum is not between the nodes; the case tests nothing");
        }
    }
}

BOOST_AUTO_TEST_CASE(the_whole_cell_check_sees_a_dip_between_the_nodes)
{
    // k = 2, one cell's nodal values (1, 0.01, 0.2) at s = (-sqrt3/2, 0,
    // sqrt3/2): all positive, but the parabola through them dips to -0.0578
    // at s = 0.294. The nodal check passes it; WholeCell names it, with the
    // place and value of the closed-form minimum.
    Grid grid(0.0, 1.0, 3);
    SaturatingSink problem(VariableBound::Positive);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "variable_bound_dip");
    sys.initialiseMatrices();
    sys.resolveBounds();

    DGSoln Y(1, grid, k);
    std::vector<double> mem(Y.getDoF(), 1.0);
    Y.Map(mem.data());
    const Vector nodes = NodalBasis::getBasis(k).getNodes();
    const double h = std::sqrt(3.0) / 2.0;
    BOOST_REQUIRE(std::abs(nodes(0) + h) < 1e-14 && std::abs(nodes(1)) < 1e-14 && std::abs(nodes(2) - h) < 1e-14);
    Y.u(0).getCoeff(1).second << 1.0, 0.01, 0.2;

    // p(s) = c0 + c1 s + c2 s^2 through the three values.
    const double c0 = 0.01, c1 = (0.2 - 1.0) / (2.0 * h), c2 = ((0.2 + 1.0) / 2.0 - 0.01) / (h * h);
    const double sMin = -c1 / (2.0 * c2), pMin = c0 - c1 * c1 / (4.0 * c2);
    BOOST_REQUIRE(pMin < -0.05);

    sys.setBoundCheck(SystemSolver::BoundCheck::Nodal);
    BOOST_TEST(!sys.boundViolation(Y).has_value());

    sys.setBoundCheck(SystemSolver::BoundCheck::WholeCell);
    const auto v = sys.boundViolation(Y);
    BOOST_REQUIRE(v.has_value());
    BOOST_TEST(v->betweenNodes);
    BOOST_TEST(v->value == pMin, boost::test_tools::tolerance(1e-12));
    BOOST_TEST(v->x == grid[1].fromRef(sMin), boost::test_tools::tolerance(1e-10));
    BOOST_TEST(sys.describe(*v).find("n = ") != std::string::npos, sys.describe(*v));

    // And a breach at a node is found by both, as the node.
    Y.u(0).getCoeff(2).second(1) = -0.5;
    sys.setBoundCheck(SystemSolver::BoundCheck::Nodal);
    const auto w = sys.boundViolation(Y);
    BOOST_REQUIRE(w.has_value());
    BOOST_TEST(!w->betweenNodes);
    BOOST_TEST(w->value == -0.5);
}

BOOST_AUTO_TEST_CASE(a_bound_keeps_a_steady_solve_physical)
{
    // The fixture's first damped step goes far below zero with a finite
    // residual. Unbounded, the physics is evaluated there; bounded, every such
    // step is refused, dt is cut, and the solve converges to the same u*
    // without the case ever being handed a breaching value.
    for (auto test : {SystemSolver::SteadyTest::NewtonStep, SystemSolver::SteadyTest::Residual})
    {
        const SteadyAnswer free = solveSaturating(VariableBound::Uncontrolled, test);
        BOOST_TEST_MESSAGE(std::format("{} Uncontrolled: converged {}, least u seen {:.3g}, error {:.2e}, "
                                       "{} steps {}",
                                       name(test), free.converged, free.minSeen, free.maxError, free.steps,
                                       free.message));
        BOOST_TEST(free.minSeen < -1.0, "the unbounded solve never went negative, so this tests nothing");
        BOOST_TEST(free.rejections == 0);

        for (auto bound : {VariableBound::Positive, VariableBound::NonNegative})
        {
            const SteadyAnswer a = solveSaturating(bound, test);
            BOOST_TEST_MESSAGE(std::format("{} {}: converged {}, least u seen {:.3g}, {} states refused, "
                                           "error {:.2e}, {} steps {}",
                                           name(test), name(bound), a.converged, a.minSeen, a.rejections,
                                           a.maxError, a.steps, a.message));
            BOOST_TEST(a.converged, name(test) << " " << name(bound) << ": " << a.message);
            BOOST_TEST(a.rejections > 0);
            // Positive: never at or below zero. NonNegative: never below zero,
            // since anything in [-eps, 0) is handed over as 0.
            if (bound == VariableBound::Positive)
                BOOST_TEST(a.minSeen > 0.0);
            else
                BOOST_TEST(a.minSeen >= 0.0);
            BOOST_TEST(a.maxError < 1e-5);
        }
    }
}

BOOST_AUTO_TEST_CASE(a_bound_reads_u_star_under_the_superconvergent_scheme)
{
    // The physics reads u* at the star nodes, so that is what the bound
    // constrains; WholeCell checks u*'s interpolant.
    for (auto check : {SystemSolver::BoundCheck::Nodal, SystemSolver::BoundCheck::WholeCell})
    {
        const SteadyAnswer a =
            solveSaturating(VariableBound::Positive, SystemSolver::SteadyTest::NewtonStep, true, check);
        BOOST_TEST_MESSAGE(std::format("superconvergent, {}: converged {}, least u* seen {:.3g}, {} refused {}",
                                       check == SystemSolver::BoundCheck::Nodal ? "Nodal" : "WholeCell",
                                       a.converged, a.minSeen, a.rejections, a.message));
        BOOST_TEST(a.converged, a.message);
        BOOST_TEST(a.minSeen > 0.0);
        BOOST_TEST(a.rejections > 0);
        BOOST_TEST(a.maxError < 1e-5);
    }
}

namespace
{
// d_t u = u_xx - 1 between Dirichlet ends at 0.05, from 0.05 + 0.8 x (1 - x):
// the steady state, 0.05 - x (1 - x)/2, is negative in the middle, so the
// solution crosses zero at a finite time and no step size can stop it. A
// Positive bound must therefore stop the run there and say why.
class UniformSink : public TransportSystem
{
public:
    explicit UniformSink(VariableBound bound) : TransportSystem({.variables = {bounded(bound)}}) {}

    Value LowerBoundary(Index, Time) const override { return 0.05; }
    Value UpperBoundary(Index, Time) const override { return 0.05; }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        minSeen = std::min(minSeen, s.u(0));
        return s.q(0);
    }
    Value Sources(Index, const State &, Position, Time) override { return -1.0; }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 1.0; }
    void dSigmaFn_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}

    Value InitialValue(Index, Position x) const override { return 0.05 + 0.8 * x * (1.0 - x); }
    Value InitialDerivative(Index, Position x) const override { return 0.8 * (1.0 - 2.0 * x); }

    double minSeen = std::numeric_limits<double>::infinity();
};
} // namespace

BOOST_AUTO_TEST_CASE(a_time_march_honours_a_bound)
{
    auto march = [](VariableBound bound, std::string &message, double &minSeen, double &minFinal,
                    long &rejections, std::optional<SystemSolver::BoundViolation> &final)
    {
        const std::string stem = "variable_bound_march";
        Grid grid(0.0, 1.0, nCells);
        UniformSink problem(bound);
        SystemSolver sys(grid, k, &problem);
        configure(sys, stem);
        sys.setOutputCadence(0.1);
        {
            CapturedOutput quiet;
            sys.initialize();
            try
            {
                sys.integrate(2.0);
            }
            catch (std::exception const &e)
            {
                message = e.what();
            }
        }
        minSeen = problem.minSeen;
        minFinal = std::numeric_limits<double>::infinity();
        for (Index i = 0; i < nCells; ++i)
            minFinal = std::min(minFinal, sys.y.u(0).getCoeff(i).second.minCoeff());
        rejections = sys.boundRejections();
        final = sys.boundViolation(sys.y);
        {
            CapturedOutput quiet;
            sys.destroySundials();
        }
        removeOutput(stem);
    };

    std::string message;
    double minSeen, minFinal;
    long rejections;
    std::optional<SystemSolver::BoundViolation> final;

    march(VariableBound::Uncontrolled, message, minSeen, minFinal, rejections, final);
    BOOST_TEST_MESSAGE(std::format("Uncontrolled: least u seen {:.3g}, least u at the end {:.3g}; {}", minSeen,
                                   minFinal, message));
    BOOST_TEST(message.empty());
    BOOST_TEST(minFinal < -0.05, "the unbounded run never crossed zero, so this tests nothing");

    march(VariableBound::Positive, message, minSeen, minFinal, rejections, final);
    BOOST_TEST_MESSAGE(std::format("Positive: least u seen {:.3g}, least u at the end {:.3g}, {} refused; {}",
                                   minSeen, minFinal, rejections, message));
    // The run stops where the solution reaches the bound, and says so.
    BOOST_TEST(!message.empty());
    BOOST_TEST(message.find("n = ") != std::string::npos, message);
    BOOST_TEST(message.find("Positive") != std::string::npos, message);
    BOOST_TEST(rejections > 0);
    // No state IDA accepted breaches it, and the physics never saw one.
    BOOST_TEST(minSeen > 0.0);
    BOOST_TEST(minFinal > 0.0);
    BOOST_TEST(!final.has_value());
}

namespace
{
// The nonlinear system ycor - target = 0, with the identity for its linear
// solve: one Newton iteration lands ycor on target exactly.
struct TargetSystem
{
    N_Vector target;
};
int targetSys(N_Vector ycor, N_Vector F, void *mem)
{
    N_VLinearSum(1.0, ycor, -1.0, static_cast<TargetSystem *>(mem)->target, F);
    return 0;
}
int identitySolve(N_Vector, void *) { return 0; }
int alwaysConverged(SUNNonlinearSolver, N_Vector, N_Vector, sunrealtype, N_Vector, void *) { return SUN_SUCCESS; }
} // namespace

BOOST_AUTO_TEST_CASE(the_corrector_is_checked_before_ida_accepts_it)
{
    // IDA takes y0 + ycor from its nonlinear solver as the step's result and
    // never evaluates it, so the residual callback cannot see a breach there.
    // The solver it is given checks it: a converged solve whose result
    // breaches is a recoverable convergence failure, which IDA answers by
    // cutting the step. Driven directly, since in every IDA run in this file
    // the residual callback refuses the breaching iterate first.
    Grid grid(0.0, 1.0, 3);
    SaturatingSink problem(VariableBound::Positive);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "variable_bound_corrector");
    sys.initialiseMatrices();
    sys.resolveBounds();

    DGSoln shape(1, grid, k);
    const Index n = shape.getDoF();
    N_Vector y0 = N_VNew_Serial(n, sys.ctx), ycor = N_VClone(y0), w = N_VClone(y0);
    TargetSystem target{N_VClone(y0)};
    N_VConst(1.0, y0);
    N_VConst(1.0, w);

    SUNNonlinearSolver NLS = sys.makeBoundedNonlinearSolver(y0);
    SUNNonlinSolSetSysFn(NLS, targetSys);
    SUNNonlinSolSetLSolveFn(NLS, identitySolve);
    SUNNonlinSolSetConvTestFn(NLS, alwaysConverged, nullptr);
    SUNNonlinSolSetMaxIters(NLS, 4);
    BOOST_REQUIRE(SUNNonlinSolInitialize(NLS) == SUN_SUCCESS);

    // A correction that keeps every u above zero is accepted...
    N_VConst(-0.5, target.target);
    N_VConst(0.0, ycor);
    BOOST_TEST(SUNNonlinSolSolve(NLS, y0, ycor, w, 1.0, SUNFALSE, &target) == SUN_SUCCESS);
    BOOST_TEST(sys.boundRejections() == 0);

    // ...and one that takes a single u coefficient below it is not.
    DGSoln t(1, grid, k, N_VGetArrayPointer(target.target));
    t.u(0).getCoeff(1).second(2) = -1.25;
    N_VConst(0.0, ycor);
    BOOST_TEST(SUNNonlinSolSolve(NLS, y0, ycor, w, 1.0, SUNFALSE, &target) == SUN_NLS_CONV_RECVR);
    BOOST_TEST(sys.boundRejections() == 1);
    BOOST_REQUIRE(sys.lastBoundViolation().has_value());
    BOOST_TEST(sys.lastBoundViolation()->value == -0.25, boost::test_tools::tolerance(1e-14));

    SUNNonlinSolFree(NLS);
    for (N_Vector v : {y0, ycor, w, target.target})
        N_VDestroy(v);
}

BOOST_AUTO_TEST_CASE(a_time_march_within_a_bound_completes)
{
    // The saturating sink marched to its steady state: the true solution
    // stays above u* > 0, but IDA's corrector on a long first step overshoots
    // far below zero, as the damped steady step does. Bounded, the run
    // completes to the same answer with the physics never handed a breach.
    for (auto bound : {VariableBound::Uncontrolled, VariableBound::Positive, VariableBound::NonNegative})
    {
        const std::string stem = "variable_bound_march_ok";
        Grid grid(0.0, 1.0, nCells);
        SaturatingSink problem(bound);
        SystemSolver sys(grid, k, &problem);
        configure(sys, stem);
        sys.setTolerances({1e-3}, 1e-2);
        sys.setOutputCadence(0.5);
        // A first step long enough for the corrector to see the bump's
        // linearisation, which throws it far below zero as the damped steady
        // step does.
        sys.setInitialTimestep(0.5);
        std::string message;
        {
            CapturedOutput quiet;
            sys.initialize();
            try
            {
                sys.integrate(1.0);
            }
            catch (std::exception const &e)
            {
                message = e.what();
            }
        }
        double err = 0.0;
        for (Index i = 0; i < 9; ++i)
            err = std::max(err, std::abs(sys.y.u(0)(0.1 * (i + 1)) - SaturatingSink::uStar()));
        BOOST_TEST_MESSAGE(std::format("{}: least u seen {:.3g}, {} refused, error {:.2e} {}", name(bound),
                                       problem.minSeen, sys.boundRejections(), err, message));
        BOOST_TEST(message.empty(), name(bound) << ": " << message);
        BOOST_TEST(err < 1e-2);
        if (bound == VariableBound::Uncontrolled)
            BOOST_TEST(problem.minSeen < -1.0, "the unbounded run never went negative, so this tests nothing");
        else
            BOOST_TEST(sys.boundRejections() > 0);
        if (bound == VariableBound::Positive)
            BOOST_TEST(problem.minSeen > 0.0);
        if (bound == VariableBound::NonNegative)
            BOOST_TEST(problem.minSeen >= 0.0);
        {
            CapturedOutput quiet;
            sys.destroySundials();
        }
        removeOutput(stem);
    }
}

namespace
{
// sigma = (1 + u) q, S = sin(3x) - 2 u - u^2, every derivative in u nonzero at
// u = 0, so a Jacobian that kept them at a clamped node would differ from the
// residual's, which does not move there. The initial state dips below zero
// over the right half of the domain.
class ClampedDiffusion : public TransportSystem
{
public:
    ClampedDiffusion() : TransportSystem({.variables = {bounded(VariableBound::NonNegative)}}) {}

    Value LowerBoundary(Index, Time) const override { return 0.3; }
    Value UpperBoundary(Index, Time) const override { return -0.2; }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        minSeen = std::min(minSeen, s.u(0));
        return (1.0 + s.u(0)) * s.q(0);
    }
    Value Sources(Index, const State &s, Position x, Time) override
    {
        return std::sin(3.0 * x) - 2.0 * s.u(0) - s.u(0) * s.u(0);
    }

    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override { v[0] = 1.0 + s.u(0); }
    void dSigmaFn_du(Index, VectorRef v, const State &s, Position, Time) override { v[0] = s.q(0); }
    void dSources_du(Index, VectorRef v, const State &s, Position, Time) override { v[0] = -2.0 - 2.0 * s.u(0); }
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}

    Value InitialValue(Index, Position x) const override { return 0.3 * std::cos(std::numbers::pi * x * 0.9) - 0.05; }
    Value InitialDerivative(Index, Position x) const override
    {
        return -0.27 * std::numbers::pi * std::sin(std::numbers::pi * x * 0.9);
    }

    double minSeen = std::numeric_limits<double>::infinity();
};

double clampJacobianMismatch(bool superconvergent, double cj, long &negativeNodes, double &minSeen)
{
    const Index kk = 2, n4 = 4;
    const double t = 0.0;
    Grid grid(0.0, 1.0, n4);
    ClampedDiffusion problem;
    SystemSolver sys(grid, kk, &problem);
    sys.setTau(0.5);
    sys.setSuperconvergent(superconvergent);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);
    DGSoln shape(1, grid, kk);
    const Index n = shape.getDoF();
    N_Vector Y = N_VNew_Serial(n, ctx), dYdt = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    // The values the physics reads, and how many of them the clamp moves; none
    // within the difference step of zero, where the clamp has no derivative.
    {
        DGSoln Yh(1, grid, kk, N_VGetArrayPointer(Y));
        auto nodes = sys.physicsNodesAt(Yh, t, false);
        negativeNodes = (nodes.states.Variable().array() < 0.0).count();
        BOOST_REQUIRE((nodes.states.Variable().array().abs() > 1e-4).all());
    }

    sys.setJacTime(t);
    sys.setAlpha(cj);
    sys.setJacEvalY(Y, dYdt);
    sys.updateBoundaryConditions(t);
    sys.updateMatricesForJacSolve();

    problem.minSeen = std::numeric_limits<double>::infinity();
    const Matrix J = fdjac::jacobian(sys, Y, dYdt, t, cj);
    minSeen = problem.minSeen;

    N_Vector g = N_VNew_Serial(n, ctx), dy = N_VClone(g);
    double *ga = N_VGetArrayPointer(g);
    for (Index i = 0; i < n; ++i)
        ga[i] = std::sin(1.0 + i * 0.7);
    const Vector gVec = Eigen::Map<Vector>(ga, n);
    sys.solveHDGJac(g, dy);
    const Vector dyVec = Eigen::Map<Vector>(N_VGetArrayPointer(dy), n);

    const Vector r = J * dyVec - gVec;
    double num = 0.0, den = 0.0;
    for (Index i = 0; i < n; ++i)
    {
        if (J.row(i).norm() == 0.0)
            continue;
        num += r(i) * r(i);
        den += gVec(i) * gVec(i);
    }

    N_VDestroy(g);
    N_VDestroy(dy);
    N_VDestroy(Y);
    N_VDestroy(dYdt);
    SUNContext_Free(&ctx);
    return std::sqrt(num / den);
}
} // namespace

BOOST_AUTO_TEST_CASE(the_clamp_jacobian_matches_a_finite_difference_of_its_residual)
{
    // J dy = g against a central difference of residual() at a state where
    // the clamp is active at several physics nodes. The difference sees no
    // change at a clamped node, so the Jacobian must carry none.
    for (bool sc : {false, true})
        for (double cj : {0.0, 3.7})
        {
            long negative = 0;
            double minSeen = 0.0;
            const double mismatch = clampJacobianMismatch(sc, cj, negative, minSeen);
            BOOST_TEST_MESSAGE(std::format("superconvergent {}, cj {}: {} clamped nodes, least u seen {:.3g}, "
                                           "||J dy - g|| / ||g|| = {:.2e}",
                                           sc, cj, negative, minSeen, mismatch));
            BOOST_TEST(negative > 2, "the clamp never acted, so this tests nothing");
            BOOST_TEST(minSeen >= 0.0);
            BOOST_TEST(mismatch < 1e-6);
        }
}

BOOST_AUTO_TEST_CASE(a_breaching_initial_condition_is_refused_by_name)
{
    // The clamp lets the physics be evaluated at the initial state, so nothing
    // else would stop the run starting; the bound does.
    Grid grid(0.0, 1.0, 4);
    ClampedDiffusion problem; // dips to -0.05 - ... at the right
    SystemSolver sys(grid, k, &problem);
    configure(sys, "variable_bound_initial");
    sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
    sys.setSteadyStateTolerance(1e-3);
    std::string message;
    {
        CapturedOutput quiet;
        try
        {
            sys.initialize();
        }
        catch (std::invalid_argument const &e)
        {
            message = e.what();
        }
        sys.destroySundials();
    }
    BOOST_TEST_MESSAGE(message);
    BOOST_TEST(message.find("initial condition breaches a bound") != std::string::npos, message);
    BOOST_TEST(message.find("NonNegative") != std::string::npos, message);
    removeOutput("variable_bound_initial");
}

BOOST_AUTO_TEST_CASE(an_unbounded_case_takes_none_of_this_path)
{
    // No bounded variable: nothing is resolved, nothing checked, IDA keeps
    // its own Newton.
    Grid grid(0.0, 1.0, nCells);
    SaturatingSink problem(VariableBound::Uncontrolled);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "variable_bound_none");
    {
        CapturedOutput quiet;
        sys.initialize();
    }
    BOOST_TEST(!sys.boundsActive());
    BOOST_TEST(sys.clampedVars.empty());
    BOOST_TEST(sys.idaBoundedNLS == nullptr);
    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
    removeOutput("variable_bound_none");
}

BOOST_AUTO_TEST_CASE(a_margin_without_a_bound_or_below_zero_is_refused)
{
    FieldSpec f{"n", "", ""};
    f.boundEpsilon = 1e-3;
    BOOST_CHECK_THROW((SystemSpec{.variables = {f}}.validate()), std::invalid_argument);
    f.bound = VariableBound::Positive;
    BOOST_CHECK_NO_THROW((SystemSpec{.variables = {f}}.validate()));
    f.boundEpsilon = -1e-3;
    BOOST_CHECK_THROW((SystemSpec{.variables = {f}}.validate()), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
