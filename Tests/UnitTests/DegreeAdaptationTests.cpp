// Choosing the global polynomial degree by solving and looking at the answer.
//
// Three layers, and they are separable on purpose: Giorgiani's rule is
// arithmetic and testable with no solver at all; the accuracy indicator needs a
// solution but not a loop; the driver needs both. Each is exercised at its own
// level here, so a failure says which of the three moved.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "DegreeAdaptation.hpp"
#include "SolverConfig.hpp"
#include "SystemSolver.hpp"
#include "TestDiffusion.hpp"

#include <boost/math/quadrature/gauss.hpp>

#include <cmath>
#include <filesystem>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

using std::numbers::pi;

namespace
{
// A steady state the polynomial space cannot hold exactly.
//
// TestDiffusion's own steady state is u = 1 - x -- degree 1, so every space
// with k >= 1 holds it outright and the indicator is zero at the first level.
// That is a case worth testing and a poor one to test *adaptation* with, since
// nothing ever needs raising. A source of sin(pi x) gives
// u'' = -sin(pi x)/kappa, hence u = sin(pi x)/pi^2 + (1 - x), which no
// polynomial space holds.
//
// Sources is independent of the state, so TestDiffusion's derivative hooks --
// all of which return zero -- stay correct without being touched.
class SineSource : public TestDiffusion
{
public:
    using TestDiffusion::TestDiffusion;

    Value Sources(Index, const State &, Position x, Time) override
    {
        return std::sin(pi * x);
    }
};

const toml::value diffusion_config = toml::parse_str(
    "[DiffusionProblem]\n"
    "Kappa = 1.0\n"
    "Centre = 0.0\n");

// A configuration for a steady solve, built through the real loader so that the
// validation under test here is the validation a run gets.
SolverConfig steadyConfig(std::string const &extra, unsigned int k = 2,
                          int cells = 8)
{
    const std::string body =
        "Polynomial_degree = " + std::to_string(k) + "\n"
        "Grid_size = " + std::to_string(cells) + "\n"
        "delta_t = 0.05\n"
        "t_final = 1.0\n"
        "Lower_boundary = 0.0\n"
        "Upper_boundary = 1.0\n"
        "TransportSystem = \"LinearDiffusion\"\n"
        "OutputFilename = \"degree_adaptation_test\"\n"
        "WriteOutput = false\n"
        "SteadyStateTolerance = 1.0e-11\n"
        "MinStepSize = 1.0e-12\n"
        "Absolute_tolerance = 1.0e-10\n" +
        // Only when the caller has not asked for something else: toml11 rejects
        // a duplicate key outright rather than letting the later one win, so a
        // test overriding this would fail on a syntax error rather than on what
        // it meant to check.
        (extra.find("SteadyStateSolver") == std::string::npos
             ? "SteadyStateSolver = \"Newton\"\n"
             : "") +
        extra;

    auto v = toml::parse_str(body);
    TomlConfigSource src(v);
    return loadSolverConfig(src, ConfigSchema::Reader::Toml);
}

void removeOutput()
{
    for (const char *ext : {".nc", ".restart.nc", ".dat"})
        std::filesystem::remove(std::string("degree_adaptation_test") + ext);
}
} // namespace

BOOST_AUTO_TEST_SUITE(degree_adaptation_tests)

// ------------------------------------------------- Giorgiani's rule -------

BOOST_AUTO_TEST_CASE(the_degree_increment_is_the_log_of_how_far_short_it_fell)
{
    // dk = ceil(log_base(E/eps)). The point of the rule is that it assumes no
    // convergence *order* -- only that one more degree buys roughly a factor of
    // `base` -- which is why it survives u* superconverging transiently where a
    // Richardson target calibrated on the coarse-grid ratio would not.
    BOOST_TEST(degreeIncrement(1e-2, 1e-8, 10.0) == 6u);
    BOOST_TEST(degreeIncrement(1e-2, 1e-8, 100.0) == 3u);

    // A larger base is a *more* aggressive assumption about what a degree buys,
    // so it asks for fewer of them. Worth stating because the name suggests the
    // opposite.
    BOOST_TEST(degreeIncrement(1e-2, 1e-8, 100.0) < degreeIncrement(1e-2, 1e-8, 10.0));

    // Exactly one decade short is one degree at base 10.
    BOOST_TEST(degreeIncrement(1e-7, 1e-8, 10.0) == 1u);
}

BOOST_AUTO_TEST_CASE(the_degree_increment_stops_at_the_target_and_never_stalls)
{
    // Met, and comfortably met: zero, which is the loop's stopping test rather
    // than an error.
    BOOST_TEST(degreeIncrement(1e-9, 1e-8, 10.0) == 0u);
    BOOST_TEST(degreeIncrement(1e-8, 1e-8, 10.0) == 0u);
    BOOST_TEST(degreeIncrement(0.0, 1e-8, 10.0) == 0u);

    // Barely missed. ceil of a positive logarithm is 1 in exact arithmetic, but
    // a ratio near 1 can round the other way, and a rule that returned 0 while
    // the target was unmet would stall the loop at a degree it had already
    // rejected -- silently, since the loop would report convergence.
    BOOST_TEST(degreeIncrement(1e-8 * (1.0 + 1e-15), 1e-8, 10.0) >= 1u);

    // An unusable solve is not an under-resolved one and more degrees will not
    // help, but returning 0 would report it as converged. One step, so the
    // caller's ceiling still bounds the loop.
    BOOST_TEST(degreeIncrement(std::numeric_limits<double>::infinity(), 1e-8, 10.0) == 1u);
    BOOST_TEST(degreeIncrement(std::numeric_limits<double>::quiet_NaN(), 1e-8, 10.0) == 1u);
}

BOOST_AUTO_TEST_CASE(the_degree_increment_refuses_a_base_or_tolerance_it_cannot_use)
{
    // base = 1 makes log(base) zero and the quotient infinite; below 1 it flips
    // the sign, so a larger error would ask for a *smaller* degree.
    BOOST_CHECK_THROW(degreeIncrement(1e-2, 1e-8, 1.0), std::invalid_argument);
    BOOST_CHECK_THROW(degreeIncrement(1e-2, 1e-8, 0.5), std::invalid_argument);
    BOOST_CHECK_THROW(degreeIncrement(1e-2, 0.0, 10.0), std::invalid_argument);
    BOOST_CHECK_THROW(degreeIncrement(1e-2, -1e-8, 10.0), std::invalid_argument);
}

// ------------------------------------------- the accuracy indicator -------

BOOST_AUTO_TEST_CASE(the_indicator_vanishes_on_a_solution_the_space_holds_exactly)
{
    // TestDiffusion's steady state is u = 1 - x, from Dirichlet ends of 1 and 0.
    // Degree 1, so every space with k >= 1 holds it outright: u* has nothing to
    // add and the estimate is round-off. The negative control, and not a
    // contrived one -- Jardin's steady state is exactly linear too.
    Grid grid(0.0, 1.0, 8);
    TestDiffusion problem(diffusion_config);
    SystemSolver sys(grid, 3, &problem);

    const SolverConfig cfg = steadyConfig("", 3, 8);
    applySolverConfig(cfg, sys);
    sys.setInputFile("degree_adaptation_test");

    {
        CapturedOutput quiet;
        sys.runSolver(1.0);
    }

    const AccuracyEstimate e = sys.accuracyEstimate(0);

    BOOST_TEST_MESSAGE("u = 1 - x at k = 3: ||u* - u_h|| = " << e.globalL2
                       << ", ||u_h|| = " << e.solutionL2);

    // Guard: the solution is not itself zero, or the line below would hold for
    // the wrong reason.
    BOOST_TEST(e.solutionL2 > 0.1);
    BOOST_TEST(e.globalL2 < 1e-12, "estimate is " << e.globalL2);
    BOOST_TEST(e.worstCell < 1e-12);

    removeOutput();
}

BOOST_AUTO_TEST_CASE(the_indicator_is_the_l2_distance_between_u_star_and_u_h)
{
    // The formula, against an independent Gauss-30 quadrature of the same two
    // fields -- and on a solution that is *not* in the space, so there is
    // something to measure. Also pins the per-cell figure as an error
    // *density*: divided by |K|, which is what makes it comparable between
    // cells of different width.
    Grid grid(0.0, 1.0, 6);
    SineSource problem(diffusion_config);
    SystemSolver sys(grid, 2, &problem);

    const SolverConfig cfg = steadyConfig("", 2, 6);
    applySolverConfig(cfg, sys);
    sys.setInputFile("degree_adaptation_test");

    {
        CapturedOutput quiet;
        sys.runSolver(1.0);
    }

    const AccuracyEstimate e = sys.accuracyEstimate(0);
    Postprocessor const *post = sys.getPostprocessor();

    boost::math::quadrature::gauss<double, 30> gauss;
    double sumsq = 0.0;

    for (Grid::Index cell = 0; cell < grid.getNCells(); ++cell)
    {
        Interval const &I = grid[cell];
        const double cellsq = gauss.integrate(
            [&](double x)
            {
                const double d = post->uStar(0)(x, I) - sys.yJac.u(0)(x, I);
                return d * d;
            },
            I.x_l, I.x_u);

        sumsq += cellsq;

        // The density, not the norm.
        BOOST_TEST(e.perCell(cell) == std::sqrt(cellsq / I.h()),
                   boost::test_tools::tolerance(1e-10));
    }

    BOOST_TEST_MESSAGE("sine source at k = 2: ||u* - u_h|| = " << e.globalL2);

    // There is something to measure, so the agreement below is not two zeros.
    BOOST_TEST(e.globalL2 > 1e-8);
    BOOST_TEST(e.globalL2 == std::sqrt(sumsq), boost::test_tools::tolerance(1e-10));
    BOOST_TEST(e.worstCell == e.perCell.maxCoeff());

    removeOutput();
}

BOOST_AUTO_TEST_CASE(the_indicator_falls_as_the_degree_rises_on_a_smooth_solution)
{
    // What the whole scheme rests on: that the estimate actually tracks how
    // well resolved the answer is. sin(pi x)/pi^2 + (1 - x) is analytic, so
    // raising the degree should collapse it.
    auto estimateAt = [](unsigned int k)
    {
        Grid grid(0.0, 1.0, 6);
        SineSource problem(diffusion_config);
        SystemSolver sys(grid, k, &problem);

        const SolverConfig cfg = steadyConfig("", k, 6);
        applySolverConfig(cfg, sys);
        sys.setInputFile("degree_adaptation_test");

        {
            CapturedOutput quiet;
            sys.runSolver(1.0);
        }
        return sys.accuracyEstimate(0).globalL2;
    };

    const double e2 = estimateAt(2), e4 = estimateAt(4), e6 = estimateAt(6);
    BOOST_TEST_MESSAGE("||u* - u_h|| at k = 2 / 4 / 6: " << e2 << " / " << e4
                       << " / " << e6);

    BOOST_TEST(e4 < e2 / 100.0, "k = 4 gave " << e4 << " against k = 2's " << e2);
    BOOST_TEST(e6 < e4 / 100.0, "k = 6 gave " << e6 << " against k = 4's " << e4);

    removeOutput();
}

// ---------------------------------------------------- the driver ---------

BOOST_AUTO_TEST_CASE(an_answer_the_space_already_holds_is_not_refined)
{
    // One solve, no bump: u = 1 - x is exact at k = 2, so there is nothing to
    // gain and the loop must not spend a level finding that out twice.
    Grid grid(0.0, 1.0, 8);
    TestDiffusion problem(diffusion_config);

    const SolverConfig cfg = steadyConfig(
        "DegreeAdaptation = true\n"
        "DegreeTolerance = 1.0e-8\n"
        "MaxPolynomialDegree = 8\n", 2, 8);

    std::string log;
    std::unique_ptr<SystemSolver> sys;
    {
        CapturedOutput capture;
        sys = runAdaptiveDegree(cfg, problem, nullptr, grid, 2, 1.0);
        log = capture.text();
    }

    const bool built = (sys != nullptr);
    BOOST_TEST(built);
    BOOST_TEST(log.find("converged at k = 2 after 1 solve") != std::string::npos, log);
    BOOST_TEST(log.find("raising k") == std::string::npos, log);

    // And the answer is still the answer.
    BOOST_TEST(sys->yJac.u(0)(0.25) == 0.75, boost::test_tools::tolerance(1e-8));

    removeOutput();
}

BOOST_AUTO_TEST_CASE(a_solution_the_space_cannot_hold_raises_the_degree_until_it_can)
{
    // The loop doing its job. Starting deliberately low, at k = 1, on an
    // analytic steady state that k = 1 cannot represent.
    Grid grid(0.0, 1.0, 6);
    SineSource problem(diffusion_config);

    const SolverConfig cfg = steadyConfig(
        "DegreeAdaptation = true\n"
        "DegreeTolerance = 1.0e-9\n"
        "MaxPolynomialDegree = 12\n", 1, 6);

    std::string log;
    std::unique_ptr<SystemSolver> sys;
    {
        CapturedOutput capture;
        sys = runAdaptiveDegree(cfg, problem, nullptr, grid, 1, 1.0);
        log = capture.text();
    }
    BOOST_TEST_MESSAGE(log);

    BOOST_TEST(log.find("raising k from 1") != std::string::npos, log);
    BOOST_TEST(log.find("converged at k =") != std::string::npos, log);
    BOOST_TEST(log.find("stopped at the ceiling") == std::string::npos, log);

    // It ended above where it started, and met the target it was given.
    BOOST_TEST(sys->getOrder() > 1u);
    BOOST_TEST(sys->accuracyEstimate(0).globalL2 < 1e-8);

    // The transfer is left disarmed. `restarting` is sticky, so a loop that did
    // not clear it would leave the *next* run on this problem resuming from the
    // second-to-last level rather than from InitialValue -- silently, and only
    // on the second run.
    const bool disarmed = !problem.isRestarting();
    BOOST_TEST(disarmed);

    removeOutput();
}

BOOST_AUTO_TEST_CASE(the_ceiling_is_reported_rather_than_thrown)
{
    // A tolerance nothing can reach at any allowed degree. The answer at the
    // ceiling is still the best available and the caller can still have it;
    // what must not happen is a run that reports success at a tolerance it
    // never met.
    Grid grid(0.0, 1.0, 6);
    SineSource problem(diffusion_config);

    const SolverConfig cfg = steadyConfig(
        "DegreeAdaptation = true\n"
        "DegreeTolerance = 1.0e-30\n"
        "MaxPolynomialDegree = 3\n", 2, 6);

    std::string log;
    std::unique_ptr<SystemSolver> sys;
    {
        CapturedOutput capture;
        BOOST_CHECK_NO_THROW(sys = runAdaptiveDegree(cfg, problem, nullptr, grid, 2, 1.0));
        log = capture.text();
    }

    const bool built = (sys != nullptr);
    BOOST_TEST(built);
    BOOST_TEST(sys->getOrder() == 3u);
    BOOST_TEST(log.find("stopped at the ceiling k = 3") != std::string::npos, log);
    BOOST_TEST(log.find("tolerance not met") != std::string::npos, log);

    removeOutput();
}

BOOST_AUTO_TEST_CASE(the_degree_may_not_start_above_its_own_ceiling)
{
    Grid grid(0.0, 1.0, 6);
    TestDiffusion problem(diffusion_config);

    // loadSolverConfig catches this for a real configuration; the driver checks
    // it too, because it takes k0 as an argument and a caller may not have got
    // it from a config at all.
    SolverConfig cfg = steadyConfig(
        "DegreeAdaptation = true\n"
        "MaxPolynomialDegree = 4\n", 2, 6);

    BOOST_CHECK_THROW(runAdaptiveDegree(cfg, problem, nullptr, grid, 5, 1.0),
                      std::invalid_argument);
}

// ------------------------------------------------- configuration ---------

BOOST_AUTO_TEST_CASE(degree_adaptation_turns_the_superconvergent_scheme_on)
{
    // The estimate is the gap between u_h and u*, and u* is only the better of
    // the two when the superconvergent scheme is on. So asking for adaptation
    // implies it rather than requiring the user to know that.
    // Wrapped in bools: Boost.Test wants to stream its operands on failure and
    // std::optional has no operator<<.
    const SolverConfig on = steadyConfig("DegreeAdaptation = true\n");
    const bool present = on.Superconvergent.has_value();
    const bool enabled = on.Superconvergent.value_or(false);
    BOOST_TEST(present);
    BOOST_TEST(enabled);

    // ...and it is still off by default without adaptation, which is what every
    // existing config relies on.
    const SolverConfig off = steadyConfig("");
    const bool defaultOff = !off.Superconvergent.value_or(false);
    BOOST_TEST(defaultOff);
}

BOOST_AUTO_TEST_CASE(asking_for_adaptation_and_against_superconvergence_is_refused)
{
    // Presence, not value, is the signal -- which is the whole reason
    // Superconvergent is a std::optional in SolverConfig. An absent key is
    // defaulted to true above; an explicit false alongside DegreeAdaptation is
    // a contradiction, and silently overriding a key the user wrote would be
    // worse than refusing it.
    BOOST_CHECK_THROW(steadyConfig("DegreeAdaptation = true\n"
                                   "Superconvergent = false\n"),
                      std::invalid_argument);

    // Explicitly agreeing with it is fine.
    BOOST_CHECK_NO_THROW(steadyConfig("DegreeAdaptation = true\n"
                                      "Superconvergent = true\n"));
}

BOOST_AUTO_TEST_CASE(degree_adaptation_refuses_configurations_it_cannot_serve)
{
    // Time marching: re-solving a transient from t_initial at a higher degree
    // gives an estimate that cannot separate spatial from temporal error, and
    // the transfer between levels drops the BDF history.
    BOOST_CHECK_THROW(steadyConfig("DegreeAdaptation = true\n"
                                   "SteadyStateSolver = \"TimeMarch\"\n"),
                      std::invalid_argument);

    // Giorgiani's range for how much a degree may be assumed to buy. Outside it
    // the rule either creeps up one degree at a time or clears the ceiling in a
    // single step.
    BOOST_CHECK_THROW(steadyConfig("DegreeAdaptation = true\n"
                                   "DegreeAdaptationBase = 2.0\n"),
                      std::invalid_argument);
    BOOST_CHECK_THROW(steadyConfig("DegreeAdaptation = true\n"
                                   "DegreeAdaptationBase = 1000.0\n"),
                      std::invalid_argument);
    BOOST_CHECK_NO_THROW(steadyConfig("DegreeAdaptation = true\n"
                                      "DegreeAdaptationBase = 100.0\n"));

    BOOST_CHECK_THROW(steadyConfig("DegreeAdaptation = true\n"
                                   "DegreeTolerance = 0.0\n"),
                      std::invalid_argument);

    // A ceiling below the starting degree leaves the loop nothing to do.
    BOOST_CHECK_THROW(steadyConfig("DegreeAdaptation = true\n"
                                   "MaxPolynomialDegree = 1\n", 3),
                      std::invalid_argument);

    // None of these is an error when adaptation is off, because none of them is
    // read. A config carrying leftover keys from an adaptive run still loads.
    BOOST_CHECK_NO_THROW(steadyConfig("DegreeAdaptationBase = 2.0\n"
                                      "DegreeTolerance = 0.0\n"
                                      "MaxPolynomialDegree = 1\n"
                                      "Superconvergent = false\n", 3));
}

BOOST_AUTO_TEST_CASE(the_degree_keys_default_to_something_usable)
{
    const SolverConfig c = steadyConfig("");
    BOOST_TEST(!c.DegreeAdaptation);
    BOOST_TEST(c.DegreeTolerance == 1.0e-6);
    BOOST_TEST(c.MaxPolynomialDegree == 10u);
    BOOST_TEST(c.DegreeAdaptationBase == 10.0);
}

BOOST_AUTO_TEST_SUITE_END()
