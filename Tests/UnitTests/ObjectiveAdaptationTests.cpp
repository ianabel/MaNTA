// Adaptation keyed off the objective: AdaptationTarget = "Objective".
//
// Three layers, as for the solution target. The control quantity and the mesh
// decision are pure functions -- of an objective estimate, and of the decay's
// decision and that quantity -- and are checked on chosen numbers. The
// drivers are checked end to end against closed forms: every problem here has a
// known steady state, so G is known to round-off and "meets its tolerance" means
// against the continuum, not against the estimate that decided it.

#include <boost/test/unit_test.hpp>

#include "AdjointProblem.hpp"
#include "CapturedOutput.hpp"
#include "DegreeAdaptation.hpp"
#include "MeshAdaptation.hpp"
#include "SmoothnessSensor.hpp"
#include "PhysicsInstance.hpp"
#include "Postprocessing.hpp"
#include "SolverConfig.hpp"
#include "SystemSolver.hpp"
#include "TransportSystem.hpp"
#include "gridStructures.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <toml.hpp>
#include <vector>

namespace
{
using std::numbers::pi;

// Two steady problems on [0, 1] with zero Dirichlet ends, both InPlace so the
// drivers may move one instance between levels:
//
//  * Smooth: -d_x[(1 + u^2) u_x] = S with u = sin(pi x) -- nonlinear, so each
//    level is a Newton solve, and analytic, so raising k converges exponentially.
//  * Axis:   -u'' = (4/9) x^(-2/3), u = x - x^(4/3) -- an x^(4/3) singularity at
//    the axis, which raising k alone cannot resolve, and the problem
//    python/Tests/test_mesh_adaptation.py grades.
enum class Problem
{
    Smooth,
    Axis,
};

class Steady : public TransportSystem
{
public:
    explicit Steady(Problem p)
        : TransportSystem(SystemSpec{.variables = numberedFields(1), .regrid = RegridPolicy::InPlace}),
          p(p)
    {
    }
    Problem p;

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        const double u = s.u(0);
        return p == Problem::Smooth ? (1.0 + u * u) * s.q(0) : s.q(0);
    }
    Value Sources(Index, const State &, Position x, Time) override
    {
        if (p == Problem::Axis)
            return (4.0 / 9.0) * std::pow(x, -2.0 / 3.0);
        const double sn = std::sin(pi * x), c = std::cos(pi * x);
        return pi * pi * sn * (1.0 + sn * sn) - 2.0 * pi * pi * sn * c * c;
    }
    void dSigmaFn_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = p == Problem::Smooth ? 2.0 * s.u(0) * s.q(0) : 0.0;
    }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = p == Problem::Smooth ? 1.0 + s.u(0) * s.u(0) : 1.0;
    }
    void dSources_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}

    Value InitialValue(Index, Position x) const override { return 0.5 * x * (1.0 - x); }
    Value InitialDerivative(Index, Position x) const override { return 0.5 - x; }
};

// G = int u dx, or int u^2 dx.
class Integral : public AdjointProblem
{
public:
    explicit Integral(bool square) : square(square)
    {
        ng = 1;
        np = 1;
    }
    bool square;

    using AdjointProblem::GFn;
    Value GFn(Index, DGSoln &y) const override
    {
        const GlobalState s = y.evalOnNodes();
        const Index n = y.getBasis().getNodes().size();
        Value G = 0.0;
        for (Index i = 0; i < static_cast<Index>(y.getGrid().getNCells()); ++i)
        {
            const Vector w = y.getBasis().getIntegrationWeights(y.getGrid()[i]);
            for (Index j = 0; j < n; ++j)
            {
                const double u = s.Variable()(0, i * n + j);
                G += w(j) * (square ? u * u : u);
            }
        }
        return G;
    }
    Value dGFndp(Index, Index, DGSoln &) const override { return 0.0; }
    Value gFn(Index, const State &s, Position) const override
    {
        return square ? s.u(0) * s.u(0) : s.u(0);
    }
    void dgFn_du(Index, VectorRef v, const State &s, Position) override
    {
        v[0] = square ? 2.0 * s.u(0) : 1.0;
    }
    void dgFn_dq(Index, VectorRef, const State &, Position) override {}
    void dgFn_dsigma(Index, VectorRef, const State &, Position) override {}
    void dgFn_dphi(Index, VectorRef, const State &, Position) override {}
    void dSigmaFn_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }
    void dSources_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }
};

// The closed forms: int sin(pi x) = 2/pi, int sin^2(pi x) = 1/2, and for
// u = x - x^(4/3), 1/2 - 3/7 = 1/14.
double exactG(Problem p, bool square)
{
    if (p == Problem::Axis)
        return 1.0 / 14.0;
    return square ? 0.5 : 2.0 / pi;
}

// A configuration through the real loader, so the validation is a run's.
SolverConfig adaptiveConfig(std::string const &extra, unsigned int k, int cells)
{
    const std::string body =
        "PolynomialDegree = " + std::to_string(k) + "\n"
        "GridSize = " + std::to_string(cells) + "\n"
        "delta_t = 1.0\nt_final = 1.0\n"
        "LowerBoundary = 0.0\nUpperBoundary = 1.0\n"
        "TransportSystem = \"LinearDiffusion\"\n"
        "OutputFilename = \"objective_adaptation_test\"\n"
        "WriteOutput = false\n"
        "SteadyStateSolve = true\nSteadyStateSolver = \"Newton\"\n"
        "Relative_tolerance = 1e-8\nAbsolute_tolerance = 1e-10\n"
        "MinStepSize = 1e-12\n" +
        // toml11 refuses a key given twice, so a test that sets the ceiling
        // replaces this one rather than adding a second.
        (extra.find("MaxPolynomialDegree") == std::string::npos ? "MaxPolynomialDegree = 12\n" : "") +
        extra;
    auto v = toml::parse_str(body);
    TomlConfigSource src(v);
    return loadSolverConfig(src, ConfigSchema::Reader::Toml);
}

const std::string toObjective = "solveAdjoint = true\nAdaptationTarget = \"Objective\"\n";

// The u*-based objective at the state a solver returned, which is what a run
// under the superconvergent scheme reports and what the estimate bounds.
double finalG(SystemSolver &sys, AdjointProblem const &objective)
{
    sys.postprocessor->computeUStar(sys.yJac);
    return objective.GFn(0, sys.yJac, *sys.postprocessor);
}

// The degrees a driver's log says it solved at, in order.
std::vector<unsigned int> levels(std::string const &log)
{
    std::vector<unsigned int> out;
    for (size_t at = log.find("  k = "); at != std::string::npos; at = log.find("  k = ", at + 1))
    {
        unsigned int k = 0;
        if (std::sscanf(log.c_str() + at, "  k = %u:", &k) == 1 && (out.empty() || out.back() != k))
            out.push_back(k);
    }
    return out;
}

struct Adapted
{
    std::unique_ptr<Grid> grid;
    std::unique_ptr<SystemSolver> sys;
    GradingDecision decision;
    std::string log;
};

Adapted adapt(SolverConfig const &config, Steady &problem, AdjointProblem &objective,
              Grid const &start, unsigned int k0)
{
    Adapted out;
    CapturedOutput capture;
    PhysicsInstance physics(problem, start, &objective);
    if (config.MeshAdaptation)
    {
        AdaptiveMeshResult r = runAdaptiveMesh(config, physics, start, k0, 1.0);
        out.decision = r.decision;
        out.sys = std::move(r.solver);
        out.grid = std::move(r.grid);
    }
    else
        out.sys = runAdaptiveDegree(config, physics, start, k0, 1.0);
    out.log = capture.text();
    return out;
}

void removeOutput()
{
    for (const char *ext : {".nc", ".restart.nc", ".dat"})
        std::filesystem::remove(std::string("objective_adaptation_test") + ext);
}
} // namespace

BOOST_AUTO_TEST_SUITE(objective_adaptation_tests)

// ------------------------------------------------- the control quantity ----

BOOST_AUTO_TEST_CASE(each_objective_is_held_to_its_own_share_of_its_tolerance)
{
    // tol_g = share * max(rel_g |G_g|, abs_g), with G the fixed point's value
    // (`corrected`), and the worst ratio drives. Objective 0 is held relatively,
    // objective 1 sits at zero and is held by its floor alone.
    Grid grid(0.0, 1.0, 4);
    Steady problem(Problem::Smooth);
    SystemSolver sys(grid, 2, &problem);
    sys.setObjectiveTolerance({1e-3, 1e-2}, {0.0, 1e-1});
    sys.setObjectiveBudget(0.25, 0.75);

    BOOST_CHECK_THROW(objectiveAccuracy(sys), std::logic_error);

    auto pair = [](double a, double b)
    {
        Vector v(2);
        v << a, b;
        return v;
    };
    SystemSolver::ObjectiveEstimate e;
    e.valid = true;
    e.value = pair(2.1, 0.0);
    e.corrected = pair(2.0, 0.0);
    e.uncertainty = pair(0.0, 0.0);
    e.discretisation.valid = true;
    e.discretisation.total = pair(1e-4, -1e-2);
    sys.objectiveEstimate = e;

    const ObjectiveAccuracy acc = objectiveAccuracy(sys);
    BOOST_TEST(acc.tolerance(0) == 0.25 * 1e-3 * 2.0, boost::test_tools::tolerance(1e-15));
    BOOST_TEST(acc.tolerance(1) == 0.25 * 1e-1, boost::test_tools::tolerance(1e-15));
    // 1e-4 / 5e-4 = 0.2 against 1e-2 / 2.5e-2 = 0.4: the second is the worse.
    BOOST_TEST(acc.worst == 1);
    BOOST_TEST(acc.ratio == 0.4, boost::test_tools::tolerance(1e-14));

    // An objective at zero with no floor asks for an accuracy nothing meets, and
    // reads as infinitely far from it rather than as a division by zero.
    sys.setObjectiveTolerance({1e-3}, {0.0});
    BOOST_TEST(std::isinf(objectiveAccuracy(sys).ratio));
}

// ------------------------------------------------- the mesh decision -------

BOOST_AUTO_TEST_CASE(the_objective_says_whether_to_grade_and_the_decay_where)
{
    // A rough lower end by its decay, 5.6x rougher than the interior.
    std::vector<CellSmoothness> cells(6);
    for (size_t i = 0; i < cells.size(); ++i)
        cells[i].decayRate = i == 0 ? 1.0 : 5.6;
    const GradingDecision decay = gradingDecision(cells, 4, 2.0);
    BOOST_TEST_REQUIRE((decay.verdict == GradingVerdict::GradeLower));

    // Not met: the decay's verdict, every number of it.
    const GradingDecision unmet = objectiveGradingDecision(decay, 3.0);
    BOOST_TEST((unmet.verdict == GradingVerdict::GradeLower));
    BOOST_TEST((unmet.lowerEnd.reading == EndReading::Rough));
    BOOST_TEST(unmet.lowerRatio == decay.lowerRatio);
    BOOST_TEST(unmet.objectiveRatio == 3.0);

    // Met, at the tolerance or inside it: nothing graded, however rough.
    for (double ratio : {1.0, 1e-3})
    {
        const GradingDecision met = objectiveGradingDecision(decay, ratio);
        BOOST_TEST((met.verdict == GradingVerdict::Uniform));
        BOOST_TEST((met.lowerEnd.reading == EndReading::Smooth));
        BOOST_TEST(met.lowerRatio == decay.lowerRatio);
    }

    // And a smooth mesh is not graded however far from its tolerance.
    std::vector<CellSmoothness> smooth(6);
    for (auto &c : smooth)
        c.decayRate = 5.6;
    BOOST_TEST((objectiveGradingDecision(gradingDecision(smooth, 4, 2.0), 1e6).verdict ==
                GradingVerdict::Uniform));

    // An end the decay could not judge stays so: that is about the mesh.
    std::vector<double> widths = {0.4, 0.05, 0.05, 0.05, 0.05, 0.4};
    const GradingDecision wide = gradingDecision(smooth, widths, 4, 2.0);
    BOOST_TEST_REQUIRE((wide.lowerEnd.reading == EndReading::Undecidable));
    BOOST_TEST((objectiveGradingDecision(wide, 0.5).lowerEnd.reading == EndReading::Undecidable));
}

// ------------------------------------------------- the degree loop ---------

BOOST_AUTO_TEST_CASE(the_degree_loop_meets_the_objective_tolerance_against_a_closed_form)
{
    // From k = 1 on 4 cells, at three tolerances and two objectives. Each run
    // ends converged, inside its tolerance against the closed form, and the
    // tighter tolerances raise the degree further.
    Grid grid(0.0, 1.0, 4);
    for (bool square : {false, true})
    {
        unsigned int previous = 0, loosest = 0;
        for (double tol : {1e-3, 1e-5, 1e-7})
            BOOST_TEST_CONTEXT((square ? "int u^2" : "int u") << " / tolerance " << tol)
            {
                Steady problem(Problem::Smooth);
                Integral objective(square);
                const SolverConfig config = adaptiveConfig(
                    toObjective + "DegreeAdaptation = true\nObjectiveTolerance = " +
                        std::format("{:g}", tol) + "\n",
                    1, 4);
                BOOST_TEST(config.EstimateDiscretisationError);
                BOOST_TEST(config.SteadyStateTest == "Objective");

                Adapted run = adapt(config, problem, objective, grid, 1);
                BOOST_TEST_MESSAGE(run.log);
                BOOST_TEST(run.log.find("converged at k =") != std::string::npos, run.log);

                const double G = finalG(*run.sys, objective);
                const double exact = exactG(Problem::Smooth, square);
                BOOST_TEST(std::abs(G - exact) <= tol * std::abs(exact),
                           "G = " << G << " against " << exact);
                BOOST_TEST(objectiveAccuracy(*run.sys).ratio <= 1.0);

                // Never lower for a tighter tolerance, and higher by the
                // tightest: int u at 1e-3 is met by k = 1 itself, int u^2 at
                // 1e-5 and 1e-7 by the same k = 4.
                const unsigned int k = run.sys->getOrder();
                BOOST_TEST(k >= previous);
                if (tol == 1e-3)
                    loosest = k;
                else if (tol == 1e-7)
                    BOOST_TEST(k > loosest);
                previous = k;
            }
    }
    removeOutput();
}

BOOST_AUTO_TEST_CASE(the_target_decides_and_the_solution_indicator_does_not)
{
    // The same problem, two targets, opposite answers, in both directions -- so
    // the key reaches the decision rather than sitting beside it. At k = 4 on 4
    // cells u* - u_h is 5.9e-4 relative and int u is 7.5e-8 relative from the
    // closed form.
    Grid grid(0.0, 1.0, 4);
    auto run = [&](std::string const &extra)
    {
        Steady problem(Problem::Smooth);
        Integral objective(false);
        Adapted a = adapt(adaptiveConfig("DegreeAdaptation = true\n" + extra, 4, 4), problem,
                          objective, grid, 4);
        return levels(a.log);
    };

    // A loose objective is met at once where a tight solution tolerance climbs...
    BOOST_TEST(run(toObjective + "ObjectiveTolerance = 1e-4\n").size() == 1u);
    BOOST_TEST(run("DegreeTolerance = 1e-9\n").size() > 1u);
    // ...and a tight objective climbs where a loose solution tolerance stops.
    BOOST_TEST(run(toObjective + "ObjectiveTolerance = 1e-10\n").size() > 1u);
    BOOST_TEST(run("DegreeTolerance = 1e-2\n").size() == 1u);
    removeOutput();
}

BOOST_AUTO_TEST_CASE(the_ceiling_keeps_the_estimate_within_max_physics_batch)
{
    // Four cells under the superconvergent scheme hand the residual 4 (k + 2)
    // points and the estimate 4 (k + 4). A cap of 32 admits the residual to
    // k = 6 and the estimate to k = 4, so the objective target's ceiling is 4:
    // a level past it would be refused by initialize() before its first
    // evaluation. The tolerance is out of reach, so the loop runs to it.
    Grid grid(0.0, 1.0, 4);
    Steady problem(Problem::Smooth);
    Integral objective(false);
    // NewtonStep, named: an objective tolerance this far out of reach would
    // stall the Objective steady test at round-off on every level.
    const SolverConfig config = adaptiveConfig(
        toObjective + "DegreeAdaptation = true\nObjectiveTolerance = 1e-14\n"
                      "SteadyStateTest = \"NewtonStep\"\n"
                      "MaxPolynomialDegree = 8\nMaxPhysicsBatch = 32\n",
        2, 4);
    Adapted run;
    BOOST_CHECK_NO_THROW(run = adapt(config, problem, objective, grid, 2));
    BOOST_TEST(run.log.find("ceiling 4 (MaxPhysicsBatch = 32)") != std::string::npos, run.log);
    BOOST_TEST(run.sys->getOrder() == 4u);

    // The solution target counts only the residual's batch, and keeps k = 6.
    Steady other(Problem::Smooth);
    Integral otherObjective(false);
    Adapted solution = adapt(adaptiveConfig("DegreeAdaptation = true\nDegreeTolerance = 1e-14\n"
                                            "MaxPolynomialDegree = 8\nMaxPhysicsBatch = 32\n",
                                            2, 4),
                             other, otherObjective, grid, 2);
    BOOST_TEST(solution.log.find("ceiling 6 (MaxPhysicsBatch = 32)") != std::string::npos,
               solution.log);
    removeOutput();
}

// ------------------------------------------------- p -> h -> p -------------

BOOST_AUTO_TEST_CASE(mesh_adaptation_grades_where_the_objective_error_is_and_meets_it)
{
    // u = x - x^(4/3) on 10 cells at k = 3: the uniform sample's estimate is 1.5x
    // the discretisation share of 1e-5, and the decay reads the axis rough. So
    // the lower end is graded, and the degree loop finishes the job.
    Grid uniform(0.0, 1.0, 10);
    Steady problem(Problem::Axis);
    Integral objective(false);
    const SolverConfig config =
        adaptiveConfig(toObjective + "MeshAdaptation = true\nObjectiveTolerance = 1e-5\n", 3, 10);
    Adapted run = adapt(config, problem, objective, uniform, 3);
    BOOST_TEST_MESSAGE(run.log);

    BOOST_TEST((run.decision.verdict == GradingVerdict::GradeLower));
    BOOST_TEST(run.decision.objectiveRatio > 1.0);
    BOOST_TEST((*run.grid)[0].h() < 1e-4);
    BOOST_TEST(run.grid->getNCells() == 10u);

    const double G = finalG(*run.sys, objective);
    BOOST_TEST(std::abs(G - 1.0 / 14.0) <= 1e-5 / 14.0, "G = " << G);
    BOOST_TEST(objectiveAccuracy(*run.sys).ratio <= 1.0);

    // At 1e-3 the sample already meets it: nothing to grade for, and one solve.
    Steady again(Problem::Axis);
    Integral againObjective(false);
    Adapted met = adapt(
        adaptiveConfig(toObjective + "MeshAdaptation = true\nObjectiveTolerance = 1e-3\n", 3, 10),
        again, againObjective, uniform, 3);
    BOOST_TEST((met.decision.verdict == GradingVerdict::Uniform));
    BOOST_TEST((*met.grid == uniform));
    BOOST_TEST(levels(met.log).size() == 1u);
    BOOST_TEST(std::abs(finalG(*met.sys, againObjective) - 1.0 / 14.0) <= 1e-3 / 14.0);
    removeOutput();
}

BOOST_AUTO_TEST_CASE(a_given_mesh_has_its_singular_end_split_only_when_the_tolerance_asks)
{
    // A mesh given as boundaries keeps every face, and a rough end cell is split
    // into a layer of its own. At 1e-5 the sample reads 2x its share and the
    // decay reads the axis rough, so the end is split and the tolerance met at
    // k = 3; at 1e-3 the sample already meets it and nothing moves.
    const Grid given(std::vector<Position>{0.0, 0.1, 0.3, 0.6, 1.0});
    for (double tol : {1e-5, 1e-3})
        BOOST_TEST_CONTEXT("tolerance " << tol)
        {
            Steady problem(Problem::Axis);
            Integral objective(false);
            Adapted run = adapt(adaptiveConfig(toObjective + "MeshAdaptation = true\nObjectiveTolerance = " +
                                                   std::format("{:g}", tol) + "\n",
                                               3, 4),
                                problem, objective, given, 3);
            BOOST_TEST_MESSAGE(run.log);
            const bool split = tol < 1e-4;
            BOOST_TEST(run.grid->getNCells() == (split ? 7u : 4u));
            BOOST_TEST(((*run.grid)[0].h() < 0.1) == split);
            BOOST_TEST(run.grid->upperBoundary() == 1.0);
            BOOST_TEST(std::abs(finalG(*run.sys, objective) - 1.0 / 14.0) <= tol / 14.0);
        }
    removeOutput();
}

BOOST_AUTO_TEST_CASE(a_smooth_problem_is_left_uniform_however_far_from_its_tolerance)
{
    // The negative control: a smooth problem's ends read smooth, so a tolerance
    // the sample is far from is met by the degree loop on the uniform mesh.
    Grid uniform(0.0, 1.0, 10);
    Steady problem(Problem::Smooth);
    Integral objective(true);
    Adapted run = adapt(
        adaptiveConfig(toObjective + "MeshAdaptation = true\nObjectiveTolerance = 1e-11\n", 3, 10),
        problem, objective, uniform, 3);
    BOOST_TEST_MESSAGE(run.log);
    BOOST_TEST(run.decision.objectiveRatio > 1.0);
    BOOST_TEST((run.decision.verdict == GradingVerdict::Uniform));
    BOOST_TEST((*run.grid == uniform));
    BOOST_TEST(levels(run.log).size() > 1u);
    BOOST_TEST(std::abs(finalG(*run.sys, objective) - 0.5) <= 1e-11 * 0.5);
    removeOutput();
}

// ------------------------------------------------- the solution target -----

BOOST_AUTO_TEST_CASE(naming_the_solution_target_changes_nothing)
{
    // "Solution" is the default, so a configuration that names it is the one
    // that does not, bit for bit, through both drivers: the degree loop on the
    // smooth problem, and p -> h -> p on the singular one, which it grades.
    for (bool mesh : {false, true})
        BOOST_TEST_CONTEXT((mesh ? "MeshAdaptation" : "DegreeAdaptation"))
        {
            const std::string driver = mesh ? "MeshAdaptation = true\nDegreeTolerance = 1e-6\n"
                                            : "DegreeAdaptation = true\nDegreeTolerance = 1e-9\n";
            Grid uniform(0.0, 1.0, 10);
            std::vector<std::vector<double>> states;
            std::vector<unsigned int> orders;
            for (std::string const &named : {std::string{}, std::string{"AdaptationTarget = \"Solution\"\n"}})
            {
                const SolverConfig config = adaptiveConfig(driver + named, 3, 10);
                BOOST_TEST(config.AdaptationTarget == "Solution");
                BOOST_TEST(!config.EstimateDiscretisationError);
                Steady problem(mesh ? Problem::Axis : Problem::Smooth);
                Integral objective(false);
                Adapted run = adapt(config, problem, objective, uniform, 3);
                states.push_back(run.sys->stateVector());
                orders.push_back(run.sys->getOrder());
            }
            BOOST_TEST(orders[0] == orders[1]);
            BOOST_TEST(states[0] == states[1]);
        }
    removeOutput();
}

BOOST_AUTO_TEST_SUITE_END()
