#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "DegreeAdaptation.hpp"
#include "MeshAdaptation.hpp"
#include "ParallelFill.hpp"
#include "PhysicsInstance.hpp"
#include "SolverConfig.hpp"
#include "SystemSolver.hpp"
#include "TransportSystem.hpp"
#include "gridStructures.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <toml.hpp>
#include <vector>

/*
    PhysicsParallelism: the cost of a level in rounds of N-wide physics, and the
    controllers filling each level they choose to the largest that costs no more.

    The plans compared are SystemSolver::evaluationPlanFor's, so the first thing
    pinned is that it is exactly the plan a solver built at that level makes --
    without that, a fill would be costing a formula rather than the sites.
 */

BOOST_AUTO_TEST_SUITE(parallel_fill_tests)

namespace
{
// u = x - x^(4/3) at steady state: -u'' = (4/9) x^(-2/3) with zero Dirichlet
// ends, singular at the axis, so MeshAdaptation grades towards x = 0 and the
// degree loop has something to climb. Records the plans it is handed.
class AxisSingular : public TransportSystem
{
public:
    AxisSingular()
        : TransportSystem(SystemSpec{.variables = numberedFields(1),
                                     .regrid = RegridPolicy::InPlace})
    {
    }

    std::vector<EvaluationPlan> plans;
    void prepareEvaluation(EvaluationPlan const &plan) override { plans.push_back(plan); }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
    Value SigmaFn(Index, const State &s, Position, Time) override { return s.q(0); }
    void dSigmaFn_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 1.0; }
    Value Sources(Index, const State &, Position x, Time) override
    {
        return (4.0 / 9.0) * std::pow(x, -2.0 / 3.0);
    }
    void dSources_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}
    Value InitialValue(Index, Position) const override { return 0.0; }
    Value InitialDerivative(Index, Position) const override { return 0.0; }
};

SolverConfig configFrom(std::string const &extra)
{
    const std::string body =
        "PolynomialDegree = 4\nGridSize = 5\ndelta_t = 0.1\nt_final = 1.0\n"
        "LowerBoundary = 0.0\nUpperBoundary = 1.0\n"
        "TransportSystem = \"LinearDiffusion\"\nWriteOutput = false\n"
        "SteadyStateSolver = \"Newton\"\nSteadyStateTolerance = 1e-11\n"
        "Absolute_tolerance = 1e-10\nMinStepSize = 1e-12\n" + extra;
    auto v = toml::parse_str(body);
    TomlConfigSource src(v);
    return loadSolverConfig(src, ConfigSchema::Reader::Toml);
}

std::unique_ptr<SystemSolver> configured(SolverConfig const &config, TransportSystem &problem,
                                         Grid const &grid, unsigned int k)
{
    auto s = std::make_unique<SystemSolver>(grid, k, &problem);
    applySolverConfig(config, *s);
    return s;
}

EvaluationSite site(EvaluationCadence cadence, Index calls, Index batch,
                    EvaluationEntry entry = EvaluationEntry::ComputePhysics)
{
    return {EvaluationKind::Residual, entry, cadence, calls, 1,
            std::vector<Position>(static_cast<size_t>(batch), 0.0)};
}

double narrowestCell(Grid const &g)
{
    double h = g.upperBoundary() - g.lowerBoundary();
    for (Grid::Index i = 0; i < g.getNCells(); ++i)
        h = std::min(h, g[i].h());
    return h;
}
} // namespace

BOOST_AUTO_TEST_CASE(rounds_count_each_recurring_batched_site_and_nothing_else)
{
    // calls x ceil(batch / width) per recurring cadence. Pointwise hooks run in
    // a serial loop, and once-per-run sites are not a cost per iteration, so
    // neither is counted.
    EvaluationPlan plan;
    plan.sites = {site(EvaluationCadence::PerResidual, 1, 30),
                  site(EvaluationCadence::PerResidual, 1, 10),
                  site(EvaluationCadence::PerJacobianBuild, 4, 10),
                  site(EvaluationCadence::PerContinuationStep, 1, 65),
                  site(EvaluationCadence::PerResidual, 1, 1000, EvaluationEntry::Pointwise),
                  site(EvaluationCadence::OncePerRun, 2, 1000),
                  site(EvaluationCadence::PerAdjointSolve, 1, 1000)};

    BOOST_TEST((roundsOf(plan, 64) == EvaluationRounds{2, 4, 2}));
    BOOST_TEST((roundsOf(plan, 1) == EvaluationRounds{40, 40, 65}));
    BOOST_TEST(residualOccupancy(plan, 64) == 40.0 / 128.0);
    BOOST_TEST(residualOccupancy(plan, 1) == 1.0);

    // Each cadence on its own: fewer residual rounds do not pay for more
    // Jacobian ones.
    BOOST_TEST((EvaluationRounds{1, 4, 2}.fitsWithin({2, 4, 2})));
    BOOST_TEST(!(EvaluationRounds{1, 5, 2}.fitsWithin({2, 4, 2})));
}

BOOST_AUTO_TEST_CASE(a_plan_for_another_level_is_the_plan_a_solver_there_makes)
{
    // Bit for bit, in every configuration that moves a site: the plan a solver
    // reports for another level and the plan a solver built at that level makes.
    for (std::string const &extra :
         {std::string{}, std::string{"Superconvergent = true\n"},
          std::string{"Superconvergent = true\ntauScaling = \"Diffusive\"\n"},
          std::string{"tauScaling = \"Diffusive\"\ntauUpdate = \"ContinuationStep\"\n"}})
    {
        BOOST_TEST_CONTEXT(extra)
        {
            const SolverConfig config = configFrom(extra);
            AxisSingular problem;
            const Grid here(0.0, 1.0, 5);
            const Grid there(gradedMeshPoints(0.0, 1.0, 9, 4, 0.2, 0.2, 0.3, GradedEnd::Lower));
            const auto planner = configured(config, problem, here, 4);
            for (unsigned int k : {2u, 4u, 7u})
            {
                const auto built = configured(config, problem, there, k);
                BOOST_TEST((planner->evaluationPlanFor(there, k) == built->evaluationPlan()));
            }
            BOOST_TEST((planner->evaluationPlanFor(here, 4) == planner->evaluationPlan()));
        }
    }
}

BOOST_AUTO_TEST_CASE(a_level_fills_to_the_largest_that_costs_no_more_rounds)
{
    // Superconvergent: 5 cells at k take 5 (k + 2) points per residual.
    const SolverConfig config = configFrom("Superconvergent = true\n");
    AxisSingular problem;
    const Grid grid(0.0, 1.0, 5);
    const auto planner = configured(config, problem, grid, 4);
    const LevelPlan planAt = [&](Grid const &g, unsigned int k)
    { return planner->evaluationPlanFor(g, k); };

    // 64 wide: 30 points is one round, and so is anything up to 5 x 12 = 60.
    BOOST_TEST(filledDegree(planAt, grid, 4, 20, 64) == 10u);
    BOOST_TEST(filledDegree(planAt, grid, 6, 20, 64) == 10u);
    BOOST_TEST(filledDegree(planAt, grid, 4, 8, 64) == 8u); // the ceiling holds
    BOOST_TEST(filledDegree(planAt, grid, 4, 20, 1) == 4u); // and 1 is no change
    BOOST_TEST(filledDegree(planAt, grid, 4, 20, 30) == 4u); // already full

    // Cells at k = 4: 6 points each, so 10 in one round of 64 and 42 in two of 256.
    BOOST_TEST(filledCellCount(planAt, grid, 4, 1000, 64) == 10u);
    BOOST_TEST(filledCellCount(planAt, grid, 4, 1000, 256) == 42u);
    BOOST_TEST(filledCellCount(planAt, grid, 4, 7, 64) == 7u);
    BOOST_TEST(filledCellCount(planAt, grid, 4, 1000, 1) == 5u);
    BOOST_TEST(filledCellCount(planAt, grid, 4, 1000, 30) == 5u);
}

BOOST_AUTO_TEST_CASE(the_tau_faces_are_counted_in_a_real_plan)
{
    // Under a Diffusive tau updated per residual the faces are a second batch
    // per residual, 2 nCells wide, and 1 + 3 nVars + nAux more per Jacobian
    // build: at 5 cells, k = 4 and width 64 that is 1 + 1 residual rounds and
    // 1 + 4 Jacobian ones. (They never bind a fill on their own -- the
    // residual's n (k + 2) points outgrow their 2n first, and they do not depend
    // on k -- but they are part of what a level costs.)
    const SolverConfig config =
        configFrom("Superconvergent = true\ntauScaling = \"Diffusive\"\ntauUpdate = \"Residual\"\n");
    AxisSingular problem;
    const Grid grid(0.0, 1.0, 5);
    const auto planner = configured(config, problem, grid, 4);
    BOOST_TEST((roundsOf(planner->evaluationPlan(), 64) == EvaluationRounds{2, 5, 0}));

    const auto continuation = configured(
        configFrom("Superconvergent = true\ntauScaling = \"Diffusive\"\n"
                   "tauUpdate = \"ContinuationStep\"\n"),
        problem, grid, 4);
    BOOST_TEST((roundsOf(continuation->evaluationPlan(), 64) == EvaluationRounds{1, 1, 1}));
}

BOOST_AUTO_TEST_CASE(an_underused_configured_level_is_warned_about_and_a_full_one_is_not)
{
    const SolverConfig config = configFrom("Superconvergent = true\n");
    AxisSingular problem;
    const Grid grid(0.0, 1.0, 5);
    const auto planner = configured(config, problem, grid, 4);
    const LevelPlan planAt = [&](Grid const &g, unsigned int k)
    { return planner->evaluationPlanFor(g, k); };

    const std::string w = underuseWarning(planAt, grid, 4, 10, 64);
    BOOST_TEST(w.find("47%") != std::string::npos, w);
    BOOST_TEST(w.find("PolynomialDegree up to 10") != std::string::npos, w);
    BOOST_TEST(w.find("GridSize up to 10") != std::string::npos, w);

    BOOST_TEST(underuseWarning(planAt, grid, 4, 10, 30).empty());
    BOOST_TEST(underuseWarning(planAt, grid, 4, 10, 1).empty());
}

BOOST_AUTO_TEST_CASE(the_degree_loop_fills_each_level_it_raises_to)
{
    // The rule asks for a few degrees at a time; filled, the first raise goes
    // straight to the largest degree in the same round, here the ceiling.
    const SolverConfig config = configFrom(
        "DegreeAdaptation = true\nDegreeTolerance = 1e-9\nMaxPolynomialDegree = 10\n"
        "PhysicsParallelism = 64\n");
    Grid grid(0.0, 1.0, 5);
    AxisSingular problem;
    PhysicsInstance physics(problem, grid);
    {
        CapturedOutput quiet;
        auto final = runAdaptiveDegree(config, physics, grid, 4, 1.0);
    }
    BOOST_TEST_REQUIRE(problem.plans.size() == 2u);
    BOOST_TEST(problem.plans[0].k == 4);
    BOOST_TEST(problem.plans[1].k == 10);
}

BOOST_AUTO_TEST_CASE(a_graded_mesh_fills_its_layer_without_thinning_the_wall_cell)
{
    // 5 cells at k = 4 is 30 points; 64 wide takes 10 cells for the same round.
    // The layer takes as many of the extra cells as keep its wall cell the one
    // the sample's count would have given it at a ratio of at most 1/2, and the
    // bulk the rest.
    auto run = [](unsigned int width)
    {
        const SolverConfig config = configFrom(
            "MeshAdaptation = true\nDegreeTolerance = 1e-2\nMaxPolynomialDegree = 4\n"
            "PhysicsParallelism = " + std::to_string(width) + "\n");
        Grid uniform(0.0, 1.0, 5);
        AxisSingular problem;
        PhysicsInstance physics(problem, uniform);
        CapturedOutput quiet;
        AdaptiveMeshResult r = runAdaptiveMesh(config, physics, uniform, 4, 1.0);
        BOOST_TEST_REQUIRE((r.decision.verdict == GradingVerdict::GradeLower));
        return Grid(*r.grid);
    };
    const Grid unfilled = run(1);
    const Grid filled = run(64);

    BOOST_TEST(unfilled.getNCells() == 5u);
    BOOST_TEST(filled.getNCells() == 10u);
    BOOST_TEST(narrowestCell(filled) == narrowestCell(unfilled),
               boost::test_tools::tolerance(1e-12));
    BOOST_TEST(narrowestCell(filled) == filled[0].h(), boost::test_tools::tolerance(1e-12));
    // Widths grow away from the wall through the layer.
    for (Grid::Index i = 0; i + 1 < filled.getNCells() && filled[i + 1].x_u <= 0.2 + 1e-12; ++i)
        BOOST_TEST(filled[i].h() <= filled[i + 1].h());
}

BOOST_AUTO_TEST_CASE(a_ladder_rung_filled_into_the_next_level_is_dropped)
{
    // 3 cells at k = 2 before 5 at k = 4. 64 wide, the rung fills to k = 4 (18
    // points) and then to 5 cells (30): the configured level itself, so it goes,
    // and the case is handed one plan rather than two.
    for (unsigned int width : {1u, 64u})
    {
        BOOST_TEST_CONTEXT("width " << width)
        {
            const SolverConfig config = configFrom(
                "GridLadder = [3]\nDegreeLadder = [2]\nPhysicsParallelism = " +
                std::to_string(width) + "\n");
            Grid grid(0.0, 1.0, 5);
            AxisSingular problem;
            PhysicsInstance physics(problem, grid);
            {
                CapturedOutput quiet;
                auto final = runLadder(config, physics, grid, 4, 1.0);
            }
            BOOST_TEST(problem.plans.size() == (width == 1 ? 2u : 1u));
            BOOST_TEST(problem.plans.back().k == 4);
            BOOST_TEST(problem.plans.back().grid.getNCells() == 5u);
        }
    }
}

BOOST_AUTO_TEST_CASE(physics_parallelism_must_be_at_least_one)
{
    BOOST_CHECK_THROW(configFrom("PhysicsParallelism = 0\n"), std::invalid_argument);
    BOOST_TEST(configFrom("").PhysicsParallelism == 1u);
}

BOOST_AUTO_TEST_SUITE_END()
