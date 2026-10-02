// The evaluation plan (EvaluationPlan.hpp) and regridding (SystemSpec::supportsRegrid).
//
// The plan is only worth having if it is complete: a case may compile once per
// batch shape, or tabulate a profile on the announced points, and a call
// anywhere else would then be either slow or wrong. So the test that matters
// here records every call a case is actually handed -- each batched entry with
// its abscissae, and the positions of the pointwise hooks the plan lists -- and
// requires both directions: every call was announced, and every announced
// batched site was actually used. A new physics call in the solver that is not
// added to SystemSolver::evaluationPlan fails the first half; a site the solver
// stopped using fails the second.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "DegreeAdaptation.hpp"
#include "MeshAdaptation.hpp"
#include "SolverConfig.hpp"
#include "SystemSolver.hpp"
#include "TransportSystem.hpp"
#include "gridStructures.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <toml.hpp>
#include <vector>

namespace
{
struct Call
{
    EvaluationEntry entry;
    std::vector<Position> points;
};

std::string name(EvaluationEntry e)
{
    switch (e)
    {
    case EvaluationEntry::ComputePhysics: return "ComputePhysics";
    case EvaluationEntry::ComputePhysicsDerivatives: return "ComputePhysicsDerivatives";
    case EvaluationEntry::ComputeSourceTimeDerivatives: return "ComputeSourceTimeDerivatives";
    case EvaluationEntry::ScalarG: return "ScalarG";
    case EvaluationEntry::ScalarGPrime: return "ScalarGPrime";
    case EvaluationEntry::InitialScalarDerivative: return "InitialScalarDerivative";
    case EvaluationEntry::Pointwise: return "Pointwise";
    }
    return "?";
}

// What a recording case optionally carries beyond plain diffusion: a source that
// reads du/dt, and a global scalar the source depends on. Each adds sites.
struct Shape
{
    bool readsUdot = false;
    bool scalar = false;
};

SystemSpec specFor(Shape shape)
{
    SystemSpec spec{.variables = numberedFields(1)};
    spec.variables[0].sourceReadsTimeDerivatives = shape.readsUdot;
    if (shape.scalar)
        spec.scalars = numberedScalars(1);
    return spec;
}

// sigma = (1 + u^2/4) q, S = 1 + mu [+ udot / 10], zero Dirichlet ends, and with
// a scalar the algebraic constraint mu = 1/2. Nonlinear, so a steady solve takes
// more than one Newton step and every per-step site comes round more than once.
//
// It records every batched call by overriding the batched entry points and
// passing straight through to the base, which is the pointwise fallback -- so
// the physics is unchanged by the recording.
class RecordingCase : public TransportSystem
{
public:
    explicit RecordingCase(Shape shape = {}) : TransportSystem(specFor(shape)), shape(shape) {}

    Shape shape;
    std::vector<Call> calls;
    mutable std::vector<Position> initialPoints; // InitialValue, InitialDerivative
    std::vector<Position> massPoints;            // aFn
    std::vector<Position> scalarCouplingPoints;  // dSources_dScalars
    std::vector<EvaluationPlan> plans;
    // How many evaluations of any kind had happened when each plan arrived.
    std::vector<size_t> evaluationsBeforePlan;

    size_t evaluations() const
    {
        return calls.size() + initialPoints.size() + massPoints.size() +
               scalarCouplingPoints.size();
    }

    void prepareEvaluation(EvaluationPlan const &plan) override
    {
        evaluationsBeforePlan.push_back(evaluations());
        plans.push_back(plan);
    }

    PhysicsOutput ComputePhysics(GlobalState const &states, std::vector<Position> const &x,
                                 Time t) override
    {
        calls.push_back({EvaluationEntry::ComputePhysics, x});
        return TransportSystem::ComputePhysics(states, x, t);
    }
    void ComputePhysicsDerivatives(std::array<std::reference_wrapper<GlobalStateMatrix>,
                                              NPHYSICS_FUNCTIONS> &&out,
                                   GlobalState const &states, std::vector<Position> const &x,
                                   Time t) override
    {
        calls.push_back({EvaluationEntry::ComputePhysicsDerivatives, x});
        TransportSystem::ComputePhysicsDerivatives(std::move(out), states, x, t);
    }
    void ComputeSourceTimeDerivatives(GlobalStateMatrix &out, GlobalState const &states,
                                      std::vector<Position> const &x, Time t) override
    {
        calls.push_back({EvaluationEntry::ComputeSourceTimeDerivatives, x});
        TransportSystem::ComputeSourceTimeDerivatives(out, states, x, t);
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value aFn(Index, Position x) override
    {
        massPoints.push_back(x);
        return 1.0;
    }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return (1.0 + 0.25 * s.u(0) * s.u(0)) * s.q(0);
    }
    void dSigmaFn_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 0.5 * s.u(0) * s.q(0);
    }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 1.0 + 0.25 * s.u(0) * s.u(0);
    }

    Value Sources(Index, const State &s, Position, Time) override
    {
        return 1.0 + (shape.scalar ? s.scalar(0) : 0.0) + (shape.readsUdot ? 0.1 * s.udot(0) : 0.0);
    }
    void dSources_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dudot(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.1;
    }

    Value InitialValue(Index, Position x) const override
    {
        initialPoints.push_back(x);
        return 0.5 * x * (1.0 - x);
    }
    Value InitialDerivative(Index, Position x) const override
    {
        initialPoints.push_back(x);
        return 0.5 - x;
    }

    // mu = 1/2, algebraic.
    Value InitialScalarValue(Index) const override { return 0.5; }
    Value ScalarG(Index, GlobalState const &y, GlobalState const &, std::vector<Position> const &x,
                  Values const &, Matrix const &, Time) override
    {
        calls.push_back({EvaluationEntry::ScalarG, x});
        return y.Scalars()(0) - 0.5;
    }
    void ScalarGPrime(GlobalStateMatrix &dG, GlobalStateMatrix &, GlobalState const &,
                      GlobalState const &, std::vector<Position> const &x, Values const &,
                      Matrix const &, Time) override
    {
        calls.push_back({EvaluationEntry::ScalarGPrime, x});
        dG[0].Scalars()(0) = 1.0;
    }
    void dSources_dScalars(Index, VectorRef v, const State &, Position x, Time) override
    {
        scalarCouplingPoints.push_back(x);
        v[0] = 1.0;
    }
};

// G = int u^2 dx, for an adjoint solve to make its own calls.
class SquareObjective : public AdjointProblem
{
public:
    SquareObjective()
    {
        ng = 1;
        np = 1;
    }
    Value GFn(Index, DGSoln &) const override { return 0.0; }
    Value dGFndp(Index, Index, DGSoln &) const override { return 0.0; }
    Value gFn(Index, const State &s, Position) const override { return s.u(0) * s.u(0); }
    void dgFn_du(Index, VectorRef v, const State &s, Position) override { v[0] = 2.0 * s.u(0); }
    void dgFn_dq(Index, VectorRef, const State &, Position) override {}
    void dgFn_dsigma(Index, VectorRef, const State &, Position) override {}
    void dgFn_dphi(Index, VectorRef, const State &, Position) override {}
    void dSigmaFn_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }
    void dSources_dp(Index, Index, Value &out, const State &, Position) override { out = 1.0; }
};

struct Run
{
    std::string label;
    bool superconvergent = false;
    bool steady = false;
    std::optional<SystemSolver::TauUpdate> diffusive; // empty: a constant tau
    Shape shape;
    bool adjoint = false;
};

constexpr Index nCells = 5, k = 4;

void configure(SystemSolver &sys, Run const &run, AdjointProblem *adjoint)
{
    sys.setTau(1.0);
    sys.setInputFile("evaluation_plan_" + run.label);
    sys.setWriteOutput(false);
    sys.setOutputCadence(0.05);
    sys.setNOutput(2);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-12);
    sys.setTolerances({1e-8}, 1e-6);
    sys.setSuperconvergent(run.superconvergent);
    if (run.diffusive)
        sys.setTauScaling(SystemSolver::TauScaling::Diffusive, 1e-3, *run.diffusive);
    if (run.steady)
    {
        sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
        sys.setSteadyStateTolerance(1e-9);
    }
    if (adjoint != nullptr)
    {
        sys.setAdjointProblem(adjoint);
        sys.setSolveAdjoint(true);
    }
}

bool sameSet(std::vector<Position> a, std::vector<Position> b)
{
    std::sort(a.begin(), a.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
    std::sort(b.begin(), b.end());
    b.erase(std::unique(b.begin(), b.end()), b.end());
    return a == b;
}

// The two directions, for one run's plan against one run's calls. The second --
// every batched site used -- holds only of a run that does some work at every
// cadence it announces: a steady solve that starts converged never builds a
// Jacobian, and the plan announcing one is still right.
void checkAgainstPlan(EvaluationPlan const &plan, RecordingCase const &problem,
                      bool everySiteUsed = true)
{
    for (Call const &c : problem.calls)
        BOOST_TEST(plan.announces(c.entry, c.points),
                   name(c.entry) << " was called at " << c.points.size()
                                 << " points the plan did not announce");

    for (EvaluationSite const &site : plan.sites)
    {
        if (!everySiteUsed || !site.isBatched())
            continue;
        // The C++ hook takes a DGSoln, so there are no positions to record.
        if (site.entry == EvaluationEntry::InitialScalarDerivative)
            continue;
        const bool seen = std::any_of(problem.calls.begin(), problem.calls.end(),
                                      [&](Call const &c)
                                      { return c.entry == site.entry && c.points == site.points; });
        BOOST_TEST(seen, toString(site.kind) << " through " << name(site.entry)
                                             << " was announced and never called");
    }

    auto pointwise = [&](EvaluationKind kind, std::vector<Position> const &seen)
    {
        if (plan.has(kind))
            BOOST_TEST(sameSet(plan.points(kind), seen),
                       toString(kind) << ": the hook was called at points the plan does not list, or not at all");
        else
            BOOST_TEST(seen.empty(), toString(kind) << " was evaluated without being announced");
    };
    pointwise(EvaluationKind::InitialProjection, problem.initialPoints);
    pointwise(EvaluationKind::MassMatrix, problem.massPoints);
    pointwise(EvaluationKind::ScalarCoupling, problem.scalarCouplingPoints);
}
} // namespace

BOOST_AUTO_TEST_SUITE(evaluation_plan_tests)

BOOST_AUTO_TEST_CASE(every_call_is_announced_and_every_announced_site_is_used)
{
    using U = SystemSolver::TauUpdate;
    const std::vector<Run> runs = {
        {.label = "march"},
        {.label = "march_superconvergent", .superconvergent = true},
        {.label = "march_diffusive", .diffusive = U::Residual},
        {.label = "march_udot", .shape = {.readsUdot = true}},
        {.label = "march_scalar", .shape = {.scalar = true}},
        {.label = "march_adjoint", .adjoint = true},
        {.label = "steady", .steady = true},
        {.label = "steady_superconvergent_diffusive", .superconvergent = true, .steady = true,
         .diffusive = U::Residual},
        {.label = "steady_continuation_tau", .steady = true, .diffusive = U::ContinuationStep},
        {.label = "steady_jacobian_tau", .superconvergent = true, .steady = true,
         .diffusive = U::JacobianBuild},
        {.label = "steady_scalar_superconvergent", .superconvergent = true, .steady = true,
         .shape = {.scalar = true}},
        {.label = "steady_adjoint_superconvergent", .superconvergent = true, .steady = true,
         .adjoint = true},
    };

    for (Run const &run : runs)
    {
        BOOST_TEST_CONTEXT(run.label)
        {
            Grid grid(std::vector<Position>{0.0, 0.1, 0.3, 0.6, 0.8, 1.0});
            RecordingCase problem(run.shape);
            SquareObjective objective;
            SystemSolver sys(grid, k, &problem);
            configure(sys, run, run.adjoint ? &objective : nullptr);

            {
                CapturedOutput quiet;
                sys.runSolver(0.1);
            }

            BOOST_TEST_REQUIRE(problem.plans.size() == 1u);
            EvaluationPlan const &plan = problem.plans.front();
            BOOST_TEST(problem.evaluationsBeforePlan.front() == 0u);
            BOOST_TEST(problem.evaluationPlan() != nullptr);

            BOOST_TEST((plan.grid == grid));
            BOOST_TEST(plan.k == k);
            BOOST_TEST(plan.superconvergent == run.superconvergent);
            BOOST_TEST(plan.steady == run.steady);

            const Index perCell = run.superconvergent ? k + 2 : k + 1;
            BOOST_TEST(plan.points(EvaluationKind::Residual).size() == size_t(nCells * perCell));
            BOOST_TEST(plan.points(EvaluationKind::Jacobian).size() == size_t(nCells * perCell));
            BOOST_TEST(plan.has(EvaluationKind::TauFaces) == run.diffusive.has_value());
            if (run.diffusive)
                BOOST_TEST(plan.points(EvaluationKind::TauFaces).size() == size_t(2 * nCells));

            checkAgainstPlan(plan, problem);
        }
    }
}

BOOST_AUTO_TEST_CASE(a_reused_solver_delivers_a_fresh_plan_each_run)
{
    // Per run, because what the plan says moves between runs on one solver: the
    // mass matrix is integrated only by the first, so aFn is announced only
    // there. And the second run's calls are still all announced -- by the plan
    // that run was handed, not the first.
    Grid grid(0.0, 1.0, nCells);
    RecordingCase problem;
    SystemSolver sys(grid, k, &problem);
    configure(sys, {.label = "reused"}, nullptr);

    {
        CapturedOutput quiet;
        sys.runSolver(0.1);
    }
    BOOST_TEST_REQUIRE(problem.plans.size() == 1u);
    BOOST_TEST(problem.plans[0].has(EvaluationKind::MassMatrix));

    problem.calls.clear();
    problem.initialPoints.clear();
    problem.massPoints.clear();
    {
        CapturedOutput quiet;
        sys.runSolver(0.1);
    }
    BOOST_TEST_REQUIRE(problem.plans.size() == 2u);
    BOOST_TEST(!problem.plans[1].has(EvaluationKind::MassMatrix));
    BOOST_TEST(problem.evaluationsBeforePlan[1] == 0u);
    checkAgainstPlan(problem.plans[1], problem);
}

BOOST_AUTO_TEST_CASE(the_plan_follows_a_restart)
{
    // A restart copied at the same discretisation needs neither the projection
    // of the initial values nor AssignSigma, so neither is announced; a steady
    // solve skips the initial du/dt solve too, which leaves the initial
    // condition no evaluation at all. Resumed from a converged state, the solve
    // stops at its first residual, so the Jacobian it announces is never built.
    Grid grid(0.0, 1.0, nCells);
    RecordingCase first;
    SystemSolver a(grid, k, &first);
    configure(a, {.label = "restart_a", .steady = true}, nullptr);
    {
        CapturedOutput quiet;
        a.runSolver(0.1);
    }

    RecordingCase problem;
    problem.setRestartValues(a.stateVector(), a.derivativeVector(), grid, k);
    SystemSolver b(grid, k, &problem);
    configure(b, {.label = "restart_b", .steady = true}, nullptr);
    {
        CapturedOutput quiet;
        b.runSolver(0.1);
    }

    BOOST_TEST_REQUIRE(problem.plans.size() == 1u);
    EvaluationPlan const &plan = problem.plans.front();
    BOOST_TEST(!plan.has(EvaluationKind::InitialProjection));
    BOOST_TEST(!plan.has(EvaluationKind::InitialCondition));
    checkAgainstPlan(plan, problem, false);
}

BOOST_AUTO_TEST_CASE(batch_sizes_are_the_shapes_to_compile_for)
{
    // What a case compiled per shape asks the plan for. Five cells at k = 4,
    // superconvergent with a Diffusive tau: the residual and Jacobian on the 30
    // star nodes, the initial condition on the 25 basis nodes, the faces on 10.
    Grid grid(0.0, 1.0, nCells);
    RecordingCase problem;
    SystemSolver sys(grid, k, &problem);
    configure(sys, {.label = "shapes", .superconvergent = true,
                    .diffusive = SystemSolver::TauUpdate::Residual},
              nullptr);
    const EvaluationPlan plan = sys.evaluationPlan();

    BOOST_TEST((plan.batchSizes(EvaluationEntry::ComputePhysics) == std::vector<Index>{25, 30}));
    BOOST_TEST((plan.batchSizes(EvaluationEntry::ComputePhysicsDerivatives) == std::vector<Index>{10, 30}));
    BOOST_TEST(plan.batchSizes(EvaluationEntry::ScalarG).empty());

    // The face sites, and how often each comes round: one call per residual,
    // 1 + 3 nVars + nAux per Jacobian build for the forward differences, and one
    // for the initial du/dt solve of a time march.
    std::vector<EvaluationSite> faces = plan.sitesOf(EvaluationKind::TauFaces);
    BOOST_TEST_REQUIRE(faces.size() == 3u);
    BOOST_TEST((faces[0].cadence == EvaluationCadence::PerResidual));
    BOOST_TEST(faces[0].calls == 1);
    BOOST_TEST((faces[1].cadence == EvaluationCadence::PerJacobianBuild));
    BOOST_TEST(faces[1].calls == 4);
    BOOST_TEST((faces[2].cadence == EvaluationCadence::OncePerRun));

    // A plan is the solver's statement about itself, and building one has no
    // effect on the solver or the case.
    BOOST_TEST(problem.plans.empty());
    BOOST_TEST(problem.evaluations() == 0u);
}

// ------------------------------------------------------------- regridding ----

namespace
{
// u = x - x^(4/3) at steady state: -u'' = (4/9) x^(-2/3) with zero Dirichlet
// ends, singular at the axis, so MeshAdaptation grades towards x = 0. The same
// problem python/Tests/test_mesh_adaptation.py's AxisSingular solves.
//
// Records every regrid it is told about, and the grid each plan names.
class AxisSingular : public TransportSystem
{
public:
    explicit AxisSingular(bool regriddable)
        : TransportSystem(SystemSpec{.variables = numberedFields(1), .supportsRegrid = regriddable})
    {
    }

    struct Regrid
    {
        Grid grid;
        Index k;
        Grid planGrid;
        size_t plansBefore;
    };
    std::vector<Regrid> regrids;
    std::vector<Grid> planGrids;

    void regrid(Grid const &grid, Index k, EvaluationPlan const &plan) override
    {
        regrids.push_back({grid, k, plan.grid, planGrids.size()});
    }
    void prepareEvaluation(EvaluationPlan const &plan) override { planGrids.push_back(plan.grid); }

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

SolverConfig adaptiveConfig(std::string const &extra)
{
    const std::string body =
        "PolynomialDegree = 4\nGridSize = 10\ndelta_t = 0.1\nt_final = 1.0\n"
        "LowerBoundary = 0.0\nUpperBoundary = 1.0\n"
        "TransportSystem = \"LinearDiffusion\"\nWriteOutput = false\n"
        "SteadyStateSolver = \"Newton\"\nSteadyStateTolerance = 1e-11\n"
        "Absolute_tolerance = 1e-10\nMinStepSize = 1e-12\n" + extra;
    auto v = toml::parse_str(body);
    TomlConfigSource src(v);
    return loadSolverConfig(src, ConfigSchema::Reader::Toml);
}
} // namespace

BOOST_AUTO_TEST_CASE(mesh_adaptation_tells_a_regriddable_case_about_the_graded_mesh)
{
    // The sample solve is on the mesh the case was built for, so it is no
    // regrid. The graded mesh is, and the case hears about it -- with that mesh,
    // the degree, and a plan for exactly that solve -- before the solve's own
    // plan arrives, and so before anything is evaluated there.
    const SolverConfig config = adaptiveConfig("MeshAdaptation = true\nDegreeTolerance = 1e-2\n");
    Grid uniform(0.0, 1.0, 10);
    AxisSingular problem(true);

    std::optional<AdaptiveMeshResult> result;
    {
        CapturedOutput quiet;
        result.emplace(runAdaptiveMesh(config, problem, nullptr, uniform, 4, 1.0));
    }

    BOOST_TEST_REQUIRE((result->decision.verdict == GradingVerdict::GradeLower));
    BOOST_TEST_REQUIRE(problem.regrids.size() == 1u);

    auto const &r = problem.regrids.front();
    BOOST_TEST((r.grid == *result->grid));
    BOOST_TEST((r.grid != uniform));
    BOOST_TEST(r.grid.getNCells() == uniform.getNCells());
    BOOST_TEST(r.k == 4);
    BOOST_TEST((r.planGrid == r.grid));

    // One plan for the sample, then the regrid, then the graded solve's plan.
    BOOST_TEST(r.plansBefore == 1u);
    BOOST_TEST_REQUIRE(problem.planGrids.size() >= 2u);
    BOOST_TEST((problem.planGrids[0] == uniform));
    BOOST_TEST((problem.planGrids[1] == r.grid));
}

BOOST_AUTO_TEST_CASE(a_case_that_does_not_declare_it_is_reused_without_a_regrid)
{
    // The same sequence, and the same answer, with nothing told -- which is the
    // behaviour every case in the tree relies on. Its plans still name the
    // graded mesh, because a plan arrives with every solve whatever the spec says.
    const SolverConfig config = adaptiveConfig("MeshAdaptation = true\nDegreeTolerance = 1e-2\n");
    Grid uniform(0.0, 1.0, 10);
    AxisSingular problem(false);

    std::optional<AdaptiveMeshResult> result;
    {
        CapturedOutput quiet;
        result.emplace(runAdaptiveMesh(config, problem, nullptr, uniform, 4, 1.0));
    }

    BOOST_TEST_REQUIRE((result->decision.verdict == GradingVerdict::GradeLower));
    BOOST_TEST(problem.regrids.empty());
    BOOST_TEST_REQUIRE(problem.planGrids.size() >= 2u);
    BOOST_TEST((problem.planGrids[1] == *result->grid));
}

BOOST_AUTO_TEST_CASE(a_grid_ladder_regrids_each_rung_and_moves_the_case_back)
{
    // Built against the final mesh, solved first on a coarser one: a regrid onto
    // the rung, and one back for the last solve. A DegreeLadder rung on the same
    // mesh is no regrid at all.
    const SolverConfig config = adaptiveConfig("GridLadder = [5]\nDegreeLadder = [2]\n");
    Grid fine(0.0, 1.0, 10);
    AxisSingular problem(true);

    {
        CapturedOutput quiet;
        auto system = runLadder(config, problem, nullptr, fine, 4, 1.0);
    }

    BOOST_TEST_REQUIRE(problem.regrids.size() == 2u);
    BOOST_TEST(problem.regrids[0].grid.getNCells() == 5u);
    BOOST_TEST(problem.regrids[0].k == 2);
    BOOST_TEST((problem.regrids[1].grid == fine));
    BOOST_TEST(problem.regrids[1].k == 4);
}

BOOST_AUTO_TEST_CASE(moving_a_case_that_does_not_declare_it_off_its_domain_is_refused)
{
    // The one property of the grid a constructor in this tree reads is the
    // domain, so that is the one a driver must not change behind a case's back.
    // No driver does; this is what stops one starting.
    Grid unit(0.0, 1.0, 4);
    AxisSingular fixed(false), moving(true);

    SystemSolver sameDomain(Grid(std::vector<Position>{0.0, 0.1, 0.5, 1.0}), 2, &fixed);
    SystemSolver otherDomain(Grid(0.0, 2.0, 4), 2, &fixed);

    Grid current = unit;
    BOOST_CHECK_NO_THROW(moveCaseToMesh(fixed, sameDomain, current));
    BOOST_TEST((current == sameDomain.getGrid()));
    BOOST_TEST(fixed.regrids.empty());

    current = unit;
    BOOST_CHECK_THROW(moveCaseToMesh(fixed, otherDomain, current), std::invalid_argument);
    BOOST_TEST((current == unit));

    // A case that declares it may go anywhere, and is told.
    SystemSolver elsewhere(Grid(0.0, 2.0, 4), 3, &moving);
    current = unit;
    BOOST_CHECK_NO_THROW(moveCaseToMesh(moving, elsewhere, current));
    BOOST_TEST_REQUIRE(moving.regrids.size() == 1u);
    BOOST_TEST(moving.regrids[0].k == 3);
    BOOST_TEST((moving.regrids[0].grid == Grid(0.0, 2.0, 4)));

    // And the same mesh again is nothing to do.
    BOOST_CHECK_NO_THROW(moveCaseToMesh(moving, elsewhere, current));
    BOOST_TEST(moving.regrids.size() == 1u);
}

BOOST_AUTO_TEST_SUITE_END()
