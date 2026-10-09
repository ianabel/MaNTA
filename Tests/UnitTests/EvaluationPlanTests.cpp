// The evaluation plan (EvaluationPlan.hpp) and regridding (RegridPolicy, PhysicsInstance).
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
#include "PhysicsCases.hpp"
#include "PhysicsInstance.hpp"
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
    SystemSolver::TauKappa kappa = SystemSolver::TauKappa::Nodal;
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
        sys.setTauScaling(SystemSolver::TauScaling::Diffusive, 1e-3, *run.diffusive, run.kappa);
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
        {
            BOOST_TEST(plan.announces(EvaluationEntry::Pointwise, seen),
                       toString(kind) << ": the hook was called at points the plan does not list");
            if (everySiteUsed)
                BOOST_TEST(sameSet(plan.points(kind), seen),
                           toString(kind) << ": announced points were never visited");
        }
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
    constexpr auto Face = SystemSolver::TauKappa::Face;
    const std::vector<Run> runs = {
        {.label = "march"},
        {.label = "march_superconvergent", .superconvergent = true},
        {.label = "march_diffusive", .diffusive = U::Residual},
        {.label = "march_diffusive_face", .diffusive = U::Residual, .kappa = Face},
        {.label = "march_udot", .shape = {.readsUdot = true}},
        {.label = "march_scalar", .shape = {.scalar = true}},
        {.label = "march_adjoint", .adjoint = true},
        {.label = "steady", .steady = true},
        {.label = "steady_superconvergent_diffusive", .superconvergent = true, .steady = true,
         .diffusive = U::Residual},
        {.label = "steady_superconvergent_diffusive_face", .superconvergent = true, .steady = true,
         .diffusive = U::Residual, .kappa = Face},
        {.label = "steady_continuation_tau", .steady = true, .diffusive = U::ContinuationStep},
        {.label = "steady_continuation_tau_face", .steady = true, .diffusive = U::ContinuationStep,
         .kappa = Face},
        {.label = "steady_jacobian_tau", .superconvergent = true, .steady = true,
         .diffusive = U::JacobianBuild},
        {.label = "steady_jacobian_tau_face", .superconvergent = true, .steady = true,
         .diffusive = U::JacobianBuild, .kappa = Face},
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
            // A nodal tau reads the physics nodes and adds no point set; a face
            // tau adds the 2 nCells face points.
            const bool faceTau = run.diffusive && run.kappa == SystemSolver::TauKappa::Face;
            const bool nodalTau = run.diffusive && run.kappa == SystemSolver::TauKappa::Nodal;
            BOOST_TEST(plan.has(EvaluationKind::TauFaces) == faceTau);
            BOOST_TEST(plan.has(EvaluationKind::TauNodes) == nodalTau);
            if (faceTau)
                BOOST_TEST(plan.points(EvaluationKind::TauFaces).size() == size_t(2 * nCells));
            if (nodalTau)
                BOOST_TEST((plan.points(EvaluationKind::TauNodes) ==
                            plan.points(EvaluationKind::Residual)));

            // One set of cell points per level: the initial condition samples
            // where the residual does, so a superconvergent level never asks for
            // the k+1 basis nodes through ComputePhysics.
            BOOST_TEST((plan.points(EvaluationKind::InitialCondition) ==
                        plan.points(EvaluationKind::Residual)));
            for (Call const &c : problem.calls)
                if (c.entry == EvaluationEntry::ComputePhysics)
                    BOOST_TEST((c.points == plan.points(EvaluationKind::Residual)));

            checkAgainstPlan(plan, problem);
        }
    }
}

BOOST_AUTO_TEST_CASE(a_rerun_with_an_equal_plan_tells_the_case_nothing)
{
    // A plan does not depend on the run, so a second run on the same solver --
    // which builds no mass matrix -- and a run on a second solver configured the
    // same way are the same plan, and the case hears nothing. Its calls are still
    // all announced, by the one plan it has. And a case whose RegridPolicy is
    // Fixed runs every time: nothing here changes its plan.
    Grid grid(0.0, 1.0, nCells);
    RecordingCase problem;
    BOOST_TEST((problem.regridPolicy() == RegridPolicy::Fixed));
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
    BOOST_TEST(problem.plans.size() == 1u);
    BOOST_TEST(problem.massPoints.empty());
    checkAgainstPlan(problem.plans[0], problem, false);

    SystemSolver again(grid, k, &problem);
    configure(again, {.label = "reused_again"}, nullptr);
    BOOST_TEST((again.evaluationPlan() == problem.plans[0]));
    {
        CapturedOutput quiet;
        BOOST_CHECK_NO_THROW(again.runSolver(0.1));
    }
    BOOST_TEST(problem.plans.size() == 1u);
}

BOOST_AUTO_TEST_CASE(a_fixed_case_is_refused_a_changed_plan)
{
    // The same instance on another degree is a regrid, which a Fixed case cannot
    // follow: refused before the solver evaluates anything, and the case keeps
    // the one plan it had.
    Grid grid(0.0, 1.0, nCells);
    RecordingCase problem;
    SystemSolver first(grid, k, &problem);
    configure(first, {.label = "fixed_first"}, nullptr);
    {
        CapturedOutput quiet;
        first.runSolver(0.1);
    }

    SystemSolver second(grid, k - 1, &problem);
    configure(second, {.label = "fixed_second"}, nullptr);
    problem.calls.clear();
    {
        CapturedOutput quiet;
        BOOST_CHECK_THROW(second.runSolver(0.1), std::invalid_argument);
    }
    BOOST_TEST(problem.plans.size() == 1u);
    BOOST_TEST(problem.calls.empty());
}

BOOST_AUTO_TEST_CASE(a_restart_is_not_a_change_of_plan)
{
    // A restart copied at the same discretisation makes neither the projection
    // of the initial values nor AssignSigma's sweep, and a steady solve skips the
    // initial du/dt solve too -- but the plan lists them all the same, as upper
    // bounds, so that it is the cold start's plan exactly. Resumed from a
    // converged state, the solve stops at its first residual, so the Jacobian it
    // announces is never built either.
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
    BOOST_TEST((plan == first.plans.front()));
    BOOST_TEST(plan.has(EvaluationKind::InitialProjection));
    BOOST_TEST(problem.initialPoints.empty());
    checkAgainstPlan(plan, problem, false);
}

BOOST_AUTO_TEST_CASE(batch_sizes_are_the_shapes_to_compile_for)
{
    // What a case compiled per shape asks the plan for. Five cells at k = 4,
    // superconvergent with a Diffusive tau read at the faces: the residual, the
    // Jacobian and the initial condition on the 30 star nodes, the faces on 10.
    Grid grid(0.0, 1.0, nCells);
    RecordingCase problem;
    SystemSolver sys(grid, k, &problem);
    configure(sys, {.label = "shapes", .superconvergent = true,
                    .diffusive = SystemSolver::TauUpdate::Residual,
                    .kappa = SystemSolver::TauKappa::Face},
              nullptr);
    const EvaluationPlan plan = sys.evaluationPlan();

    BOOST_TEST((plan.batchSizes(EvaluationEntry::ComputePhysics) == std::vector<Index>{30}));
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

    // The same with kappa read at the nodes: one shape, the star nodes, and
    // the same three cadences -- without the 1 + in the Jacobian build's count,
    // since kappa there comes off the build's own derivatives.
    SystemSolver nodal(grid, k, &problem);
    configure(nodal, {.label = "shapes_nodal", .superconvergent = true,
                      .diffusive = SystemSolver::TauUpdate::Residual},
              nullptr);
    const EvaluationPlan nodalPlan = nodal.evaluationPlan();
    BOOST_TEST((nodalPlan.batchSizes(EvaluationEntry::ComputePhysicsDerivatives) ==
                std::vector<Index>{30}));
    BOOST_TEST(!nodalPlan.has(EvaluationKind::TauFaces));
    std::vector<EvaluationSite> nodes = nodalPlan.sitesOf(EvaluationKind::TauNodes);
    BOOST_TEST_REQUIRE(nodes.size() == 3u);
    BOOST_TEST((nodes[0].cadence == EvaluationCadence::PerResidual));
    BOOST_TEST(nodes[0].calls == 1);
    BOOST_TEST((nodes[1].cadence == EvaluationCadence::PerJacobianBuild));
    BOOST_TEST(nodes[1].calls == 3);
    BOOST_TEST((nodes[2].cadence == EvaluationCadence::OncePerRun));

    // A plan is the solver's statement about itself, and building one has no
    // effect on the solver or the case.
    BOOST_TEST(problem.plans.empty());
    BOOST_TEST(problem.evaluations() == 0u);
}

// ------------------------------------------------------------- regridding ----

namespace
{
int liveAxisCases = 0;
bool anAxisCaseSawTwoPlans = false;

// u = x - x^(4/3) at steady state: -u'' = (4/9) x^(-2/3) with zero Dirichlet
// ends, singular at the axis, so MeshAdaptation grades towards x = 0. The same
// problem python/Tests/test_mesh_adaptation.py's AxisSingular solves.
//
// Records the plans it is handed, counts the instances alive, and notes whether
// any Fixed instance was ever handed a second plan -- which is what a rebuild
// exists to prevent.
class AxisSingular : public TransportSystem
{
public:
    explicit AxisSingular(RegridPolicy policy)
        : TransportSystem(SystemSpec{.variables = numberedFields(1), .regrid = policy})
    {
        ++liveAxisCases;
    }
    ~AxisSingular() override { --liveAxisCases; }

    std::vector<EvaluationPlan> plans;

    void prepareEvaluation(EvaluationPlan const &plan) override
    {
        plans.push_back(plan);
        if (regridPolicy() == RegridPolicy::Fixed && plans.size() > 1)
            anAxisCaseSawTwoPlans = true;
    }

    std::unique_ptr<AdjointProblem> createAdjointProblem() override;

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

// The objective a case hands out, remembering which case it came from -- the
// way AutodiffAdjointProblem holds a pointer back to its PhysicsProblem.
class OwnedObjective : public SquareObjective
{
public:
    explicit OwnedObjective(TransportSystem const *owner) : owner(owner) {}
    TransportSystem const *owner;
};

std::unique_ptr<AdjointProblem> AxisSingular::createAdjointProblem()
{
    return std::make_unique<OwnedObjective>(this);
}

// A Fixed AxisSingular by name, so a rebuild goes through the registry as
// runManta's and a named Runner's do.
constexpr char axisName[] = "EvaluationPlanTestsFixedAxis";
void registerAxis()
{
    static bool done = false;
    if (done)
        return;
    PhysicsCases::RegisterPhysicsCase(axisName, [](toml::value const &, Grid const &)
                                      { return std::make_unique<AxisSingular>(RegridPolicy::Fixed); });
    done = true;
}

PhysicsInstance::Rebuild fromRegistry()
{
    registerAxis();
    return [](Grid const &g) -> std::shared_ptr<TransportSystem>
    { return PhysicsCases::InstantiateProblem(axisName, toml::value{}, g); };
}

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

const std::string meshAdaptation = "MeshAdaptation = true\nDegreeTolerance = 1e-2\n";
// Tight enough that the loop raises k at least once from 4.
const std::string degreeAdaptation =
    "DegreeAdaptation = true\nDegreeTolerance = 1e-6\nMaxPolynomialDegree = 6\n";

// Consecutive plans differ: a plan is delivered only when it changes.
bool allChanges(std::vector<EvaluationPlan> const &plans)
{
    for (size_t i = 1; i < plans.size(); ++i)
        if (plans[i] == plans[i - 1])
            return false;
    return true;
}

// The case and adjoint slots a rebuildable run works through, as runManta's.
struct Owned
{
    std::shared_ptr<TransportSystem> problem;
    std::unique_ptr<AdjointProblem> adjoint;
    std::weak_ptr<TransportSystem> original;

    explicit Owned(bool withAdjoint)
    {
        registerAxis();
        problem = PhysicsCases::InstantiateProblem(axisName, toml::value{}, Grid(0.0, 1.0, 10));
        if (withAdjoint)
            adjoint = problem->createAdjointProblem();
        original = problem;
    }
    AxisSingular &axis() const { return static_cast<AxisSingular &>(*problem); }
};
} // namespace

BOOST_AUTO_TEST_CASE(an_in_place_case_follows_mesh_and_degree_adaptation)
{
    // One instance throughout, told about each new plan once: the sample on the
    // mesh it was built for, the graded mesh, and each degree the loop raises to.
    for (std::string const &mode : {meshAdaptation, degreeAdaptation})
    {
        BOOST_TEST_CONTEXT(mode)
        {
            const SolverConfig config = adaptiveConfig(mode);
            Grid uniform(0.0, 1.0, 10);
            AxisSingular problem(RegridPolicy::InPlace);
            PhysicsInstance physics(problem, uniform);

            std::unique_ptr<SystemSolver> final;
            std::optional<AdaptiveMeshResult> mesh;
            {
                CapturedOutput quiet;
                if (config.MeshAdaptation)
                    mesh.emplace(runAdaptiveMesh(config, physics, uniform, 4, 1.0));
                else
                    final = runAdaptiveDegree(config, physics, uniform, 4, 1.0);
            }

            BOOST_TEST(&physics.problem() == &problem);
            BOOST_TEST(physics.rebuilds() == 0);
            BOOST_TEST_REQUIRE(problem.plans.size() >= 2u);
            BOOST_TEST(allChanges(problem.plans));
            BOOST_TEST((problem.plans.front().grid == uniform));
            BOOST_TEST(problem.plans.front().k == 4);

            if (mesh)
            {
                BOOST_TEST_REQUIRE((mesh->decision.verdict == GradingVerdict::GradeLower));
                BOOST_TEST((problem.plans[1].grid == *mesh->grid));
                BOOST_TEST((problem.plans.back().grid == *mesh->grid));
            }
            else
            {
                BOOST_TEST(problem.plans.back().k > 4);
                for (auto const &plan : problem.plans)
                    BOOST_TEST((plan.grid == uniform));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(rebuild_physics_on_regrid_replaces_a_fixed_case)
{
    // Under the key, a case that cannot follow a new plan is destroyed and built
    // again from the registry for each one. Every instance is handed exactly one
    // plan, the original is gone, and the adjoint problem the final solver uses
    // is the one the final instance handed out.
    for (std::string const &mode : {meshAdaptation, degreeAdaptation})
    {
        BOOST_TEST_CONTEXT(mode)
        {
            const SolverConfig config = adaptiveConfig(mode + "RebuildPhysicsOnRegrid = true\n");
            BOOST_TEST(config.RebuildPhysicsOnRegrid);
            Grid uniform(0.0, 1.0, 10);
            anAxisCaseSawTwoPlans = false;

            Owned slots(true);
            PhysicsInstance physics(slots.problem, slots.adjoint, uniform, fromRegistry());

            std::unique_ptr<SystemSolver> final;
            std::optional<AdaptiveMeshResult> mesh;
            {
                CapturedOutput quiet;
                if (config.MeshAdaptation)
                    mesh.emplace(runAdaptiveMesh(config, physics, uniform, 4, 1.0));
                else
                    final = runAdaptiveDegree(config, physics, uniform, 4, 1.0);
            }
            SystemSolver &solver = mesh ? *mesh->solver : *final;

            BOOST_TEST(physics.rebuilds() >= 1);
            BOOST_TEST(slots.original.expired());
            BOOST_TEST(liveAxisCases == 1);
            BOOST_TEST(!anAxisCaseSawTwoPlans);
            BOOST_TEST(slots.axis().plans.size() == 1u);

            BOOST_TEST(solver.problem == slots.problem.get());
            BOOST_TEST(solver.adjointProblem == slots.adjoint.get());
            BOOST_TEST(static_cast<OwnedObjective *>(slots.adjoint.get())->owner ==
                       slots.problem.get());
            if (mesh)
                BOOST_TEST((slots.axis().plans.front().grid == *mesh->grid));
        }
    }
}

BOOST_AUTO_TEST_CASE(a_rebuilt_case_starts_from_where_the_old_one_finished)
{
    // The warm start a level hands on is set on the instance it was solved with,
    // so a rebuild has to carry it over or the next level starts cold. The same
    // problem adapted in place and by rebuilding then takes the same steps and
    // gives the same answer, bit for bit.
    const SolverConfig inPlaceConfig = adaptiveConfig(degreeAdaptation);
    const SolverConfig rebuildConfig =
        adaptiveConfig(degreeAdaptation + "RebuildPhysicsOnRegrid = true\n");
    Grid uniform(0.0, 1.0, 10);

    AxisSingular inPlace(RegridPolicy::InPlace);
    PhysicsInstance borrowed(inPlace, uniform);
    Owned slots(false);
    PhysicsInstance owned(slots.problem, slots.adjoint, uniform, fromRegistry());

    std::unique_ptr<SystemSolver> a, b;
    {
        CapturedOutput quiet;
        a = runAdaptiveDegree(inPlaceConfig, borrowed, uniform, 4, 1.0);
        b = runAdaptiveDegree(rebuildConfig, owned, uniform, 4, 1.0);
    }
    BOOST_TEST(owned.rebuilds() >= 1);
    BOOST_TEST(a->getOrder() == b->getOrder());
    BOOST_TEST((a->stateVector() == b->stateVector()));
    BOOST_TEST(a->lastSteadyStats().steps == b->lastSteadyStats().steps);
}

BOOST_AUTO_TEST_CASE(a_fixed_case_without_the_key_is_refused_before_the_first_solve)
{
    // Neither InPlace nor RebuildPhysicsOnRegrid: refused up front, by every
    // driver that may change the plan, before the case is handed anything. A
    // rebuild function is not enough on its own -- the key is what supplies one,
    // and without it the caller passes none.
    const std::vector<std::string> modes = {meshAdaptation, degreeAdaptation,
                                            "GridLadder = [5]\n"};
    for (std::string const &mode : modes)
    {
        BOOST_TEST_CONTEXT(mode)
        {
            const SolverConfig config = adaptiveConfig(mode);
            Grid uniform(0.0, 1.0, 10);
            AxisSingular problem(RegridPolicy::Fixed);
            PhysicsInstance physics(problem, uniform);
            {
                CapturedOutput quiet;
                if (config.MeshAdaptation)
                    BOOST_CHECK_THROW(runAdaptiveMesh(config, physics, uniform, 4, 1.0),
                                      std::invalid_argument);
                else if (config.DegreeAdaptation)
                    BOOST_CHECK_THROW(runAdaptiveDegree(config, physics, uniform, 4, 1.0),
                                      std::invalid_argument);
                else
                    BOOST_CHECK_THROW(runLadder(config, physics, uniform, 4, 1.0),
                                      std::invalid_argument);
            }
            BOOST_TEST(problem.plans.empty());
        }
    }
}

BOOST_AUTO_TEST_CASE(a_fixed_case_adapts_where_the_plan_cannot_change)
{
    // DegreeAdaptation with no room above k0 never changes the plan, so a Fixed
    // case runs under it: refusing is for runs that would move the case.
    const SolverConfig config =
        adaptiveConfig("DegreeAdaptation = true\nDegreeTolerance = 1e-2\nMaxPolynomialDegree = 4\n");
    Grid uniform(0.0, 1.0, 10);
    AxisSingular problem(RegridPolicy::Fixed);
    PhysicsInstance physics(problem, uniform);
    std::unique_ptr<SystemSolver> sys;
    {
        CapturedOutput quiet;
        BOOST_CHECK_NO_THROW(sys = runAdaptiveDegree(config, physics, uniform, 4, 1.0));
    }
    BOOST_TEST(problem.plans.size() == 1u);
}

BOOST_AUTO_TEST_CASE(a_grid_ladder_moves_an_in_place_case_and_rebuilds_a_fixed_one)
{
    // Built against the final mesh, solved first on a coarser one at a lower
    // degree, then back: two changes of plan.
    Grid fine(0.0, 1.0, 10);
    {
        const SolverConfig config = adaptiveConfig("GridLadder = [5]\nDegreeLadder = [2]\n");
        AxisSingular problem(RegridPolicy::InPlace);
        PhysicsInstance physics(problem, fine);
        {
            CapturedOutput quiet;
            auto system = runLadder(config, physics, fine, 4, 1.0);
        }
        BOOST_TEST_REQUIRE(problem.plans.size() == 2u);
        BOOST_TEST(problem.plans[0].grid.getNCells() == 5u);
        BOOST_TEST(problem.plans[0].k == 2);
        BOOST_TEST((problem.plans[1].grid == fine));
        BOOST_TEST(problem.plans[1].k == 4);
    }
    {
        const SolverConfig config = adaptiveConfig(
            "GridLadder = [5]\nDegreeLadder = [2]\nRebuildPhysicsOnRegrid = true\n");
        anAxisCaseSawTwoPlans = false;
        Owned slots(false);
        PhysicsInstance physics(slots.problem, slots.adjoint, fine, fromRegistry());
        {
            CapturedOutput quiet;
            auto system = runLadder(config, physics, fine, 4, 1.0);
        }
        BOOST_TEST(physics.rebuilds() == 2);
        BOOST_TEST(!anAxisCaseSawTwoPlans);
        BOOST_TEST((slots.axis().plans.front().grid == fine));
    }
}

BOOST_AUTO_TEST_SUITE_END()
