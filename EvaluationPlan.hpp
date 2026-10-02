#ifndef EVALUATIONPLAN_HPP
#define EVALUATIONPLAN_HPP

#include "Types.hpp"
#include "gridStructures.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

/*
    Where, through which entry point and how often the solver will evaluate a
    physics case, stated before it evaluates it.

    A *plan* rather than a layout, because the points are only half of it. A JAX
    case wants the batch shapes it will be handed so it can compile for them; a
    case that precomputes x-dependent profiles wants the abscissae; and both want
    to know which of those happen once and which happen on every residual. The
    solver decides all three at once, from its grid, its degree, the
    Superconvergent flag and the tau scaling, so they travel together.

    It is a statement of what *can* happen, not a schedule. Each site has a
    cadence -- per residual, per Jacobian build, per continuation step, once per
    run -- and a count of calls per occasion where that count is fixed by the
    discretisation. How many residuals or Jacobian builds a run takes is decided
    by Newton and IDA as it goes, and nothing here pretends to know it.

    **A plan is a function of the discretisation and the configuration alone**,
    never of the run: not of a restart, and not of whether this is a solver's
    first run. A site that only some runs reach -- the mass matrix's aFn, built
    on a solver's first run; the initial-condition sweeps a copied restart skips
    -- is listed always, with its count as an upper bound. That is what lets
    `==` mean "this case is evaluated at the same points in the same way", which
    is the test for whether a case must be told anything at all.

    SystemSolver::evaluationPlan() builds one, and initialize() hands it to the
    case through TransportSystem::deliverEvaluationPlan before the first physics
    call of the run. See docs/physics_interface.rst, "Evaluation plans".
 */

/// What an evaluation is for. Each names the solver code that makes it.
enum class EvaluationKind
{
    Residual,          // residual(): fluxes, sources and aux constraints at the physics nodes
    Jacobian,          // evaluatePhysicsDerivatives(): their derivatives, at the same nodes
    InitialCondition,  // setInitialConditions(): sigma from u and q, and the sources the
                       // initial du/dt is solved from, on the k+1 basis nodes
    TauFaces,          // faceStates(): kappa for the Diffusive tau, on both faces of every cell
    ScalarConstraint,  // ScalarG, and InitialScalarDerivative, on the k+1 basis nodes
    ScalarJacobian,    // ScalarGPrime, on the k+1 basis nodes
    ScalarCoupling,    // dSources_dScalars, pointwise at the physics nodes
    FieldCoupling,     // dSigmaFn_dGeometry, dSources_dGeometry and dAuxG_dGeometry,
                       // pointwise at the physics nodes, with a field model attached
    Adjoint,           // initializeMatricesForAdjointSolve(): derivatives at the physics nodes
    InitialProjection, // InitialValue, InitialDerivative and InitialAuxValue, pointwise at
                       // each cell's Gauss points, L2-projected onto the basis
    MassMatrix,        // aFn, pointwise at each cell's Gauss points
};

/// Which TransportSystem entry point receives the points.
enum class EvaluationEntry
{
    ComputePhysics,               // batched; `points` is the abscissae vector exactly
    ComputePhysicsDerivatives,    // batched; likewise
    ComputeSourceTimeDerivatives, // batched; only when a variable reads du/dt
    ScalarG,                      // batched, one call per scalar
    ScalarGPrime,                 // batched, one call for every scalar
    InitialScalarDerivative,      // once per differential scalar; a Python case is
                                  // handed the solution on these points
    Pointwise,                    // a pointwise hook, called at each point of `points`
};

/// When the evaluation happens. A cadence, not a count.
enum class EvaluationCadence
{
    PerResidual,         // every residual evaluation: IDA's, KINSOL's, every norm taken
    PerJacobianBuild,    // every Jacobian build, including IDACalcIC's
    PerContinuationStep, // every pseudo-transient continuation step of a steady solve
    PerAdjointSolve,     // every adjoint solve, including the objective estimate a
                         // steady solve makes on its way out
    OncePerRun,          // inside initialize(), at most `calls` times: a restart or a
                         // steady solve skips some of them
    OncePerSolver,       // the first initialize() of a SystemSolver, at most
};

/// One kind of evaluation, through one entry point, at one cadence.
struct EvaluationSite
{
    EvaluationKind kind;
    EvaluationEntry entry;
    EvaluationCadence cadence;

    // Calls per occasion. For a batched entry, how many batched calls are made
    // with these points each time the cadence comes round -- one ScalarG per
    // scalar per residual, say, or 1 + 3 nVars + nAux face calls per Jacobian
    // build under tauUpdate = "Residual", which differences tau. For a pointwise
    // entry, how many times each point is visited per variable index: the L2
    // projection evaluates InitialValue k+1 times at each Gauss point, once per
    // basis function it is tested against.
    Index calls = 1;

    // Points per cell. `points` is cell-major, so cell i owns
    // points[i * pointsPerCell, (i + 1) * pointsPerCell).
    Index pointsPerCell = 0;

    // The abscissae, in the order a batched entry receives them. For a pointwise
    // entry, the positions the hook is called at, in cell order.
    std::vector<Position> points;

    // The batch shape a batched entry is handed: the GlobalState's point count.
    Index batchSize() const { return static_cast<Index>(points.size()); }

    bool isBatched() const { return entry != EvaluationEntry::Pointwise; }

    bool operator==(EvaluationSite const &) const = default;
};

struct EvaluationPlan
{
    // The discretisation the plan is for. `grid` is a copy, so the plan outlives
    // the solver that built it.
    Grid grid;
    Index k = 0;
    bool superconvergent = false;
    bool steady = false; // a steady solve: PseudoTransient or Newton, not a time march

    std::vector<EvaluationSite> sites;

    /// Plain equality, field by field: the same evaluations at the same points.
    /// Nothing in a plan depends on the run, so two runs that evaluate a case the
    /// same way compare equal and one that does not compares unequal.
    bool operator==(EvaluationPlan const &) const = default;

    /// Does any evaluation of this kind happen in this run?
    bool has(EvaluationKind kind) const
    {
        return std::any_of(sites.begin(), sites.end(),
                           [kind](EvaluationSite const &s) { return s.kind == kind; });
    }

    /// Every site of one kind, in the order the plan lists them.
    std::vector<EvaluationSite> sitesOf(EvaluationKind kind) const
    {
        std::vector<EvaluationSite> out;
        for (auto const &s : sites)
            if (s.kind == kind)
                out.push_back(s);
        return out;
    }

    /// The points every evaluation of a kind is made at. All the sites of one
    /// kind share them, which is what makes a kind the unit to precompute for.
    std::vector<Position> const &points(EvaluationKind kind) const
    {
        for (auto const &s : sites)
            if (s.kind == kind)
                return s.points;
        throw std::out_of_range("This run makes no evaluation of that kind; ask has() first.");
    }

    /// Was this call announced? The plan is meant to be complete -- a case may
    /// compile per batch shape, or tabulate a profile on the announced points,
    /// and a call anywhere else is a solver bug rather than a fallback to
    /// tolerate -- so this is what a case asserts in its own ComputePhysics if
    /// it wants to be told. For a batched entry, `points` must equal some site's
    /// point set exactly, in order and bit for bit, which it does because the
    /// solver builds both with the same code. For Pointwise, every point must
    /// appear in some pointwise site.
    bool announces(EvaluationEntry entry, std::vector<Position> const &points) const
    {
        if (entry != EvaluationEntry::Pointwise)
            return std::any_of(sites.begin(), sites.end(), [&](EvaluationSite const &s)
                               { return s.entry == entry && s.points == points; });

        return std::all_of(points.begin(), points.end(), [&](Position x)
                           {
                               return std::any_of(sites.begin(), sites.end(), [&](EvaluationSite const &s)
                                                  {
                                                      return s.entry == entry &&
                                                             std::find(s.points.begin(), s.points.end(), x) != s.points.end();
                                                  });
                           });
    }

    /// The distinct batch sizes an entry is called with, ascending. For a case
    /// compiled per shape, this is the list of shapes to compile for.
    std::vector<Index> batchSizes(EvaluationEntry entry) const
    {
        std::vector<Index> out;
        for (auto const &s : sites)
            if (s.entry == entry && s.isBatched())
                out.push_back(s.batchSize());
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }
};

inline char const *toString(EvaluationKind kind)
{
    switch (kind)
    {
    case EvaluationKind::Residual: return "Residual";
    case EvaluationKind::Jacobian: return "Jacobian";
    case EvaluationKind::InitialCondition: return "InitialCondition";
    case EvaluationKind::TauFaces: return "TauFaces";
    case EvaluationKind::ScalarConstraint: return "ScalarConstraint";
    case EvaluationKind::ScalarJacobian: return "ScalarJacobian";
    case EvaluationKind::ScalarCoupling: return "ScalarCoupling";
    case EvaluationKind::FieldCoupling: return "FieldCoupling";
    case EvaluationKind::Adjoint: return "Adjoint";
    case EvaluationKind::InitialProjection: return "InitialProjection";
    case EvaluationKind::MassMatrix: return "MassMatrix";
    }
    return "?";
}

#endif // EVALUATIONPLAN_HPP
