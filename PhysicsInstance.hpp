#ifndef PHYSICSINSTANCE_HPP
#define PHYSICSINSTANCE_HPP

#include <functional>
#include <memory>
#include <string_view>

#include "gridStructures.hpp"

class AdjointProblem;
class SystemSolver;
class TransportSystem;
struct EvaluationPlan;

/*
    The physics case an adaptation driver solves, and, if the configuration
    allows it, how to replace it.

    MeshAdaptation, DegreeAdaptation and the ladders solve one problem on a
    sequence of discretisations, each a new evaluation plan. A case whose spec
    declares RegridPolicy::InPlace follows each one through prepareEvaluation.
    Any other case may be evaluated by its first plan only; for those, and only
    when the configuration sets RebuildPhysicsOnRegrid, a driver destroys the
    case and builds a new instance for the new grid -- which is how a case
    written before evaluation plans existed runs under adaptation at all.

    A rebuild needs more than the case:

      * a way to construct a case for a given grid, which only the caller has --
        runManta and a PyRunner named after a case know the name and the table
        InstantiateProblem wants, while a Python case object handed to Runner
        cannot be rebuilt at all;
      * the adjoint problem, which a case hands out and which may point back into
        it (AutodiffAdjointProblem::PhysicsProblem), so a new case needs a new one;
      * the restart state a driver put on the old instance to warm-start the next
        solve, which has to reach the new instance.

    This holds all three, through the *caller's own* owning pointers, so that a
    rebuilt case and its adjoint are what the caller holds once the driver
    returns, and the solver it returns points at them. A borrowed case -- a
    TransportSystem& with no slots -- cannot be rebuilt.

    Non-copyable and non-movable: the borrowing form points into itself.
 */
class PhysicsInstance
{
public:
    // A fresh case for a grid: InstantiateProblem with the caller's name and
    // table.
    using Rebuild = std::function<std::shared_ptr<TransportSystem>(Grid const &)>;

    // Borrow a case built on `builtFor`, and its adjoint problem if it has one.
    // It cannot be rebuilt.
    PhysicsInstance(TransportSystem &problem, Grid const &builtFor,
                    AdjointProblem *adjoint = nullptr);

    // Work through the caller's slots. `problem` holds a case built on
    // `builtFor` and `adjoint` its adjoint problem or null. `rebuild` is empty
    // unless the configuration sets RebuildPhysicsOnRegrid and the case can be
    // built by name; a rebuild replaces both slots' contents.
    PhysicsInstance(std::shared_ptr<TransportSystem> &problem,
                    std::unique_ptr<AdjointProblem> &adjoint, Grid const &builtFor,
                    Rebuild rebuild);

    PhysicsInstance(PhysicsInstance const &) = delete;
    PhysicsInstance &operator=(PhysicsInstance const &) = delete;

    // The current case and adjoint problem. Both may change across solverFor, so
    // a driver asks again after each one rather than holding on to either.
    TransportSystem &problem() const { return **problem_; }
    AdjointProblem *adjoint() const;

    bool canRebuild() const { return static_cast<bool>(rebuild_); }

    // How many times this has replaced the case.
    int rebuilds() const { return rebuilds_; }

    // Refuse, before any solve, a driver that may change the plan of a case that
    // can follow no change and may not be rebuilt: std::invalid_argument naming
    // `driver` and both remedies.
    void requireAdaptable(std::string_view driver) const;

    // A solver for one solve on `grid` at degree `k`: built around the current
    // case, configured by `configure`, and given the adjoint problem. If that
    // solver's plan is one the case may not be evaluated by and a rebuild is
    // allowed, the solver is discarded, the case rebuilt for `grid` -- taking
    // over the old instance's restart state, with a fresh adjoint problem -- and
    // the solver built again around the new one; otherwise it is refused.
    // Constructing a SystemSolver evaluates no physics, so the first solver is a
    // probe that costs nothing a case could notice. The previous solve's solver
    // must be gone first: a rebuild destroys the case it pointed at.
    std::unique_ptr<SystemSolver> solverFor(Grid const &grid, unsigned int k,
                                            std::function<void(SystemSolver &)> const &configure);

private:
    bool usable(EvaluationPlan const &plan) const;
    void rebuild(Grid const &grid);

    std::shared_ptr<TransportSystem> borrowed_;
    std::shared_ptr<TransportSystem> *problem_;
    std::unique_ptr<AdjointProblem> *adjointSlot_ = nullptr;
    AdjointProblem *borrowedAdjoint_ = nullptr;
    Grid builtFor_;
    Rebuild rebuild_;
    int rebuilds_ = 0;
};

#endif // PHYSICSINSTANCE_HPP
