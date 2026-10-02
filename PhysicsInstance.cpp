#include "PhysicsInstance.hpp"

#include "AdjointProblem.hpp"
#include "SystemSolver.hpp"
#include "TransportSystem.hpp"

#include <format>
#include <stdexcept>

PhysicsInstance::PhysicsInstance(TransportSystem &problem, Grid const &builtFor,
                                 AdjointProblem *adjoint)
    // Aliasing a shared_ptr that owns nothing, so problem() is one expression
    // whichever constructor ran.
    : borrowed_(std::shared_ptr<TransportSystem>(), &problem), problem_(&borrowed_),
      borrowedAdjoint_(adjoint), builtFor_(builtFor)
{
}

PhysicsInstance::PhysicsInstance(std::shared_ptr<TransportSystem> &problem,
                                 std::unique_ptr<AdjointProblem> &adjoint, Grid const &builtFor,
                                 Rebuild rebuild)
    : problem_(&problem), adjointSlot_(&adjoint), builtFor_(builtFor),
      rebuild_(std::move(rebuild))
{
    if (!problem)
        throw std::logic_error("PhysicsInstance was handed an empty physics case.");
}

AdjointProblem *PhysicsInstance::adjoint() const
{
    return adjointSlot_ != nullptr ? adjointSlot_->get() : borrowedAdjoint_;
}

void PhysicsInstance::requireAdaptable(std::string_view driver) const
{
    if (problem().regridPolicy() == RegridPolicy::InPlace || canRebuild())
        return;
    throw std::invalid_argument(std::format(
        "{} may solve this physics case on more than one mesh or degree, but the case "
        "can be evaluated on one only: its RegridPolicy is Fixed, and the "
        "configuration does not set RebuildPhysicsOnRegrid. Either declare RegridPolicy "
        "InPlace in its spec (regrid = manta.Regrid.InPlace in Python), if "
        "prepareEvaluation rebuilds whatever depends on where the case is evaluated "
        "or nothing does; or set RebuildPhysicsOnRegrid = true, to have a new instance "
        "built from the registry for each discretisation -- which needs a registered "
        "case, named rather than handed over as an object.",
        driver));
}

bool PhysicsInstance::usable(EvaluationPlan const &plan) const
{
    TransportSystem const &p = problem();
    if (p.regridPolicy() == RegridPolicy::InPlace)
        return true;
    if (EvaluationPlan const *last = p.evaluationPlan())
        return *last == plan;
    // Never evaluated yet, so its only commitment is the grid it was built on.
    return plan.grid == builtFor_;
}

void PhysicsInstance::rebuild(Grid const &grid)
{
    // The new case first: if its construction throws, the old one is untouched.
    std::shared_ptr<TransportSystem> fresh = rebuild_(grid);
    if (!fresh)
        throw std::runtime_error("Rebuilding the physics case produced no instance.");
    fresh->copyRestartFrom(problem());

    // The adjoint before the case it may point into, and the case last: once the
    // slot lets go of the old instance nothing refers to it, and it is destroyed
    // here.
    const bool hadAdjoint = adjoint() != nullptr;
    adjointSlot_->reset();
    *problem_ = std::move(fresh);
    if (hadAdjoint)
        *adjointSlot_ = problem().createAdjointProblem();

    builtFor_ = grid;
    ++rebuilds_;
}

std::unique_ptr<SystemSolver> PhysicsInstance::solverFor(
    Grid const &grid, unsigned int k, std::function<void(SystemSolver &)> const &configure)
{
    auto build = [&]
    {
        auto system = std::make_unique<SystemSolver>(grid, k, &problem());
        configure(*system);
        if (AdjointProblem *a = adjoint())
            system->setAdjointProblem(a);
        return system;
    };

    auto system = build();
    if (usable(system->evaluationPlan()))
        return system;

    if (!canRebuild())
        throw std::invalid_argument(TransportSystem::regridRefusal());

    system.reset();
    rebuild(grid);
    return build();
}
