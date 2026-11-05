#ifndef FIELDMODELEQUILIBRIUM_HPP
#define FIELDMODELEQUILIBRIUM_HPP

#include "EquilibriumModel.hpp"
#include "FieldModel.hpp"

#include <memory>

/*
    Any residual-form FieldModel, run as a tier-0 equilibrium.

    Solve is Newton on the model's field rows alone, R(psi; p) = 0, with the
    transport state held fixed: the model's own dR/dpsi through its
    updateFieldJacobian and solveB, at alpha = 0. It starts from the previous
    equilibrium's psi when there is one and from InitialFieldValue when there is
    not, and iterates to round-off -- the field work is free on the cost that
    matters here, and a looser equilibrium would put its error into every
    transport residual.

    A model with a differential DOF is refused: a field with dynamics of its own
    is not an instantaneous function of the transport state, which is all tier 0
    can express.

    The response is the model's own, exactly (tier 1, EquilibriumResponse::Model):
    the field rows hold R(psi(p); p) = 0, so

        dpsi = -(dR/dpsi)^-1 (dR/dp) dp,   dg = (dGeometry/dpsi) dpsi,

    the tier-2 coupling blocks with psi eliminated -- which is what makes the
    theta pinch at tier 1 checkable against tier 2. `differencedResponse` keeps
    the base class's forward difference of Solve instead, so that the default
    every black-box model gets is exercised on a real field too.
 */
class FieldModelEquilibrium : public EquilibriumModel
{
public:
    explicit FieldModelEquilibrium(std::shared_ptr<FieldModel> model, bool differencedResponse = false);

    std::unique_ptr<Equilibrium> Solve(GlobalState const &states, std::vector<Position> const &points,
                                       Vector const &weights, Time t, Equilibrium const *warm) override;

    void Geometry(MatrixRef g, Equilibrium const &e, std::vector<Position> const &points) override;

    void applyResponse(MatrixRef dg, Equilibrium const &e, GlobalState const &states, GlobalState const &dp,
                       std::vector<Position> const &inputPoints, Vector const &weights, Time t,
                       std::vector<Position> const &geometryPoints) override;

    void resetForRun() override { model->resetForRun(); }

    FieldModel const &fieldModel() const { return *model; }

    /// The field unknowns of an equilibrium this adapter produced.
    static Vector const &psi(Equilibrium const &e);

    /// Newton iterations over the adapter's lifetime: the field-side cost.
    long newtonIterations() const { return iterations; }

    // Newton stops when a step is this small relative to psi, or gives up and
    // throws after maxIterations.
    static constexpr double relativeStep = 1e-13;
    static constexpr int maxIterations = 50;

private:
    std::shared_ptr<FieldModel> model;
    long iterations = 0;
    // The equilibrium whose dR/dpsi the model's own factorisation currently
    // holds, or null. Solve refactorises it at every Newton step, so a response
    // applied after another Solve has to put its own back first.
    Equilibrium const *factorisedFor = nullptr;
};

#endif // FIELDMODELEQUILIBRIUM_HPP
