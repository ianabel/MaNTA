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
 */
class FieldModelEquilibrium : public EquilibriumModel
{
public:
    explicit FieldModelEquilibrium(std::shared_ptr<FieldModel> model);

    std::unique_ptr<Equilibrium> Solve(GlobalState const &states, std::vector<Position> const &points,
                                       Vector const &weights, Time t, Equilibrium const *warm) override;

    void Geometry(MatrixRef g, Equilibrium const &e, std::vector<Position> const &points) override;

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
};

#endif // FIELDMODELEQUILIBRIUM_HPP
