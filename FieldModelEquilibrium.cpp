#include "FieldModelEquilibrium.hpp"

#include <format>
#include <stdexcept>

namespace
{
struct FieldModelState : public Equilibrium
{
    Vector psi;
    Time t = 0.0;
};

EquilibriumSpec equilibriumSpecOf(FieldModel const &model)
{
    FieldModelSpec const &f = model.getSpec();
    return {.geometry = f.geometry, .label = f.label, .name = f.name};
}

FieldModelState const &asState(Equilibrium const &e)
{
    auto const *s = dynamic_cast<FieldModelState const *>(&e);
    if (!s)
        throw std::logic_error("FieldModelEquilibrium was handed an equilibrium it did not produce");
    return *s;
}
} // namespace

FieldModelEquilibrium::FieldModelEquilibrium(std::shared_ptr<FieldModel> model_)
    : EquilibriumModel(equilibriumSpecOf(*model_)), model(std::move(model_))
{
    for (Index f = 0; f < model->nFieldDOF(); ++f)
        if (model->isFieldDOFDifferential(f))
            throw std::invalid_argument(
                "The field model's DOF '" + model->getSpec().dofs[f].name +
                "' is differential, so its field is not an instantaneous function of the "
                "transport state and cannot be run as an equilibrium (FieldCoupling = "
                "\"consistent\"). Couple it instead.");
}

Vector const &FieldModelEquilibrium::psi(Equilibrium const &e) { return asState(e).psi; }

std::unique_ptr<Equilibrium> FieldModelEquilibrium::Solve(GlobalState const &states,
                                                          std::vector<Position> const &points,
                                                          Vector const &weights, Time t,
                                                          Equilibrium const *warm)
{
    const Index nField = model->nFieldDOF();

    auto out = std::make_unique<FieldModelState>();
    out->t = t;
    if (warm)
        out->psi = asState(*warm).psi;
    else
    {
        out->psi = Vector::Zero(nField);
        model->InitialFieldValue(out->psi);
    }
    Vector &psi = out->psi;

    // dR/d(state) comes with dR/dpsi and is not wanted here, but
    // FieldResidualPrime needs somewhere zeroed to write it.
    GlobalStateMatrix dR(nField), dRdot(nField);
    for (Index f = 0; f < nField; ++f)
    {
        dR.add(states.zeroedLike());
        dRdot.add(states.zeroedLike());
    }
    const Vector zero = Vector::Zero(nField);
    Vector R(nField), step(nField);
    Matrix dRdpsi(nField, nField), dRddpsidt(nField, nField);

    for (int it = 0; it < maxIterations; ++it)
    {
        ++iterations;
        R.setZero();
        model->FieldResidual(R, psi, zero, states, points, weights, t);
        dRdpsi.setZero();
        dRddpsidt.setZero();
        model->FieldResidualPrime(dR, dRdot, dRdpsi, dRddpsidt, psi, zero, states, points, weights, t);
        model->updateFieldJacobian(dRdpsi, dRddpsidt, 0.0);
        model->solveB(step, R);
        psi -= step;
        if (step.norm() <= relativeStep * std::max(psi.norm(), 1.0))
            return out;
    }
    throw std::runtime_error(std::format("FieldModelEquilibrium: Newton on the field rows of '{}' did not "
                                         "converge in {} iterations",
                                         model->getSpec().name, maxIterations));
}

void FieldModelEquilibrium::Geometry(MatrixRef g, Equilibrium const &e, std::vector<Position> const &points)
{
    FieldModelState const &s = asState(e);
    Vector slot(nGeometry());
    for (size_t j = 0; j < points.size(); ++j)
    {
        slot.setZero();
        model->Geometry(slot, s.psi, points[j], s.t);
        g.col(static_cast<Eigen::Index>(j)) = slot;
    }
}
