#include "FieldModelEquilibrium.hpp"

#include <format>
#include <stdexcept>

namespace
{
struct FieldModelState : public Equilibrium
{
    Vector psi;
    Time t = 0.0;

    // The linearisation applyResponse needs, taken once per equilibrium on first
    // use: d R / d(state) at the converged psi, dR/dpsi beside it, and
    // dGeometry/dpsi at the points last asked for. Mutable because an
    // equilibrium is handed back const, and this is a cache of quantities it
    // already determines.
    mutable bool linearised = false;
    mutable GlobalStateMatrix dR{0};
    mutable Matrix dRdpsi, dRddpsidt;
    mutable std::vector<Position> geometryPoints;
    mutable std::vector<Matrix> dGdpsi;
};

EquilibriumSpec equilibriumSpecOf(FieldModel const &model, bool differenced)
{
    FieldModelSpec const &f = model.getSpec();
    return {.geometry = f.geometry,
            .label = f.label,
            .name = f.name,
            .response = differenced ? EquilibriumResponse::Differenced : EquilibriumResponse::Model};
}

FieldModelState const &asState(Equilibrium const &e)
{
    auto const *s = dynamic_cast<FieldModelState const *>(&e);
    if (!s)
        throw std::logic_error("FieldModelEquilibrium was handed an equilibrium it did not produce");
    return *s;
}

// The contraction (dR_f/d state) . dp over every nodal component R can read.
double contract(GlobalState const &dRf, GlobalState const &dp)
{
    return dRf.Variable().cwiseProduct(dp.Variable()).sum() +
           dRf.Derivative().cwiseProduct(dp.Derivative()).sum() +
           dRf.Flux().cwiseProduct(dp.Flux()).sum() + dRf.Aux().cwiseProduct(dp.Aux()).sum();
}
} // namespace

FieldModelEquilibrium::FieldModelEquilibrium(std::shared_ptr<FieldModel> model_, bool differencedResponse)
    : EquilibriumModel(equilibriumSpecOf(*model_, differencedResponse)), model(std::move(model_))
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
    factorisedFor = nullptr;

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

// See the class comment. Everything here is field-side work, so it is taken at
// the converged psi exactly rather than at the last Newton iterate's, whose
// factorisation Solve leaves behind.
void FieldModelEquilibrium::applyResponse(MatrixRef dg, Equilibrium const &e, GlobalState const &states,
                                          GlobalState const &dp, std::vector<Position> const &inputPoints,
                                          Vector const &weights, Time t,
                                          std::vector<Position> const &geometryPoints)
{
    if (spec.response == EquilibriumResponse::Differenced)
    {
        EquilibriumModel::applyResponse(dg, e, states, dp, inputPoints, weights, t, geometryPoints);
        return;
    }

    FieldModelState const &s = asState(e);
    const Index nField = model->nFieldDOF();

    if (!s.linearised)
    {
        s.dR = GlobalStateMatrix(nField);
        GlobalStateMatrix dRdot(nField);
        for (Index f = 0; f < nField; ++f)
        {
            s.dR.add(states.zeroedLike());
            dRdot.add(states.zeroedLike());
        }
        s.dRdpsi = Matrix::Zero(nField, nField);
        s.dRddpsidt = Matrix::Zero(nField, nField);
        model->FieldResidualPrime(s.dR, dRdot, s.dRdpsi, s.dRddpsidt, s.psi, Vector::Zero(nField), states,
                                  inputPoints, weights, s.t);
        s.linearised = true;
    }
    if (s.geometryPoints != geometryPoints)
    {
        s.geometryPoints = geometryPoints;
        s.dGdpsi.assign(geometryPoints.size(), Matrix::Zero(nGeometry(), nField));
        for (size_t j = 0; j < geometryPoints.size(); ++j)
            model->dGeometry_dpsi(s.dGdpsi[j], s.psi, geometryPoints[j], s.t);
    }
    if (factorisedFor != &e)
    {
        // Copies: the hook takes its arguments by mutable reference.
        Matrix dRdpsi = s.dRdpsi, dRddpsidt = s.dRddpsidt;
        model->updateFieldJacobian(dRdpsi, dRddpsidt, 0.0);
        factorisedFor = &e;
    }

    Vector rhs(nField);
    for (Index f = 0; f < nField; ++f)
        rhs(f) = contract(s.dR[f], dp);
    Vector dpsi(nField);
    model->solveB(dpsi, rhs);

    for (size_t j = 0; j < geometryPoints.size(); ++j)
        dg.col(static_cast<Eigen::Index>(j)) = -(s.dGdpsi[j] * dpsi);
}
