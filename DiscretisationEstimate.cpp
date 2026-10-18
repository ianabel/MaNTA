// The dual-weighted estimate of an objective's discretisation error.
//
// docs/adjoints.rst ("The discretisation error") has the derivation, its sign
// conventions and the measured effectivity; the declarations in SystemSolver.hpp
// say what each part of the answer is. In outline: for the exact solution u and
// any state U~ near it, G(U~) - G(u) = z^T R(U~) to first order, where R is the
// continuum residual and z the continuum adjoint. The scheme's own residual and
// adjoint are no use for this -- F_h(u_h) = 0 at a converged state -- so both are
// taken from a richer discretisation over the same mesh: R from the residual of a
// degree-(k + increment) HDG scheme assembled at the lifted state, and z from
// that scheme's adjoint. What the enriched residual sees and F_h does not is
// everything that distinguishes the continuum from this discretisation: the
// physics between the scheme's nodes, and the test functions of degree above k.

#include "SystemSolver.hpp"

#include <format>
#include <stdexcept>

SystemSolver &SystemSolver::enrichedSolverFor(unsigned int increment)
{
    const unsigned int ke = k + increment;
    if (enrichedSolver && enrichedSolver->k == ke)
        return *enrichedSolver;

    enrichedSolver = std::make_unique<SystemSolver>(grid, ke, problem);
    SystemSolver &e = *enrichedSolver;
    e.setTau(tauc);
    e.setZeroFlux(zeroFlux);
    e.massFromHook = false;
    e.initialiseMatrices();

    // The lifts: each basis evaluated at the enriched nodes, which maps a
    // coefficient vector to the coefficients of the same polynomial in the
    // enriched nodal basis. At increment 1 the enriched nodes are the star nodes,
    // so the degree-k lift is Postprocessor's own V and the star lift the
    // identity, written so rather than evaluated.
    NodalBasis const &basis = y.getBasis();
    NodalBasis const &star = postprocessor->getStarBasis();
    const std::vector<Position> nodes = e.yJac.getPoints();
    liftFromBasis.assign(nCells, Matrix());
    liftFromStar.assign(nCells, Matrix());
    for (Index i = 0; i < nCells; ++i)
    {
        Interval const &I = grid[i];
        if (increment == 1)
        {
            liftFromBasis[i] = postprocessor->V(i);
            liftFromStar[i] = Matrix::Identity(k + 2, k + 2);
            continue;
        }
        Matrix Lb(ke + 1, k + 1), Ls(ke + 1, k + 2);
        for (Index m = 0; m < static_cast<Index>(ke) + 1; ++m)
        {
            const Position x = nodes[i * (ke + 1) + m];
            for (Index j = 0; j < static_cast<Index>(k) + 1; ++j)
                Lb(m, j) = basis.Evaluate(I, j, x);
            for (Index j = 0; j < static_cast<Index>(k) + 2; ++j)
                Ls(m, j) = star.Evaluate(I, j, x);
        }
        liftFromBasis[i] = std::move(Lb);
        liftFromStar[i] = std::move(Ls);
    }
    return e;
}

namespace
{
// The flat [ sigma | q | u | aux ] cell vector's blocks, n dofs per field.
struct CellLayout
{
    Index nVars, nAux, n;
    Index sigma(Index v) const { return v * n; }
    Index q(Index v) const { return (nVars + v) * n; }
    Index u(Index v) const { return (2 * nVars + v) * n; }
    Index aux(Index a) const { return (3 * nVars + a) * n; }
    Index size() const { return (3 * nVars + nAux) * n; }
};
} // namespace

SystemSolver::DiscretisationEstimate
SystemSolver::estimateDiscretisationError(AdjointProblem &objective, DGSoln const &state,
                                          DGSoln const &delta, Time tEval)
{
    if (!estimateDiscretisation)
        throw std::logic_error(
            "estimateDiscretisationError on a solver not configured to estimate "
            "(EstimateDiscretisationError): its evaluation plan does not announce the "
            "enriched evaluations the estimate makes.");
    if (nScalars > 0)
        throw std::invalid_argument(
            "The discretisation estimate cannot be made with global scalars: its adjoint "
            "does not carry the scalar bordering.");
    if (fieldModel)
        throw std::invalid_argument(
            "The discretisation estimate cannot be made with a field model attached: the "
            "enriched residual has no field rows.");
    if (tauScaling != TauScaling::Constant)
        throw std::invalid_argument(
            "The discretisation estimate cannot be made under tauScaling = \"Diffusive\": "
            "its adjoint carries no d tau / dy.");
    if (k < 1 || !postprocessor)
        throw std::invalid_argument(
            "The discretisation estimate needs PolynomialDegree >= 1.");

    const unsigned int increment = enrichmentIncrement();
    const Index ng = objective.getNg();
    const Index ke = k + increment;
    SystemSolver &e = enrichedSolverFor(increment);

    const CellLayout base{static_cast<Index>(nVars), static_cast<Index>(nAux), k + 1};
    const CellLayout rich{static_cast<Index>(nVars), static_cast<Index>(nAux), ke + 1};

    // ---- The lifted state U~: every degree-k field as the same polynomial in
    // the enriched basis, with u* in place of u_h under the superconvergent
    // scheme -- the u its residual and its objective are functions of.
    if (superconvergent)
        postprocessor->computeUStar(state);
    std::vector<double> liftedMem(DGSoln::getDoF(nVars, nCells, ke, 0, nAux, 0), 0.0);
    DGSoln lifted(nVars, grid, ke, liftedMem.data(), 0, nAux, 0);
    for (Index i = 0; i < nCells; ++i)
    {
        Matrix const &Lb = liftFromBasis[i];
        for (Index v = 0; v < nVars; ++v)
        {
            lifted.sigma(v).getCoeff(i).second = Lb * state.sigma(v).getCoeff(i).second;
            lifted.q(v).getCoeff(i).second = Lb * state.q(v).getCoeff(i).second;
            if (superconvergent)
                lifted.u(v).getCoeff(i).second =
                    liftFromStar[i] * postprocessor->uStar(v).getCoeff(i).second;
            else
                lifted.u(v).getCoeff(i).second = Lb * state.u(v).getCoeff(i).second;
        }
        for (Index a = 0; a < nAux; ++a)
            lifted.Aux(a).getCoeff(i).second = Lb * state.Aux(a).getCoeff(i).second;
    }
    for (Index v = 0; v < nVars; ++v)
        lifted.lambda(v) = state.lambda(v);

    GlobalState liftedStates = lifted.evalOnNodes();
    const std::vector<Position> richPoints = lifted.getPoints();
    // The state is steady, and a case that reads du/dt is handed zeros.
    if (problem->anySourceReadsTimeDerivatives())
        liftedStates.setVariableDot(Matrix::Zero(nVars, richPoints.size()));

    // ---- F_e(U~): the transport rows at degree ke, from the physics at the
    // enriched nodes -- the first DiscretisationEstimate site.
    const PhysicsOutput richValues = problem->ComputePhysics(liftedStates, richPoints, tEval);
    std::vector<double> richResMem(liftedMem.size(), 0.0), zeroDotMem(liftedMem.size(), 0.0);
    DGSoln richRes(nVars, grid, ke, richResMem.data(), 0, nAux, 0);
    DGSoln zeroDot(nVars, grid, ke, zeroDotMem.data(), 0, nAux, 0);
    e.updateBoundaryConditions(tEval);
    e.assembleTransportResidual(lifted, zeroDot, richValues, richRes);

    // ---- The improved adjoint's ingredients, shared by every objective.
    if (discretisationAdjoint == DiscretisationAdjoint::Enriched)
    {
        // The enriched operator, from the derivatives at the enriched nodes --
        // the second DiscretisationEstimate site.
        PhysicsDerivatives rec;
        rec.dSigma = GlobalStateMatrix(nVars);
        rec.dSource = GlobalStateMatrix(nVars);
        rec.dAux = GlobalStateMatrix(nAux);
        for (Index v = 0; v < nVars; ++v)
        {
            rec.dSigma.add(nCells, ke, nVars, 0, nAux);
            rec.dSource.add(nCells, ke, nVars, 0, nAux);
        }
        for (Index a = 0; a < nAux; ++a)
            rec.dAux.add(nCells, ke, nVars, 0, nAux);
        problem->ComputePhysicsDerivatives({rec.dSigma, rec.dSource, rec.dAux}, liftedStates,
                                           richPoints, tEval);
        rec.t = tEval;
        rec.valid = true;
        e.assembleAdjointOperator(rec);
    }
    else
    {
        // z_h, from the last Jacobian build, with no physics call.
        std::vector<Index> all(ng);
        for (Index g = 0; g < ng; ++g)
            all[g] = g;
        solveAdjointAt(objective, all, state);
    }

    // ---- The objective's per-cell value under the scheme's own rule and under
    // the enriched one: g at the scheme's physics nodes against its weights (u*
    // on the star nodes under the superconvergent scheme, as GFn reports it), and
    // g at the enriched nodes against theirs.
    Matrix quadrature(ng, nCells);
    {
        PhysicsNodes own = physicsNodesAt(state, tEval);
        const Index ownPerCell = superconvergent ? k + 2 : k + 1;
        for (Index g = 0; g < ng; ++g)
        {
            const Values gOwn = objective.gFn(g, own.states, own.points);
            const Values gRich = objective.gFn(g, liftedStates, richPoints);
            for (Index i = 0; i < nCells; ++i)
            {
                const Vector wOwn = superconvergent
                                        ? postprocessor->starWeights(i)
                                        : y.getBasis().getIntegrationWeights(grid[i]);
                const Vector wRich = lifted.getBasis().getIntegrationWeights(grid[i]);
                quadrature(g, i) = wOwn.dot(gOwn.segment(i * ownPerCell, ownPerCell)) -
                                   wRich.dot(gRich.segment(i * (ke + 1), ke + 1));
            }
        }
    }

    DiscretisationEstimate out;
    out.perCell = Matrix::Zero(ng, nCells);
    out.residual = Vector::Zero(ng);
    out.algebraic = Vector::Zero(ng);
    out.quadrature = quadrature.rowwise().sum();

    for (Index g = 0; g < ng; ++g)
    {
        // z+ per cell at degree ke, and on the faces.
        std::vector<Vector> zPlus(nCells);
        Vector zPlusLambda;
        if (discretisationAdjoint == DiscretisationAdjoint::Enriched)
        {
            e.G_y = e.assembleObjectiveGradient(objective, g, lifted, tEval);
            e.G_field.resize(0);
            e.solveAdjointState();
            zPlus = e.adjoint_squ;
            zPlusLambda = e.adjoint_lambdas;
        }
        else
        {
            AdjointSolution const *zh = adjointSolution(g, state);
            if (zh == nullptr)
                throw std::logic_error("estimateDiscretisationError: z_h was not stored.");
            // Paper I eq. (7) applied to (z_sigma, z_u): the adjoint's sigma column
            // reads (z_sigma, w) = -(z_u, w') - <z_lambda, w n>, so z_sigma is the
            // weak derivative of z_u as q is of u.
            for (Index i = 0; i < nCells; ++i)
            {
                Vector const &zc = zh->squ[i];
                Matrix const &Lb = liftFromBasis[i];
                Vector &zp = zPlus[i];
                zp.resize(rich.size());
                for (Index v = 0; v < nVars; ++v)
                {
                    const Vector zs = zc.segment(base.sigma(v), k + 1);
                    const Vector zu = zc.segment(base.u(v), k + 1);
                    zp.segment(rich.sigma(v), ke + 1) = Lb * zs;
                    zp.segment(rich.q(v), ke + 1) = Lb * zc.segment(base.q(v), k + 1);
                    zp.segment(rich.u(v), ke + 1) =
                        liftFromStar[i] *
                        (postprocessor->B11(i) * zs + postprocessor->B12(i) * zu);
                }
                for (Index a = 0; a < nAux; ++a)
                    zp.segment(rich.aux(a), ke + 1) = Lb * zc.segment(base.aux(a), k + 1);
            }
            zPlusLambda = zh->lambda;
        }

        // The residual part, z+ . F_e(U~), cell by cell; each face's trace rows
        // shared between the cells either side of it, all of a boundary face's
        // going to its one cell.
        Vector resCells(nCells);
        for (Index i = 0; i < nCells; ++i)
            resCells(i) = zPlus[i].dot(
                Eigen::Map<const Vector>(richResMem.data() + i * rich.size(), rich.size()));
        for (Index v = 0; v < nVars; ++v)
            for (Index f = 0; f <= static_cast<Index>(nCells); ++f)
            {
                const double t = zPlusLambda(v * (nCells + 1) + f) * richRes.lambda(v)(f);
                if (f == 0)
                    resCells(0) += t;
                else if (f == static_cast<Index>(nCells))
                    resCells(nCells - 1) += t;
                else
                {
                    resCells(f - 1) += 0.5 * t;
                    resCells(f) += 0.5 * t;
                }
            }

        // The algebraic part, (dG/dy) . delta: dG/dy has no trace block, so it is
        // cellwise already.
        const std::vector<Vector> Gy = assembleObjectiveGradient(objective, g, state, tEval);
        Vector algCells(nCells);
        for (Index i = 0; i < nCells; ++i)
            algCells(i) = Gy[i].dot(
                Eigen::Map<const Vector>(delta.data() + i * base.size(), base.size()));

        out.residual(g) = resCells.sum();
        out.algebraic(g) = algCells.sum();
        out.perCell.row(g) = (resCells + quadrature.row(g).transpose() - algCells).transpose();
    }

    out.total = out.residual + out.quadrature - out.algebraic;
    out.valid = true;
    return out;
}
