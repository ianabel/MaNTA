
#include <cassert>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "Types.hpp"
#include "SystemSolver.hpp"

// dG/dZ for one cell: the integration weights, elementwise.
//
// This used to apply InterpolateOntoBasis -- the mass matrix -- and the
// distinction is not cosmetic. GFn reports
//
//     G = Int I_h[g] dx = sum_m w_m g(Z_m)
//
// so the exact derivative with respect to the nodal coefficient Z_i is
// w_i dg/dZ|_i, i.e. diag(w) dg/dZ. InterpolateOntoBasis gives M dg/dZ instead.
// M 1 = w exactly -- a mass matrix's row sums *are* the quadrature weights -- so
// the two agree whenever dg/dZ is constant across a cell and differ otherwise by
// (M - diag(w)) dg/dZ, an operator that annihilates constants.
//
// That last property is why this survived: the discrepancy sums to zero over
// each cell, so every aggregate stayed exact. The scalar-parameter gradient
// agreed with finite differences to 2e-8 and with a closed form to 7e-16, and
// nothing else in the suite looks at anything finer. It is visible only per
// node, which means only through spatial adjoint parameters, where it showed as
// an error depending solely on the intra-cell node index -- symmetric,
// alternating in sign, summing to zero -- and decaying as O(h^4) because a
// refined cell is one across which dg/dZ varies less. A convergence rate of an
// inconsistency, not a discretisation order: refining hid it, nothing removed it.
//
// The superconvergent branch of initializeMatricesForAdjointSolve already did
// this correctly (`cwiseProduct(b1)` against the star weights), and
// AdjointProblem::GFn's comment already promised it -- "the same quadrature is
// what initializeMatricesForAdjointSolve differentiates to build G_y, so the
// reported objective and the reported gradient are exactly a function and its
// derivative". This is the plain branch keeping that promise.
void SystemSolver::DerivativeSubVector(Index, Vector &Vec, Eigen::Ref<Matrix> const dX_dZ, DGSoln const &Y, Index intervalIndex)
{
    Interval const &I(grid[intervalIndex]);
    const Vector weights = Y.getBasis().getIntegrationWeights(I);
    for (Index XVar = 0; XVar < nVars; XVar++)
        Vec.block(XVar * (k + 1), 0, (k + 1), 1) =
            dX_dZ.row(XVar).transpose().cwiseProduct(weights);
}


// The same operator as DerivativeSubVector, over the auxiliary variables. It is a
// separate function only because the loop bound differs: there are nAux of these
// and DerivativeSubVector runs to nVars. The two coincide in every fixture except
// test_adjoint_aux.py, which is how the version this replaced came to carry two
// confusions between them.
//
// That version integrated the *pointwise* dgFn_dphi against the basis functions
// on the basis's own Gauss rule -- Int dg/dphi phi_j dx, the derivative of
// Int g dx and so of a functional GFn does not report, the last survivor of the
// family the comment above describes. A C++ case's dgFn_dphi still reaches this
// via AdjointProblem::dg's default, which samples it at the nodes; a Python case
// supplies dg directly.
void SystemSolver::dGdaux_Vec(Index, Vector &Vec, Eigen::Ref<Matrix> const dX_dZ, DGSoln const &Y, Index intervalIndex)
{
    // One (k+1)-block per auxiliary variable, which is what
    // initializeMatricesForAdjointSolve and dGdt both size their vectors for.
    assert(Vec.size() == nAux * (k + 1));

    Interval const &I(grid[intervalIndex]);
    const Vector weights = Y.getBasis().getIntegrationWeights(I);
    for (Index XAux = 0; XAux < nAux; XAux++)
        Vec.block(XAux * (k + 1), 0, (k + 1), 1) =
            dX_dZ.row(XAux).transpose().cwiseProduct(weights);
}


// dG/dt for one objective, by the chain rule
//
//     dG/dt = Int ( dg/du . u' + dg/dq . q' + dg/dsigma . sigma' + dg/dphi . phi' ) dx
//
// DerivativeSubVector turns the nodal values of one dg/dZ into
// dG/d(coefficients) for that field in one cell -- the projection against every
// basis function -- so contracting each with the matching block of Ydot and
// summing over cells is the whole derivative. It is the same quantity the adjoint
// solve assembles into G_y, reused rather than rebuilt.
//
// Assembled here rather than asked of AdjointProblem because it needs the grid,
// the basis and the projection, none of which AdjointProblem can reach; what it
// does own -- dg, giving dg/dZ at the nodes -- is already required of every
// adjoint case, so nothing has to implement anything new. Going through the
// derivatives rather than through the objective functional is also what makes
// this correct for a g that is nonlinear in the state: evaluating the functional
// *on* the derivative vector instead, as origin/optimize-mode's version did,
// gives Int g(u',q',...) dx, which coincides with dG/dt only when g is linear.
//
// Three limits worth knowing.
//
// There is no scalar term, because AdjointProblem has no dgFn_dscalars to go with
// the other four -- an objective depending on the global scalars mu loses that
// contribution silently. This also inherits AdjointProblem's standing assumption
// that G = Int g dx (AdjointProblem.hpp).
//
// And the caller decides how much of the sum is real, by what it puts in Ydot.
// IDA's own dydt is not enough at t0: q, sigma and phi are algebraic, IDA's
// IDA_YA_YDP_INIT produces no derivative for an algebraic component, and those
// blocks are exactly zero there -- so three quarters of the chain rule below
// multiply by nothing, and an objective depending on q alone comes out at exactly
// 0.0. That is what at_t0_only_the_differential_part_of_dydt_exists in
// SolverLifecycleTests.cpp pins, and why the gate passes dydtComplete rather than
// dydt: computeAlgebraicTimeDerivatives() solves the differentiated constraints
// for the missing blocks. Pass IDA's dydt here and the result is still correct,
// including for a nonlinear g -- it is just answering about a derivative with
// three of its four parts set to zero.
Value SystemSolver::dGdt(Index gIndex, DGSoln const &Y, DGSoln const &Ydot)
{
    if (!adjointProblem)
        throw std::logic_error("dG/dt requested with no AdjointProblem set; there is no objective to differentiate.");

    // Superconvergence changes both the node count dg is sampled on and the
    // projection the adjoint assembly uses (the pp.V / pp.B11 / pp.B12 branch of
    // initializeMatricesForAdjointSolve). None of that is handled here, and a
    // quietly inconsistent dG/dt is worse than none.
    if (superconvergent)
        throw std::logic_error("dG/dt is not implemented for Superconvergent = true; the objective would be differentiated through the wrong projection.");

    // Via the *batched* dg hook and the interpolatory DerivativeSubVector, which
    // is the same route initializeMatricesForAdjointSolve takes to build G_y.
    //
    // Not the quadrature overload and the pointwise dgFn_du/dq/dsigma/dphi hooks
    // behind it, even though those give the exact integral rather than the
    // integral of an interpolant. Two reasons, and the first is decisive: a Python
    // AdjointProblem does not implement the pointwise hooks at all -- the
    // trampoline raises "Individual derivative function \"dgFn_du\" deprecated;
    // use vectorized version dg instead" -- so the quadrature route works only for
    // C++ cases. The second is consistency: the adjoint operator is built from the
    // interpolatory form, so a gate built from the quadrature form would answer a
    // slightly different question from the gradients beside it.
    GlobalState dGdvars(nCells, k, nVars, nScalars, nAux);
    adjointProblem->dg(gIndex, dGdvars, Y.evalOnNodes(), Y.getPoints());

    Vector projected(nVars * (k + 1));
    Vector projectedAux(nAux * (k + 1));
    Value total = 0.0;

    for (Index i = 0; i < nCells; ++i)
    {
        DerivativeSubVector(gIndex, projected, dGdvars.cellwiseVariable(i), Y, i);
        for (Index var = 0; var < nVars; ++var)
            total += projected(Eigen::seqN(var * (k + 1), k + 1))
                         .dot(Ydot.u(var).getCoeff(i).second);

        DerivativeSubVector(gIndex, projected, dGdvars.cellwiseDerivative(i), Y, i);
        for (Index var = 0; var < nVars; ++var)
            total += projected(Eigen::seqN(var * (k + 1), k + 1))
                         .dot(Ydot.q(var).getCoeff(i).second);

        DerivativeSubVector(gIndex, projected, dGdvars.cellwiseFlux(i), Y, i);
        for (Index var = 0; var < nVars; ++var)
            total += projected(Eigen::seqN(var * (k + 1), k + 1))
                         .dot(Ydot.sigma(var).getCoeff(i).second);

        // Through dGdaux_Vec, which is DerivativeSubVector over nAux rather than
        // nVars. This used to apply InterpolateOntoBasis inline -- the mass
        // matrix, i.e. the discrepancy the comment at the top of this file
        // describes, in the one block that had not been converted. It differed
        // from the rest of the chain rule by (M - diag(w)) dg/dphi and so
        // answered a slightly different question from the gradients beside it.
        if (nAux > 0)
        {
            dGdaux_Vec(gIndex, projectedAux, dGdvars.cellwiseAux(i), Y, i);
            for (Index a = 0; a < nAux; ++a)
                total += projectedAux(Eigen::seqN(a * (k + 1), k + 1))
                             .dot(Ydot.Aux(a).getCoeff(i).second);
        }
    }

    return total;
}

// void SystemSolver::dSigmadp_Vec(Index i, Vector &Vec, DGSoln const &Y, Index I)
// {
//     DerivativeSubVector(i, Vec, &AdjointProblem::dSigmaFn_dp, Y, I);
// }
// void SystemSolver::dSourcesdp_Vec(Index i, Vector &Vec, DGSoln const &Y, Index I)
// {
//     DerivativeSubVector(i, Vec, &AdjointProblem::dSources_dp, Y, I);
// }
