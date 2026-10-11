
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
    // initializeMatricesForAdjointSolve sizes its vectors for.
    assert(Vec.size() == nAux * (k + 1));

    Interval const &I(grid[intervalIndex]);
    const Vector weights = Y.getBasis().getIntegrationWeights(I);
    for (Index XAux = 0; XAux < nAux; XAux++)
        Vec.block(XAux * (k + 1), 0, (k + 1), 1) =
            dX_dZ.row(XAux).transpose().cwiseProduct(weights);
}



// void SystemSolver::dSigmadp_Vec(Index i, Vector &Vec, DGSoln const &Y, Index I)
// {
//     DerivativeSubVector(i, Vec, &AdjointProblem::dSigmaFn_dp, Y, I);
// }
// void SystemSolver::dSourcesdp_Vec(Index i, Vector &Vec, DGSoln const &Y, Index I)
// {
//     DerivativeSubVector(i, Vec, &AdjointProblem::dSources_dp, Y, I);
// }
