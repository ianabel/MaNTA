
#include <cassert>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "Types.hpp"
#include "SystemSolver.hpp"

void SystemSolver::DerivativeSubVector(Index gIndex, Vector &Vec, Eigen::Ref<Matrix> const dX_dZ, DGSoln const &Y, Index intervalIndex)
{
    Interval const &I(grid[intervalIndex]);
    for (Index XVar = 0; XVar < nVars; XVar++)
    {
        auto const &dX_dZ_vec = dX_dZ(XVar, Eigen::all);
        Vec.block(XVar * (k + 1), 0, (k + 1), 1) = Y.getBasis().InterpolateOntoBasis(I, dX_dZ_vec);
    }
}

void SystemSolver::DerivativeSubVector(Index gIndex, Vector &Vec, void (AdjointProblem::*dX_dZ)(Index, VectorRef, const State &, Position), DGSoln const &Y, Index intervalIndex)
{
    Interval const &I(grid[intervalIndex]);
    auto const &x_vals = y.getBasis().abscissae();
    auto const &x_wgts = y.getBasis().weights();
    const size_t n_abscissa = x_vals.size();

    // ASSERT vec.shape == ( nVars * ( k + 1) )
    assert(Vec.size() == nVars * (k + 1));

    Vec.setZero();

    // Phi are basis fn's
    // M( nVars * K + k, nVars * J + j ) = Int_I ( d sigma_fn_K / d u_J * Phi_k * Phi_j )

    for (Index XVar = 0; XVar < nVars; XVar++)
    {
        Values dX_dZ_vals1(nVars);
        Values dX_dZ_vals2(nVars);
        dX_dZ_vals1.setZero();
        dX_dZ_vals2.setZero();

        for (size_t i = 0; i < n_abscissa; ++i)
        {
            // Pull the loop over the gaussian integration points
            // outside so we can evaluate u, q, dX_dZ once and store the values

            // All for loops inside here can be parallelised as they all
            // write to separate entries in mat

            double wgt = x_wgts[i] * (I.h() / 2.0);

            double y_plus = I.x_l + (1.0 + x_vals[i]) * (I.h() / 2.0);
            double y_minus = I.x_l + (1.0 - x_vals[i]) * (I.h() / 2.0);

            State Y_plus = Y.eval(y_plus), Y_minus = Y.eval(y_minus);

            (adjointProblem->*dX_dZ)(gIndex, dX_dZ_vals1, Y_plus, y_plus);
            (adjointProblem->*dX_dZ)(gIndex, dX_dZ_vals2, Y_minus, y_minus);

            for (Index j = 0; j < k + 1; ++j)
            {
                Vec(XVar * (k + 1) + j) +=
                    wgt * dX_dZ_vals1[XVar] * y.getBasis().Evaluate(I, j, y_plus);
                Vec(XVar * (k + 1) + j) +=
                    wgt * dX_dZ_vals2[XVar] * y.getBasis().Evaluate(I, j, y_minus);
            }
        }
    }
}

void SystemSolver::dGdu_Vec(Index gIndex, Vector &Vec, DGSoln const &Y, Index I)
{
    DerivativeSubVector(gIndex, Vec, &AdjointProblem::dgFn_du, Y, I);
}

void SystemSolver::dGdq_Vec(Index gIndex, Vector &Vec, DGSoln const &Y, Index I)
{
    DerivativeSubVector(gIndex, Vec, &AdjointProblem::dgFn_dq, Y, I);
}

void SystemSolver::dGdsigma_Vec(Index gIndex, Vector &Vec, DGSoln const &Y, Index I)
{
    DerivativeSubVector(gIndex, Vec, &AdjointProblem::dgFn_dsigma, Y, I);
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
// And the caller decides how much of the sum is real, by what it puts in Ydot. At
// t0 in particular only the differential part of dydt carries anything: q, sigma
// and phi are algebraic, IDA's IDA_YA_YDP_INIT produces no derivative for an
// algebraic component, and setInitialConditions leaves those blocks at zero. So
// the dG/dt gate, which runs there, is differentiating through the u dependence
// alone -- correctly, including for a nonlinear g, but not completely. The
// function itself is the full chain rule and is tested as such; see
// at_t0_only_the_differential_part_of_dydt_exists in SolverLifecycleTests.cpp for
// the gap and TODO for what closing it needs.
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

        // Aux is projected here rather than through DerivativeSubVector because
        // that function's loop is hardcoded to nVars, and there are nAux of these.
        // The two coincide in every fixture except test_adjoint_aux, which is how
        // dGdaux_Vec came to carry two confusions between them.
        if (nAux > 0)
        {
            Interval const &I(grid[i]);
            auto dAux = dGdvars.cellwiseAux(i);
            for (Index a = 0; a < nAux; ++a)
            {
                Vector const nodal = dAux.row(a).transpose();
                total += Y.getBasis().InterpolateOntoBasis(I, nodal)
                             .dot(Ydot.Aux(a).getCoeff(i).second);
            }
        }
    }

    return total;
}

void SystemSolver::dGdaux_Vec(Index gIndex, Vector &Vec, DGSoln const &Y, Index intervalIndex)
{
    Interval const &I(grid[intervalIndex]);
    auto const &x_vals = y.getBasis().abscissae();
    auto const &x_wgts = y.getBasis().weights();
    const size_t n_abscissa = x_vals.size();

    // This writes one (k+1)-block per *auxiliary* variable (the loop below runs
    // to nAux), and its only caller sizes the vector nAux * (k + 1) --
    // initializeMatricesForAdjointSolve in SystemSolver.cpp. The bound here read
    // nVars, so a system with nAux != nVars aborted on a correctly-sized
    // vector. Nothing defines NDEBUG in any build variant, so that abort was
    // live in release builds too; it went unnoticed because every aux case in
    // the suite happens to have nAux == nVars.
    assert(Vec.size() == nAux * (k + 1));

    Vec.setZero();

    // Phi are basis fn's
    // M( nVars * K + k, nVars * J + j ) = Int_I ( d sigma_fn_K / d u_J * Phi_k * Phi_j )

    for (Index XVar = 0; XVar < nAux; XVar++)
    {
        // nAux, not nVars: dgFn_dphi fills one entry per auxiliary variable, and
        // the reads below index these with XVar, which runs to nAux. Sized nVars
        // this read past the end whenever nAux > nVars, and -- because the hook
        // takes a VectorRef -- an implementation that assigns the whole vector
        // rather than writing elementwise tripped Eigen's "Ref cannot be
        // resized" assert instead. The C++ mocks in the unit tests all write
        // elementwise and have nAux <= nVars, so neither symptom appeared there.
        Values dX_dZ_vals1(nAux);
        Values dX_dZ_vals2(nAux);
        dX_dZ_vals1.setZero();
        dX_dZ_vals2.setZero();

        for (size_t i = 0; i < n_abscissa; ++i)
        {
            // Pull the loop over the gaussian integration points
            // outside so we can evaluate u, q, dX_dZ once and store the values

            // All for loops inside here can be parallelised as they all
            // write to separate entries in mat

            double wgt = x_wgts[i] * (I.h() / 2.0);

            double y_plus = I.x_l + (1.0 + x_vals[i]) * (I.h() / 2.0);
            double y_minus = I.x_l + (1.0 - x_vals[i]) * (I.h() / 2.0);

            State Y_plus = Y.eval(y_plus), Y_minus = Y.eval(y_minus);

            (adjointProblem->dgFn_dphi)(gIndex, dX_dZ_vals1, Y_plus, y_plus);
            (adjointProblem->dgFn_dphi)(gIndex, dX_dZ_vals2, Y_minus, y_minus);

            for (Index j = 0; j < k + 1; ++j)
            {
                Vec(XVar * (k + 1) + j) +=
                    wgt * dX_dZ_vals1[XVar] * y.getBasis().Evaluate(I, j, y_plus);
                Vec(XVar * (k + 1) + j) +=
                    wgt * dX_dZ_vals2[XVar] * y.getBasis().Evaluate(I, j, y_minus);
            }
        }
    }
}
// void SystemSolver::dSigmadp_Vec(Index i, Vector &Vec, DGSoln const &Y, Index I)
// {
//     DerivativeSubVector(i, Vec, &AdjointProblem::dSigmaFn_dp, Y, I);
// }
// void SystemSolver::dSourcesdp_Vec(Index i, Vector &Vec, DGSoln const &Y, Index I)
// {
//     DerivativeSubVector(i, Vec, &AdjointProblem::dSources_dp, Y, I);
// }
