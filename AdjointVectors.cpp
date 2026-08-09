
#include <cassert>

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
