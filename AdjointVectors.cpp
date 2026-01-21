
#include <cassert>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "Types.hpp"
#include "SystemSolver.hpp"

void SystemSolver::DerivativeSubVector(Index pIndex, Vector &Vec, void (AdjointProblem::*dX_dZ)(Index, Values &, const State &, Position), DGSoln const &Y, Interval I)
{
    auto const &x_vals = DGApprox::Integrator().abscissa();
    auto const &x_wgts = DGApprox::Integrator().weights();
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

            (adjointProblem->*dX_dZ)(pIndex, dX_dZ_vals1, Y_plus, y_plus);
            (adjointProblem->*dX_dZ)(pIndex, dX_dZ_vals2, Y_minus, y_minus);

            for (Index j = 0; j < k + 1; ++j)
            {
                Vec(XVar * (k + 1) + j) +=
                    wgt * dX_dZ_vals1[XVar] * LegendreBasis::Evaluate(I, j, y_plus);
                Vec(XVar * (k + 1) + j) +=
                    wgt * dX_dZ_vals2[XVar] * LegendreBasis::Evaluate(I, j, y_minus);
            }
        }
    }
}
void SystemSolver::dGdu_Vec(Index i, Vector &Vec, DGSoln const &Y, Interval I)
{
    DerivativeSubVector(i, Vec, &AdjointProblem::dgFn_du, Y, I);
}
void SystemSolver::dGdq_Vec(Index i, Vector &Vec, DGSoln const &Y, Interval I)
{
    DerivativeSubVector(i, Vec, &AdjointProblem::dgFn_dq, Y, I);
}
void SystemSolver::dGdsigma_Vec(Index i, Vector &Vec, DGSoln const &Y, Interval I)
{
    DerivativeSubVector(i, Vec, &AdjointProblem::dgFn_dsigma, Y, I);
}
void SystemSolver::dGdaux_Vec(Index pIndex, Vector &Vec, DGSoln const &Y, Interval I)
{
    auto const &x_vals = DGApprox::Integrator().abscissa();
    auto const &x_wgts = DGApprox::Integrator().weights();
    const size_t n_abscissa = x_vals.size();

    // ASSERT vec.shape == ( nVars * ( k + 1) )
    assert(Vec.size() == nVars * (k + 1));

    Vec.setZero();

    // Phi are basis fn's
    // M( nVars * K + k, nVars * J + j ) = Int_I ( d sigma_fn_K / d u_J * Phi_k * Phi_j )

    for (Index XVar = 0; XVar < nAux; XVar++)
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

            (adjointProblem->dgFn_dphi)(pIndex, dX_dZ_vals1, Y_plus, y_plus);
            (adjointProblem->dgFn_dphi)(pIndex, dX_dZ_vals2, Y_minus, y_minus);

            for (Index j = 0; j < k + 1; ++j)
            {
                Vec(XVar * (k + 1) + j) +=
                    wgt * dX_dZ_vals1[XVar] * LegendreBasis::Evaluate(I, j, y_plus);
                Vec(XVar * (k + 1) + j) +=
                    wgt * dX_dZ_vals2[XVar] * LegendreBasis::Evaluate(I, j, y_minus);
            }
        }
    }
}
// void SystemSolver::dSigmadp_Vec(Index i, Vector &Vec, DGSoln const &Y, Interval I)
// {
//     DerivativeSubVector(i, Vec, &AdjointProblem::dSigmaFn_dp, Y, I);
// }
// void SystemSolver::dSourcesdp_Vec(Index i, Vector &Vec, DGSoln const &Y, Interval I)
// {
//     DerivativeSubVector(i, Vec, &AdjointProblem::dSources_dp, Y, I);
// }