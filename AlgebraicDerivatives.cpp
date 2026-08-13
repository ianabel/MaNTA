// The algebraic time derivatives: q', sigma', phi' and lambda', obtained by
// differentiating the constraints that define them.
//
// IDA never produces them. IDA_YA_YDP_INIT computes algebraic *values* and
// differential *derivatives*, so at t0 those blocks of its dYdt are identically
// zero -- which at_t0_only_the_differential_part_of_dydt_exists pins as
// structural rather than as a defect in the id vector. Anything differentiating
// the solution in time then sees only the u term of its chain rule; an objective
// depending on q alone gets exactly zero.
//
// The algebraic residual rows (SystemSolver::residual) are
//
//     res.sigma  = A sigma + Pi( sigmaHat(u, q, x, t) )               = 0
//     res.q      = -A q - B^T u + C^T lambda - RF(t)                  = 0
//     res.Aux    = Pi( G(phi, u, q, sigma, x, t) )                    = 0
//     res.lambda = Csigma sigma + Cq q + G_c u + H lambda - L(t)      = 0
//
// Only res.u carries a time derivative, through X u'. Differentiating the rest
// in time gives dF/dy . ydot = -dF/dt, a linear system in (sigma', q', phi',
// lambda') once u' -- which IDA does have -- is treated as data. So the matrix is
// the residual Jacobian with no mass term, i.e. the alpha = 0 case, the
// differential rows are replaced by the identity with the known derivative on the
// right, and one dense factorisation finishes it.
//
// Three things here are easy to get wrong and are worth stating:
//
//   * The right-hand side carries the *explicit* d/dt terms only. dSigmaHat/du u',
//     B^T u' and the aux constraint's dG/du u' are already in the matrix and are
//     supplied automatically once u' is pinned by the identity rows; putting them
//     in the right-hand side as well would double them.
//   * The Dirichlet trace rows are not in the residual at all -- lambda = g_D(t)
//     is imposed inside the linear solve -- so they arrive structurally zero and
//     have to be given their own identity row and dg_D/dt, or the matrix is
//     singular by exactly the number of Dirichlet boundaries.
//   * The result goes to dydtComplete, never to IDA's dYdt. That is the state IDA
//     takes its first step from.
//
// This runs once per armed run, which is why a dense factorisation of the whole
// system is affordable and static condensation is not worth the branch it would
// need in solveHDGJac.

#include <Eigen/Core>
#include <Eigen/Dense>

#include "SystemSolver.hpp"
#include "Types.hpp"

#include <nvector/nvector_serial.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
/// Runs its callable when the enclosing scope ends, however it ends.
///
/// updateBoundaryConditions() writes RF_cellwise and L_global in place and those
/// are what the forward residual reads, so the differences below have to put them
/// back -- and a trailing call would be skipped by an exception out of residual(),
/// leaving every later residual evaluation reading boundary data from t - h with
/// nothing to say so.
template <class F>
class ScopeGuard
{
public:
    explicit ScopeGuard(F f) : run(std::move(f)) {}
    ScopeGuard(ScopeGuard const &) = delete;
    ScopeGuard &operator=(ScopeGuard const &) = delete;
    ~ScopeGuard() { run(); }

private:
    F run;
};
} // namespace

// The whole Jacobian, densely. Every block comes from the same place the forward
// solve gets it -- assembleCellMatrix for the cell blocks, CEBlocks for their
// lambda columns, CG_cellwise and H_cellwise for the trace rows, and
// assembleScalarCoupling for the border -- so this cannot drift from what the
// solver actually applies. That it does not is checked directly, by
// the_assembled_jacobian_matches_a_finite_difference_of_the_residual.
Matrix SystemSolver::assembleDenseJacobian(DGSoln const &Y, DGSoln const &Ydot, Time tEval,
                                           double alphaValue)
{
    const Index n = static_cast<Index>(Y.getDoF());
    const Index cellDoF = static_cast<Index>(localDOF);
    const Index lambdaOffset = cellDoF * static_cast<Index>(nCells);
    const Index scalarOffset = lambdaOffset + static_cast<Index>(nVars) * (nCells + 1);

    Matrix J = Matrix::Zero(n, n);

    GlobalStateMatrix dSigma_vals(nVars);
    GlobalStateMatrix dSource_vals(nVars);
    GlobalStateMatrix dAux_vals(nAux);
    const PhysicsNodes nodes =
        evaluatePhysicsDerivatives(Y, tEval, dSigma_vals, dSource_vals, dAux_vals);

    for (Index i = 0; i < nCells; ++i)
    {
        // The cell's own [ sigma | q | u | aux ] block, exactly as the forward
        // solve factorises it.
        J.block(i * cellDoF, i * cellDoF, cellDoF, cellDoF) =
            assembleCellMatrix(i, Y, dSigma_vals, dSource_vals, dAux_vals, alphaValue);

        Matrix const &CE = CEBlocks[i];
        for (Index var = 0; var < nVars; ++var)
        {
            // Columns 2*var and 2*var+1 of the cell-major trace blocks are this
            // cell's two ends, which are entries i and i+1 of the variable-major
            // global lambda.
            const Index lam = lambdaOffset + var * (nCells + 1) + i;

            // How this cell's rows depend on its trace values, and how the trace
            // equations depend on this cell. Accumulated because two cells share a
            // node, and each contributes to that node's equation.
            J.block(i * cellDoF, lam, cellDoF, 2) += CE.middleCols(2 * var, 2);
            J.block(lam, i * cellDoF, 2, cellDoF) += CG_cellwise[i].middleRows(2 * var, 2);
            J.block(lam, lam, 2, 2) += H_cellwise[i].block(2 * var, 2 * var, 2, 2);
        }
    }

    if (nScalars > 0)
    {
        // Assembled into scratch rather than into v, w and N_global: those belong
        // to the forward solve, which reads them again on its next Newton step.
        std::vector<double> vMem(static_cast<size_t>(nScalars) * n, 0.0);
        std::vector<double> wMem(static_cast<size_t>(nScalars) * n, 0.0);
        std::vector<DGSoln> v_map, w_map;
        v_map.reserve(nScalars);
        w_map.reserve(nScalars);
        for (Index j = 0; j < nScalars; ++j)
        {
            v_map.emplace_back(nVars, grid, k, vMem.data() + j * n, nScalars, nAux);
            w_map.emplace_back(nVars, grid, k, wMem.data() + j * n, nScalars, nAux);
        }

        Matrix N_local = Matrix::Zero(nScalars, nScalars);
        assembleScalarCoupling(Y, Ydot, nodes, tEval, alphaValue, v_map, w_map, N_local);

        for (Index j = 0; j < nScalars; ++j)
        {
            // dF_HDG/dmu is *minus* v. solveJacEq carries that sign inside its
            // Woodbury elimination -- it inverts ( A + v N^-1 w^T ), which is the
            // Schur complement of the system whose off-diagonal block is -v -- so
            // the stored v has the opposite sign to the Jacobian entry. The
            // residual agrees: res.u subtracts the source, and v holds
            // +dSources/dmu.
            J.block(0, scalarOffset + j, scalarOffset, 1) =
                -Eigen::Map<const Vector>(vMem.data() + j * n, scalarOffset);
            J.block(scalarOffset + j, 0, 1, scalarOffset) =
                Eigen::Map<const Vector>(wMem.data() + j * n, scalarOffset).transpose();
        }
        J.block(scalarOffset, scalarOffset, nScalars, nScalars) = N_local;
    }

    return J;
}

double SystemSolver::timeDifferenceStep(Time tEval)
{
    // cbrt(eps), not sqrt(eps).
    //
    // A central difference has truncation O(h^2 F''') and round-off O(eps |F| /
    // h), and the two balance at h ~ eps^(1/3) -- giving an error of order
    // eps^(2/3), about 4e-11. sqrt(eps) is the *one-sided* choice, where the
    // truncation is O(h F'') instead; used here it leaves round-off at eps / h =
    // 1.5e-8 against a truncation of 2e-16, eight orders apart rather than
    // comparable, and the design document that specified it said "comparable".
    //
    // It is worth 2.5 orders of magnitude and it is measured, not argued:
    // the_derivatives_match_a_manufactured_solution reports q' off its closed
    // form by 3.4e-8 (k = 2) and 2.5e-8 (k = 3) with sqrt(eps), and by 5.6e-11 and
    // 8.8e-12 with this -- on a problem whose explicit time dependence is linear
    // in t and therefore has no truncation error at any step at all, so what
    // shrank is entirely the round-off.
    //
    // Scaled by |t| so that h is a relative perturbation once t is large, and
    // floored at 1 so it does not collapse near t = 0.
    return std::cbrt(std::numeric_limits<double>::epsilon()) *
           std::max(1.0, std::abs(tEval));
}

// The explicit d/dt terms, with the state held fixed.
//
// Nothing exposes them analytically: TransportSystem::LowerBoundary has no
// derivative counterpart and there is no dSigmaFn_dt or dAuxG_dt. Differencing
// residual() itself picks up all of them at once -- RF, L, sigmaHat, the sources
// and the aux constraint -- and asks nothing of the physics case.
Vector SystemSolver::differenceResidualInTime(Time tEval, double h)
{
    const Index n = static_cast<Index>(y.getDoF());

    // residual() calls updateBoundaryConditions(t) on the way in, so both calls
    // below leave RF_cellwise and L_global at tEval - h.
    ScopeGuard restoreBoundaries([this, tEval] { updateBoundaryConditions(tEval); });

    N_Vector fPlus = N_VClone(Y);
    N_Vector fMinus = N_VClone(Y);
    ScopeGuard freeVectors(
        [&]
        {
            N_VDestroy(fPlus);
            N_VDestroy(fMinus);
        });

    residual(tEval + h, Y, dYdt, fPlus);
    residual(tEval - h, Y, dYdt, fMinus);

    const double *fp = N_VGetArrayPointer(fPlus);
    const double *fm = N_VGetArrayPointer(fMinus);

    Vector dFdt(n);
    for (Index i = 0; i < n; ++i)
        dFdt(i) = (fp[i] - fm[i]) / (2.0 * h);

    return dFdt;
}

void SystemSolver::computeAlgebraicTimeDerivatives()
{
    if (Y == nullptr || dYdt == nullptr)
        throw std::logic_error(
            "computeAlgebraicTimeDerivatives() differentiates the constraints at the "
            "current state, so it can only be called between initialize() and "
            "destroySundials()");

    const Index n = static_cast<Index>(y.getDoF());
    const Index cellDoF = static_cast<Index>(localDOF);
    const Index lambdaOffset = cellDoF * static_cast<Index>(nCells);
    const Index scalarOffset = lambdaOffset + static_cast<Index>(nVars) * (nCells + 1);

    const double tNow = t;
    const double h = timeDifferenceStep(tNow);

    // ---- the explicit d/dt terms, with the state held fixed.
    const Vector dFdt = differenceResidualInTime(tNow, h);

    // ---- dF/dy, with no mass term. That is what alpha = 0 means here: the only
    // place alpha enters the block assembly is the X matrix in the u row.
    Matrix J = assembleDenseJacobian(y, dydt, tNow, 0.0);
    Vector rhs = -dFdt;

    // ---- the differential rows, which are data rather than equations.
    //
    // Differentiating them instead would bring in u'' and mu'', which are neither
    // available nor wanted. The identity plus the known derivative is what makes
    // the rest of the system determinate, and the u block coming back out
    // unchanged is a free check that the substitution is right.
    for (Index var = 0; var < nVars; ++var)
    {
        for (Index i = 0; i < nCells; ++i)
        {
            const Index base = i * cellDoF + 2 * nVars * (k + 1) + var * (k + 1);
            for (Index j = 0; j < k + 1; ++j)
            {
                J.row(base + j).setZero();
                J(base + j, base + j) = 1.0;
                rhs(base + j) = dydt.u(var).getCoeff(i).second(j);
            }
        }
    }

    for (Index s = 0; s < nScalars; ++s)
    {
        if (!problem->isScalarDifferential(s))
            continue;
        const Index row = scalarOffset + s;
        J.row(row).setZero();
        J(row, row) = 1.0;
        rhs(row) = dydt.Scalar(s);
    }

    // ---- the Dirichlet trace values, which the residual does not constrain.
    //
    // At a Dirichlet end, initialiseMatrices zeroes that node's C, E, G, Csigma and
    // H entries, so the trace unknown has an identically zero row *and* column: the
    // constraint lambda = g_D(t) lives in ApplyDirichletBCs and H_global instead.
    // Differentiating it gives lambda' = dg_D/dt, and without these rows the matrix
    // is singular by exactly the number of Dirichlet boundaries.
    for (Index var = 0; var < nVars; ++var)
    {
        if (problem->isLowerBoundaryDirichlet(var))
        {
            const Index row = lambdaOffset + var * (nCells + 1);
            J.row(row).setZero();
            J(row, row) = 1.0;
            rhs(row) = (problem->LowerBoundary(var, tNow + h) -
                        problem->LowerBoundary(var, tNow - h)) /
                       (2.0 * h);
        }
        if (problem->isUpperBoundaryDirichlet(var))
        {
            const Index row = lambdaOffset + var * (nCells + 1) + nCells;
            J.row(row).setZero();
            J(row, row) = 1.0;
            rhs(row) = (problem->UpperBoundary(var, tNow + h) -
                        problem->UpperBoundary(var, tNow - h)) /
                       (2.0 * h);
        }
    }

    Eigen::FullPivLU<Matrix> lu(J);
    if (!lu.isInvertible())
        throw std::runtime_error(
            "The differentiated algebraic constraints are singular (rank " +
            std::to_string(lu.rank()) + " of " + std::to_string(n) +
            "), so the algebraic time derivatives cannot be solved for. Either the "
            "system is not index 1 at this state or a block is missing from "
            "assembleDenseJacobian.");

    const Vector ydot = lu.solve(rhs);

    // The whole vector, u block included. Writing dYdt's u' back over it instead
    // would make the round-trip check vacuous, and the two agree to the
    // factorisation's round-off by construction.
    std::copy_n(ydot.data(), n, dydtCompleteMem);
}
