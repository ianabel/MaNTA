#ifndef FINITEDIFFERENCEJACOBIAN_HPP
#define FINITEDIFFERENCEJACOBIAN_HPP

// Shared helpers for the tests that check the linear solve against an
// independently computed Jacobian (SolveJacTests.cpp, ScalarJacobianTests.cpp).
//
// MaNTA never assembles its Jacobian: solveHDGJac and solveJacEq apply the
// inverse directly. So the only way to check them is to build the Jacobian some
// other way -- here by finite-differencing SystemSolver::residual -- and require
// the vector the solve returns to satisfy J dy = g.

#include "SystemSolver.hpp"
#include "Types.hpp"

#include <nvector/nvector_serial.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace fdjac
{

/// Build J = dF/dY + cj dF/dY' by central differences, matching IDA's
/// convention: perturbing column j means perturbing Y[j] by h and Y'[j] by cj*h
/// together.
inline Matrix jacobian(SystemSolver &sys, N_Vector Y, N_Vector dYdt, double t, double cj)
{
    const Index n = N_VGetLength(Y);
    Matrix J(n, n);

    N_Vector Yp = N_VClone(Y), dYp = N_VClone(dYdt);
    N_Vector Fplus = N_VClone(Y), Fminus = N_VClone(Y);

    const double *Y0 = N_VGetArrayPointer(Y);
    const double *dY0 = N_VGetArrayPointer(dYdt);
    double *Ya = N_VGetArrayPointer(Yp);
    double *dYa = N_VGetArrayPointer(dYp);

    for (Index j = 0; j < n; ++j)
    {
        const double h = 1e-6 * std::max(1.0, std::abs(Y0[j]));

        std::copy(Y0, Y0 + n, Ya);
        std::copy(dY0, dY0 + n, dYa);
        Ya[j] += h;
        dYa[j] += cj * h;
        sys.residual(t, Yp, dYp, Fplus);

        std::copy(Y0, Y0 + n, Ya);
        std::copy(dY0, dY0 + n, dYa);
        Ya[j] -= h;
        dYa[j] -= cj * h;
        sys.residual(t, Yp, dYp, Fminus);

        const double *fp = N_VGetArrayPointer(Fplus);
        const double *fm = N_VGetArrayPointer(Fminus);
        for (Index i = 0; i < n; ++i)
            J(i, j) = (fp[i] - fm[i]) / (2.0 * h);
    }

    N_VDestroy(Yp);
    N_VDestroy(dYp);
    N_VDestroy(Fplus);
    N_VDestroy(Fminus);
    return J;
}

/// Rows the residual never writes.
///
/// residual() does not touch the Dirichlet boundary rows -- those constraints
/// are imposed inside the linear solve, via H_global, not in the residual. The
/// finite-differenced Jacobian is therefore rank-deficient by exactly the number
/// of Dirichlet boundaries, and those rows carry no information to compare
/// against.
inline std::vector<Index> undefinedRows(Matrix const &J)
{
    std::vector<Index> rows;
    for (Index i = 0; i < J.rows(); ++i)
        if (J.row(i).norm() == 0.0)
            rows.push_back(i);
    return rows;
}

inline bool isUndefined(std::vector<Index> const &rows, Index i)
{
    return std::find(rows.begin(), rows.end(), i) != rows.end();
}

/// ||J dy - g|| / ||g||, over the rows the residual actually defines.
inline double relativeResidual(Matrix const &J, Vector const &dy, Vector const &g,
                               std::vector<Index> const &skip)
{
    const Vector r = J * dy - g;
    double num = 0.0, den = 0.0;
    for (Index i = 0; i < g.size(); ++i)
    {
        if (isUndefined(skip, i))
            continue;
        num += r(i) * r(i);
        den += g(i) * g(i);
    }
    return std::sqrt(num) / std::sqrt(den);
}

inline Vector toVector(N_Vector v)
{
    const Index n = N_VGetLength(v);
    const double *a = N_VGetArrayPointer(v);
    Vector out(n);
    for (Index i = 0; i < n; ++i)
        out(i) = a[i];
    return out;
}

} // namespace fdjac

#endif // FINITEDIFFERENCEJACOBIAN_HPP
