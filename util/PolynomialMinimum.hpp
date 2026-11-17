#ifndef MANTA_UTIL_POLYNOMIALMINIMUM_HPP
#define MANTA_UTIL_POLYNOMIALMINIMUM_HPP

// The minimum over [-1, 1] of the polynomial that interpolates nodal values.
//
// This is what BoundCheck = "WholeCell" asks of every cell: a degree-p
// polynomial known by its values at p+1 reference nodes, and whether it dips
// below a bound anywhere in the cell rather than only at the nodes. The answer
// is exact, not sampled. A polynomial's minimum on a closed interval is taken
// at an endpoint or at a real root of its derivative, so the minimum over
// {-1, 1} and every root of p' that lies in [-1, 1] *is* the minimum. The
// roots are the eigenvalues of the derivative's colleague matrix (the
// Chebyshev analogue of the companion matrix; Good 1961, Boyd 2002), and the
// real part of every eigenvalue is tried, clipped into [-1, 1]: every
// candidate is a point of the interval, so a spurious one can only raise the
// minimum it is compared against, never lower it, and no tolerance on the
// imaginary part is needed to recognise a real root. Rounding moves a computed
// root by delta and the value there by O(p'' delta^2), so the guarantee is the
// exact minimum up to round-off.
//
// The polynomial is carried in the Chebyshev basis rather than monomials: the
// nodes here are Chebyshev points (NodalBasis), on which the Chebyshev
// Vandermonde is close to orthogonal and the monomial one is not.

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>

namespace manta
{

class PolynomialMinimum
{
public:
    struct Result
    {
        double value; // the minimum
        double s;     // where, in [-1, 1]
    };

    // `nodes` are the reference abscissae in [-1, 1], distinct.
    explicit PolynomialMinimum(Eigen::VectorXd const &nodes)
    {
        const Eigen::Index n = nodes.size();
        Eigen::MatrixXd V(n, n);
        for (Eigen::Index i = 0; i < n; ++i)
            for (Eigen::Index j = 0; j < n; ++j)
                V(i, j) = std::cos(static_cast<double>(j) * std::acos(std::clamp(nodes(i), -1.0, 1.0)));
        lu.compute(V);
    }

    Result minimum(Eigen::VectorXd const &values) const
    {
        return minimumOfSeries(lu.solve(values).eval());
    }

    // The value at s of sum_n a_n T_n(s), by Clenshaw's recurrence.
    static double evaluate(Eigen::VectorXd const &a, double s)
    {
        double b1 = 0.0, b2 = 0.0;
        for (Eigen::Index n = a.size() - 1; n >= 1; --n)
        {
            const double b0 = a(n) + 2.0 * s * b1 - b2;
            b2 = b1;
            b1 = b0;
        }
        return a.size() == 0 ? 0.0 : a(0) + s * b1 - b2;
    }

    // The minimum over [-1, 1] of sum_n a_n T_n(s).
    static Result minimumOfSeries(Eigen::VectorXd const &a)
    {
        Result best{std::numeric_limits<double>::infinity(), -1.0};
        auto consider = [&](double s)
        {
            s = std::clamp(s, -1.0, 1.0);
            const double v = evaluate(a, s);
            // Written so that a NaN coefficient gives a NaN minimum, which every
            // bound test then reads as a breach.
            if (!(v >= best.value))
            {
                best.value = v;
                best.s = s;
            }
        };
        consider(-1.0);
        consider(1.0);
        if (!a.allFinite())
            return Result{std::numeric_limits<double>::quiet_NaN(), -1.0};

        // The derivative's coefficients: b_{n-1} = b_{n+1} + 2 n a_n, b_0 halved.
        const Eigen::Index N = a.size() - 1;
        if (N < 2)
            return best; // constant or linear: the endpoints are all there is
        Eigen::VectorXd b = Eigen::VectorXd::Zero(N + 1);
        for (Eigen::Index n = N; n >= 1; --n)
            b(n - 1) = (n + 1 <= N ? b(n + 1) : 0.0) + 2.0 * static_cast<double>(n) * a(n);
        b(0) *= 0.5;
        b.conservativeResize(N); // degree N - 1

        // Its true degree: trailing coefficients at round-off of the largest
        // are noise, and dividing by one would put a spurious root anywhere.
        const double scale = b.cwiseAbs().maxCoeff();
        if (!(scale > 0.0))
            return best;
        Eigen::Index d = b.size() - 1;
        while (d > 0 && std::abs(b(d)) <= 1e-13 * scale)
            --d;
        if (d == 0)
            return best;
        if (d == 1)
        {
            consider(-b(0) / b(1));
            return best;
        }

        // Colleague matrix: s T_0 = T_1, s T_j = (T_{j-1} + T_{j+1}) / 2, and at
        // a root T_d = -sum_{j<d} b_j T_j / b_d.
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(d, d);
        C(0, 1) = 1.0;
        for (Eigen::Index i = 1; i < d; ++i)
        {
            C(i, i - 1) = 0.5;
            if (i + 1 < d)
                C(i, i + 1) = 0.5;
        }
        for (Eigen::Index j = 0; j < d; ++j)
            C(d - 1, j) -= 0.5 * b(j) / b(d);

        Eigen::EigenSolver<Eigen::MatrixXd> es(C, false);
        if (es.info() != Eigen::Success)
        {
            // Not expected for a matrix this small; fall back to a dense
            // sampling rather than report an unchecked cell as clean.
            for (int m = 0; m <= 64 * static_cast<int>(d); ++m)
                consider(-1.0 + 2.0 * m / (64.0 * static_cast<double>(d)));
            return best;
        }
        for (Eigen::Index i = 0; i < d; ++i)
            consider(es.eigenvalues()(i).real());
        return best;
    }

private:
    Eigen::PartialPivLU<Eigen::MatrixXd> lu;
};

} // namespace manta

#endif // MANTA_UTIL_POLYNOMIALMINIMUM_HPP
