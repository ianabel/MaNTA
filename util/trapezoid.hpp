#pragma once

/*
 * Autodiff compatible implementation of the trapezoid rule
 *  Doubles the number of points at each iteration until error abs(DN) < tol, or maximum number of refinements is reached
 *  Theoretical convergence |DN| <= C/N^2
 */

template <class T, class F>
T trapezoid(const F &f, T a, T b, double tol = 1e-6, size_t max_refinements = 12)
{
    if (b == a)
        return static_cast<T>(0);
    else if (a > b)
        throw std::logic_error("Right endpoint must be larger than left");

    size_t n_refinements = 0;
    int N = 1;

    T dx = (b - a) / N;

    T TN = dx / 2.0 * (f(a) + f(b));

    T DN, MN;
    do
    {
        MN = 0.0;
        for (auto i = 0; i < N; ++i)
            MN += dx * f(a + dx / 2.0 + i * dx);
        DN = 0.5 * (MN - TN);
        TN += DN;
        dx /= 2;
        N *= 2;
        n_refinements++;
    } while (abs(DN) > tol && n_refinements < max_refinements);

    return TN;
}