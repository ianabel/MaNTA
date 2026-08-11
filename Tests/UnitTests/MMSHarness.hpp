#ifndef MMSHARNESS_HPP
#define MMSHARNESS_HPP

// The shared machinery for the method-of-manufactured-solutions order studies.
//
// Extracted from MMSConvergenceTests.cpp when the aux-variable and scalar
// studies were added, because those need the same sweep and the same
// least-squares fit but a different set of things to measure. Nothing here is
// specific to a particular manufactured problem; the problems themselves live
// in the test files.
//
// Every study in the tree shares one exact solution,
//
//     u(x, t) = sin(pi x) * (1 + t)      on [0, 1]
//
// which vanishes at both ends for every t, so it is consistent with the
// homogeneous Dirichlet boundary conditions. That matters: an MMS whose exact
// solution does not satisfy the boundary conditions imposed by the physics case
// converges at the wrong rate, or not at all.
//
// A problem that has a third thing worth measuring -- an auxiliary field, a
// global scalar -- declares
//
//     static double extraError(SystemSolver &, Grid const &, double t);
//
// and the harness picks it up through the HasExtraError concept below. A
// problem that declares nothing is entirely unaffected.

#include "CapturedOutput.hpp"
#include "SystemSolver.hpp"
#include "Types.hpp"

#include <boost/math/quadrature/gauss.hpp>

#include <cmath>
#include <cstdio>
#include <format>
#include <functional>
#include <numbers>
#include <string>
#include <typeinfo>
#include <vector>

namespace mms
{

using std::numbers::pi;

inline double exactSolution(double x, double t) { return std::sin(pi * x) * (1.0 + t); }
inline double exactDerivative(double x, double t)
{
    return pi * std::cos(pi * x) * (1.0 + t);
}

/// The manufactured forcing for sigma_hat = (1 + u^2) q, i.e. for
///
///     u_t - d_x[ (1 + u^2) u_x ] = S
///
/// Shared because two cases solve this same PDE: ManufacturedNonlinearFlux
/// directly, and ManufacturedAux with the (1 + u^2) factor routed through an
/// auxiliary variable. Having one definition is what makes that pair a
/// controlled comparison rather than two problems that are meant to match.
///
/// Note the minus sign in front of the divergence: the stored sigma is
/// -sigma_hat, so a source derived from u_t + d_x[sigma_hat] gives an
/// anti-diffusion equation that still converges, at the right rate, to the wrong
/// function.
inline double nonlinearFluxSource(double x, double t)
{
    const double A = 1.0 + t;
    const double s = std::sin(pi * x), c = std::cos(pi * x);
    return s + A * pi * pi * s * (1.0 + A * A * s * s) -
           2.0 * A * A * A * pi * pi * s * c * c;
}

/// Cell-by-cell Gauss-30 quadrature of (f - exact)^2. Independent of the basis's
/// own integration weights, which are part of what is under test.
inline double l2ErrorAgainst(std::function<double(double)> f,
                             std::function<double(double, double)> exact,
                             Grid const &grid, double t)
{
    boost::math::quadrature::gauss<double, 30> gauss;
    double total = 0.0;
    for (size_t cell = 0; cell < grid.getNCells(); ++cell)
    {
        Interval const &I = grid[cell];
        auto integrand = [&](double x)
        {
            const double d = f(x) - exact(x, t);
            return d * d;
        };
        total += gauss.integrate(integrand, I.x_l, I.x_u);
    }
    return std::sqrt(total);
}

/// The same, against the shared manufactured solution.
inline double l2ErrorOf(std::function<double(double)> f, Grid const &grid, double t)
{
    return l2ErrorAgainst(std::move(f), exactSolution, grid, t);
}

inline double l2Error(SystemSolver &sys, Grid const &grid, double t)
{
    return l2ErrorOf([&](double x) { return sys.yJac.u(0)(x); }, grid, t);
}

/// A problem that has something to measure beyond u and u*.
template <class P>
concept HasExtraError = requires(SystemSolver &s, Grid const &g) {
    { P::extraError(s, g, 0.0) } -> std::convertible_to<double>;
};

// The two errors every run reports, plus the optional third. HDG gives k+1 for
// the first; the second is k+2 when the method is superconvergent, and that
// difference is the whole point of the feature.
struct Errors
{
    double u;
    double uStar;
    double extra = 0.0;
};

// The time-integration tolerances a run is given.
//
// Parameterised rather than hardcoded because on a fine enough grid the spatial
// error drops below the temporal one, and then the sweep stops measuring space
// at all: the errors flatten and the observed order collapses, which looks
// exactly like a genuine loss of superconvergence. The only way to tell the two
// apart is to re-run the finest grid at a different tolerance and see whether
// the error moves. `atTolerance` below exists for that.
struct Tolerances
{
    // Tight enough that the temporal error is far below the spatial error over
    // the standard sweeps, so the measured rate is the spatial one.
    //
    // Not tighter by default: at 1e-12 IDA cannot get off the ground for
    // k >= 2, failing at t = 0 with "the error test failed repeatedly or with
    // |h| = hmin". That is a real limit of this solver, not of the manufactured
    // problem.
    double absolute = 1e-11;
    double relative = 1e-9;
};

// Run to tFinal on a uniform grid of nCells cells at degree k, and return the L2
// errors of the final solution and of its postprocessing.
//
// runSolver writes <stem>.nc / .dat / .restart.nc into the working directory,
// so the output name is unique per case and the files are removed afterwards.
template <class Problem>
Errors solveAndMeasureBoth(Index k, Index nCells, double tFinal,
                           bool superconvergent = false, Tolerances tol = {})
{
    Grid grid(0.0, 1.0, nCells);
    Problem problem;

    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.setSuperconvergent(superconvergent);
    sys.resetCoeffs();

    const std::string stem = "mms_" + std::string(typeid(Problem).name()) + "_k" +
                             std::to_string(k) + "_n" + std::to_string(nCells) +
                             (superconvergent ? "_sc" : "");
    sys.setInputFile(stem);

    sys.setOutputCadence(tFinal);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-14);
    sys.setTolerances({tol.absolute}, tol.relative);

    {
        // runSolver reports its step counts and IDACalcIC warnings; sixteen
        // integrations of that is a hundred lines of noise around a passing
        // test. The measured orders are reported by BOOST_TEST_MESSAGE instead.
        CapturedOutput quiet;
        sys.runSolver(tFinal);
    }

    // u* was last reconstructed from `y`, whose N_Vector runSolver has since
    // destroyed. Rebuild it from yJac, which the solver owns.
    sys.postprocessor->computeUStar(sys.yJac);

    Errors err{
        l2Error(sys, grid, tFinal),
        l2ErrorOf([&](double x) { return sys.getPostprocessor()->uStar(0)(x); }, grid,
                  tFinal)};

    if constexpr (HasExtraError<Problem>)
        err.extra = Problem::extraError(sys, grid, tFinal);

    for (const char *suffix : {".nc", ".dat", ".restart.nc"})
        std::remove((stem + suffix).c_str());

    return err;
}

/// The order between two consecutive grids, log(e0/e1) / log(n1/n0).
///
/// The least-squares fit below reports one number for a whole sweep, which is
/// the right summary only when the rate is constant across it. When it is not --
/// and the flag-off postprocessing of a nonlinear flux is not -- the fit reports
/// something in between and looks like a rate the method never has. Read these.
inline double localOrder(Index n0, Index n1, double e0, double e1)
{
    return std::log(e0 / e1) / std::log(static_cast<double>(n1) / static_cast<double>(n0));
}

/// Least-squares slope of log(error) against log(1/nCells) -- the observed order.
inline double observedOrder(std::vector<Index> const &cellCounts,
                            std::vector<double> const &errors)
{
    const size_t n = cellCounts.size();
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double x = std::log(1.0 / static_cast<double>(cellCounts[i]));
        const double y = std::log(errors[i]);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

struct Rates
{
    double uOff, starOff, uOn, starOn;
    double extraOff = 0.0, extraOn = 0.0;

    // The errors the fit was made from. A slope alone cannot distinguish a
    // converged rate from a pre-asymptotic sweep, and that distinction is
    // exactly what decides whether an observed order is worth asserting on.
    std::string detail;

    // ...and the raw sweep, so a test can assert on the local orders directly
    // rather than on a single fit that averages a changing rate away.
    std::vector<Index> cells;
    std::vector<double> uOffErr, starOffErr, uOnErr, starOnErr;

    /// The order of the flag-off u* column between grids i-1 and i.
    double localStarOff(size_t i) const
    {
        return localOrder(cells[i - 1], cells[i], starOffErr[i - 1], starOffErr[i]);
    }
    /// The same for the flag-on column.
    double localStarOn(size_t i) const
    {
        return localOrder(cells[i - 1], cells[i], starOnErr[i - 1], starOnErr[i]);
    }
};

/// Refine, fit both orders, flag off and flag on, and report all four (or six).
template <class Problem>
Rates measureRates(Index k, std::vector<Index> const &cells, double tFinal,
                   Tolerances tol = {})
{
    std::vector<double> uOff, starOff, uOn, starOn, extraOff, extraOn;
    for (Index n : cells)
    {
        const Errors off = solveAndMeasureBoth<Problem>(k, n, tFinal, false, tol);
        const Errors on = solveAndMeasureBoth<Problem>(k, n, tFinal, true, tol);
        uOff.push_back(off.u);
        starOff.push_back(off.uStar);
        uOn.push_back(on.u);
        starOn.push_back(on.uStar);
        extraOff.push_back(off.extra);
        extraOn.push_back(on.extra);
    }

    Rates r{observedOrder(cells, uOff), observedOrder(cells, starOff),
            observedOrder(cells, uOn), observedOrder(cells, starOn)};
    r.cells = cells;
    r.uOffErr = uOff;
    r.starOffErr = starOff;
    r.uOnErr = uOn;
    r.starOnErr = starOn;

    // Scientific, not std::to_string: these errors reach 1e-9 and a fixed six
    // decimal places rounds the interesting end of every sweep to "0.000000".
    // Each row also carries the order against the previous grid, which is what
    // shows a rate that changes partway through a sweep.
    for (size_t i = 0; i < cells.size(); ++i)
    {
        r.detail += std::format("\n      n={:<4} off u {:.3e} u* {:.3e}  on u {:.3e} u* {:.3e}",
                                cells[i], uOff[i], starOff[i], uOn[i], starOn[i]);
        if constexpr (HasExtraError<Problem>)
            r.detail += std::format("  extra off {:.3e} on {:.3e}", extraOff[i], extraOn[i]);
        if (i > 0)
            r.detail += std::format(
                "\n            local order: off u {:.2f} u* {:.2f}  on u {:.2f} u* {:.2f}",
                localOrder(cells[i - 1], cells[i], uOff[i - 1], uOff[i]),
                localOrder(cells[i - 1], cells[i], starOff[i - 1], starOff[i]),
                localOrder(cells[i - 1], cells[i], uOn[i - 1], uOn[i]),
                localOrder(cells[i - 1], cells[i], starOn[i - 1], starOn[i]));
    }

    // Only fit the extra column when there is one. log(0) is -inf, so fitting a
    // vector of zeros would put a NaN in the report of every problem that has
    // nothing extra to measure.
    if constexpr (HasExtraError<Problem>)
    {
        r.extraOff = observedOrder(cells, extraOff);
        r.extraOn = observedOrder(cells, extraOn);
    }

    return r;
}

/// One line per degree, for BOOST_TEST_MESSAGE. `extraName` names the third
/// column when the problem has one.
inline std::string report(Index k, Rates const &r, const char *extraName = nullptr)
{
    std::string s = "k = " + std::to_string(k) + ":  flag off  u " +
                    std::to_string(r.uOff) + "  u* " + std::to_string(r.starOff) +
                    "   |   flag on  u " + std::to_string(r.uOn) + "  u* " +
                    std::to_string(r.starOn);
    if (extraName != nullptr)
        s += "   |   " + std::string(extraName) + "  off " + std::to_string(r.extraOff) +
             "  on " + std::to_string(r.extraOn);
    s += "   (u should be " + std::to_string(k + 1) + ", u* with the flag on " +
         std::to_string(k + 2) + ")" + r.detail;
    return s;
}

} // namespace mms

#endif // MMSHARNESS_HPP
