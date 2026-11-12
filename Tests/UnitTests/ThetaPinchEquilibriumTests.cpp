#include <Eigen/Core>
#include <Eigen/Dense>

#include "../../PhysicsCases/ThetaPinchEquilibrium.hpp"
#include "../../DGSoln.hpp"
#include "../../Types.hpp"

#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/test/unit_test.hpp>

#include <cmath>
#include <tuple>
#include <vector>

/*
    The theta-pinch field model on its own, with no solver: its spline, its
    derivatives against finite differences, its block solve against a dense one,
    and its equilibrium against the closed form

        V'(x) = Phi / sqrt(C - p(x)),   C fixed by Int_0^1 V' dx = pi a^2.
 */

namespace
{

using Model = ThetaPinchEquilibrium;

/// The transport nodes of a grid at degree k, with a single auxiliary variable
/// holding p(x) at each one: the shape FieldResidual is handed.
template <typename F>
std::tuple<GlobalState, std::vector<Position>, Vector> pressureOnNodes(Grid const &grid, Index k, F p)
{
    DGSoln soln(1, grid, k, Index(0), Index(1));
    std::vector<double> mem(soln.getDoF(), 0.0);
    soln.Map(mem.data());
    GlobalState states = soln.evalOnNodes();
    std::vector<Position> points = soln.getPoints();
    for (size_t j = 0; j < points.size(); ++j)
        states.Aux()(0, static_cast<Index>(j)) = p(points[j]);
    Vector weights = Vector::Zero(static_cast<Index>(points.size())); // unused by the model
    return {states, points, weights};
}

/// A V' near the vacuum one, nonconstant and positive.
Vector perturbedField(Model const &m)
{
    Vector psi(m.nCoefficients() + 1);
    for (Index n = 0; n < m.nCoefficients(); ++n)
        psi(n) = m.wallArea() * (1.0 + 0.2 * std::sin(1.3 * n + 0.4));
    psi(m.nCoefficients()) = 1.1;
    return psi;
}

GlobalStateMatrix zeroDerivatives(Index nRows, Grid const &grid, Index k)
{
    GlobalStateMatrix out(nRows);
    for (Index m = 0; m < nRows; ++m)
        out.add(grid.getNCells(), k, 1, 0, 1);
    return out;
}

const Grid unitGrid(0.0, 1.0, 5);
// Unequal cells, so the knots are not uniform and the end extensions differ.
const Grid gradedGrid(std::vector<double>{0.0, 0.1, 0.25, 0.45, 0.7, 1.0});

/// The distinct knots of a model on `grid` with m intervals per cell: what the
/// spline is piecewise polynomial between.
std::vector<double> knotsOf(Grid const &grid, Index m)
{
    std::vector<double> out{0.0};
    for (Index i = 0; i < static_cast<Index>(grid.getNCells()); ++i)
        for (Index j = 1; j <= m; ++j)
            out.push_back(grid[i].x_l + (grid[i].x_u - grid[i].x_l) * static_cast<double>(j) / static_cast<double>(m));
    return out;
}

} // namespace

BOOST_AUTO_TEST_SUITE(theta_pinch_equilibrium_tests)

BOOST_AUTO_TEST_CASE(the_spline_is_a_partition_of_unity_with_consistent_derivatives)
{
    Model m({.intervalsPerCell = 2}, gradedGrid);
    const double eps = 1e-6;
    for (double x : {0.0, 0.03, 0.05, 0.1, 0.31, 0.45, 0.5, 0.77, 0.85, 0.999, 1.0})
    {
        BOOST_TEST_CONTEXT("x = " << x)
        {
            const auto l = m.localBasis(x);
            BOOST_TEST(l.d.row(0).sum() == 1.0, boost::test_tools::tolerance(1e-14));
            BOOST_TEST(std::abs(l.d.row(1).sum()) < 1e-11);

            // First derivatives against second-order differences of the values,
            // one-sided where x is at an end.
            auto value = [&](double y, Index n)
            {
                const auto ly = m.localBasis(y);
                const Index r = n - ly.first;
                return (r >= 0 && r < 4) ? ly.d(0, r) : 0.0;
            };
            for (Index r = 0; r < 4; ++r)
            {
                const Index n = l.first + r;
                double fd;
                if (x - eps < 0.0)
                    fd = (-3.0 * value(x, n) + 4.0 * value(x + eps, n) - value(x + 2 * eps, n)) / (2 * eps);
                else if (x + eps > 1.0)
                    fd = (3.0 * value(x, n) - 4.0 * value(x - eps, n) + value(x - 2 * eps, n)) / (2 * eps);
                else
                    fd = (value(x + eps, n) - value(x - eps, n)) / (2 * eps);
                BOOST_TEST(std::abs(l.d(1, r) - fd) < 1e-6 * (1.0 + std::abs(fd)),
                           "basis " << n << ": " << l.d(1, r) << " against " << fd);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(the_antiderivative_is_the_integral_of_each_basis_function)
{
    Model m({.intervalsPerCell = 2}, gradedGrid);
    const std::vector<double> knots = knotsOf(gradedGrid, 2);
    using GK = boost::math::quadrature::gauss_kronrod<double, 31>;
    for (double x : {0.0, 0.1, 0.175, 0.42, 0.5, 0.9, 1.0})
    {
        BOOST_TEST_CONTEXT("x = " << x)
        {
            const Vector w = m.antiderivativeWeights(x);
            for (Index n = 0; n < m.nCoefficients(); ++n)
            {
                // Piecewise polynomial, so integrate knot interval by interval.
                double ref = 0.0;
                for (size_t c = 0; c + 1 < knots.size(); ++c)
                {
                    const double a = knots[c], b = std::min(x, knots[c + 1]);
                    if (b <= a)
                        break;
                    ref += GK::integrate([&](double y)
                                         {
                                             const auto l = m.localBasis(y);
                                             const Index r = n - l.first;
                                             return (r >= 0 && r < 4) ? l.d(0, r) : 0.0; },
                                         a, b, 0, 1e-15);
                }
                BOOST_TEST(w(n) == ref, boost::test_tools::tolerance(1e-13));
            }
        }
    }
    // A partition of unity integrates to the length of [0, 1]; and on uniform
    // knots an interior basis function integrates to one interval width.
    BOOST_TEST(m.antiderivativeWeights(1.0).sum() == 1.0, boost::test_tools::tolerance(1e-14));
    const Model uniform({.intervalsPerCell = 2}, unitGrid);
    BOOST_TEST(uniform.antiderivativeWeights(1.0)(3) == 0.1, boost::test_tools::tolerance(1e-14));
}

BOOST_AUTO_TEST_CASE(the_vacuum_state_is_a_uniform_field)
{
    Model m({.intervalsPerCell = 2, .wallRadius = 1.5, .vacuumField = 2.0}, gradedGrid);
    Vector psi(m.nCoefficients() + 1);
    m.InitialFieldValue(psi);
    Vector g(4);
    for (double x : {0.0, 0.3, 1.0})
    {
        g.setZero();
        m.Geometry(g, psi, x, 0.0);
        BOOST_TEST(g(Model::Vprime) == M_PI * 2.25, boost::test_tools::tolerance(1e-14));
        BOOST_TEST(std::abs(g(Model::dVprime_dx)) < 1e-12);
        BOOST_TEST(g(Model::Bfield) == 2.0, boost::test_tools::tolerance(1e-14));
        // x = r^2 / a^2, so |grad x|^2 = 4 r^2 / a^4 = 4 x / a^2.
        BOOST_TEST(g(Model::gradx2) == 4.0 * x / 2.25, boost::test_tools::tolerance(1e-13));
    }

    // And with no pressure it is the equilibrium: every row vanishes.
    auto [states, points, weights] = pressureOnNodes(gradedGrid, 3, [](double) { return 0.0; });
    Vector R = Vector::Zero(m.nCoefficients() + 1);
    m.FieldResidual(R, psi, Vector::Zero(psi.size()), states, points, weights, 0.0);
    BOOST_TEST(R.cwiseAbs().maxCoeff() < 1e-12);
}

BOOST_AUTO_TEST_CASE(dgeometry_dpsi_matches_finite_differences)
{
    Model m({.intervalsPerCell = 2}, gradedGrid);
    const Vector psi = perturbedField(m);
    const Index n = psi.size();
    Matrix dG(4, n);
    Vector gp(4), gm(4);
    for (double x : {0.0, 0.07, 0.125, 0.48, 0.91, 1.0})
    {
        BOOST_TEST_CONTEXT("x = " << x)
        {
            dG.setZero();
            m.dGeometry_dpsi(dG, psi, x, 0.0);
            for (Index c = 0; c < n; ++c)
            {
                const double eps = 1e-6 * std::max(1.0, std::abs(psi(c)));
                Vector p = psi, q = psi;
                p(c) += eps;
                q(c) -= eps;
                gp.setZero();
                gm.setZero();
                m.Geometry(gp, p, x, 0.0);
                m.Geometry(gm, q, x, 0.0);
                const Vector fd = (gp - gm) / (2.0 * eps);
                for (Index s = 0; s < 4; ++s)
                    BOOST_TEST(std::abs(dG(s, c) - fd(s)) < 1e-7 * (1.0 + std::abs(fd(s))),
                               "slot " << s << ", dof " << c << ": " << dG(s, c) << " against " << fd(s));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(the_residual_derivatives_match_finite_differences)
{
    Model m({.intervalsPerCell = 2}, gradedGrid);
    const Index k = 3;
    auto [states, points, weights] = pressureOnNodes(gradedGrid, k, [](double x) { return 0.3 * (1.0 - x * x); });
    const Vector psi = perturbedField(m);
    const Index n = psi.size();
    const Vector zero = Vector::Zero(n);

    auto R = [&](Vector const &ps, GlobalState const &st)
    {
        Vector out = Vector::Zero(n);
        m.FieldResidual(out, ps, zero, st, points, weights, 0.0);
        return out;
    };

    GlobalStateMatrix dR = zeroDerivatives(n, gradedGrid, k), dRdot = zeroDerivatives(n, gradedGrid, k);
    Matrix dRdpsi = Matrix::Zero(n, n), dRddpsidt = Matrix::Zero(n, n);
    m.FieldResidualPrime(dR, dRdot, dRdpsi, dRddpsidt, psi, zero, states, points, weights, 0.0);

    // The field block.
    for (Index c = 0; c < n; ++c)
    {
        const double eps = 1e-6 * std::max(1.0, std::abs(psi(c)));
        Vector p = psi, q = psi;
        p(c) += eps;
        q(c) -= eps;
        const Vector fd = (R(p, states) - R(q, states)) / (2.0 * eps);
        BOOST_TEST((dRdpsi.col(c) - fd).cwiseAbs().maxCoeff() < 1e-7 * (1.0 + fd.cwiseAbs().maxCoeff()),
                   "column " << c);
    }
    BOOST_TEST(dRddpsidt.cwiseAbs().maxCoeff() == 0.0);

    // Through the pressure. The residual is affine in p, so this is exact.
    for (Index j = 0; j < static_cast<Index>(points.size()); ++j)
    {
        GlobalState plus = states, minus = states;
        plus.Aux()(0, j) += 1e-3;
        minus.Aux()(0, j) -= 1e-3;
        const Vector fd = (R(psi, plus) - R(psi, minus)) / 2e-3;
        for (Index row = 0; row < n; ++row)
            BOOST_TEST(std::abs(dR[row].Aux()(0, j) - fd(row)) < 1e-12, "row " << row << ", node " << j);
        BOOST_TEST(dR[n - 1].Aux()(0, j) == 0.0);
    }
}

BOOST_AUTO_TEST_CASE(the_pressure_projection_integrates_the_interpolant_exactly)
{
    // A polynomial of degree k is its own interpolant, so P p must reproduce
    // Int phi_m p dx to rounding, with field knots falling inside transport
    // cells (three intervals per cell, on unequal cells).
    Model m({.intervalsPerCell = 3}, gradedGrid);
    const std::vector<double> knots = knotsOf(gradedGrid, 3);
    const Index k = 4;
    auto p = [](double x) { return 1.0 + x - 2.0 * x * x + 0.5 * std::pow(x, 4); };
    auto [states, points, weights] = pressureOnNodes(gradedGrid, k, p);
    const Vector pv = states.Aux().row(0).transpose();
    const Vector got = m.pressureProjection(points) * pv;

    using GK = boost::math::quadrature::gauss_kronrod<double, 31>;
    for (Index n = 0; n < m.nCoefficients(); ++n)
    {
        double ref = 0.0;
        for (size_t c = 0; c + 1 < knots.size(); ++c)
            ref += GK::integrate([&](double y)
                                 {
                                     const auto l = m.localBasis(y);
                                     const Index r = n - l.first;
                                     return ((r >= 0 && r < 4) ? l.d(0, r) : 0.0) * p(y); },
                                 knots[c], knots[c + 1], 0, 1e-15);
        BOOST_TEST(got(n) == ref, boost::test_tools::tolerance(1e-12));
    }
}

BOOST_AUTO_TEST_CASE(the_field_block_solves_agree_with_a_dense_solve)
{
    Model m({.intervalsPerCell = 2}, unitGrid);
    auto [states, points, weights] = pressureOnNodes(unitGrid, 3, [](double x) { return 0.2 * (1.0 - x); });
    const Vector psi = perturbedField(m);
    const Index n = psi.size();
    GlobalStateMatrix dR = zeroDerivatives(n, unitGrid, 3), dRdot = zeroDerivatives(n, unitGrid, 3);
    Matrix dRdpsi = Matrix::Zero(n, n), dRddpsidt = Matrix::Zero(n, n);
    m.FieldResidualPrime(dR, dRdot, dRdpsi, dRddpsidt, psi, Vector::Zero(n), states, points, weights, 0.0);
    m.updateFieldJacobian(dRdpsi, dRddpsidt, 0.0);

    Vector rhs(n);
    for (Index i = 0; i < n; ++i)
        rhs(i) = std::cos(0.7 * i + 0.1);
    Vector out(n);

    Vector ref = dRdpsi.fullPivLu().solve(rhs);
    m.solveB(out, rhs);
    BOOST_TEST((out - ref).norm() < 1e-11 * ref.norm());

    ref = dRdpsi.transpose().fullPivLu().solve(rhs);
    m.solveBTranspose(out, rhs);
    BOOST_TEST((out - ref).norm() < 1e-11 * ref.norm());

    m.applyB(out, rhs);
    BOOST_TEST((out - dRdpsi * rhs).norm() < 1e-13 * (dRdpsi * rhs).norm());
    m.applyBTranspose(out, rhs);
    BOOST_TEST((out - dRdpsi.transpose() * rhs).norm() < 1e-13 * (dRdpsi.transpose() * rhs).norm());

    // Not vacuous: the block is not symmetric (the border's signs differ), so a
    // transpose solve that ignored the transpose is wrong by far more than the
    // 1e-11 the right one agrees to. Measured: 1e-4 relative.
    Vector wrong(n);
    m.solveB(wrong, rhs);
    BOOST_TEST((wrong - ref).norm() > 1e-6 * ref.norm());
}

BOOST_AUTO_TEST_CASE(the_equilibrium_converges_to_the_closed_form_at_fourth_order)
{
    // p = beta cos^2(pi x / 2): peaked on axis, zero at the wall.
    const double beta = 0.5;
    auto p = [beta](double x) { const double c = std::cos(0.5 * M_PI * x); return beta * c * c; };

    // a = B0 = 1, so Phi = pi and Int_0^1 pi / sqrt(C - p) dx = pi.
    using GK = boost::math::quadrature::gauss_kronrod<double, 61>;
    auto constraint = [&](double C)
    { return GK::integrate([&](double x) { return 1.0 / std::sqrt(C - p(x)); }, 0.0, 1.0, 0, 1e-15) - 1.0; };
    boost::math::tools::eps_tolerance<double> tol(50);
    std::uintmax_t iters = 100;
    const auto bracket = boost::math::tools::bisect(constraint, beta + 1e-6, 4.0, tol, iters);
    const double Cexact = 0.5 * (bracket.first + bracket.second);
    auto VpExact = [&](double x) { return M_PI / std::sqrt(Cexact - p(x)); };

    // The pressure at a degree high enough that its own interpolation error,
    // O(h^9), stays well below the spline's O(h^4).
    const Index k = 8;

    std::vector<double> errors;
    const std::vector<Index> Ms{4, 8, 16, 32, 64};
    for (Index M : Ms)
    {
        const Grid fine(0.0, 1.0, M);
        Model m({.intervalsPerCell = 1}, fine);
        auto [states, points, weights] = pressureOnNodes(fine, k, p);
        const Index n = m.nCoefficients() + 1;
        Vector psi(n), R(n), dpsi(n);
        m.InitialFieldValue(psi);
        GlobalStateMatrix dR = zeroDerivatives(n, fine, k), dRdot = zeroDerivatives(n, fine, k);
        Matrix dRdpsi(n, n), dRddpsidt = Matrix::Zero(n, n);
        for (int it = 0; it < 30; ++it)
        {
            R.setZero();
            m.FieldResidual(R, psi, Vector::Zero(n), states, points, weights, 0.0);
            if (R.cwiseAbs().maxCoeff() < 1e-14)
                break;
            dRdpsi.setZero();
            m.FieldResidualPrime(dR, dRdot, dRdpsi, dRddpsidt, psi, Vector::Zero(n), states, points,
                                 weights, 0.0);
            m.updateFieldJacobian(dRdpsi, dRddpsidt, 0.0);
            m.solveB(dpsi, R);
            psi -= dpsi;
        }
        BOOST_TEST_REQUIRE(R.cwiseAbs().maxCoeff() < 1e-12, "Newton did not converge at M = " << M);

        double err = 0.0;
        Vector g(4);
        for (Index s = 0; s <= 1000; ++s)
        {
            const double x = s / 1000.0;
            g.setZero();
            m.Geometry(g, psi, x, 0.0);
            err = std::max(err, std::abs(g(Model::Vprime) - VpExact(x)));
        }
        errors.push_back(err);
        BOOST_TEST_MESSAGE("M = " << M << ": max |V' - V'_exact| = " << err << ", C - C_exact = "
                                  << psi(n - 1) - Cexact);
    }

    // Local orders, not a fitted slope: a fit would average away a rate that
    // is changing.
    for (size_t i = 1; i < errors.size(); ++i)
    {
        const double order = std::log2(errors[i - 1] / errors[i]);
        BOOST_TEST_MESSAGE("local order " << Ms[i - 1] << " -> " << Ms[i] << ": " << order);
        // Pre-asymptotic below M = 16 (measured 5.3, then 3.4); 3.8 and 3.9 after.
        if (Ms[i - 1] >= 16)
            BOOST_TEST(order > 3.7, "local order " << order << " between M = " << Ms[i - 1] << " and " << Ms[i]);
    }
}

BOOST_AUTO_TEST_CASE(a_non_positive_vprime_is_a_recoverable_failure)
{
    Model m({.intervalsPerCell = 1}, unitGrid);
    auto [states, points, weights] = pressureOnNodes(unitGrid, 3, [](double) { return 0.0; });
    Vector psi = perturbedField(m);
    psi(4) = -10.0;
    Vector R = Vector::Zero(psi.size());
    // runtime_error, which static_residual catches and turns into IDA's
    // recoverable return; not invalid_argument, which would mean misuse.
    BOOST_CHECK_THROW(m.FieldResidual(R, psi, Vector::Zero(psi.size()), states, points, weights, 0.0),
                      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(it_is_registered_and_reads_its_own_table)
{
    using namespace toml::literals::toml_literals;
    const toml::value config = u8R"(
        [ThetaPinchEquilibrium]
        IntervalsPerCell = 3
        PressureAux = 0
        WallRadius = 2.0
    )"_toml;
    auto model = FieldModels::InstantiateFieldModel("ThetaPinchEquilibrium", config, unitGrid);
    BOOST_TEST(model->nFieldDOF() == 5 * 3 + 3 + 1);
    BOOST_TEST(model->nGeometry() == 4);
    auto const *tp = dynamic_cast<Model const *>(model.get());
    BOOST_REQUIRE(tp != nullptr);
    BOOST_TEST(tp->wallArea() == 4.0 * M_PI, boost::test_tools::tolerance(1e-15));

    BOOST_CHECK_THROW(Model(Model::Parameters{}, Grid(0.0, 2.0, 4)), std::invalid_argument);
    BOOST_CHECK_THROW(Model({.intervalsPerCell = 0}, unitGrid), std::invalid_argument);

    // A key the model does not read is refused, not run on the default.
    const toml::value misspelt = u8R"(
        [ThetaPinchEquilibrium]
        Intervals = 16
    )"_toml;
    BOOST_CHECK_THROW(FieldModels::InstantiateFieldModel("ThetaPinchEquilibrium", misspelt, unitGrid),
                      std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
