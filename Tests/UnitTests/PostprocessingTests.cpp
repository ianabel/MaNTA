// Tests for the element-local superconvergent postprocessing.
//
// The whole superconvergence feature rests on four small per-cell matrices --
// B11, B12, V and A9 -- and none of them produces an obviously wrong answer when
// it is subtly wrong. A sign error in the A2 block (MaNTA's q = +d_x u against
// the paper's q = -grad u) still yields a u* that satisfies the mean constraint
// and looks like a plausible profile; it just converges at the wrong rate,
// hundreds of lines away in an integration test.
//
// So the anchor here is polynomial exactness: the reconstruction
//
//     ( d_x u*, d_x z )_K + ( eta, z )_K = ( q_h, d_x z )_K   for z in P_{k+1}
//     ( u*, w )_K                        = ( u_h, w )_K       for w in P_0
//
// reproduces any polynomial of degree <= k+1 exactly when it is fed that
// polynomial's own u_h and q_h. That single property pins the sign, the scaling
// with h, and both blocks at once.

#include <boost/test/unit_test.hpp>

#include "Postprocessing.hpp"
#include "DGSoln.hpp"
#include "gridStructures.hpp"

#include <boost/math/quadrature/gauss.hpp>

#include <cmath>
#include <functional>
#include <numbers>
#include <vector>

namespace
{

using std::numbers::pi;

// A DGSoln with u and q set from functions, on memory we own.
struct Soln
{
    std::vector<double> mem;
    DGSoln y;

    // The 4th argument must be a typed Index: a bare 0 is ambiguous between
    // DGSoln's (nScalars, nAux) and (double *memory, ...) constructors.
    Soln(Index nVars, Grid const &grid, Index k, Index nAux = 0)
        : mem(DGSoln(nVars, grid, k, Index{0}, nAux).getDoF(), 0.0),
          y(nVars, grid, k, mem.data(), Index{0}, nAux)
    {
    }

    void assign(std::function<double(Index, double)> u_fn,
                std::function<double(Index, double)> q_fn)
    {
        y.AssignU(u_fn);
        y.AssignQ(q_fn);
    }
};

// Cell-by-cell Gauss-30 quadrature, independent of anything the postprocessor
// uses internally.
double cellIntegral(Interval const &I, std::function<double(double)> f)
{
    static boost::math::quadrature::gauss<double, 30> gauss;
    return gauss.integrate(f, I.x_l, I.x_u);
}

double l2CellError(Interval const &I, std::function<double(double)> a,
                   std::function<double(double)> b)
{
    return std::sqrt(cellIntegral(I, [&](double x)
                                  { const double d = a(x) - b(x); return d * d; }));
}

// The two grids every test runs on: uniform, and the strongly non-uniform grid
// Grid's High_Grid_Boundary option builds. The per-cell matrices scale with h,
// so a mistake in that scaling is invisible on a uniform grid.
std::vector<Grid> testGrids()
{
    std::vector<Grid> out;
    out.emplace_back(0.0, 1.0, 5);
    out.emplace_back(-2.0, 3.0, 9, true, 0.2, 0.2);
    return out;
}

} // namespace

BOOST_AUTO_TEST_SUITE(postprocessing_tests)

BOOST_AUTO_TEST_CASE(reconstruction_is_exact_for_polynomials_of_degree_k_plus_one)
{
    // The headline property. u = x^(k+1) is in P_{k+1} but not in P_k, so u_h
    // cannot represent it and u* must recover it from the gradient information
    // carried by q_h.
    for (Grid const &grid : testGrids())
    {
        for (Index k = 1; k <= 4; ++k)
        {
            auto u_exact = [k](double x) { return std::pow(x, k + 1) + 0.5 * x - 1.25; };
            auto q_exact = [k](double x)
            { return (k + 1) * std::pow(x, k) + 0.5; };

            Soln s(1, grid, k);
            s.assign([&](Index, double x) { return u_exact(x); },
                     [&](Index, double x) { return q_exact(x); });

            Postprocessor pp(grid, k, 1);
            pp.computeUStar(s.y);

            for (Index cell = 0; cell < static_cast<Index>(grid.getNCells()); ++cell)
            {
                Interval const &I(grid[cell]);
                const double err = l2CellError(
                    I, [&](double x) { return pp.uStar(0)(x); }, u_exact);
                const double scale = std::sqrt(cellIntegral(
                    I, [&](double x) { return u_exact(x) * u_exact(x); }));

                BOOST_TEST(err < 1e-10 * std::max(scale, 1.0),
                           "k = " << k << ", cell " << cell << " on ["
                                  << I.x_l << ", " << I.x_u
                                  << "]: u* missed a degree-" << k + 1
                                  << " polynomial by " << err);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(reconstruction_reproduces_a_degree_k_solution)
{
    // The weaker consistency check: if u is already representable in P_k, u*
    // must equal it. This one would still pass with B11 scaled wrongly (q's
    // contribution is then also representable), which is why the test above
    // exists -- but it fails outright if B12 is wrong.
    for (Grid const &grid : testGrids())
    {
        for (Index k = 1; k <= 4; ++k)
        {
            auto u_exact = [k](double x) { return std::pow(x, k) - 2.0 * x; };
            auto q_exact = [k](double x) { return k * std::pow(x, k - 1) - 2.0; };

            Soln s(1, grid, k);
            s.assign([&](Index, double x) { return u_exact(x); },
                     [&](Index, double x) { return q_exact(x); });

            Postprocessor pp(grid, k, 1);
            pp.computeUStar(s.y);

            for (Index cell = 0; cell < static_cast<Index>(grid.getNCells()); ++cell)
            {
                Interval const &I(grid[cell]);
                const double err = l2CellError(
                    I, [&](double x) { return pp.uStar(0)(x); }, u_exact);
                BOOST_TEST(err < 1e-10,
                           "k = " << k << ", cell " << cell << ": err " << err);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(the_cell_mean_of_u_star_matches_the_cell_mean_of_u)
{
    // The second equation of the reconstruction, checked directly. This is what
    // fixes the constant that the pure-Neumann first equation leaves free, so if
    // it does not hold the Lagrange-multiplier row of the bordered system is
    // wired up wrongly.
    for (Grid const &grid : testGrids())
    {
        for (Index k = 1; k <= 4; ++k)
        {
            Soln s(1, grid, k);
            s.assign([](Index, double x) { return std::exp(-x) * std::sin(3.0 * x); },
                     [](Index, double x)
                     { return -std::exp(-x) * std::sin(3.0 * x) +
                              3.0 * std::exp(-x) * std::cos(3.0 * x); });

            Postprocessor pp(grid, k, 1);
            pp.computeUStar(s.y);

            for (Index cell = 0; cell < static_cast<Index>(grid.getNCells()); ++cell)
            {
                Interval const &I(grid[cell]);
                const double meanStar =
                    cellIntegral(I, [&](double x) { return pp.uStar(0)(x); });
                const double meanU =
                    cellIntegral(I, [&](double x) { return s.y.u(0)(x); });

                BOOST_TEST(std::abs(meanStar - meanU) <
                               1e-11 * std::max(std::abs(meanU), 1.0),
                           "k = " << k << ", cell " << cell << ": means "
                                  << meanStar << " vs " << meanU);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(a9_projects_a_star_field_onto_the_p_k_test_space)
{
    // A9 replaces the mass matrix that InterpolateOntoBasis applies when the
    // interpolation is into P_k: given the nodal values of a P_{k+1} function it
    // must return ( that function, phi_i )_K. Checked against direct quadrature.
    for (Grid const &grid : testGrids())
    {
        for (Index k = 1; k <= 4; ++k)
        {
            Postprocessor pp(grid, k, 1);
            NodalBasis const &starBasis = pp.getStarBasis();
            NodalBasis basis = NodalBasis::getBasis(k);

            auto g = [](double x) { return std::cos(2.0 * x) + x * x; };

            for (Index cell = 0; cell < static_cast<Index>(grid.getNCells()); ++cell)
            {
                Interval const &I(grid[cell]);

                // g interpolated into P_{k+1}: nodal values at the star nodes.
                Vector vals(k + 2);
                for (Index m = 0; m < k + 2; ++m)
                    vals(m) = g(I.fromRef(starBasis.getNodes()(m)));

                const Vector projected = pp.A9(cell) * vals;

                auto interpolant = [&](double x)
                { return starBasis.Evaluate(I, vals, x); };

                for (Index i = 0; i < k + 1; ++i)
                {
                    const double expected = cellIntegral(
                        I, [&](double x)
                        { return interpolant(x) * basis.Evaluate(I, i, x); });
                    BOOST_TEST(std::abs(projected(i) - expected) < 1e-12,
                               "k = " << k << ", cell " << cell << ", i = " << i
                                      << ": " << projected(i) << " vs " << expected);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(a9_on_a_degree_k_field_agrees_with_interpolate_onto_basis)
{
    // The bridge between the old and new residual assembly: for a function that
    // P_k already represents exactly, projecting its P_{k+1} interpolant with A9
    // must give the same vector InterpolateOntoBasis gives from its P_k nodal
    // values. This is precisely the invariance the flag-on residual relies on.
    for (Grid const &grid : testGrids())
    {
        for (Index k = 1; k <= 4; ++k)
        {
            Postprocessor pp(grid, k, 1);
            NodalBasis const &starBasis = pp.getStarBasis();
            NodalBasis basis = NodalBasis::getBasis(k);

            // A polynomial of degree exactly k -- in P_k, so both interpolations
            // reproduce it.
            auto g = [k](double x) { return std::pow(x, k) + 3.0 * x - 1.0; };

            for (Index cell = 0; cell < static_cast<Index>(grid.getNCells()); ++cell)
            {
                Interval const &I(grid[cell]);

                Vector starVals(k + 2);
                for (Index m = 0; m < k + 2; ++m)
                    starVals(m) = g(I.fromRef(starBasis.getNodes()(m)));

                Vector localVals(k + 1);
                for (Index j = 0; j < k + 1; ++j)
                    localVals(j) = g(I.fromRef(basis.getNodes()(j)));

                const Vector viaA9 = pp.A9(cell) * starVals;
                const Vector viaInterp = basis.InterpolateOntoBasis(I, localVals);

                for (Index i = 0; i < k + 1; ++i)
                    BOOST_TEST(std::abs(viaA9(i) - viaInterp(i)) <
                                   1e-11 * std::max(std::abs(viaInterp(i)), 1.0),
                               "k = " << k << ", cell " << cell << ", i = " << i
                                      << ": " << viaA9(i) << " vs " << viaInterp(i));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(v_evaluates_a_degree_k_field_at_the_star_nodes)
{
    for (Grid const &grid : testGrids())
    {
        for (Index k = 1; k <= 4; ++k)
        {
            Postprocessor pp(grid, k, 1);
            NodalBasis const &starBasis = pp.getStarBasis();

            Soln s(1, grid, k);
            s.assign([](Index, double x) { return 0.0; },
                     [](Index, double x) { return std::sin(pi * x) + x; });

            for (Index cell = 0; cell < static_cast<Index>(grid.getNCells()); ++cell)
            {
                Interval const &I(grid[cell]);
                const Vector atStar =
                    pp.V(cell) * s.y.q(0).getCoeff(cell).second;

                for (Index m = 0; m < k + 2; ++m)
                {
                    const double x = I.fromRef(starBasis.getNodes()(m));
                    BOOST_TEST(std::abs(atStar(m) - s.y.q(0)(x)) < 1e-12,
                               "k = " << k << ", cell " << cell << ", node " << m);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(star_points_are_the_star_nodes_in_cell_major_order)
{
    for (Grid const &grid : testGrids())
    {
        for (Index k = 1; k <= 4; ++k)
        {
            Postprocessor pp(grid, k, 1);
            auto const &points = pp.starPoints();

            BOOST_TEST(points.size() == grid.getNCells() * (k + 2));
            BOOST_TEST(pp.starDoF() == k + 2);

            for (Index cell = 0; cell < static_cast<Index>(grid.getNCells()); ++cell)
                for (Index m = 0; m < k + 2; ++m)
                {
                    const double expected =
                        grid[cell].fromRef(pp.getStarBasis().getNodes()(m));
                    BOOST_TEST(points[cell * (k + 2) + m] == expected,
                               boost::test_tools::tolerance(1e-14));
                    // and inside the cell it belongs to
                    BOOST_TEST(grid[cell].contains(points[cell * (k + 2) + m]));
                }
        }
    }
}

BOOST_AUTO_TEST_CASE(eval_on_star_nodes_carries_every_field)
{
    // The state handed to ComputePhysics: u must be u*, while q, sigma and the
    // auxiliary variables must be the solver's own fields sampled at the star
    // nodes. Getting u from u_h instead of u* here is exactly the bug that
    // silently reverts the method to the non-superconvergent one.
    const Index nVars = 2, nAux = 1, k = 3;
    Grid grid(0.0, 1.0, 4);

    Soln s(nVars, grid, k, nAux);
    s.assign([](Index v, double x) { return std::exp(-x) + v; },
             [](Index v, double x) { return -std::exp(-x); });
    for (Index v = 0; v < nVars; ++v)
        s.y.sigma(v) = [v](double x) { return 0.75 * x + v; };
    s.y.AssignAux([](Index, double x) { return std::cos(x); });

    Postprocessor pp(grid, k, nVars, 0, nAux);
    pp.computeUStar(s.y);
    GlobalState gs = pp.evalOnStarNodes(s.y);

    BOOST_TEST(gs.size() == grid.getNCells() * (k + 2));

    auto const &points = pp.starPoints();
    for (size_t i = 0; i < points.size(); ++i)
    {
        const double x = points[i];
        const State st = gs[i];
        for (Index v = 0; v < nVars; ++v)
        {
            BOOST_TEST(st.Variable[v] == pp.uStar(v)(x),
                       boost::test_tools::tolerance(1e-12));
            BOOST_TEST(st.Derivative[v] == s.y.q(v)(x),
                       boost::test_tools::tolerance(1e-12));
            BOOST_TEST(st.Flux[v] == s.y.sigma(v)(x),
                       boost::test_tools::tolerance(1e-12));
        }
        BOOST_TEST(st.Aux[0] == s.y.Aux(0)(x), boost::test_tools::tolerance(1e-12));
    }

    // u* is a genuinely different field from u_h -- if this were not so the test
    // above would pass vacuously.
    double maxDiff = 0.0;
    for (double x : points)
        maxDiff = std::max(maxDiff, std::abs(pp.uStar(0)(x) - s.y.u(0)(x)));
    BOOST_TEST(maxDiff > 0.0);
}

BOOST_AUTO_TEST_CASE(the_variables_do_not_alias_each_other)
{
    // uStar_ maps one flat buffer with a stride of nVars*(k+2); an off-by-one
    // there aliases two variables onto the same coefficients, the same class of
    // bug DGSolnTests pins for DGSoln::Map.
    const Index nVars = 3, k = 2;
    Grid grid(0.0, 1.0, 3);

    Soln s(nVars, grid, k);
    s.assign([](Index v, double x) { return (v + 1) * x * x; },
             [](Index v, double x) { return 2.0 * (v + 1) * x; });

    Postprocessor pp(grid, k, nVars);
    pp.computeUStar(s.y);

    for (double x : {0.1, 0.45, 0.9})
        for (Index v = 0; v < nVars; ++v)
            BOOST_TEST(pp.uStar(v)(x) == (v + 1) * x * x,
                       boost::test_tools::tolerance(1e-11));
}

BOOST_AUTO_TEST_CASE(degree_zero_is_rejected_rather_than_silently_wrong)
{
    // NodalBasis::getBasis(0) returns early without building Vandermonde or
    // BarycentricWeights (Basis.hpp:369-377), so Evaluate() off-node reads an
    // empty vector. Paper I needs k >= 1 for the superconvergence in any case.
    Grid grid(0.0, 1.0, 4);
    BOOST_CHECK_THROW(Postprocessor(grid, 0, 1), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
