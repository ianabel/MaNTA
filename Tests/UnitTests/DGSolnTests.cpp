// Tests for DGSoln / DGApprox -- the views that impose structure on the flat
// double array SUNDIALS hands us.
//
// DGSolnImpl::Map is the single most fragile pure function in the solver: an
// off-by-one in its offset arithmetic does not crash, it silently aliases two
// fields onto the same memory. DGTests.cpp covers construction and evaluation;
// what is pinned here is the memory layout itself, the DOF count, and the
// lambda trace formula.

#include <boost/test/unit_test.hpp>

#include "DGSoln.hpp"
#include "gridStructures.hpp"

#include <numeric>
#include <stdexcept>
#include <vector>

BOOST_AUTO_TEST_SUITE(dg_soln_layout_tests, *boost::unit_test::tolerance(1e-12))

namespace
{
// The layout Map() is documented to produce, restated independently here so the
// test fails if either side drifts.
struct Layout
{
    Index nVars, nAux, nScalars, nCells, k;

    Index perCell() const { return (3 * nVars + nAux) * (k + 1); }
    Index sigmaAt(Index var, Index cell) const { return cell * perCell() + var * (k + 1); }
    Index qAt(Index var, Index cell) const { return cell * perCell() + nVars * (k + 1) + var * (k + 1); }
    Index uAt(Index var, Index cell) const { return cell * perCell() + 2 * nVars * (k + 1) + var * (k + 1); }
    Index auxAt(Index a, Index cell) const { return cell * perCell() + 3 * nVars * (k + 1) + a * (k + 1); }
    Index lambdaAt(Index var) const { return perCell() * nCells + var * (nCells + 1); }
    Index scalarAt() const { return perCell() * nCells + nVars * (nCells + 1); }
    Index total() const { return perCell() * nCells + nVars * (nCells + 1) + nScalars; }
};
} // namespace

BOOST_AUTO_TEST_CASE(dof_count_matches_the_layout)
{
    for (Index nVars : {1, 2, 4})
        for (Index nAux : {0, 1, 3})
            for (Index nScalars : {0, 1, 5})
                for (Index nCells : {1, 4, 9})
                    for (Index k : {0, 1, 3, 5})
                    {
                        Grid grid(0.0, 1.0, nCells);
                        DGSoln y(nVars, grid, k, nScalars, nAux);
                        Layout L{nVars, nAux, nScalars, nCells, k};

                        BOOST_TEST(y.getDoF() == static_cast<size_t>(L.total()));
                        BOOST_TEST(y.getNumVars() == nVars);
                        BOOST_TEST(y.getAux() == nAux);
                        BOOST_TEST(y.getScalars() == nScalars);
                    }
}

BOOST_AUTO_TEST_CASE(map_places_every_field_at_the_documented_offset)
{
    // Fill the backing array with its own indices, then check that each view
    // reads back exactly the indices the layout predicts. This catches aliasing
    // (two fields sharing memory) as well as plain off-by-ones.
    const Index nVars = 3, nAux = 2, nScalars = 4, nCells = 5, k = 2;
    Layout L{nVars, nAux, nScalars, nCells, k};

    Grid grid(0.0, 1.0, nCells);
    DGSoln y(nVars, grid, k, nScalars, nAux);
    BOOST_TEST(y.getDoF() == static_cast<size_t>(L.total()));

    std::vector<double> Y(y.getDoF());
    std::iota(Y.begin(), Y.end(), 0.0);
    y.Map(Y.data());

    for (Index var = 0; var < nVars; ++var)
    {
        for (Index cell = 0; cell < nCells; ++cell)
        {
            auto const &sig = y.sigma(var).getCoeff(cell).second;
            auto const &qq = y.q(var).getCoeff(cell).second;
            auto const &uu = y.u(var).getCoeff(cell).second;

            BOOST_TEST(sig.size() == k + 1);
            for (Index j = 0; j < k + 1; ++j)
            {
                BOOST_TEST(sig(j) == static_cast<double>(L.sigmaAt(var, cell) + j));
                BOOST_TEST(qq(j) == static_cast<double>(L.qAt(var, cell) + j));
                BOOST_TEST(uu(j) == static_cast<double>(L.uAt(var, cell) + j));
            }
        }

        BOOST_TEST(y.lambda(var).size() == nCells + 1);
        for (Index j = 0; j < nCells + 1; ++j)
            BOOST_TEST(y.lambda(var)(j) == static_cast<double>(L.lambdaAt(var) + j));
    }

    for (Index a = 0; a < nAux; ++a)
        for (Index cell = 0; cell < nCells; ++cell)
        {
            auto const &ax = y.Aux(a).getCoeff(cell).second;
            for (Index j = 0; j < k + 1; ++j)
                BOOST_TEST(ax(j) == static_cast<double>(L.auxAt(a, cell) + j));
        }

    BOOST_TEST(y.Scalars().size() == nScalars);
    for (Index s = 0; s < nScalars; ++s)
        BOOST_TEST(y.Scalar(s) == static_cast<double>(L.scalarAt() + s));
}

BOOST_AUTO_TEST_CASE(map_views_alias_the_original_buffer)
{
    // The views must write through to the caller's memory -- SUNDIALS owns it.
    const Index nVars = 2, nAux = 1, nScalars = 2, nCells = 3, k = 1;
    Layout L{nVars, nAux, nScalars, nCells, k};

    Grid grid(0.0, 1.0, nCells);
    DGSoln y(nVars, grid, k, nScalars, nAux);
    std::vector<double> Y(y.getDoF(), 0.0);
    y.Map(Y.data());

    y.u(1).getCoeff(2).second(0) = 42.0;
    BOOST_TEST(Y[L.uAt(1, 2)] == 42.0);

    y.Scalar(1) = -7.5;
    BOOST_TEST(Y[L.scalarAt() + 1] == -7.5);

    y.lambda(0)(nCells) = 3.25;
    BOOST_TEST(Y[L.lambdaAt(0) + nCells] == 3.25);

    y.Aux(0).getCoeff(1).second(1) = 9.0;
    BOOST_TEST(Y[L.auxAt(0, 1) + 1] == 9.0);

    // No field may overlap another: exactly the four cells we wrote are set.
    Index nonZero = 0;
    for (double v : Y)
        if (v != 0.0)
            ++nonZero;
    BOOST_TEST(nonZero == 4);
}

BOOST_AUTO_TEST_CASE(map_handles_the_degenerate_shapes)
{
    // nAux = 0 and nScalars = 0 are the common case and must not leave gaps.
    const Index nVars = 2, nCells = 3, k = 2;
    Grid grid(0.0, 1.0, nCells);
    DGSoln y(nVars, grid, k);

    Layout L{nVars, 0, 0, nCells, k};
    BOOST_TEST(y.getDoF() == static_cast<size_t>(L.total()));

    std::vector<double> Y(y.getDoF());
    std::iota(Y.begin(), Y.end(), 0.0);
    y.Map(Y.data());

    BOOST_TEST(y.Scalars().size() == 0);
    for (Index var = 0; var < nVars; ++var)
        for (Index cell = 0; cell < nCells; ++cell)
            for (Index j = 0; j < k + 1; ++j)
                BOOST_TEST(y.u(var).getCoeff(cell).second(j) ==
                           static_cast<double>(L.uAt(var, cell) + j));
}

// ------------------------------------------------------------ lambda trace --

BOOST_AUTO_TEST_CASE(evaluate_lambda_averages_u_across_faces)
{
    // lambda = {{u}} on interior faces, and the one-sided trace of u at the
    // domain boundaries.
    const Index nVars = 1, nCells = 4, k = 3;
    Grid grid(0.0, 1.0, nCells);
    DGSoln y(nVars, grid, k);
    std::vector<double> Y(y.getDoF(), 0.0);
    y.Map(Y.data());

    // Use a cubic: with k = 3 the L2 projection is exact, so the trace values
    // must match the analytic function to round-off rather than to the
    // discretisation error. (A transcendental function here only measures the
    // projection error, which is ~1e-4 on this grid.)
    auto f = [](Index, double x) { return 1.0 + 2.0 * x - 0.5 * x * x + 0.25 * x * x * x; };
    y.AssignU(f);
    y.EvaluateLambda();

    // A polynomial is continuous across faces, so {{u}} equals the pointwise
    // value there, and the boundary entries are the one-sided traces.
    BOOST_TEST(y.lambda(0)(0) == f(0, 0.0), boost::test_tools::tolerance(1e-10));
    BOOST_TEST(y.lambda(0)(nCells) == f(0, 1.0), boost::test_tools::tolerance(1e-10));
    for (Index i = 1; i < nCells; ++i)
        BOOST_TEST(y.lambda(0)(i) == f(0, grid[i].x_l), boost::test_tools::tolerance(1e-10));
}

BOOST_AUTO_TEST_CASE(evaluate_lambda_with_tau_adds_the_flux_jump)
{
    // With tau, lambda = {{u}} + [[q.n]]/(2 tau). For a continuous q the two
    // one-sided contributions at an interior face cancel, so the result must
    // match the no-tau version there; the tau term only bites on a jump.
    const Index nVars = 1, nCells = 4, k = 3;
    const double tau = 2.5;
    Grid grid(0.0, 1.0, nCells);

    DGSoln plain(nVars, grid, k);
    std::vector<double> Ya(plain.getDoF(), 0.0);
    plain.Map(Ya.data());

    DGSoln withTau(nVars, grid, k);
    std::vector<double> Yb(withTau.getDoF(), 0.0);
    withTau.Map(Yb.data());

    auto u = [](Index, double x) { return std::cos(2.0 * x); };
    auto q = [](Index, double x) { return 1.0 + x; }; // continuous

    plain.AssignU(u);
    plain.AssignQ(q);
    plain.EvaluateLambda();

    withTau.AssignU(u);
    withTau.AssignQ(q);
    withTau.EvaluateLambda(tau);

    for (Index i = 1; i < nCells; ++i)
        BOOST_TEST(withTau.lambda(0)(i) == plain.lambda(0)(i),
                   boost::test_tools::tolerance(1e-9));

    // Boundaries are overwritten with the trace of u in both overloads.
    BOOST_TEST(withTau.lambda(0)(0) == plain.lambda(0)(0), boost::test_tools::tolerance(1e-12));
    BOOST_TEST(withTau.lambda(0)(nCells) == plain.lambda(0)(nCells),
               boost::test_tools::tolerance(1e-12));
}

// --------------------------------------------------------- copy and += --

BOOST_AUTO_TEST_CASE(copy_transfers_every_field)
{
    const Index nVars = 2, nAux = 1, nScalars = 2, nCells = 3, k = 2;
    Grid grid(0.0, 1.0, nCells);

    DGSoln src(nVars, grid, k, nScalars, nAux);
    std::vector<double> Ys(src.getDoF());
    std::iota(Ys.begin(), Ys.end(), 1.0);
    src.Map(Ys.data());

    DGSoln dst(nVars, grid, k, nScalars, nAux);
    std::vector<double> Yd(dst.getDoF(), 0.0);
    dst.Map(Yd.data());

    dst.copy(src);

    // copy() covers u, q, sigma, lambda, scalars and aux -- i.e. the whole
    // vector -- so the buffers must match element for element.
    for (size_t i = 0; i < Ys.size(); ++i)
        BOOST_TEST(Yd[i] == Ys[i]);
}

BOOST_AUTO_TEST_CASE(copy_and_add_reject_mismatched_shapes)
{
    Grid grid(0.0, 1.0, 4);
    Grid other(0.0, 2.0, 4);

    DGSoln a(2, grid, 2);
    DGSoln differentVars(3, grid, 2);
    DGSoln differentGrid(2, other, 2);

    std::vector<double> Ya(a.getDoF(), 0.0), Yv(differentVars.getDoF(), 0.0),
        Yg(differentGrid.getDoF(), 0.0);
    a.Map(Ya.data());
    differentVars.Map(Yv.data());
    differentGrid.Map(Yg.data());

    BOOST_CHECK_THROW(a.copy(differentVars), std::invalid_argument);
    BOOST_CHECK_THROW(a.copy(differentGrid), std::invalid_argument);
    BOOST_CHECK_THROW(a += differentVars, std::invalid_argument);
    BOOST_CHECK_THROW(a += differentGrid, std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(operator_plus_equals_covers_u_q_sigma_and_lambda_only)
{
    // NOTE: unlike copy(), operator+= deliberately skips the auxiliary fields
    // and the global scalars. This function currently has no callers in the
    // solver; the asymmetry is pinned here so that if it is ever wired into the
    // Newton update the omission is a visible decision rather than a surprise.
    const Index nVars = 1, nAux = 1, nScalars = 1, nCells = 2, k = 1;
    Layout L{nVars, nAux, nScalars, nCells, k};
    Grid grid(0.0, 1.0, nCells);

    DGSoln a(nVars, grid, k, nScalars, nAux);
    std::vector<double> Ya(a.getDoF(), 1.0);
    a.Map(Ya.data());

    DGSoln b(nVars, grid, k, nScalars, nAux);
    std::vector<double> Yb(b.getDoF(), 10.0);
    b.Map(Yb.data());

    a += b;

    BOOST_TEST(a.u(0).getCoeff(0).second(0) == 11.0);
    BOOST_TEST(a.q(0).getCoeff(0).second(0) == 11.0);
    BOOST_TEST(a.sigma(0).getCoeff(0).second(0) == 11.0);
    BOOST_TEST(a.lambda(0)(0) == 11.0);

    // Untouched:
    BOOST_TEST(a.Aux(0).getCoeff(0).second(0) == 1.0);
    BOOST_TEST(a.Scalar(0) == 1.0);
}

// --------------------------------------------------------------- DGApprox --

BOOST_AUTO_TEST_CASE(dgapprox_map_rejects_a_stride_that_would_overlap)
{
    const Index k = 3;
    Grid grid(0.0, 1.0, 4);
    const DGSoln::basis_type basis = DGSoln::basis_type::getBasis(k);

    std::vector<double> buffer(4 * (k + 1), 0.0);
    DGApproxImpl<DGSoln::basis_type> approx(grid, basis);

    // A stride shorter than k+1 makes consecutive cells overlap.
    BOOST_CHECK_THROW(approx.Map(buffer.data(), k), std::invalid_argument);
    BOOST_CHECK_NO_THROW(approx.Map(buffer.data(), k + 1));
}

BOOST_AUTO_TEST_CASE(eval_and_get_points_agree_with_the_nodal_values)
{
    const Index nVars = 2, nCells = 3, k = 3;
    Grid grid(0.0, 1.0, nCells);
    DGSoln y(nVars, grid, k);
    std::vector<double> Y(y.getDoF(), 0.0);
    y.Map(Y.data());

    auto f = [](Index v, double x) { return (v + 1) * x * x - 0.25 * x; };
    y.AssignU(f);

    // getPoints lists the nodal abscissae in cell-major order; evaluating there
    // must reproduce the projected function (exactly, since f is degree 2 <= k).
    auto pts = y.getPoints();
    BOOST_TEST(pts.size() == static_cast<size_t>(nCells * (k + 1)));

    for (auto x : pts)
        for (Index v = 0; v < nVars; ++v)
            BOOST_TEST(y.eval(x).u(v) == f(v, x), boost::test_tools::tolerance(1e-9));
}

BOOST_AUTO_TEST_SUITE_END()
