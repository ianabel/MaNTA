// Tests for the modal-decay smoothness sensor (SmoothnessSensor.{hpp,cpp}) and
// for the NodalBasis::ToModal primitive it is built on.
//
// The sensor is deliberately testable without running a solve: it is a pure
// function of one cell's nodal values, so a test can hand it a polynomial it
// knows the answer for. Everything here does that, except the last case, which
// drives the DGSoln wrapper over a real grid.

#include <boost/test/unit_test.hpp>

#include "SmoothnessSensor.hpp"

#include "Basis.hpp"
#include "DGSoln.hpp"
#include "Types.hpp"
#include "gridStructures.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
// A field's nodal values on the reference cell: the basis is nodal at the
// Chebyshev points, so sampling f there *is* building its interpolant.
Vector onNodes(NodalBasis const &basis, std::function<double(double)> f)
{
    Vector const &nodes = basis.getNodes();
    Vector out(nodes.size());
    for (Index i = 0; i < nodes.size(); ++i)
        out(i) = f(nodes(i));
    return out;
}

// Shestakov's axis behaviour, which is what MESH-REFINEMENT.md section 6
// measures the regularity limit on: x^(4/3) has an unbounded second derivative
// at the branch point and caps how far raising k can pay.
//
// The branch point is at the *edge* of the reference cell, deliberately. That
// is where it sits in the case this is modelled on -- Shestakov's singularity
// is at the domain boundary, so it falls on cell 0's left edge -- and putting
// it at the centre instead would make the function even, which is a different
// test with a different answer: see
// a_spectrum_with_structural_zeros_is_not_fitted_through_the_gaps.
double singular(double x) { return std::pow(x + 1.0, 4.0 / 3.0); }

} // namespace

BOOST_AUTO_TEST_SUITE(smoothness_sensor_tests, *boost::unit_test::tolerance(1e-12))

BOOST_AUTO_TEST_CASE(to_modal_recovers_the_legendre_coefficients_of_a_known_polynomial)
{
    // The primitive the whole sensor rests on. A polynomial written in the
    // Legendre basis, sampled at the nodes, must decompose back to the
    // coefficients it was written with -- exactly, since it lies in the space.
    const NodalBasis basis = NodalBasis::getBasis(4);

    const Vector want = (Vector(5) << 1.0, -2.0, 3.0, 0.0, 0.5).finished();

    const Vector nodal = onNodes(basis, [&](double x)
    {
        double sum = 0.0;
        for (Index j = 0; j < want.size(); ++j)
            sum += want(j) * LegendreBasis::Evaluate(j, x);
        return sum;
    });

    const Vector got = basis.ToModal(nodal);

    BOOST_TEST(got.size() == want.size());
    for (Index j = 0; j < want.size(); ++j)
        BOOST_TEST(got(j) == want(j));

    // No interval anywhere in that call. The reference map's h/2 is common to
    // the nodal and modal representations and cancels, so the decomposition is
    // scale-free -- which is what lets the sensor be read across a non-uniform
    // mesh without a per-cell normalisation.
}

BOOST_AUTO_TEST_CASE(to_modal_refuses_a_basis_or_a_vector_it_cannot_decompose)
{
    BOOST_CHECK_THROW(NodalBasis::getBasis(0).ToModal(Vector::Zero(1)),
                      std::invalid_argument);

    const NodalBasis basis = NodalBasis::getBasis(3);
    BOOST_CHECK_THROW(basis.ToModal(Vector::Zero(3)), std::invalid_argument);
    BOOST_CHECK_NO_THROW(basis.ToModal(Vector::Zero(4)));
}

BOOST_AUTO_TEST_CASE(a_polynomial_below_the_basis_degree_reports_itself_resolved)
{
    // The negative control, and it is a real case rather than a contrived one:
    // MESH-REFINEMENT.md notes that Jardin's steady state is exactly degree 1,
    // so every cell of it lands here. An adaptive driver must leave such a mesh
    // alone rather than chase round-off.
    const NodalBasis basis = NodalBasis::getBasis(5);

    const Vector nodal = onNodes(basis, [](double x)
    { return 1.0 + 2.0 * x - 0.5 * x * x; });

    const CellSmoothness s = cellSmoothness(basis, nodal);

    // Nothing in the top mode, to round-off.
    BOOST_TEST(s.modalEnergyFraction < 1e-28,
               "top-mode energy share is " << s.modalEnergyFraction);

    // And the rate is reported as infinite rather than fitted. The top mode is
    // at the round-off floor, so this polynomial is exactly representable below
    // degree k and there is no decay left to measure -- as against a fit over
    // the surviving modes, which for this quadratic would run over j = 1 and 2
    // alone and return about 2.6, indistinguishable from a genuinely singular
    // cell.
    BOOST_TEST(s.decayRate == std::numeric_limits<double>::infinity(),
               "decay rate is " << s.decayRate);
}

BOOST_AUTO_TEST_CASE(a_spectrum_with_structural_zeros_is_not_fitted_through_the_gaps)
{
    // A function even about the cell centre has every *odd* Legendre
    // coefficient identically zero. Those land on the round-off floor, and
    // fitting the resulting alternating floor/signal sequence as though it were
    // data gives a line with the wrong sign: measured s = -8.33 at k = 6 and
    // -51.8 at k = 2 for a spectrum that plainly decays. A negative rate means
    // "the coefficients grow", so this is not a small error -- the sensor was
    // reporting the opposite of the truth, and reporting it about the sharpest
    // feature in the tree.
    //
    // The fix is that a floored coefficient is not a measurement; it was
    // clamped precisely because it is round-off. The fit skips them.
    const NodalBasis basis = NodalBasis::getBasis(6);

    const Vector nodal = onNodes(basis, [](double x)
    { return std::pow(std::abs(x), 4.0 / 3.0); });
    const Vector uhat = basis.ToModal(nodal);

    // The premise: the odd modes really are round-off, and the even ones really
    // are not. Measured 2.6e-16, 4.7e-17 and 2.6e-17 of the scale against
    // 7.2e-1, 1.3e-1 and 5.2e-2 -- eleven orders between the two populations,
    // which is what makes the floor's exact position uncritical.
    const double scale = uhat.cwiseAbs().maxCoeff();
    BOOST_TEST_MESSAGE("even |x|^(4/3) modes, relative to " << scale << ":");
    for (Index j = 0; j <= 6; ++j)
        BOOST_TEST_MESSAGE("  uhat(" << j << ")/scale = " << uhat(j) / scale);

    for (Index j = 1; j <= 5; j += 2)
        BOOST_TEST(std::abs(uhat(j)) < 1e-14 * scale,
                   "mode " << j << " is " << uhat(j) << ", so this fixture no longer "
                   "has the structural zeros it is testing");
    for (Index j = 2; j <= 6; j += 2)
        BOOST_TEST(std::abs(uhat(j)) > 1e-3 * scale,
                   "mode " << j << " is " << uhat(j) << ", which is too close to the "
                   "floor for this fixture to be testing what it claims");

    const CellSmoothness s = cellSmoothness(basis, nodal);
    BOOST_TEST_MESSAGE("even |x|^(4/3): s = " << s.decayRate);

    // Positive, and it lands on the rate theory predicts: the surviving modes
    // are 2, 4 and 6, and |x|^a has Legendre coefficients falling like
    // j^-(a+1) = j^-2.33 for a = 4/3. Measured 2.41. Getting the exponent right
    // to within 3% off three points is a stronger check than any sign test --
    // it says the fit is measuring decay, not merely avoiding the old defect.
    BOOST_TEST(s.decayRate == 7.0 / 3.0, boost::test_tools::tolerance(0.1));
}

BOOST_AUTO_TEST_CASE(a_cell_with_no_decay_to_measure_says_so_rather_than_reporting_zero)
{
    const NodalBasis basis = NodalBasis::getBasis(4);
    const double inf = std::numeric_limits<double>::infinity();

    // Identically zero. Every ratio here is 0/0, and this is reachable on any
    // problem with a quiescent region.
    const CellSmoothness empty = cellSmoothness(basis, Vector::Zero(5));
    BOOST_TEST(empty.modalEnergyFraction == 0.0);
    BOOST_TEST(empty.decayRate == inf);

    // Constant, which is the trap. The coefficients being fitted are then all
    // pinned at the round-off floor, so they are all *equal*, so the
    // least-squares slope is exactly zero -- and a zero decay rate is the
    // sensor's way of saying "as rough as it gets". Reporting the smoothest
    // possible field as the roughest is the sort of defect that would survive a
    // long time, since nothing about the number looks wrong.
    const CellSmoothness flat = cellSmoothness(basis, Vector::Constant(5, 3.25));
    BOOST_TEST(flat.modalEnergyFraction == 0.0);
    BOOST_TEST(flat.decayRate == inf);

    // The opposite end, and the other value that is reported rather than
    // fitted. A single Legendre mode at the top of the space has every mode
    // below it at zero, so exactly one point survives the floor and there is no
    // line to fit -- but the answer is not "unknown", it is that the spectrum
    // has no decay in it whatsoever. Zero, which sorts as the roughest thing
    // this can report, is the honest reading.
    const Vector pure = onNodes(basis, [](double x)
    { return LegendreBasis::Evaluate(4, x); });
    const CellSmoothness top = cellSmoothness(basis, pure);
    BOOST_TEST(top.modalEnergyFraction == 1.0, boost::test_tools::tolerance(1e-12));
    BOOST_TEST(top.decayRate == 0.0);
}

BOOST_AUTO_TEST_CASE(too_few_modes_to_fit_is_refused_rather_than_guessed_at)
{
    // The fit runs over j = 1..k, so k = 1 is a single point and no slope.
    // MaNTA carries one global order, so this is a property of the run and not
    // of a cell -- which is why it throws instead of returning a sentinel that
    // every caller would have to test for.
    BOOST_CHECK_THROW(cellSmoothness(NodalBasis::getBasis(1), Vector::Zero(2)),
                      std::invalid_argument);
    BOOST_CHECK_NO_THROW(cellSmoothness(NodalBasis::getBasis(2), Vector::Zero(3)));
}

BOOST_AUTO_TEST_CASE(a_singular_function_decays_more_slowly_than_an_analytic_one)
{
    // The measurement the sensor exists to make. exp(x) is entire, so its
    // Legendre coefficients fall geometrically; |x|^(4/3) has an unbounded
    // second derivative at the origin and falls algebraically.
    const NodalBasis basis = NodalBasis::getBasis(6);

    const CellSmoothness smooth = cellSmoothness(basis, onNodes(basis, [](double x)
    { return std::exp(x); }));
    const CellSmoothness rough = cellSmoothness(basis, onNodes(basis, singular));

    BOOST_TEST_MESSAGE("k = 6: exp(x) s = " << smooth.decayRate << ", S_K = "
                       << smooth.modalEnergyFraction << "; |x|^(4/3) s = "
                       << rough.decayRate << ", S_K = " << rough.modalEnergyFraction);

    BOOST_TEST(smooth.decayRate > rough.decayRate,
               "the analytic function's decay rate is " << smooth.decayRate
               << " against the singular one's " << rough.decayRate);

    // Both indicators agree in direction here. They do not always -- see the
    // next case -- which is the reason the sensor reports both.
    BOOST_TEST(smooth.modalEnergyFraction < rough.modalEnergyFraction);
}

BOOST_AUTO_TEST_CASE(the_energy_fraction_moves_orders_with_degree_where_the_decay_rate_does_not)
{
    // Why decayRate is the quantity to drive decisions from, and it is *not*
    // that it discriminates better at a given degree -- at k = 6 both do. It is
    // that S_K has no fixed scale.
    //
    // Persson & Peraire threshold S_K against S* ~ 1/k^4, calibrated for shock
    // capture. The measurement below is what makes any such fixed threshold a
    // per-degree quantity: on one function, S_K falls through orders of
    // magnitude as k rises, because it is an energy *share* of a tail that is
    // itself collapsing. The decay rate is an exponent, so it stays O(1) at
    // every degree and a rule like "s < 4 is rough" means the same thing
    // wherever it is applied.
    auto measure = [](unsigned int k, std::function<double(double)> f)
    {
        const NodalBasis basis = NodalBasis::getBasis(k);
        return cellSmoothness(basis, onNodes(basis, f));
    };

    auto smooth = [](double x) { return std::exp(x); };

    const CellSmoothness s2 = measure(2, smooth);
    const CellSmoothness s4 = measure(4, smooth);
    const CellSmoothness s6 = measure(6, smooth);

    BOOST_TEST_MESSAGE("exp(x): S_K = " << s2.modalEnergyFraction << " / "
                       << s4.modalEnergyFraction << " / " << s6.modalEnergyFraction
                       << ", s = " << s2.decayRate << " / " << s4.decayRate << " / "
                       << s6.decayRate << " at k = 2 / 4 / 6");

    // Measured 3.3e-3, 8.0e-7, 3.8e-11: nearly eight orders between k = 2 and
    // k = 6 on one unchanged function, against 2.04, 3.63, 5.39 for the rate.
    // No single S* can be the boundary between smooth and rough at both.
    BOOST_TEST(s2.modalEnergyFraction > 1e6 * s6.modalEnergyFraction,
               "S_K went from " << s2.modalEnergyFraction << " to "
               << s6.modalEnergyFraction << "; if it is now scale-stable across "
               "degrees, a fixed threshold may be safe and this test should be "
               "revisited rather than deleted");

    // ...while s stays within a factor of a few, and stays positive.
    for (CellSmoothness const &s : {s2, s4, s6})
    {
        BOOST_TEST(s.decayRate > 1.0, "decay rate " << s.decayRate);
        BOOST_TEST(s.decayRate < 20.0, "decay rate " << s.decayRate);
    }

    // What neither indicator can do, recorded so it is not discovered later as
    // a surprise: at k = 2 they do not separate these two functions at all, and
    // both get the sign of the answer wrong -- the singular function reads as
    // *smoother* than exp(x) on both measures (S_K 0.24x, s 3.35 against 2.04).
    // Two modes is not enough to see a singularity, whichever quantity is
    // formed from them. MESH-REFINEMENT.md section 7 says the same of S_K, and
    // the honest version is that it applies to both. The separation at k = 6 is
    // covered by the case above.
    const CellSmoothness rough2 = measure(2, singular);
    BOOST_TEST(rough2.decayRate > s2.decayRate,
               "at k = 2 the singular function now reads as rougher than exp(x), "
               "which is better than this test expects -- check what changed before "
               "relaxing it");
}

BOOST_AUTO_TEST_CASE(the_sensor_finds_the_cell_that_holds_the_singularity)
{
    // End to end over a real grid, and the claim MESH-REFINEMENT.md section 7
    // makes: on Shestakov's x^(4/3) the sensor puts the whole signal in cell 0,
    // which is the cell ANALYSIS.md section 7 independently identifies.
    constexpr Index k = 4, nVars = 1;
    constexpr Grid::Index nCells = 10;

    Grid grid(0.0, 1.0, nCells);
    DGSoln Y(nVars, grid, k);
    std::vector<double> mem(Y.getDoF());
    Y.Map(mem.data());
    Y.zeroCoeffs();
    Y.AssignU([](Index, double x) { return std::pow(x, 4.0 / 3.0); });

    const std::vector<CellSmoothness> s = cellSmoothness(Y, 0);
    BOOST_TEST(s.size() == static_cast<size_t>(nCells));

    Index roughest = 0;
    for (Index i = 1; i < static_cast<Index>(nCells); ++i)
        if (s[i].decayRate < s[roughest].decayRate)
            roughest = i;

    BOOST_TEST_MESSAGE("decay rate by cell: " << [&]
    {
        std::string out;
        for (auto const &c : s)
            out += std::to_string(c.decayRate) + " ";
        return out;
    }());

    BOOST_TEST(roughest == 0,
               "the roughest cell is " << roughest << ", not the one at the origin");

    // Not marginally, either: it is the singular cell against the rest of a
    // domain on which the same function is analytic.
    BOOST_TEST(s[0].decayRate < 0.8 * s[1].decayRate,
               "cell 0 has decay rate " << s[0].decayRate << " against cell 1's "
               << s[1].decayRate << "; the singular cell should stand out from "
               "its neighbour, not merely edge past it");
}

BOOST_AUTO_TEST_SUITE_END()
