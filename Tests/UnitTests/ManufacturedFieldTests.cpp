// The manufactured field models, checked against their own closed forms before
// anything in the solver depends on them.
//
// ManufacturedField is nFieldDOF = 1 with R = psi - Int u dx and one geometry
// slot g(x; psi) = 1 + psi c(x). With the harness's u = sin(pi x)(1+t) on
// [0,1], Int u dx = (2/pi)(1+t), so psi_exact(t) = (2/pi)(1+t).
//
// The coupling is genuinely two-way: A2 is dense across every u DOF, and A1 is
// nonzero because the flux is sigma_hat = g kappa q.
#include <boost/test/unit_test.hpp>

#include "ManufacturedFields.hpp"

#include "../../DGSoln.hpp"
#include "../../PyIntegrator.hpp"

#include <cmath>

// Sample u_exact on the nodes of a grid and return the GlobalState, the
// abscissae and the integration weights -- exactly what a residual call hands
// a FieldModel -- so a test can drive FieldResidual/FieldResidualPrime without
// a solver.
//
// DGSoln soln(1, grid, k, Index(0), Index(0)): the bare literal 0 that the
// fourth slot would otherwise take is a null-pointer constant as far as
// overload resolution is concerned, and is equally viable for the
// (double *memory, ...) constructor's `memory` parameter, making the call
// ambiguous. FieldDoFLayoutTests.cpp, PostprocessingTests.cpp,
// ScalarJacobianTests.cpp and AdjointProblemTests.cpp already work around the
// same trap the same way.
std::tuple<GlobalState, std::vector<Position>, Vector>
sampleExactOnNodes(Grid const &grid, Index k, Time t)
{
    DGSoln soln(1, grid, k, Index(0), Index(0));
    std::vector<double> mem(soln.getDoF(), 0.0);
    soln.Map(mem.data());
    // AssignU takes (variable index, x), not just x: DGSoln.hpp:244.
    soln.AssignU([t](Index, double x) { return manufacturedU(x, t); });

    GlobalState states = soln.evalOnNodes();
    std::vector<Position> points = soln.getPoints();
    Vector weights = Integrator::getIntegrationWeights(soln.getBasis(), grid);
    return {states, points, weights};
}

BOOST_AUTO_TEST_SUITE(manufactured_field_tests)

BOOST_AUTO_TEST_CASE(psi_exact_is_the_integral_of_u)
{
    // Int_0^1 sin(pi x) dx = 2/pi
    BOOST_CHECK_CLOSE(manufacturedPsiExact(0.0), 2.0 / M_PI, 1e-12);
    BOOST_CHECK_CLOSE(manufacturedPsiExact(1.0), 4.0 / M_PI, 1e-12);
}

BOOST_AUTO_TEST_CASE(the_residual_vanishes_at_the_exact_solution)
{
    ManufacturedField model{toml::value{}, Grid(0.0, 1.0, 16)};
    const Time t = 0.3;

    // Sample u_exact on the nodes of a fine grid and integrate with the
    // solver's own weights, exactly as the residual will.
    auto [states, points, weights] = sampleExactOnNodes(Grid(0.0, 1.0, 16), 3, t);

    Vector psi(1);
    psi(0) = manufacturedPsiExact(t);
    Vector dpsidt(1);
    dpsidt(0) = 0.0;

    Vector out = Vector::Zero(1);
    model.FieldResidual(out, psi, dpsidt, states, points, weights, t);

    // Not exactly zero: the quadrature is interpolatory on a degree-3 basis,
    // so it integrates sin(pi x) to the basis's accuracy, not exactly.
    BOOST_CHECK_SMALL(out(0), 1e-6);
}

BOOST_AUTO_TEST_CASE(dR_dpsi_is_the_identity_and_dR_dstate_is_minus_the_weights)
{
    // R = psi - Int u dx, so dR/dpsi = 1 and dR/du_j = -w_j exactly. A case
    // must use the solver's quadrature weights rather than a rule of its own;
    // ScalarTestLD3 disagreed with its own Jacobian by 8% for doing otherwise.
    ManufacturedField model{toml::value{}, Grid(0.0, 1.0, 8)};
    const Time t = 0.0;
    auto [states, points, weights] = sampleExactOnNodes(Grid(0.0, 1.0, 8), 2, t);

    GlobalStateMatrix dR(1), dRdot(1);
    dR.add(8, 2, 1, 0, 0);
    dRdot.add(8, 2, 1, 0, 0);
    Matrix dRdpsi = Matrix::Zero(1, 1), dRddpsidt = Matrix::Zero(1, 1);

    Vector psi(1); psi(0) = manufacturedPsiExact(t);
    Vector dpsidt = Vector::Zero(1);

    model.FieldResidualPrime(dR, dRdot, dRdpsi, dRddpsidt, psi, dpsidt,
                             states, points, weights, t);

    BOOST_CHECK_CLOSE(dRdpsi(0, 0), 1.0, 1e-12);
    BOOST_CHECK_SMALL(dRddpsidt(0, 0), 1e-15);
    for (Index j = 0; j < weights.size(); ++j)
        BOOST_CHECK_CLOSE(dR[0][j].u(0), -weights(j), 1e-12);
}

BOOST_AUTO_TEST_CASE(the_geometry_and_its_derivative_agree_with_the_closed_form)
{
    ManufacturedField model{toml::value{}, Grid(0.0, 1.0, 4)};
    Vector psi(1); psi(0) = 0.75;

    Vector g = Vector::Zero(1);
    model.Geometry(g, psi, 0.25, 0.0);
    BOOST_CHECK_CLOSE(g(0), 1.0 + 0.75 * manufacturedC(0.25), 1e-12);

    Matrix dg = Matrix::Zero(1, 1);
    model.dGeometry_dpsi(dg, psi, 0.25, 0.0);
    BOOST_CHECK_CLOSE(dg(0, 0), manufacturedC(0.25), 1e-12);
}

BOOST_AUTO_TEST_CASE(the_vector_model_has_a_nonscalar_b_block)
{
    // L is SPD tridiagonal, so B is genuinely a matrix and its solve is not a
    // division. This is what stops the block solve being exercised only in a
    // degenerate case.
    ManufacturedFieldVector model{toml::value{}, Grid(0.0, 1.0, 4)};
    BOOST_CHECK_EQUAL(model.nFieldDOF(), 5);

    Matrix dRdpsi = manufacturedL(5);
    Matrix dRddpsidt = Matrix::Zero(5, 5);
    model.updateFieldJacobian(dRdpsi, dRddpsidt, 0.0);

    Vector rhs = Vector::LinSpaced(5, 1.0, 5.0);
    Vector x = Vector::Zero(5);
    model.solveB(x, rhs);

    Vector back = Vector::Zero(5);
    model.applyB(back, x);
    BOOST_CHECK_SMALL((back - rhs).norm(), 1e-12);
}

BOOST_AUTO_TEST_SUITE_END()
