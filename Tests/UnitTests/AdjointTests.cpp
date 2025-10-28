#include <boost/test/unit_test.hpp>
#include "../../PhysicsCases/AdjointTestProblem.hpp"
#include "../../PhysicsCases/AutodiffAdjointProblem.hpp"
#include "Types.hpp"
#include <toml.hpp>

#include "SystemSolver.hpp"

#include <nvector/nvector_serial.h>         /* access to serial N_Vector            */
#include <sundials/sundials_linearsolver.h> /* Generic Liner Solver Interface */
#include <sundials/sundials_types.h>

using namespace toml::literals::toml_literals;

// raw string literal (`R"(...)"` is useful for this purpose)
const toml::value config_snippet = u8R"(

[AutodiffTransportSystem]

uL = [0.0]
isLowerDirichlet = true
uR = [0.0]
isUpperDirichlet = true

InitialHeights = [1.0]
InitialProfile = ["Gaussian"]

[AdjointTestProblem]
SourceCentre = 0.3
kappa = 2.0

)"_toml;

BOOST_AUTO_TEST_SUITE(adjoint_test_suite, *boost::unit_test::tolerance(1e-6))

BOOST_AUTO_TEST_CASE(autodiff_init_tests)
{
    Grid testGrid(-1.0, 1.0, 4);
    BOOST_CHECK_NO_THROW(AdjointTestProblem problem(config_snippet, testGrid));
}

BOOST_AUTO_TEST_CASE(adjoint_init_tests)
{
    Grid testGrid(-1.0, 1.0, 4);
    AdjointTestProblem *problem = new AdjointTestProblem(config_snippet, testGrid);

    BOOST_CHECK_NO_THROW(AutodiffAdjointProblem adjoint(problem));

    delete problem;
}

BOOST_AUTO_TEST_CASE(test_derivatives)
{

    Grid testGrid(-1.0, 1.0, 4);
    AdjointTestProblem *problem = new AdjointTestProblem(config_snippet, testGrid);

    AutodiffAdjointProblem adjoint(problem);

    auto gfun = [&](Position x, Real p, RealVector &u, RealVector &q, RealVector &sigma, RealVector &phi)
    {
        return problem->g(x, p, u, q, sigma, phi);
    };

    BOOST_CHECK_NO_THROW(adjoint.setG(gfun));
    Value T_s = 50;
    Value SourceWidth = 0.02;
    Value SourceCentre = 0.3;

    auto dSdc = [&](Position x)
    {
        auto y = x - SourceCentre;
        return T_s * (2 * y) / SourceWidth * exp(-y * y / SourceWidth);
    };

    Values Positions(3);
    Positions << 0.2, 0.0, -0.2;
    State s(1);
    s.zero();

    // dGdp tests
    Value p;
    adjoint.dSources_dp(0, p, s, Positions(0));
    BOOST_TEST(dSdc(Positions(0)) == p);

    adjoint.dSources_dp(0, p, s, Positions(1));
    BOOST_TEST(dSdc(Positions(1)) == p);

    adjoint.dSources_dp(0, p, s, Positions(2));
    BOOST_TEST(dSdc(Positions(2)) == p);

    s.Derivative[0] = 1.0;

    adjoint.dSigmaFn_dp(1, p, s, Positions(0));
    BOOST_TEST(p == s.Derivative[0]);
    adjoint.dSigmaFn_dp(1, p, s, Positions(1));
    BOOST_TEST(p == s.Derivative[0]);
    adjoint.dSigmaFn_dp(1, p, s, Positions(2));
    BOOST_TEST(p == s.Derivative[0]);

    // dGdy tests
    s.Variable[0] = 2.0;

    Values grad(1);
    adjoint.dgFn_du(0, grad, s, 0.0);
    BOOST_TEST(grad(0) == 2.0);

    adjoint.dgFn_dq(0, grad, s, 0.0);
    BOOST_TEST(grad(0) == 0.0);

    adjoint.dgFn_dsigma(0, grad, s, 0.0);
    BOOST_TEST(grad(0) == 0.0);

    delete problem;
}

BOOST_AUTO_TEST_CASE(systemsolver_adjoint_tests)
{
    int nGrid = 4;
    Grid testGrid(-1.0, 1.0, nGrid);
    AdjointTestProblem *problem = new AdjointTestProblem(config_snippet, testGrid);

    AdjointProblem *adjoint = problem->createAdjointProblem();

    // auto gfun = [&](Position x, Real p, RealVector &u, RealVector &q, RealVector &sigma, RealVector &phi)
    // {
    //     return problem->g(x, p, u, q, sigma, phi);
    // };

    // adjoint->setG(gfun);

    Index k = 2; // make sure it works for higher order bases

    SystemSolver *system = nullptr;

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    BOOST_CHECK_NO_THROW(system = new SystemSolver(testGrid, k, problem, adjoint));

    system->setTau(1.0);
    system->resetCoeffs();
    system->initialiseMatrices();

    N_Vector y0, y0_dot;
    y0 = N_VNew_Serial(3 * nGrid * (k + 1) + 1 * (nGrid + 1), ctx);
    y0_dot = N_VClone(y0);
    BOOST_CHECK_NO_THROW(system->setInitialConditions(y0, y0_dot));

    Vector test_Vec(k + 1);
    Vector dGdu_test(k + 1);
    Vector zeroVec(k + 1);
    zeroVec.setZero();
    for (Index i = 0; i < nGrid; ++i)
    {
        auto I = testGrid[i];

        // dG/dCij = c_ij ?
        auto yCoeffs = system->y.u(0).getCoeff(i);
        dGdu_test = yCoeffs.second;
        BOOST_CHECK_NO_THROW(system->dGdu_Vec(0, test_Vec, system->y, I));
        BOOST_TEST((dGdu_test - test_Vec).norm() < 1e-9);

        BOOST_CHECK_NO_THROW(system->dGdq_Vec(0, test_Vec, system->y, I));
        BOOST_TEST((zeroVec - test_Vec).norm() == 0.0);

        BOOST_CHECK_NO_THROW(system->dGdsigma_Vec(0, test_Vec, system->y, I));
        BOOST_TEST((zeroVec - test_Vec).norm() == 0.0);

        BOOST_CHECK_NO_THROW(system->dGdaux_Vec(0, test_Vec, system->y, I));
        BOOST_TEST((zeroVec - test_Vec).norm() == 0.0);
    }

    BOOST_CHECK_NO_THROW(system->initializeMatricesForAdjointSolve());

    delete problem;
    delete adjoint;
    delete system;
}

BOOST_AUTO_TEST_SUITE_END()
