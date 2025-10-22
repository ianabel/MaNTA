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

    auto dGdp = [&](Position x)
    {
        auto y = x - SourceCentre;
        return T_s * (2 * y) / SourceWidth * exp(-y * y / SourceWidth);
    };

    Values Positions(3);
    Positions << 0.2, 0.0, -0.2;
    State s(1);
    s.zero();
    Value p;
    adjoint.dSources_dp(0, p, s, Positions(0));
    BOOST_TEST(dGdp(Positions(0)) == p);

    adjoint.dSources_dp(0, p, s, Positions(1));
    BOOST_TEST(dGdp(Positions(1)) == p);

    adjoint.dSources_dp(0, p, s, Positions(2));
    BOOST_TEST(dGdp(Positions(2)) == p);

    adjoint.dSigmaFn_dp(0, p, s, Positions(0));
    BOOST_TEST(p == 0.0);

    s.Variable[0] = 2.0;

    Values grad(1);
    adjoint.dgFn_du(0, grad, s, 0.0);
    BOOST_TEST(grad(0) == 4.0);

    adjoint.dgFn_dq(0, grad, s, 0.0);
    BOOST_TEST(grad(0) == 0.0);

    adjoint.dgFn_dsigma(0, grad, s, 0.0);
    BOOST_TEST(grad(0) == 0.0);

    delete problem;
}

BOOST_AUTO_TEST_CASE(systemsolver_adjoint_tests)
{
    Grid testGrid(-1.0, 1.0, 4);
    AdjointTestProblem *problem = new AdjointTestProblem(config_snippet, testGrid);

    AutodiffAdjointProblem *adjoint = new AutodiffAdjointProblem(problem);

    auto gfun = [&](Position x, Real p, RealVector &u, RealVector &q, RealVector &sigma, RealVector &phi)
    {
        return problem->g(x, p, u, q, sigma, phi);
    };

    adjoint->setG(gfun);

    Index k = 1; // Start piecewise linear

    SystemSolver *system = nullptr;
    double tau = 0.5;

    BOOST_CHECK_NO_THROW(system = new SystemSolver(testGrid, k, problem, adjoint));

    // SUNContext ctx;
    // SUNContext_Create(SUN_COMM_NULL, &ctx);

    // N_Vector y0, y0_dot;

    // y0 = N_VNew_Serial(3 * 4 * (2) + 1 * (4 + 1), ctx);
    // y0_dot = N_VClone(y0);
    // system->setInitialConditions(y0, y0_dot);

    delete problem;
    delete adjoint;
    delete system;
}

BOOST_AUTO_TEST_SUITE_END()
