#include "../../PhysicsCases/AdjointTestProblem.hpp"
#include "../../PhysicsCases/AutodiffAdjointProblem.hpp"
#include "Types.hpp"
#include <boost/math/quadrature/gauss.hpp>
#include <boost/test/unit_test.hpp>
#include <toml.hpp>

#include "SystemSolver.hpp"

#include <nvector/nvector_serial.h> /* access to serial N_Vector            */
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

BOOST_AUTO_TEST_CASE(autodiff_init_tests) {
  Grid testGrid(-1.0, 1.0, 4);
  BOOST_CHECK_NO_THROW(AdjointTestProblem problem(config_snippet, testGrid));
}

BOOST_AUTO_TEST_CASE(adjoint_init_tests) {
  Grid testGrid(-1.0, 1.0, 4);
  AdjointTestProblem *problem =
      new AdjointTestProblem(config_snippet, testGrid);

  BOOST_CHECK_NO_THROW(AutodiffAdjointProblem adjoint(problem));

  delete problem;
}

BOOST_AUTO_TEST_CASE(test_derivatives) {

  Grid testGrid(-1.0, 1.0, 4);
  AdjointTestProblem *problem =
      new AdjointTestProblem(config_snippet, testGrid);

  AutodiffAdjointProblem adjoint(problem);

  auto gfun = [&](Position x, RealVector &u, RealVector &q, RealVector &sigma,
                  RealVector &phi) { return problem->g1(x, u, q, sigma, phi); };

  BOOST_CHECK_NO_THROW(adjoint.addG(gfun));
  Value T_s = 50;
  Value SourceWidth = 0.02;
  Value SourceCentre = 0.3;

  auto dSdc = [&](Position x) {
    auto y = x - SourceCentre;
    return T_s * (2 * y) / SourceWidth * exp(-y * y / SourceWidth);
  };

  Values Positions(3);
  Positions << 0.2, 0.0, -0.2;
  State s(1);
  s.zero();

  // dGdp tests
  Value p;
  adjoint.dSources_dp(0, 0, p, s, Positions(0));
  BOOST_TEST(dSdc(Positions(0)) == p);

  adjoint.dSources_dp(0, 0, p, s, Positions(1));
  BOOST_TEST(dSdc(Positions(1)) == p);

  adjoint.dSources_dp(0, 0, p, s, Positions(2));
  BOOST_TEST(dSdc(Positions(2)) == p);

  s.q(0) = 1.0;
  s.u(0) = 2.0;

  adjoint.dSigmaFn_dp(0, 1, p, s, Positions(0));
  BOOST_TEST(p == s.q(0));
  adjoint.dSigmaFn_dp(0, 1, p, s, Positions(1));
  BOOST_TEST(p == s.q(0));
  adjoint.dSigmaFn_dp(0, 1, p, s, Positions(2));
  BOOST_TEST(p == s.q(0));

  // dGdy tests

  Values grad(1);
  adjoint.dgFn_du(0, grad, s, 0.0);
  BOOST_TEST(grad(0) == 2.0);

  adjoint.dgFn_dq(0, grad, s, 0.0);
  BOOST_TEST(grad(0) == 0.0);

  adjoint.dgFn_dsigma(0, grad, s, 0.0);
  BOOST_TEST(grad(0) == 0.0);

  delete problem;
}

BOOST_AUTO_TEST_CASE(systemsolver_adjoint_tests) {
  int nGrid = 4;
  Grid testGrid(-1.0, 1.0, nGrid);
  AdjointTestProblem *problem =
      new AdjointTestProblem(config_snippet, testGrid);

  std::unique_ptr<AdjointProblem> adjoint = problem->createAdjointProblem();

  // auto gfun = [&](Position x, Real p, RealVector &u, RealVector &q,
  // RealVector &sigma, RealVector &phi)
  // {
  //     return problem->g(x, p, u, q, sigma, phi);
  // };

  // adjoint->addG(gfun);

  Index k = 2; // make sure it works for higher order bases

  SystemSolver *system = nullptr;

  SUNContext ctx;
  SUNContext_Create(SUN_COMM_NULL, &ctx);

  BOOST_CHECK_NO_THROW(system = new SystemSolver(testGrid, k, problem));

  system->setAdjointProblem(adjoint.get());

  system->setTau(1.0);
  system->resetCoeffs();
  system->initialiseMatrices();

  N_Vector y0, y0_dot;
  y0 = N_VNew_Serial(3 * nGrid * (k + 1) + 1 * (nGrid + 1), ctx);
  y0_dot = N_VClone(y0);
  BOOST_CHECK_NO_THROW(system->setInitialConditions(y0, y0_dot));
  // dGdu_Vec / dGdq_Vec / dGdsigma_Vec used to be compared here against the
  // basis's Gauss rule. They are gone: they integrated dg/dZ against the basis
  // functions, which differentiates Int g dx rather than the sum_m w_m g_m that
  // GFn reports, and nothing in a solve ever called them.
  for (Index i = 0; i < nGrid; ++i) {
    // AdjointTestProblem has nAux == 0, so dGdaux_Vec's output is empty --
    // it writes one block per auxiliary variable, not per variable. This is
    // therefore only a "does not throw" check; the substantive comparison
    // against quadrature lives in AdjointProblemTests.cpp, whose fixture has
    // nAux == 1 with nVars == 2 and so distinguishes the two lengths.
    Vector aux_Vec(0);
    BOOST_CHECK_NO_THROW(system->dGdaux_Vec(0, aux_Vec, system->y, i));
  }

  BOOST_CHECK_NO_THROW(system->initializeMatricesForAdjointSolve());

  delete problem;
  delete system;
}

BOOST_AUTO_TEST_SUITE_END()
