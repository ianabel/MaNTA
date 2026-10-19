#include "../../PhysicsCases/AdjointTestProblem.hpp"
#include "../../PhysicsCases/AutodiffAdjointProblem.hpp"
#include "Types.hpp"
#include <boost/math/quadrature/gauss.hpp>
#include <boost/test/unit_test.hpp>
#include <toml.hpp>

#include "SystemSolver.hpp"
#include "CapturedOutput.hpp"
#include <filesystem>
#include <memory>

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
    // AdjointTestProblem has nAux == 0, so both dGdaux_Vec's output and the
    // dg/dphi slice it reads are empty -- it writes one block per auxiliary
    // variable, not per variable. This is therefore only a "does not throw"
    // check; the substantive comparison lives in AdjointProblemTests.cpp, whose
    // fixture has nAux == 1 with nVars == 2 and so distinguishes the two
    // lengths.
    Vector aux_Vec(0);
    Matrix no_aux(0, k + 1);
    BOOST_CHECK_NO_THROW(system->dGdaux_Vec(0, aux_Vec, no_aux, system->y, i));
  }

  BOOST_CHECK_NO_THROW(system->initializeMatricesForAdjointSolve());

  delete problem;
  delete system;
}


// Every objective gets its own gradient, and each one matches a finite difference.
//
// The adjoint of an objective is the solve's answer, not a shared workspace, so
// each of the ng objectives needs its own dG/dy (G_y), its own adjoint state
// (adjoint_squ) and its own row of G_p. Two of those three are filled by
// emplace_back, which makes a stale prefix the natural failure: G_y[cell] and
// adjoint_squ[cell] index by *cell*, so anything left over from an earlier
// objective is what every cell reads, and the result is that objective 1 silently
// receives objective 0's gradient.
//
// Nothing about the answer says so -- G is right, the matrix is the right shape,
// and the rows are merely equal -- which is why this differences the objective
// itself. AdjointTestProblem is the fixture for it because its two objectives are
// 0.5 u^2 and 2 u^2: the same functional up to a factor of four, so a gradient
// that has been copied from the other one is off by exactly that and cannot hide
// in a tolerance.
BOOST_AUTO_TEST_CASE(each_objective_gets_its_own_gradient) {
  constexpr Index nCells = 10;
  constexpr Index order = 3;

  // One steady solve at a given parameter offset; returns G per objective and,
  // optionally, the adjoint gradients.
  auto solveWith = [&](Index pIndex, double delta, Matrix *G_p_out) {
    Grid grid(0.0, 1.0, nCells);
    auto problem = std::make_unique<AdjointTestProblem>(config_snippet, grid);
    if (delta != 0.0)
      problem->setPval(pIndex, Real(problem->getPval(pIndex).val + delta));
    auto adjoint = problem->createAdjointProblem();

    SystemSolver system(grid, order, problem.get());
    system.setTau(10.0);
    system.resetCoeffs();
    system.setInputFile("adjoint_gradient_per_objective");
    system.setOutputCadence(0.25);
    system.setNOutput(2);
    system.setInitialTime(0.0);
    system.setMinStepSize(1e-12);
    system.setTolerances({1e-8}, 1e-6);
    system.setSteadyStateTolerance(1e-10);
    system.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
    system.setSolveAdjoint(true);
    system.setAdjointProblem(adjoint.get());
    system.setWriteOutput(false);
    system.setWriteDatFile(false);
    system.setWriteDebugDatFiles(false);

    {
      CapturedOutput quiet;
      system.initialize();
      system.integrate(0.0);
    }

    Vector G(adjoint->getNg());
    for (Index g = 0; g < adjoint->getNg(); ++g)
      G(g) = adjoint->GFn(g, system.yJac);
    if (G_p_out)
      *G_p_out = system.G_p;
    {
      CapturedOutput quiet;
      system.destroySundials();
    }
    return G;
  };

  Matrix G_p;
  const Vector G0 = solveWith(0, 0.0, &G_p);

  // Computing gradients must leave the case exactly as it found it. dGFndp
  // differentiates by writing a seeded parameter back through setPval, which
  // takes a *parameter* index; writing the objective's index there instead
  // leaves one parameter holding another's value, and every later residual then
  // evaluates a different problem. Nothing announces that -- G is still a
  // number, the solve has already finished -- so it is checked directly.
  {
    Grid grid(0.0, 1.0, nCells);
    auto problem = std::make_unique<AdjointTestProblem>(config_snippet, grid);
    auto adjoint = problem->createAdjointProblem();
    std::vector<double> before;
    for (Index p = 0; p < 2; ++p)
      before.push_back(problem->getPval(p).val);
    BOOST_TEST_REQUIRE(before[0] != before[1],
                       "both parameters are " << before[0]
                           << ", so a swap between them would be invisible");

    SystemSolver system(grid, order, problem.get());
    system.setTau(10.0);
    system.resetCoeffs();
    system.setInputFile("adjoint_gradient_per_objective");
    system.setOutputCadence(0.25);
    system.setNOutput(2);
    system.setInitialTime(0.0);
    system.setMinStepSize(1e-12);
    system.setTolerances({1e-8}, 1e-6);
    system.setSteadyStateTolerance(1e-10);
    system.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
    system.setSolveAdjoint(true);
    system.setAdjointProblem(adjoint.get());
    system.setWriteOutput(false);
    system.setWriteDatFile(false);
    system.setWriteDebugDatFiles(false);

    N_Vector zero = nullptr, F = nullptr;
    double residualBefore = 0.0, residualAfter = 0.0;
    {
      CapturedOutput quiet;
      system.initialize();
      system.integrate(0.0);

      zero = N_VClone(system.Y);
      F = N_VClone(system.Y);
      N_VConst(0.0, zero);
      system.residual(system.t0, system.Y, zero, F);
      residualBefore = std::sqrt(N_VDotProd(F, F));

      system.runAdjointSolve();

      system.residual(system.t0, system.Y, zero, F);
      residualAfter = std::sqrt(N_VDotProd(F, F));
    }

    for (Index p = 0; p < 2; ++p)
      BOOST_TEST(problem->getPval(p).val == before[p],
                 "parameter " << p << " moved from " << before[p] << " to "
                     << problem->getPval(p).val << " while gradients were computed");

    BOOST_TEST(residualAfter == residualBefore,
               "the residual at an unchanged state went from " << residualBefore
                   << " to " << residualAfter << " across the gradient computation, "
                   "so the physics case is no longer the one that was solved");

    N_VDestroy(zero);
    N_VDestroy(F);
    {
      CapturedOutput quiet;
      system.destroySundials();
    }
  }
  BOOST_TEST_REQUIRE(G0.size() == 2, "this case needs two objectives to say anything");

  // Not vacuous: the two objectives must actually differ, or equal gradient rows
  // would be the right answer.
  BOOST_TEST_REQUIRE(std::abs(G0(1) - G0(0)) > 1e-6,
                     "the two objectives agree to " << std::abs(G0(1) - G0(0))
                         << ", so this fixture cannot tell a copied gradient apart");

  const double h = 1e-5;
  for (Index p = 0; p < 2; ++p) {
    const Vector Gplus = solveWith(p, +h, nullptr);
    const Vector Gminus = solveWith(p, -h, nullptr);
    for (Index g = 0; g < G0.size(); ++g) {
      const double fd = (Gplus(g) - Gminus(g)) / (2.0 * h);
      BOOST_TEST_MESSAGE("objective " << g << ", parameter " << p
                         << ": adjoint " << G_p(g, p) << " against " << fd);
      // 1e-8, not looser: GFn and dg are now one functional, and the agreement
      // measured is 6e-10. A GFn integrating g some other way moves this by
      // the difference between the two quadratures, not by rounding.
      BOOST_TEST(G_p(g, p) == fd, boost::test_tools::tolerance(1e-8));
    }
  }

  std::filesystem::remove("adjoint_gradient_per_objective.nc");
}


// The corrected objective is worth more than the tolerance it was solved at.
//
// A steady solve stops when ||F|| is small, not when G is, so G carries an error
// that nothing reports. estimateObjective() extrapolates it to the fixed point --
// G - (dG/dy) . J^-1 F, one Jacobian solve against a whole continuation -- and
// bounds what is left with ||dG/dy|| ||J^-1 F||.
//
// Both halves are checked against a solve two orders tighter, which stands in for
// the fixed point. The correction has to be a large improvement rather than a
// small one, or the extra solve is not worth making; the bound has to *hold*,
// since a bound that can be exceeded is not one.
BOOST_AUTO_TEST_CASE(the_corrected_objective_beats_the_tolerance_it_was_solved_at) {
  constexpr Index nCells = 10;
  constexpr Index order = 3;

  auto solveTo = [&](double steadyTol) {
    Grid grid(0.0, 1.0, nCells);
    auto problem = std::make_unique<AdjointTestProblem>(config_snippet, grid);
    auto adjoint = problem->createAdjointProblem();

    SystemSolver system(grid, order, problem.get());
    system.setTau(10.0);
    system.resetCoeffs();
    system.setInputFile("adjoint_objective_estimate");
    system.setOutputCadence(0.25);
    system.setNOutput(2);
    system.setInitialTime(0.0);
    system.setMinStepSize(1e-12);
    system.setTolerances({1e-8}, 1e-6);
    system.setSteadyStateTolerance(steadyTol);
    system.setSteadyTest(SystemSolver::SteadyTest::Residual);
    system.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
    system.setSolveAdjoint(true);
    system.setAdjointProblem(adjoint.get());
    system.setWriteOutput(false);
    system.setWriteDatFile(false);
    system.setWriteDebugDatFiles(false);
    {
      CapturedOutput quiet;
      system.initialize();
      system.integrate(0.0);
    }
    const auto estimate = system.lastObjectiveEstimate();
    const auto outcome = system.lastSteadyOutcome();
    {
      CapturedOutput quiet;
      system.destroySundials();
    }
    return std::make_pair(estimate, outcome);
  };

  const auto [tight, tightOutcome] = solveTo(1e-8);
  BOOST_TEST_REQUIRE((tightOutcome == SystemSolver::SteadyOutcome::Converged));
  BOOST_TEST_REQUIRE(tight.valid);
  const double reference = tight.value(0);

  const auto [loose, looseOutcome] = solveTo(1e-2);
  BOOST_TEST_REQUIRE((looseOutcome == SystemSolver::SteadyOutcome::Converged));
  BOOST_TEST_REQUIRE(loose.valid);

  const double rawError = std::abs(loose.value(0) - reference);
  const double correctedError = std::abs(loose.corrected(0) - reference);

  BOOST_TEST_MESSAGE("loose solve: raw error " << rawError << ", corrected "
                     << correctedError << ", bound " << loose.uncertainty(0));

  // Not vacuous: the loose solve has to have stopped visibly short, or there is
  // nothing for the correction to recover.
  BOOST_TEST_REQUIRE(rawError > 1e-4 * std::abs(reference),
                     "the loose solve is already accurate to " << rawError
                         << ", so this fixture cannot show a correction working");

  BOOST_TEST(correctedError < rawError / 100.0,
             "correcting moved the objective from " << rawError << " to "
                 << correctedError << " of the reference; it is supposed to be "
                 "worth orders of magnitude, not a factor");

  BOOST_TEST(loose.uncertainty(0) >= rawError,
             "the bound is " << loose.uncertainty(0) << " but the error is "
                 << rawError << ", so it does not bound it");

  // And it is a bound worth reporting rather than a vacuous one.
  BOOST_TEST(loose.uncertainty(0) < 10.0 * rawError,
             "the bound is " << loose.uncertainty(0) << " against an error of "
                 << rawError << ", too loose to tell a caller anything");

  // A converged solve reports a small one, and every objective is estimated.
  BOOST_TEST(tight.value.size() == 2);
  BOOST_TEST(tight.uncertainty(0) < 1e-6 * std::abs(reference));

  std::filesystem::remove("adjoint_objective_estimate.nc");
}

// An autodiff objective's value and its gradient are one functional.
//
// G_y is the derivative of G = sum_j w_j g(Z_j) over the nodes the objective is
// integrated on -- assembleObjectiveGradient weights the nodal dg/dZ by diag(w)
// -- so GFn has to report that sum, and dGFndp has to differentiate it. Both
// used to integrate g(u_h) exactly instead, with a Gauss-Kronrod rule over the
// whole domain: a different functional, close enough that a finite-difference
// check of the gradient at 2e-4 could not tell. These check each half at
// round-off, against a guard that the exact integral is visibly different.
namespace {

struct AutodiffObjectiveFixture {
  Grid grid;
  std::unique_ptr<AdjointTestProblem> problem;
  std::unique_ptr<AutodiffAdjointProblem> objective;
  std::unique_ptr<SystemSolver> sys;

  explicit AutodiffObjectiveFixture(bool superconvergent)
      : grid(0.0, 1.0, 6),
        problem(std::make_unique<AdjointTestProblem>(config_snippet, grid)) {
    // Nonlinear in u, varying in x, and reading a parameter, so neither the
    // nodal rule nor the star rule integrates it exactly and dG/dp is nonzero.
    objective = std::make_unique<AutodiffAdjointProblem>(problem.get());
    AdjointTestProblem *pr = problem.get();
    objective->addG([pr](Position x, RealVector &u, RealVector &, RealVector &,
                         RealVector &) -> Real {
      return pr->getPval(0) * (1.0 + x) * u(0) * u(0);
    });
    objective->setNp(1);

    sys = std::make_unique<SystemSolver>(grid, 3, problem.get());
    sys->setTau(10.0);
    sys->setInputFile("autodiff_objective_quadrature");
    sys->setWriteOutput(false);
    sys->setWriteDatFile(false);
    sys->setOutputCadence(1.0);
    sys->setNOutput(2);
    sys->setInitialTime(0.0);
    sys->setSuperconvergent(superconvergent);
    sys->setAdjointProblem(objective.get());
    sys->setSolveAdjoint(false);
    CapturedOutput quiet;
    sys->initialize();
  }

  ~AutodiffObjectiveFixture() {
    CapturedOutput quiet;
    sys->destroySundials();
  }

  // G at the state held in `buf`, as the solver would report it.
  double G(std::vector<double> &buf) {
    DGSoln v(problem->getNumVars(), grid, 3, buf.data(), 0, problem->getNumAux());
    if (sys->isSuperconvergent()) {
      sys->postprocessor->computeUStar(v);
      return objective->GFn(0, v, *sys->postprocessor);
    }
    return objective->GFn(0, v);
  }
};

} // namespace

BOOST_AUTO_TEST_CASE(the_autodiff_objective_is_the_functional_dg_differentiates) {
  for (bool superconvergent : {false, true}) {
    BOOST_TEST_CONTEXT("superconvergent = " << superconvergent) {
      AutodiffObjectiveFixture f(superconvergent);
      DGSoln const &y = f.sys->yJac;
      std::vector<double> base(y.data(), y.data() + y.getDoF());

      const auto Gy = f.sys->assembleObjectiveGradient(*f.objective, 0, y, f.sys->jt);
      const Index cellDOF = Gy[0].size();
      std::vector<double> delta(base.size(), 0.0);
      double predicted = 0.0;
      for (size_t i = 0; i < Gy.size(); ++i)
        for (Index d = 0; d < cellDOF; ++d) {
          const Index n = static_cast<Index>(i) * cellDOF + d;
          delta[n] = std::sin(1.7 * n + 0.3);
          predicted += Gy[i](d) * delta[n];
        }

      // G is quadratic in the state (u* is linear in u and q), so a central
      // difference is exact up to rounding.
      const double h = 1e-4;
      auto shifted = [&](double s) {
        std::vector<double> buf(base);
        for (size_t n = 0; n < buf.size(); ++n)
          buf[n] += s * delta[n];
        return buf;
      };
      std::vector<double> plus = shifted(h), minus = shifted(-h);
      const double fd = (f.G(plus) - f.G(minus)) / (2.0 * h);
      BOOST_TEST_MESSAGE("dG/dy . delta " << predicted << " against " << fd);
      BOOST_TEST(predicted == fd, boost::test_tools::tolerance(1e-9));

      // Not vacuous: the exact integral of g(u_h), which GFn used to report, has
      // a derivative this test would have rejected.
      if (!superconvergent) {
        auto exact = [&](std::vector<double> &buf) {
          DGSoln v(f.problem->getNumVars(), f.grid, 3, buf.data(), 0, f.problem->getNumAux());
          return v.EvaluateIntegral([&](DGSoln const &w, Position x) {
            return f.objective->gFn(0, w.eval(x), x);
          });
        };
        const double fdExact = (exact(plus) - exact(minus)) / (2.0 * h);
        BOOST_TEST(std::abs(fdExact - predicted) > 1e-6 * std::abs(predicted),
                   "the exact integral's derivative " << fdExact << " is within "
                       << std::abs(fdExact - predicted)
                       << " of dG/dy, so this fixture cannot tell the two apart");
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(the_autodiff_parameter_derivative_is_of_the_reported_objective) {
  Value basisDerivative = 0.0, starDerivative = 0.0;
  for (bool superconvergent : {false, true}) {
    BOOST_TEST_CONTEXT("superconvergent = " << superconvergent) {
      AutodiffObjectiveFixture f(superconvergent);
      DGSoln &y = f.sys->yJac;
      std::vector<double> base(y.data(), y.data() + y.getDoF());
      const double p0 = f.problem->getPval(0).val;

      Value analytic = 0.0;
      if (superconvergent) {
        f.sys->postprocessor->computeUStar(y);
        analytic = f.objective->dGFndp(0, y, *f.sys->postprocessor)(0, 0);
        starDerivative = analytic;
      } else {
        analytic = f.objective->dGFndp(0, y)(0, 0);
        BOOST_TEST(f.objective->dGFndp(0, 0, y) == analytic,
                   boost::test_tools::tolerance(1e-15));
        basisDerivative = analytic;
      }
      BOOST_TEST(f.problem->getPval(0).val == p0,
                 "dGFndp left the parameter at " << f.problem->getPval(0).val);

      // G is linear in p, so the central difference is exact up to rounding.
      const double h = 1e-3;
      f.problem->setPval(0, Real(p0 + h));
      const double Gplus = f.G(base);
      f.problem->setPval(0, Real(p0 - h));
      const double Gminus = f.G(base);
      f.problem->setPval(0, Real(p0));
      const double fd = (Gplus - Gminus) / (2.0 * h);
      BOOST_TEST_MESSAGE("dG/dp " << analytic << " against " << fd);
      BOOST_TEST(analytic == fd, boost::test_tools::tolerance(1e-10));
    }
  }

  // Not vacuous: at this k, u* differs from u_h enough that a superconvergent
  // dGFndp taken on the basis nodes, which is what it used to be, would fail.
  BOOST_TEST(std::abs(starDerivative - basisDerivative) > 1e-6 * std::abs(starDerivative),
             "the star-node and basis-node derivatives agree to "
                 << std::abs(starDerivative - basisDerivative));
}


// An autodiff case whose auxiliary equation reads an adjoint parameter:
//
//     u_t - u_xx = 1 - a,   a = p u,   u = 0 at both ends.
//
// AdjointProblem::ComputePhysicsDerivatives asks dAux_dp of every auxiliary
// variable, and AutodiffAdjointProblem used to leave it at the base default,
// which throws -- so no autodiff case with an aux variable could compute a
// gradient, whether or not the aux equation read a parameter.
namespace {

class ParametrisedAux : public AutodiffTransportSystem {
public:
  explicit ParametrisedAux(Grid const &grid)
      : AutodiffTransportSystem(toml::value(toml::table{}), grid,
                                {.variables = {{"u", "", ""}}, .aux = {{"a", "p u", ""}}}) {
    addP(rate);
  }
  Real rate = 2.0;

private:
  Value LowerBoundary(Index, Time) const override { return 0.0; }
  Value UpperBoundary(Index, Time) const override { return 0.0; }
  Real Flux(Index, RealVector, RealVector q, Real, Time) override { return q(0); }
  Real Source(Index, RealVector, RealVector, RealVector, RealVector phi, Real, Time) override {
    return 1.0 - phi(0);
  }
  Real GFunc(Index, RealVector u, RealVector, RealVector, RealVector phi, Position, Time) override {
    return phi(0) - rate * u(0);
  }
  Real2nd InitialFunction(Index, Real2nd x, Real2nd) const override { return 0.1 * sin(M_PI * x); }
  Value InitialAuxValue(Index, Position x, Time) const override {
    return rate.val * 0.1 * std::sin(M_PI * x);
  }
};

} // namespace

BOOST_AUTO_TEST_CASE(an_autodiff_aux_equation_reports_its_parameter_derivative) {
  Grid grid(0.0, 1.0, 4);
  ParametrisedAux problem(grid);
  AutodiffAdjointProblem adjoint(&problem);
  adjoint.setNp(1);

  State s(1, 0, 1);
  s.u(0) = 0.3;
  s.phi(0) = 0.5;
  Value grad = 0.0;
  BOOST_REQUIRE_NO_THROW(adjoint.dAux_dp(0, 0, grad, s, 0.4));
  // d/dp (a - p u) = -u, exactly.
  BOOST_TEST(grad == -0.3, boost::test_tools::tolerance(1e-15));
  BOOST_TEST(problem.rate.val == 2.0, "dAux_dp left the parameter at " << problem.rate.val);
}

BOOST_AUTO_TEST_CASE(an_autodiff_case_with_an_aux_variable_gets_a_gradient) {
  constexpr Index nCells = 8, order = 3;

  auto solveWith = [&](double delta, double *Gp) {
    Grid grid(0.0, 1.0, nCells);
    ParametrisedAux problem(grid);
    problem.rate = Real(problem.rate.val + delta);
    AutodiffAdjointProblem adjoint(&problem);
    adjoint.addG([](Position, RealVector &u, RealVector &, RealVector &, RealVector &) -> Real {
      return u(0) * u(0);
    });
    adjoint.setNp(1);

    SystemSolver system(grid, order, &problem);
    system.setTau(1.0);
    system.resetCoeffs();
    system.setInputFile("autodiff_aux_gradient");
    system.setOutputCadence(0.25);
    system.setNOutput(2);
    system.setInitialTime(0.0);
    system.setMinStepSize(1e-12);
    system.setTolerances({1e-10}, 1e-8);
    system.setSteadyStateTolerance(1e-11);
    system.setSteadyMode(SystemSolver::SteadyMode::Newton);
    system.setSolveAdjoint(Gp != nullptr);
    system.setAdjointProblem(&adjoint);
    system.setWriteOutput(false);
    system.setWriteDatFile(false);
    {
      CapturedOutput quiet;
      system.initialize();
      system.integrate(0.0);
    }
    if (Gp)
      *Gp = system.G_p(0, 0);
    const double G = adjoint.GFn(0, system.yJac);
    {
      CapturedOutput quiet;
      system.destroySundials();
    }
    return G;
  };

  double Gp = 0.0;
  BOOST_REQUIRE_NO_THROW(solveWith(0.0, &Gp));
  const double h = 1e-5;
  const double fd = (solveWith(+h, nullptr) - solveWith(-h, nullptr)) / (2.0 * h);
  BOOST_TEST_MESSAGE("dG/dp: adjoint " << Gp << " against " << fd);
  // Not vacuous: the rate really moves G.
  BOOST_TEST(std::abs(fd) > 1e-4);
  BOOST_TEST(Gp == fd, boost::test_tools::tolerance(1e-6));
  std::filesystem::remove("autodiff_aux_gradient.nc");
}


BOOST_AUTO_TEST_SUITE_END()
