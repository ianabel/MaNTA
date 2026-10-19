// The adjoint solved on demand, from the last Jacobian build, and what it says
// about the physics.
//
// Three properties, each of which nothing else in the suite would notice losing:
//
//  * **No physics call.** solveAdjointAt assembles J^T from the derivatives the
//    last build took, so an expensive flux model is asked for nothing. A case
//    that counts every entry point it has is the only witness: a regression that
//    re-evaluated the derivatives would give the same z.
//
//  * **w = dG/dGamma, per node.** The sensitivity of G to the flux and to the
//    source at each physics node is -z^T dF/dGamma, one column of the test-space
//    projection. A sign or operator error in it is invisible to every aggregate,
//    so it is checked against finite differences of G under a perturbation of
//    one node at a time -- and the check is guarded against the trap CLAUDE.md
//    records for the adjoint's dG/dZ: diag(w) and the mass matrix agree on
//    anything the interpolatory rule integrates exactly, so the guard requires
//    the two candidate operators to differ by far more than the tolerance.
//
//  * **The forward solve does not see an adjoint solve.** The adjoint keeps its
//    own factorisations, so a sliced steady solve with an adjoint solve between
//    every slice ends bit for bit where one without them does.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "AdjointProblem.hpp"
#include "SystemSolver.hpp"
#include "TransportSystem.hpp"

#include <boost/math/quadrature/gauss.hpp>

#include <cmath>
#include <cstring>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

namespace
{
using std::numbers::pi;

// One physics node's flux or source moved by a constant. `node` indexes the
// physics points ComputePhysics is handed, which are the same every call.
struct Perturbation
{
    Index node = -1;
    double flux = 0.0;
    double source = 0.0;
};

// -d_x( kappa(u) d_x u ) = pi^2 sin(pi x) on [0, 1], u = 0 at both ends. With
// kappa = 1 the solution is u = sin(pi x), and for G = int u^2 dx the adjoint
// solves -z'' = 2u, so z = 2 sin(pi x) / pi^2 -- not a polynomial at any degree,
// which is what separates the two candidate operators below. `nonlinear` makes
// kappa = 1 + u^2 / 4, for a solve that takes more than one step.
//
// Every entry point the solver can reach is counted.
class CountingDiffusion : public TransportSystem
{
public:
    explicit CountingDiffusion(bool nonlinear = false)
        : TransportSystem(SystemSpec{.variables = numberedFields(1)}), nonlinear(nonlinear)
    {
    }

    bool nonlinear;
    Perturbation perturbation;
    // Batched entry points, and pointwise hooks; the batched defaults reach the
    // second through the first.
    long batched = 0;
    mutable long calls = 0;

    void reset() { batched = calls = 0; }

    PhysicsOutput ComputePhysics(GlobalState const &states, std::vector<Position> const &x,
                                 Time t) override
    {
        ++batched;
        PhysicsOutput out = TransportSystem::ComputePhysics(states, x, t);
        const Index m = perturbation.node;
        if (m >= 0 && m < static_cast<Index>(x.size()))
        {
            out[0][0](m) += perturbation.flux;
            out[1][0](m) += perturbation.source;
        }
        return out;
    }
    void ComputePhysicsDerivatives(std::array<std::reference_wrapper<GlobalStateMatrix>,
                                              NPHYSICS_FUNCTIONS> &&out,
                                   GlobalState const &states, std::vector<Position> const &x,
                                   Time t) override
    {
        ++batched;
        TransportSystem::ComputePhysicsDerivatives(std::move(out), states, x, t);
    }
    void ComputeSourceTimeDerivatives(GlobalStateMatrix &out, GlobalState const &states,
                                      std::vector<Position> const &x, Time t) override
    {
        ++batched;
        TransportSystem::ComputeSourceTimeDerivatives(out, states, x, t);
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
    Value aFn(Index, Position) override
    {
        ++calls;
        return 1.0;
    }

    double kappa(double u) const { return nonlinear ? 1.0 + 0.25 * u * u : 1.0; }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        ++calls;
        return kappa(s.u(0)) * s.q(0);
    }
    Value Sources(Index, const State &, Position x, Time) override
    {
        ++calls;
        return pi * pi * std::sin(pi * x);
    }
    void dSigmaFn_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        ++calls;
        v[0] = nonlinear ? 0.5 * s.u(0) * s.q(0) : 0.0;
    }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        ++calls;
        v[0] = kappa(s.u(0));
    }
    void dSources_du(Index, VectorRef, const State &, Position, Time) override { ++calls; }
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override { ++calls; }
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override { ++calls; }

    Value InitialValue(Index, Position) const override
    {
        ++calls;
        return 0.0;
    }
    Value InitialDerivative(Index, Position) const override
    {
        ++calls;
        return 0.0;
    }
};

// G_0 = int u^2 dx and G_1 = int 3 u^2 dx, as the interpolatory quadrature GFn
// has to report for G_y to be its exact derivative. One parameter, the source's
// amplitude, so that solveAdjoint has a gradient to compute.
class SquareIntegral : public AdjointProblem
{
public:
    SquareIntegral()
    {
        ng = 2;
        np = 1;
    }

    static double scale(Index g) { return g == 0 ? 1.0 : 3.0; }

    // The u*-based objective is the base class's, over gFn below.
    using AdjointProblem::GFn;

    Value GFn(Index g, DGSoln &y) const override
    {
        Value G = 0.0;
        for (Index i = 0; i < static_cast<Index>(y.getGrid().getNCells()); ++i)
        {
            const Vector w = y.getBasis().getIntegrationWeights(y.getGrid()[i]);
            const auto &u = y.u(0).getCoeff(i).second;
            for (Index j = 0; j < w.size(); ++j)
                G += w(j) * u(j) * u(j);
        }
        return scale(g) * G;
    }
    Value dGFndp(Index, Index, DGSoln &) const override { return 0.0; }
    Value gFn(Index g, const State &s, Position) const override
    {
        return scale(g) * s.u(0) * s.u(0);
    }
    void dgFn_du(Index g, VectorRef v, const State &s, Position) override
    {
        v[0] = 2.0 * scale(g) * s.u(0);
    }
    void dgFn_dq(Index, VectorRef, const State &, Position) override {}
    void dgFn_dsigma(Index, VectorRef, const State &, Position) override {}
    void dgFn_dphi(Index, VectorRef, const State &, Position) override {}
    void dSigmaFn_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }
    void dSources_dp(Index, Index, Value &out, const State &, Position x) override
    {
        out = pi * pi * std::sin(pi * x);
    }
};

struct Options
{
    Index nCells = 4;
    Index k = 2;
    bool superconvergent = false;
    bool nonlinear = false;
    bool estimate = true;   // the steady solve estimates G on its way out
    bool solveAdjoint = false;
    Perturbation perturbation;
};

// A steady solve run to round-off, with the SUNDIALS objects left alive so that
// a test can go on to drive the solver. Destroyed with the object.
struct Run
{
    std::unique_ptr<Grid> grid;
    std::unique_ptr<CountingDiffusion> problem;
    std::unique_ptr<SquareIntegral> objective;
    std::unique_ptr<SystemSolver> sys;

    SystemSolver *operator->() { return sys.get(); }

    ~Run()
    {
        if (sys)
        {
            CapturedOutput quiet;
            sys->destroySundials();
        }
    }

    // G at the state getSolution() would read, as the run reports it.
    double G(Index g = 0)
    {
        if (sys->isSuperconvergent())
        {
            sys->postprocessor->computeUStar(sys->yJac);
            return objective->GFn(g, sys->yJac, *sys->postprocessor);
        }
        return objective->GFn(g, sys->yJac);
    }
};

std::unique_ptr<Run> configure(Options const &o, std::string const &stem)
{
    auto run = std::make_unique<Run>();
    run->grid = std::make_unique<Grid>(0.0, 1.0, o.nCells);
    run->problem = std::make_unique<CountingDiffusion>(o.nonlinear);
    run->problem->perturbation = o.perturbation;
    run->objective = std::make_unique<SquareIntegral>();
    run->sys = std::make_unique<SystemSolver>(*run->grid, o.k, run->problem.get());

    SystemSolver &sys = *run->sys;
    sys.setTau(1.0);
    sys.setInputFile(stem);
    sys.setWriteOutput(false);
    sys.setWriteDatFile(false);
    sys.setOutputCadence(1.0);
    sys.setNOutput(2);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-14);
    sys.setTolerances({1e-12}, 1e-10);
    sys.setSuperconvergent(o.superconvergent);
    sys.setSteadyMode(o.nonlinear ? SystemSolver::SteadyMode::PseudoTransient
                                  : SystemSolver::SteadyMode::Newton);
    sys.setSteadyStateTolerance(1e-10);
    sys.setAdjointProblem(run->objective.get());
    sys.setSolveAdjoint(o.solveAdjoint);
    sys.setEstimateObjectiveOnFinish(o.estimate);
    return run;
}

std::unique_ptr<Run> solve(Options const &o, std::string const &stem)
{
    auto run = configure(o, stem);
    CapturedOutput quiet;
    run->sys->initialize();
    run->sys->integrate(0.0);
    return run;
}

std::vector<double> flatten(SystemSolver::AdjointSolution const &z)
{
    std::vector<double> out;
    for (Vector const &c : z.squ)
        out.insert(out.end(), c.data(), c.data() + c.size());
    out.insert(out.end(), z.lambda.data(), z.lambda.data() + z.lambda.size());
    out.insert(out.end(), z.field.data(), z.field.data() + z.field.size());
    return out;
}

bool bitwiseEqual(std::vector<double> const &a, std::vector<double> const &b)
{
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

} // namespace

BOOST_AUTO_TEST_SUITE(adjoint_on_demand_tests)

BOOST_AUTO_TEST_CASE(an_adjoint_solve_makes_no_physics_call)
{
    for (bool superconvergent : {false, true})
    {
        BOOST_TEST_CONTEXT("superconvergent = " << superconvergent)
        {
            auto run = solve({.superconvergent = superconvergent}, "adjoint_on_demand_calls");
            SystemSolver &sys = *run->sys;

            run->problem->reset();
            sys.solveAdjointAt(*run->objective, {0, 1}, sys.yJac);
            auto const *z = sys.adjointSolution(1, sys.yJac);
            BOOST_REQUIRE(z != nullptr);
            const auto w = sys.objectiveSensitivity(*z);
            const auto Gy = sys.assembleObjectiveGradient(*run->objective, 0, sys.yJac, sys.jt);
            BOOST_TEST(run->problem->batched == 0);
            BOOST_TEST(run->problem->calls == 0,
                       "the adjoint solve, its sensitivity and dG/dy asked the case for "
                       << run->problem->calls << " pointwise evaluations");

            // Not vacuous: z is a solution, not a zero vector.
            BOOST_TEST(z->operatorAtState);
            BOOST_TEST(Vector(w.sources.row(0)).cwiseAbs().maxCoeff() > 1e-3);
        }
    }
}

BOOST_AUTO_TEST_CASE(the_end_of_run_adjoint_reuses_the_estimate_build)
{
    // The objective estimate builds the Jacobian at the converged state, and that
    // is the state the end-of-run adjoint is defined at, so its derivatives are
    // reused and the adjoint costs no evaluation of the case -- for any number of
    // objectives. Without the estimate the last build was an earlier iterate, and
    // the adjoint evaluates the derivatives once, not once per objective.
    for (bool estimate : {true, false})
    {
        BOOST_TEST_CONTEXT("estimate = " << estimate)
        {
            auto run = configure({.estimate = estimate}, "adjoint_on_demand_end_of_run");
            SystemSolver &sys = *run->sys;
            {
                CapturedOutput quiet;
                sys.initialize();
                sys.integrate(0.0);
            }
            run->problem->reset();
            sys.setSolveAdjoint(true);
            {
                CapturedOutput quiet;
                sys.runAdjointSolve();
            }
            BOOST_TEST(run->problem->batched == (estimate ? 0 : 1));
        }
    }

    // And the estimate itself: one residual and one Jacobian build, whatever ng.
    auto run = solve({}, "adjoint_on_demand_estimate");
    run->problem->reset();
    const auto e = run->sys->estimateObjective();
    BOOST_TEST(e.valid);
    BOOST_TEST(e.value.size() == 2);
    BOOST_TEST(run->problem->batched == 2);
}

BOOST_AUTO_TEST_CASE(the_adjoint_on_demand_is_the_end_of_run_adjoint)
{
    // At the final state the two are the same operator from the same derivatives
    // and the same right-hand side, so they agree bit for bit.
    for (bool superconvergent : {false, true})
    {
        BOOST_TEST_CONTEXT("superconvergent = " << superconvergent)
        {
            auto run = solve({.superconvergent = superconvergent, .solveAdjoint = true},
                             "adjoint_on_demand_same");
            SystemSolver &sys = *run->sys;
            std::vector<std::vector<double>> endOfRun;
            for (Index g = 0; g < 2; ++g)
            {
                auto const *z = sys.adjointSolution(g, sys.yJac);
                BOOST_REQUIRE(z != nullptr);
                BOOST_TEST(z->operatorAtState);
                endOfRun.push_back(flatten(*z));
            }

            sys.solveAdjointAt(*run->objective, {0, 1}, sys.yJac);
            for (Index g = 0; g < 2; ++g)
                BOOST_TEST(bitwiseEqual(flatten(*sys.adjointSolution(g, sys.yJac)), endOfRun[g]));

            // The second objective is three times the first, and so is its z.
            const auto z0 = flatten(*sys.adjointSolution(0, sys.yJac));
            const auto z1 = flatten(*sys.adjointSolution(1, sys.yJac));
            double worst = 0.0, scale = 0.0;
            for (size_t i = 0; i < z0.size(); ++i)
            {
                worst = std::max(worst, std::abs(z1[i] - 3.0 * z0[i]));
                scale = std::max(scale, std::abs(z1[i]));
            }
            BOOST_TEST(worst <= 1e-13 * scale);
        }
    }
}

BOOST_AUTO_TEST_CASE(the_operator_of_an_earlier_iterate_says_so)
{
    // Without the estimate, the last build is the final Newton iteration's, at
    // the state before the last correction -- so the operator is not at the state
    // dG/dy is taken at, and the solution says so rather than passing for exact.
    auto run = solve({.estimate = false}, "adjoint_on_demand_lagged");
    SystemSolver &sys = *run->sys;
    sys.solveAdjointAt(*run->objective, {0}, sys.yJac);
    BOOST_TEST(!sys.adjointSolution(0, sys.yJac)->operatorAtState);

    // ...and an answer for one state is not offered for another.
    std::vector<double> moved(sys.yJac.data(), sys.yJac.data() + sys.yJac.getDoF());
    moved[0] += 1e-3;
    DGSoln elsewhere(1, *run->grid, 2, moved.data());
    BOOST_TEST(sys.adjointSolution(0, elsewhere) == nullptr);
}

BOOST_AUTO_TEST_CASE(a_solver_that_never_built_a_jacobian_refuses)
{
    auto run = configure({}, "adjoint_on_demand_unbuilt");
    BOOST_CHECK_THROW(run->sys->solveAdjointAt(*run->objective, {0}, run->sys->yJac),
                      std::logic_error);
}

BOOST_AUTO_TEST_CASE(the_sensitivity_matches_finite_differences_node_by_node)
{
    // w_m = dG/dGamma_m, against (G(Gamma_m + h) - G(Gamma_m - h)) / 2h with the
    // whole steady solve rerun. The problem is linear and G quadratic in the
    // state, so G is quadratic in h and the central difference is exact: what is
    // left is round-off, which is what the tolerance is set against.
    //
    // k = 2 at four cells, and z = 2 sin(pi x) / pi^2: the interpolatory rule on
    // three Chebyshev points is exact to degree three, and phi_m z is not a
    // polynomial at all, so the mass-matrix projection and diag(w) differ here --
    // the guard below measures by how much.
    for (bool superconvergent : {false, true})
    {
        BOOST_TEST_CONTEXT("superconvergent = " << superconvergent)
        {
            Options base{.nCells = 4, .k = 2, .superconvergent = superconvergent};
            auto run = solve(base, "adjoint_on_demand_fd");
            SystemSolver &sys = *run->sys;
            sys.solveAdjointAt(*run->objective, {0}, sys.yJac);
            auto const *z = sys.adjointSolution(0, sys.yJac);
            BOOST_REQUIRE(z != nullptr);
            BOOST_REQUIRE(z->operatorAtState);
            const auto w = sys.objectiveSensitivity(*z);
            const Index nPoints = static_cast<Index>(w.points.size());
            BOOST_REQUIRE(nPoints == base.nCells * (base.k + (superconvergent ? 2 : 1)));

            // The other candidate: the adjoint function's value at the node times
            // that node's quadrature weight. It is what diag(w) would give, and
            // it agrees with the projection on anything the rule integrates
            // exactly.
            const Index perCell = superconvergent ? base.k + 2 : base.k + 1;
            std::vector<double> zMem(sys.yJac.getDoF(), 0.0);
            DGSoln zh(1, *run->grid, base.k, zMem.data());
            for (Index i = 0; i < base.nCells; ++i)
                std::copy(z->squ[i].data(), z->squ[i].data() + z->squ[i].size(),
                          zMem.data() + i * z->squ[i].size());
            Matrix diagFlux(1, nPoints), diagSource(1, nPoints);
            for (Index i = 0; i < base.nCells; ++i)
            {
                const Vector weights =
                    superconvergent ? Vector(sys.postprocessor->starWeights(i))
                                    : sys.yJac.getBasis().getIntegrationWeights((*run->grid)[i]);
                for (Index j = 0; j < perCell; ++j)
                {
                    const Index m = i * perCell + j;
                    diagFlux(0, m) = -weights(j) * zh.sigma(0)(w.points[m]);
                    diagSource(0, m) = weights(j) * zh.u(0)(w.points[m]);
                }
            }

            const double G0 = run->G();
            const double h = 1e-3;
            double worstFlux = 0.0, worstSource = 0.0, guardFlux = 0.0, guardSource = 0.0;
            double scaleFlux = 0.0, scaleSource = 0.0;
            for (Index m = 0; m < nPoints; ++m)
            {
                for (int channel = 0; channel < 2; ++channel)
                {
                    double G[2];
                    for (int side = 0; side < 2; ++side)
                    {
                        Options o = base;
                        o.estimate = false;
                        o.perturbation.node = m;
                        (channel == 0 ? o.perturbation.flux : o.perturbation.source) =
                            side == 0 ? h : -h;
                        G[side] = solve(o, "adjoint_on_demand_fd")->G();
                    }
                    const double fd = (G[0] - G[1]) / (2.0 * h);
                    if (channel == 0)
                    {
                        worstFlux = std::max(worstFlux, std::abs(fd - w.flux(0, m)));
                        guardFlux = std::max(guardFlux, std::abs(fd - diagFlux(0, m)));
                        scaleFlux = std::max(scaleFlux, std::abs(fd));
                    }
                    else
                    {
                        worstSource = std::max(worstSource, std::abs(fd - w.sources(0, m)));
                        guardSource = std::max(guardSource, std::abs(fd - diagSource(0, m)));
                        scaleSource = std::max(scaleSource, std::abs(fd));
                    }
                }
            }
            BOOST_TEST_MESSAGE("G = " << G0 << "; flux: |w - fd| " << worstFlux
                               << ", |diag(w) - fd| " << guardFlux << " of " << scaleFlux
                               << "; sources: |w - fd| " << worstSource
                               << ", |diag(w) - fd| " << guardSource << " of " << scaleSource);

            BOOST_TEST(worstFlux <= 1e-9 * scaleFlux);
            BOOST_TEST(worstSource <= 1e-9 * scaleSource);
            // The guard: the finite difference can tell the two operators apart,
            // by orders of magnitude more than the tolerance above.
            BOOST_TEST(guardFlux >= 1e-5 * scaleFlux);
            BOOST_TEST(guardSource >= 1e-5 * scaleSource);
        }
    }
}

BOOST_AUTO_TEST_CASE(the_sensitivity_is_the_continuum_one)
{
    // The closed form behind the discrete one. With z = 2 sin(pi x) / pi^2,
    //
    //     dG/dS(x)       =  z(x)
    //     dG/dsigma_hat  = -z'(x)
    //
    // as densities: moving the source by eps l_m(x) -- the basis function of node
    // m, which is what moving one nodal value of the interpolant does -- moves G
    // by eps Int z l_m dx, and the flux by -eps Int z' l_m dx (integrate the
    // perturbed equation by parts; z vanishes at both ends). So w converges to
    // those integrals at the rate the discrete adjoint does. That rate is a
    // property of the discretisation rather than of w -- which the finite
    // differences above pin exactly -- and it is lower than the solution's:
    // measured from 8 to 16 cells, the per-node error falls at second order for
    // odd k without the superconvergent scheme and at third for even k (k = 1 to
    // 4: 2.1, 3.1, 2.0, 3.0 for the sources), and at 2.1, 3.0, 4.0, 3.0 with it.
    // k = 3 is the slowest of those, so it is the one held to second order here.
    for (bool superconvergent : {false, true})
    {
        BOOST_TEST_CONTEXT("superconvergent = " << superconvergent)
        {
            const Index k = 3;
            auto relativeError = [&](Index nCells, double &source, double &flux)
            {
                auto run = solve({.nCells = nCells, .k = k, .superconvergent = superconvergent},
                                 "adjoint_on_demand_closed_form");
                SystemSolver &sys = *run->sys;
                sys.solveAdjointAt(*run->objective, {0}, sys.yJac);
                const auto w = sys.objectiveSensitivity(*sys.adjointSolution(0, sys.yJac));

                const NodalBasis basis = NodalBasis::getBasis(superconvergent ? k + 1 : k);
                const Index perCell = superconvergent ? k + 2 : k + 1;
                boost::math::quadrature::gauss<double, 20> quad;
                double worstSource = 0.0, worstFlux = 0.0, scaleSource = 0.0, scaleFlux = 0.0;
                for (Index i = 0; i < nCells; ++i)
                {
                    Interval const &I = (*run->grid)[i];
                    for (Index j = 0; j < perCell; ++j)
                    {
                        const double exactSource = quad.integrate(
                            [&](double x)
                            { return 2.0 * std::sin(pi * x) / (pi * pi) * basis.Evaluate(I, j, x); },
                            I.x_l, I.x_u);
                        const double exactFlux = -quad.integrate(
                            [&](double x)
                            { return 2.0 * std::cos(pi * x) / pi * basis.Evaluate(I, j, x); },
                            I.x_l, I.x_u);
                        const Index m = i * perCell + j;
                        worstSource = std::max(worstSource, std::abs(w.sources(0, m) - exactSource));
                        worstFlux = std::max(worstFlux, std::abs(w.flux(0, m) - exactFlux));
                        scaleSource = std::max(scaleSource, std::abs(exactSource));
                        scaleFlux = std::max(scaleFlux, std::abs(exactFlux));
                    }
                }
                source = worstSource / scaleSource;
                flux = worstFlux / scaleFlux;
            };

            double source8, flux8, source16, flux16;
            relativeError(8, source8, flux8);
            relativeError(16, source16, flux16);
            BOOST_TEST_MESSAGE("against the continuum at 8 and 16 cells: sources "
                               << source8 << " -> " << source16 << " (order "
                               << std::log2(source8 / source16) << "), flux " << flux8 << " -> "
                               << flux16 << " (order " << std::log2(flux8 / flux16) << ")");
            BOOST_TEST(source8 < 1e-3);
            BOOST_TEST(flux8 < 1e-3);
            BOOST_TEST(std::log2(source8 / source16) > 1.8);
            BOOST_TEST(std::log2(flux8 / flux16) > 1.8);
        }
    }
}

BOOST_AUTO_TEST_CASE(an_adjoint_solve_between_slices_leaves_the_forward_solve_alone)
{
    // Two sliced steady solves of a nonlinear case, identical but for an adjoint
    // solve between every slice in one of them. They end at the same state bit
    // for bit, after the same number of steps.
    auto sliced = [](bool withAdjoint, SystemSolver::SteadyTest test)
    {
        auto run = configure({.nonlinear = true, .estimate = false}, "adjoint_on_demand_sliced");
        SystemSolver &sys = *run->sys;
        sys.setSteadyTest(test);
        sys.setPseudoTransientInitialStep(1e-3);
        sys.setMaxContinuationSteps(2);
        int slices = 0;
        {
            CapturedOutput quiet;
            sys.initialize();
            for (bool resume = false;; resume = true)
            {
                try
                {
                    resume ? sys.continueSteadyState() : sys.solveSteadyState();
                }
                catch (...)
                {
                    if (sys.lastSteadyOutcome() != SystemSolver::SteadyOutcome::OutOfSteps)
                        throw;
                }
                ++slices;
                sys.captureState();
                if (sys.lastSteadyOutcome() == SystemSolver::SteadyOutcome::Converged)
                    break;
                if (withAdjoint)
                    sys.solveAdjointAt(*run->objective, {0, 1}, sys.yJac);
                BOOST_REQUIRE(slices < 100);
            }
        }
        std::vector<double> state(sys.yJac.data(), sys.yJac.data() + sys.yJac.getDoF());
        return std::make_pair(state, slices);
    };

    for (auto test : {SystemSolver::SteadyTest::NewtonStep, SystemSolver::SteadyTest::Residual})
    {
        const auto [plain, plainSlices] = sliced(false, test);
        const auto [interleaved, interleavedSlices] = sliced(true, test);
        BOOST_TEST_MESSAGE("converged in " << plainSlices << " slices");
        BOOST_TEST_REQUIRE(plainSlices > 2, "the solve has to be sliced for this to say anything");
        BOOST_TEST(interleavedSlices == plainSlices);
        BOOST_TEST(bitwiseEqual(interleaved, plain));
    }
}

BOOST_AUTO_TEST_SUITE_END()
