// The dual-weighted estimate of an objective's discretisation error, against
// closed forms.
//
// Every problem here is steady, on [0, 1], with u = sin(pi x) exactly and u = 0
// at both ends, and every objective is an integral of a known function of u and
// x -- so G(u) is known to round-off, and the error the estimate is checked
// against is G_h - G with nothing estimated on that side. Two quadratures agreeing
// is not evidence that either is right; a closed form is.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "AdjointProblem.hpp"
#include "SystemSolver.hpp"
#include "TransportSystem.hpp"

#include <boost/math/quadrature/gauss.hpp>

#include <chrono>
#include <cmath>
#include <format>
#include <functional>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

namespace
{
using std::numbers::pi;

enum class Physics
{
    Linear,     // sigma_hat = kappa q, kappa = 0.7
    Nonlinear,  // sigma_hat = (1 + u^2) q
    Reaction,   // sigma_hat = q, S = f(x) - u^3
    Variable,   // sigma_hat = (1 + x) q
    Aux,        // sigma_hat = (1 + phi) q with the aux constraint phi = u^2, and a
                // source reading phi: the nonlinear flux, through an aux variable
};

std::string name(Physics p)
{
    switch (p)
    {
    case Physics::Linear: return "linear";
    case Physics::Nonlinear: return "nonlinear flux";
    case Physics::Reaction: return "reaction";
    case Physics::Variable: return "variable kappa";
    case Physics::Aux: return "aux variable";
    }
    return "?";
}

constexpr double KAPPA = 0.7;

// -d_x sigma_hat = S with u = sin(pi x). The stored sigma is -sigma_hat, so the
// equation integrated is -d_x[ sigma_hat ] = S, and S is written for that sign.
class Manufactured : public TransportSystem
{
public:
    explicit Manufactured(Physics p)
        : TransportSystem({.variables = numberedFields(1),
                           .aux = p == Physics::Aux ? numberedAux(1) : std::vector<AuxSpec>{}}),
          p(p)
    {
    }
    Physics p;
    long values = 0, derivatives = 0;

    PhysicsOutput ComputePhysics(GlobalState const &s, std::vector<Position> const &x,
                                 Time t) override
    {
        ++values;
        return TransportSystem::ComputePhysics(s, x, t);
    }
    void ComputePhysicsDerivatives(std::array<std::reference_wrapper<GlobalStateMatrix>,
                                              NPHYSICS_FUNCTIONS> &&out,
                                   GlobalState const &s, std::vector<Position> const &x,
                                   Time t) override
    {
        ++derivatives;
        TransportSystem::ComputePhysicsDerivatives(std::move(out), s, x, t);
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index, const State &st, Position x, Time) override
    {
        const double u = st.u(0), q = st.q(0);
        switch (p)
        {
        case Physics::Linear: return KAPPA * q;
        case Physics::Nonlinear: return (1.0 + u * u) * q;
        case Physics::Reaction: return q;
        case Physics::Variable: return (1.0 + x) * q;
        case Physics::Aux: return (1.0 + st.phi(0)) * q;
        }
        return 0.0;
    }
    Value Sources(Index, const State &st, Position x, Time) override
    {
        const double s = std::sin(pi * x), c = std::cos(pi * x);
        switch (p)
        {
        case Physics::Linear: return KAPPA * pi * pi * s;
        case Physics::Nonlinear: return pi * pi * s * (1.0 + s * s) - 2.0 * pi * pi * s * c * c;
        case Physics::Reaction:
        {
            const double u = st.u(0);
            return pi * pi * s + s * s * s - u * u * u;
        }
        case Physics::Variable: return -pi * c + (1.0 + x) * pi * pi * s;
        case Physics::Aux:
            return pi * pi * s * (1.0 + s * s) - 2.0 * pi * pi * s * c * c + s * s * s * s -
                   st.phi(0) * st.phi(0);
        }
        return 0.0;
    }
    Value AuxG(Index, const State &st, Position, Time) override
    {
        return st.phi(0) - st.u(0) * st.u(0);
    }
    void AuxGPrime(Index, State &out, const State &st, Position, Time) override
    {
        out.phi(0) = 1.0;
        out.u(0) = -2.0 * st.u(0);
    }
    void dSigma_dPhi(Index, VectorRef v, const State &st, Position, Time) override
    {
        v[0] = p == Physics::Aux ? st.q(0) : 0.0;
    }
    void dSources_dPhi(Index, VectorRef v, const State &st, Position, Time) override
    {
        v[0] = p == Physics::Aux ? -2.0 * st.phi(0) : 0.0;
    }
    Value InitialAuxValue(Index, Position x) const override
    {
        const double s = std::sin(pi * x);
        return s * s;
    }
    void dSigmaFn_dq(Index, VectorRef v, const State &st, Position x, Time) override
    {
        const double u = st.u(0);
        switch (p)
        {
        case Physics::Linear: v[0] = KAPPA; break;
        case Physics::Nonlinear: v[0] = 1.0 + u * u; break;
        case Physics::Reaction: v[0] = 1.0; break;
        case Physics::Variable: v[0] = 1.0 + x; break;
        case Physics::Aux: v[0] = 1.0 + st.phi(0); break;
        }
    }
    void dSigmaFn_du(Index, VectorRef v, const State &st, Position, Time) override
    {
        v[0] = p == Physics::Nonlinear ? 2.0 * st.u(0) * st.q(0) : 0.0;
    }
    void dSources_du(Index, VectorRef v, const State &st, Position, Time) override
    {
        v[0] = p == Physics::Reaction ? -3.0 * st.u(0) * st.u(0) : 0.0;
    }
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}

    Value InitialValue(Index, Position x) const override { return std::sin(pi * x); }
    Value InitialDerivative(Index, Position x) const override { return pi * std::cos(pi * x); }
};

enum class Functional
{
    Mean,      // int u dx            = 2 / pi
    Weighted,  // int x u dx          = 1 / pi
    Square,    // int u^2 dx          = 1 / 2
    Flux,      // int x q dx          = int x pi cos(pi x) dx = -2 / pi
};

std::string name(Functional f)
{
    switch (f)
    {
    case Functional::Mean: return "int u";
    case Functional::Weighted: return "int x u";
    case Functional::Square: return "int u^2";
    case Functional::Flux: return "int x q";
    }
    return "?";
}

double exactG(Functional f)
{
    switch (f)
    {
    case Functional::Mean: return 2.0 / pi;
    case Functional::Weighted: return 1.0 / pi;
    case Functional::Square: return 0.5;
    case Functional::Flux: return -2.0 / pi;
    }
    return 0.0;
}

class Objective : public AdjointProblem
{
public:
    explicit Objective(Functional f) : f(f)
    {
        ng = 1;
        np = 1;
    }
    Functional f;

    double g(double u, double q, double x) const
    {
        switch (f)
        {
        case Functional::Mean: return u;
        case Functional::Weighted: return x * u;
        case Functional::Square: return u * u;
        case Functional::Flux: return x * q;
        }
        return 0.0;
    }

    using AdjointProblem::GFn;
    Value GFn(Index, DGSoln &y) const override
    {
        const GlobalState s = y.evalOnNodes();
        const std::vector<Position> x = y.getPoints();
        const Index n = y.getBasis().getNodes().size();
        Value G = 0.0;
        for (Index i = 0; i < static_cast<Index>(y.getGrid().getNCells()); ++i)
        {
            const Vector w = y.getBasis().getIntegrationWeights(y.getGrid()[i]);
            for (Index j = 0; j < n; ++j)
                G += w(j) * g(s.Variable()(0, i * n + j), s.Derivative()(0, i * n + j), x[i * n + j]);
        }
        return G;
    }
    Value dGFndp(Index, Index, DGSoln &) const override { return 0.0; }
    Value gFn(Index, const State &s, Position x) const override { return g(s.u(0), s.q(0), x); }
    void dgFn_du(Index, VectorRef v, const State &s, Position x) override
    {
        switch (f)
        {
        case Functional::Mean: v[0] = 1.0; break;
        case Functional::Weighted: v[0] = x; break;
        case Functional::Square: v[0] = 2.0 * s.u(0); break;
        case Functional::Flux: v[0] = 0.0; break;
        }
    }
    void dgFn_dq(Index, VectorRef v, const State &, Position x) override
    {
        v[0] = f == Functional::Flux ? x : 0.0;
    }
    void dgFn_dsigma(Index, VectorRef, const State &, Position) override {}
    void dgFn_dphi(Index, VectorRef, const State &, Position) override {}
    void dSigmaFn_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }
    void dSources_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }
};

struct Setup
{
    Physics physics = Physics::Linear;
    Functional functional = Functional::Mean;
    Index nCells = 4;
    Index k = 2;
    bool superconvergent = false;
};

// A steady solve run to round-off, with the SUNDIALS objects left alive so the
// estimate can be taken at the state it stopped in.
struct Run
{
    std::unique_ptr<Grid> grid;
    std::unique_ptr<Manufactured> problem;
    std::unique_ptr<Objective> objective;
    std::unique_ptr<SystemSolver> sys;

    ~Run()
    {
        if (sys)
        {
            CapturedOutput quiet;
            sys->destroySundials();
        }
    }

    double G()
    {
        if (sys->isSuperconvergent())
        {
            sys->postprocessor->computeUStar(sys->yJac);
            return objective->GFn(0, sys->yJac, *sys->postprocessor);
        }
        return objective->GFn(0, sys->yJac);
    }
};

// `estimate`: the objective estimate on the way out; `discretisation`: whether
// it, and the solver, estimate the discretisation error.
std::unique_ptr<Run> solve(Setup const &s, bool estimate = true, bool discretisation = true)
{
    auto run = std::make_unique<Run>();
    run->grid = std::make_unique<Grid>(0.0, 1.0, s.nCells);
    run->problem = std::make_unique<Manufactured>(s.physics);
    run->objective = std::make_unique<Objective>(s.functional);
    run->sys = std::make_unique<SystemSolver>(*run->grid, s.k, run->problem.get());
    SystemSolver &sys = *run->sys;
    sys.setTau(1.0);
    sys.setInputFile("discretisation_estimate");
    sys.setWriteOutput(false);
    sys.setWriteDatFile(false);
    sys.setOutputCadence(1.0);
    sys.setNOutput(2);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-14);
    sys.setTolerances({1e-12}, 1e-10);
    sys.setSuperconvergent(s.superconvergent);
    sys.setSteadyMode(SystemSolver::SteadyMode::Newton);
    sys.setSteadyStateTolerance(1e-11);
    sys.setAdjointProblem(run->objective.get());
    sys.setEstimateObjectiveOnFinish(estimate);
    sys.setEstimateDiscretisationError(discretisation);
    CapturedOutput quiet;
    sys.initialize();
    sys.integrate(0.0);
    return run;
}

// Every candidate at the state a run stopped in.
struct Candidates
{
    double error;  // G_h - G, against the closed form
    double G;
    std::vector<SystemSolver::DiscretisationEstimate> estimates;
};

using Which = SystemSolver::DiscretisationAdjoint;
struct Variant
{
    Which which;
    unsigned int increment;
    std::string label;
};

Candidates measure(Run &run, std::vector<Variant> const &variants)
{
    // At alpha = 0 and zero dydt, as estimateObjective builds it.
    SystemSolver &sys = *run.sys;
    N_Vector zero = N_VClone(sys.Y), F = N_VClone(sys.Y), delta = N_VClone(sys.Y);
    N_VConst(0.0, zero);
    sys.setAlpha(0.0);
    sys.setJacEvalY(sys.Y, zero);
    sys.updateMatricesForJacSolve();
    sys.residual(sys.t0, sys.Y, zero, F);
    sys.solveJacEq(F, delta);
    DGSoln deltaH(1, *run.grid, sys.k, N_VGetArrayPointer(delta), 0, sys.nAux, 0);

    Candidates out;
    out.G = run.G();
    out.error = out.G - exactG(run.objective->f);
    for (Variant const &v : variants)
    {
        sys.discretisationAdjoint = v.which;
        sys.discretisationIncrement = v.increment;
        out.estimates.push_back(
            sys.estimateDiscretisationError(*run.objective, sys.y, deltaH, sys.t0));
    }
    sys.discretisationAdjoint = Which::Enriched;
    sys.discretisationIncrement = 0;
    N_VDestroy(zero);
    N_VDestroy(F);
    N_VDestroy(delta);
    return out;
}

} // namespace

BOOST_AUTO_TEST_SUITE(discretisation_estimate_tests)

BOOST_AUTO_TEST_CASE(the_estimate_tracks_the_error_against_closed_forms)
{
    // The estimate the steady solve makes on its way out, against G_h - G with G
    // known: four objectives, three physics, both schemes, k = 1 to 4 on eight
    // cells. measure_effectivity below has the whole table; this is the band its
    // N = 8 rows fall in (0.97 to 1.15), with room.
    for (bool sc : {false, true})
        for (Physics p : {Physics::Linear, Physics::Nonlinear, Physics::Reaction})
            for (Functional f :
                 {Functional::Mean, Functional::Weighted, Functional::Square, Functional::Flux})
                for (Index k = 1; k <= 4; ++k)
                    BOOST_TEST_CONTEXT(name(p) << " / " << name(f) << " / k = " << k
                                               << " / superconvergent = " << sc)
                    {
                        auto run = solve({p, f, 8, k, sc});
                        const auto e = run->sys->lastObjectiveEstimate();
                        BOOST_TEST_REQUIRE(e.discretisation.valid);
                        const double error = e.corrected(0) - exactG(f);
                        const double effectivity = e.discretisation.total(0) / error;
                        BOOST_TEST(effectivity > 0.9);
                        BOOST_TEST(effectivity < 1.2);
                    }
}

BOOST_AUTO_TEST_CASE(it_reaches_the_aux_rows)
{
    // The same flux routed through an auxiliary variable: the enriched residual's
    // aux rows and the enriched adjoint's aux blocks are on the path. Coarser
    // meshes are not asymptotic here -- the error changes sign between 4 and 8
    // cells at low k -- so this is the 16-cell band, 0.94 to 1.08 measured.
    for (bool sc : {false, true})
        for (Functional f : {Functional::Mean, Functional::Square, Functional::Flux})
            for (Index k = 2; k <= 4; ++k)
                BOOST_TEST_CONTEXT(name(f) << " / k = " << k << " / superconvergent = " << sc)
                {
                    auto run = solve({Physics::Aux, f, 16, k, sc});
                    const auto e = run->sys->lastObjectiveEstimate();
                    const double effectivity =
                        e.discretisation.total(0) / (e.corrected(0) - exactG(f));
                    BOOST_TEST(effectivity > 0.9);
                    BOOST_TEST(effectivity < 1.2);
                }
}

BOOST_AUTO_TEST_CASE(on_a_linear_problem_and_a_linear_objective_it_is_all_but_exact)
{
    // The dual-weighted representation is exact here up to how well the enriched
    // discretisation stands in for the continuum, which two degrees up is very
    // well: every N = 8 row of the table is within 1e-3 of 1.
    for (bool sc : {false, true})
        for (Functional f : {Functional::Mean, Functional::Weighted, Functional::Flux})
            for (Index k = 1; k <= 4; ++k)
                BOOST_TEST_CONTEXT(name(f) << " / k = " << k << " / superconvergent = " << sc)
                {
                    auto run = solve({Physics::Linear, f, 8, k, sc});
                    const auto e = run->sys->lastObjectiveEstimate();
                    const double effectivity =
                        e.discretisation.total(0) / (e.corrected(0) - exactG(f));
                    BOOST_TEST(std::abs(effectivity - 1.0) < 2e-3);
                }
}

BOOST_AUTO_TEST_CASE(the_per_cell_indicators_sum_to_the_total)
{
    for (bool sc : {false, true})
        for (Physics p : {Physics::Linear, Physics::Nonlinear})
            BOOST_TEST_CONTEXT(name(p) << " / superconvergent = " << sc)
            {
                auto run = solve({p, Functional::Square, 6, 2, sc});
                auto const &d = run->sys->lastObjectiveEstimate().discretisation;
                BOOST_TEST_REQUIRE(d.valid);
                BOOST_TEST(d.perCell.cols() == 6);
                const double scale = d.perCell.cwiseAbs().sum();
                BOOST_TEST(std::abs(d.perCell.row(0).sum() - d.total(0)) < 1e-13 * scale);
                BOOST_TEST(std::abs(d.residual(0) + d.quadrature(0) - d.algebraic(0) -
                                    d.total(0)) < 1e-13 * scale);
                // Not a trivial split: the cells disagree, by sign as well as size
                // for this objective, whose weight 2u changes across the domain.
                BOOST_TEST(d.perCell.cwiseAbs().maxCoeff() > 0.05 * std::abs(d.total(0)));
            }
}

BOOST_AUTO_TEST_CASE(the_estimate_makes_exactly_its_two_physics_calls)
{
    // One ComputePhysics and one ComputePhysicsDerivatives at the enriched nodes,
    // and nothing else of the case's: everything else is the objective's own, or
    // assembled from those two. The postprocessed comparison saves the second.
    for (bool sc : {false, true})
        BOOST_TEST_CONTEXT("superconvergent = " << sc)
        {
            auto run = solve({Physics::Nonlinear, Functional::Square, 4, 2, sc}, false);
            const std::vector<Variant> variants = {{Which::Enriched, 0, ""}};
            run->problem->values = run->problem->derivatives = 0;
            SystemSolver &sys = *run->sys;
            // measure() builds the Jacobian and the residual first, as
            // estimateObjective does: one of each.
            measure(*run, variants);
            BOOST_TEST(run->problem->values == 1 + 1);
            BOOST_TEST(run->problem->derivatives == 1 + 1);

            run->problem->values = run->problem->derivatives = 0;
            measure(*run, {{Which::Postprocessed, 0, ""}});
            BOOST_TEST(run->problem->values == 1 + 1);
            BOOST_TEST(run->problem->derivatives == 1 + 0);
            BOOST_TEST(sys.discretisationIncrement == 0u);
        }
}

BOOST_AUTO_TEST_CASE(estimating_moves_no_other_answer)
{
    // The value, the algebraic correction and its bound are bit for bit those of
    // a run that does not estimate the discretisation error.
    for (bool sc : {false, true})
        BOOST_TEST_CONTEXT("superconvergent = " << sc)
        {
            auto with = solve({Physics::Nonlinear, Functional::Weighted, 4, 3, sc});
            auto without = solve({Physics::Nonlinear, Functional::Weighted, 4, 3, sc}, true, false);
            const auto a = with->sys->lastObjectiveEstimate();
            const auto b = without->sys->lastObjectiveEstimate();
            BOOST_TEST(a.discretisation.valid);
            BOOST_TEST(!b.discretisation.valid);
            BOOST_TEST(a.value(0) == b.value(0));
            BOOST_TEST(a.corrected(0) == b.corrected(0));
            BOOST_TEST(a.uncertainty(0) == b.uncertainty(0));
        }
}

BOOST_AUTO_TEST_CASE(the_estimate_refuses_a_solver_not_configured_for_it)
{
    // Its evaluations are announced by the plan only on a solver that asked, so
    // one that did not is refused rather than handed unannounced batches.
    auto run = solve({Physics::Linear, Functional::Mean, 4, 2, false}, true, false);
    SystemSolver &sys = *run->sys;
    N_Vector zero = N_VClone(sys.Y);
    N_VConst(0.0, zero);
    DGSoln z(1, *run->grid, 2, N_VGetArrayPointer(zero), 0, 0, 0);
    BOOST_CHECK_THROW(sys.estimateDiscretisationError(*run->objective, sys.y, z, sys.t0),
                      std::logic_error);
    N_VDestroy(zero);

    // And a level whose enriched batch is past MaxPhysicsBatch is refused before
    // its first evaluation: 4 cells at k = 2 are 12 points per residual and 20 at
    // the enriched nodes.
    Grid grid(0.0, 1.0, 4);
    Manufactured problem(Physics::Linear);
    Objective objective(Functional::Mean);
    SystemSolver capped(grid, 2, &problem);
    capped.setTau(1.0);
    capped.setInputFile("discretisation_estimate_capped");
    capped.setWriteOutput(false);
    capped.setSteadyMode(SystemSolver::SteadyMode::Newton);
    capped.setSteadyStateTolerance(1e-11);
    capped.setAdjointProblem(&objective);
    capped.setEstimateDiscretisationError(true);
    capped.setMaxPhysicsBatch(16);
    CapturedOutput quiet;
    BOOST_CHECK_THROW(capped.initialize(), std::invalid_argument);
    capped.destroySundials();
}

// The table docs/adjoints.rst quotes. Disabled by default, since it is a
// measurement rather than a check and takes a while; run it with
//
//     UnitTests --run_test=discretisation_estimate_tests/measure_effectivity --log_level=message
BOOST_AUTO_TEST_CASE(measure_effectivity, *boost::unit_test::disabled())
{
    for (bool sc : {false, true})
    {
        const std::vector<Variant> variants =
            sc ? std::vector<Variant>{{Which::Enriched, 2, "en+2"},
                                      {Which::Enriched, 3, "en+3"},
                                      {Which::Enriched, 4, "en+4"},
                                      {Which::Postprocessed, 3, "pp+3"}}
               : std::vector<Variant>{{Which::Enriched, 1, "en+1"},
                                      {Which::Enriched, 2, "en+2"},
                                      {Which::Enriched, 3, "en+3"},
                                      {Which::Postprocessed, 2, "pp+2"}};
        for (Physics p : {Physics::Linear, Physics::Nonlinear, Physics::Reaction, Physics::Variable,
                          Physics::Aux})
            for (Functional f : {Functional::Mean, Functional::Weighted, Functional::Square, Functional::Flux})
            {
                BOOST_TEST_MESSAGE(std::format("\n{} / {} / superconvergent = {}", name(p), name(f), sc));
                std::string head = std::format("{:>3} {:>4} {:>11}", "k", "N", "G_h - G");
                for (auto const &v : variants)
                    head += std::format(" {:>8}", v.label);
                BOOST_TEST_MESSAGE(head);
                for (Index k = 1; k <= 5; ++k)
                    for (Index n : {2, 4, 8, 16})
                    {
                        auto run = solve({p, f, n, k, sc}, false);
                        const Candidates c = measure(*run, variants);
                        std::string line = std::format("{:>3} {:>4} {:>11.3e}", k, n, c.error);
                        for (auto const &e : c.estimates)
                            line += std::format(" {:>8.4f}", e.total(0) / c.error);
                        BOOST_TEST_MESSAGE(line);
                    }
            }
    }
}

// What an estimate costs against what a Newton iteration costs -- one residual
// and one Jacobian build -- on a case whose physics is cheap, so that what is
// timed is MaNTA's own work. Best of five. Disabled for the same reason as the
// above.
BOOST_AUTO_TEST_CASE(measure_cost, *boost::unit_test::disabled())
{
    using clock = std::chrono::steady_clock;
    auto best = [](auto &&f)
    {
        double t = 1e300;
        for (int rep = 0; rep < 5; ++rep)
        {
            const auto t0 = clock::now();
            f();
            t = std::min(t, std::chrono::duration<double, std::milli>(clock::now() - t0).count());
        }
        return t;
    };
    BOOST_TEST_MESSAGE(std::format("{:>5} {:>3} {:>4} {:>10} {:>10} {:>10} {:>10}", "sc", "k", "N",
                                   "residual", "jacobian", "estimate", "ratio"));
    for (bool sc : {false, true})
        for (auto [k, n] : std::vector<std::pair<Index, Index>>{{2, 64}, {3, 32}, {5, 16}, {8, 8}})
        {
            auto run = solve({Physics::Nonlinear, Functional::Square, n, k, sc}, false);
            SystemSolver &sys = *run->sys;
            N_Vector zero = N_VClone(sys.Y), F = N_VClone(sys.Y), delta = N_VClone(sys.Y);
            N_VConst(0.0, zero);
            sys.setAlpha(0.0);
            sys.setJacEvalY(sys.Y, zero);
            const double tRes = best([&] { sys.residual(sys.t0, sys.Y, zero, F); });
            const double tJac = best([&] { sys.updateMatricesForJacSolve(); });
            sys.solveJacEq(F, delta);
            DGSoln deltaH(1, *run->grid, k, N_VGetArrayPointer(delta), 0, sys.nAux, 0);
            const double tEst = best(
                [&] { sys.estimateDiscretisationError(*run->objective, sys.y, deltaH, sys.t0); });
            BOOST_TEST_MESSAGE(std::format("{:>5} {:>3} {:>4} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.2f}",
                                           sc, k, n, tRes, tJac, tEst, tEst / (tRes + tJac)));
            N_VDestroy(zero);
            N_VDestroy(F);
            N_VDestroy(delta);
        }
}

BOOST_AUTO_TEST_SUITE_END()
