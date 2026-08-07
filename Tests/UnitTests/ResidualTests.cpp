// Tests for SystemSolver::residual and the two helpers that hang off it.
//
// residual() is what IDA sees; everything else in the solver exists to drive
// Newton towards making it zero. It had no direct coverage: the regression
// suite exercises it end to end, but nothing pinned *what it computes*, so a
// block written into the wrong slice of the residual vector would show up only
// as a converged-but-wrong answer.
//
// The strongest available check is a consistency one. setInitialConditions
// constructs sigma, q, u, lambda and dudt so that the residual is satisfied at
// t = 0; with an initial profile the basis represents exactly (a polynomial of
// degree <= k, with the derivative supplied consistently) there is no
// projection error anywhere and the residual must come out at round-off. That
// single assertion couples setInitialConditions, the A/B/C/D/E/H block
// assembly, and every branch of residual(). A perturbation case follows it, so
// that "the residual is zero" cannot be satisfied by a residual that is blind
// to part of its input.

#include <boost/test/unit_test.hpp>

#include "../../PhysicsCases/ScalarTestLD3.hpp"
#include "CapturedOutput.hpp"
#include "SystemSolver.hpp"
#include "TestDiffusion.hpp"
#include "Types.hpp"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <toml.hpp>

// Defined in SystemSolver.cpp with external linkage but no declaration in any
// header -- it is only ever handed to IDA as a function pointer. Declared here
// so the exception path can be driven directly.
int static_residual(sunrealtype tres, N_Vector Y, N_Vector dYdt, N_Vector resval,
                    void *user_data);

using namespace toml::literals::toml_literals;

namespace
{

const toml::value diffusion_config = u8R"(
    [DiffusionProblem]
    Kappa = 1.0
    Centre = 0.0
)"_toml;

const toml::value scalar_config = u8R"(
    [DiffusionProblem]
    Kappa = 1.0
    alpha = 0.2
    beta = 1.0
    gamma = 1.0
    u0 = 0.1
)"_toml;

// Linear diffusion whose initial condition is a cubic -- exactly representable
// for k >= 3, so the projection in setInitialConditions is lossless and the
// residual it produces should be zero to round-off rather than to O(h^{k+1}).
//
// u(x) = x(1-x)(1+x) vanishes at both endpoints, matching the Dirichlet data.
class PolynomialDiffusion : public TransportSystem
{
public:
    explicit PolynomialDiffusion(double kappa_ = 1.3) : kappa(kappa_) { nVars = 1; }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
    bool isLowerBoundaryDirichlet(Index) const override { return true; }
    bool isUpperBoundaryDirichlet(Index) const override { return true; }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return kappa * s.Derivative[0];
    }
    Value Sources(Index, const State &, Position x, Time) override
    {
        // Any smooth source is fine; make it a polynomial too so the
        // interpolation onto the basis is also exact.
        return 0.5 + 2.0 * x - x * x;
    }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = kappa;
    }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }

    // u = x - x^3, so q = du/dx = 1 - 3x^2. These have to agree exactly:
    // the q block of the residual *is* the statement that q is the derivative
    // of u, so an inconsistent pair leaves a residual of order one.
    Value InitialValue(Index, Position x) const override { return x - x * x * x; }
    Value InitialDerivative(Index, Position x) const override
    {
        return 1.0 - 3.0 * x * x;
    }

    double kappa;
};

// Throws on the first flux evaluation. static_residual must convert that into
// IDA's "recoverable error" return code rather than letting it escape into C.
class ThrowingDiffusion : public TestDiffusion
{
public:
    using TestDiffusion::TestDiffusion;

    Value SigmaFn(Index, const State &, Position, Time) override
    {
        throw std::runtime_error("deliberate physics failure");
    }
};

// An aux-carrying system with a linear constraint, so the aux rows of the
// residual can be predicted exactly: G = a - c*u, projected onto the basis.
class AuxResidualMock : public TransportSystem
{
public:
    AuxResidualMock()
    {
        nVars = 1;
        nAux = 1;
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
    bool isLowerBoundaryDirichlet(Index) const override { return true; }
    bool isUpperBoundaryDirichlet(Index) const override { return true; }

    Value SigmaFn(Index, const State &s, Position, Time) override { return s.Derivative[0]; }
    Value Sources(Index, const State &s, Position, Time) override { return s.Aux[0]; }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 1.0; }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dPhi(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 1.0;
    }
    void dSigma_dPhi(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }

    Value AuxG(Index, const State &s, Position, Time) override
    {
        return s.Aux[0] - auxCoeff * s.Variable[0];
    }
    void AuxGPrime(Index, State &out, const State &, Position, Time) override
    {
        out.zero();
        out.Variable[0] = -auxCoeff;
        out.Aux[0] = 1.0;
    }

    Value InitialValue(Index, Position x) const override { return x * (1.0 - x); }
    Value InitialDerivative(Index, Position x) const override { return 1.0 - 2.0 * x; }
    // Deliberately NOT the value that satisfies G = 0, so the aux rows of the
    // residual are nonzero and can be checked against a hand computation.
    Value InitialAuxValue(Index, Position x) const override { return 0.4 + 0.1 * x; }

    double auxCoeff = 2.5;
};

} // namespace

BOOST_AUTO_TEST_SUITE(residual_tests)

BOOST_AUTO_TEST_CASE(residual_vanishes_on_an_exactly_representable_initial_state)
{
    // k = 3 represents the cubic initial condition exactly, so there is no
    // projection error anywhere and every row of the residual must be at
    // round-off. The companion case below perturbs the state to show this is
    // not simply a residual that is always zero.
    const Index k = 3, nCells = 5;
    Grid grid(0.0, 1.0, nCells);
    PolynomialDiffusion problem;

    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(problem.getNumVars(), grid, k);
    const Index n = shape.getDoF();

    N_Vector Y = N_VNew_Serial(n, ctx), dYdt = N_VClone(Y), res = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    BOOST_TEST(sys.residual(0.0, Y, dYdt, res) == 0);

    const double *r = N_VGetArrayPointer(res);
    double worst = 0.0;
    Index worstRow = 0;
    for (Index i = 0; i < n; ++i)
        if (std::abs(r[i]) > worst)
        {
            worst = std::abs(r[i]);
            worstRow = i;
        }

    BOOST_TEST(worst < 1e-11, "largest residual entry " << worst << " at row " << worstRow);

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    N_VDestroy(res);
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_CASE(residual_responds_to_a_perturbation_of_every_block)
{
    // The case above asserts the residual is zero. Taken alone that would also
    // be satisfied by a residual that is *always* zero, or by one that only
    // looks at some of its inputs -- so perturb each block of the state
    // separately and require the residual to notice each time.
    //
    // (A first attempt at this used k = 2, expecting the cubic to become
    // inexact and the residual to grow to O(h^3). It does not: at k = 2 the
    // residual still comes out at 2e-15, because q = 1 - 3x^2 is itself
    // representable and the u-projection error happens not to survive the
    // weak-derivative pairing. Perturbing the state is the check that actually
    // discriminates.)
    const Index k = 3, nCells = 4;
    Grid grid(0.0, 1.0, nCells);
    PolynomialDiffusion problem;

    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(problem.getNumVars(), grid, k);
    const Index n = shape.getDoF();

    N_Vector Y = N_VNew_Serial(n, ctx), dYdt = N_VClone(Y), res = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    std::vector<double> baseline(N_VGetArrayPointer(Y), N_VGetArrayPointer(Y) + n);
    double *ya = N_VGetArrayPointer(Y);

    // Blocks in DGSoln's layout: per cell [sigma | q | u], then all of lambda.
    const Index perCell = 3 * (k + 1);
    struct Block
    {
        const char *name;
        Index row;
    };
    const Block blocks[] = {
        {"sigma", 0},
        {"q", k + 1},
        {"u", 2 * (k + 1)},
        {"lambda", nCells * perCell + 1}, // an interior face, not a Dirichlet one
    };

    for (auto const &b : blocks)
    {
        std::copy(baseline.begin(), baseline.end(), ya);
        ya[b.row] += 0.1;
        sys.residual(0.0, Y, dYdt, res);
        BOOST_TEST(N_VMaxNorm(res) > 1e-4,
                   "perturbing the " << b.name << " block left the residual at "
                                     << N_VMaxNorm(res));
    }

    // Restoring the state restores the zero residual -- so the growth above is
    // the perturbation, not accumulated state inside the solver.
    std::copy(baseline.begin(), baseline.end(), ya);
    sys.residual(0.0, Y, dYdt, res);
    BOOST_TEST(N_VMaxNorm(res) < 1e-11);

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    N_VDestroy(res);
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_CASE(residual_is_affine_for_a_linear_problem)
{
    // TestDiffusion is linear in (sigma, q, u, lambda), so the residual is
    // affine and must satisfy F(a Y1 + (1-a) Y2) = a F(Y1) + (1-a) F(Y2).
    // This catches any state that leaks between residual evaluations -- the
    // physics case caches sources, and the DGSoln views are non-owning maps
    // over SUNDIALS memory, so aliasing here is a live risk.
    const Index k = 2, nCells = 4;
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(diffusion_config);

    SystemSolver sys(grid, k, &problem);
    sys.setTau(0.8);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(problem.getNumVars(), grid, k);
    const Index n = shape.getDoF();

    N_Vector Y0 = N_VNew_Serial(n, ctx), dY0 = N_VClone(Y0);
    N_VConst(0.0, Y0);
    N_VConst(0.0, dY0);
    sys.setInitialConditions(Y0, dY0);

    N_Vector Y1 = N_VClone(Y0), Y2 = N_VClone(Y0), Ym = N_VClone(Y0);
    N_Vector dY1 = N_VClone(Y0), dY2 = N_VClone(Y0), dYm = N_VClone(Y0);
    N_Vector F1 = N_VClone(Y0), F2 = N_VClone(Y0), Fm = N_VClone(Y0);

    double *y1 = N_VGetArrayPointer(Y1), *y2 = N_VGetArrayPointer(Y2);
    double *d1 = N_VGetArrayPointer(dY1), *d2 = N_VGetArrayPointer(dY2);
    double *ym = N_VGetArrayPointer(Ym), *dm = N_VGetArrayPointer(dYm);

    const double a = 0.37;
    for (Index i = 0; i < n; ++i)
    {
        y1[i] = std::sin(0.7 * i + 0.2);
        y2[i] = std::cos(0.4 * i - 0.1);
        d1[i] = 0.3 * std::cos(0.9 * i);
        d2[i] = -0.2 * std::sin(0.6 * i);
        ym[i] = a * y1[i] + (1.0 - a) * y2[i];
        dm[i] = a * d1[i] + (1.0 - a) * d2[i];
    }

    sys.residual(0.25, Y1, dY1, F1);
    sys.residual(0.25, Y2, dY2, F2);
    sys.residual(0.25, Ym, dYm, Fm);

    const double *f1 = N_VGetArrayPointer(F1);
    const double *f2 = N_VGetArrayPointer(F2);
    const double *fm = N_VGetArrayPointer(Fm);

    double worst = 0.0;
    for (Index i = 0; i < n; ++i)
        worst = std::max(worst, std::abs(fm[i] - (a * f1[i] + (1.0 - a) * f2[i])));
    BOOST_TEST(worst < 1e-12, "affinity violated by " << worst);

    // Repeating an evaluation must give the identical answer -- no accumulation.
    N_Vector Frepeat = N_VClone(Y0);
    sys.residual(0.25, Y1, dY1, Frepeat);
    const double *fr = N_VGetArrayPointer(Frepeat);
    for (Index i = 0; i < n; ++i)
        BOOST_TEST(fr[i] == f1[i]);

    for (N_Vector v : {Y0, dY0, Y1, Y2, Ym, dY1, dY2, dYm, F1, F2, Fm, Frepeat})
        N_VDestroy(v);
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_CASE(residual_scalar_rows_are_scalar_g_extended)
{
    // res.Scalar(j) = ScalarGExtended(j, Y, dYdt, t), verbatim. Cheap to pin,
    // and it is the row block the Woodbury elimination borders onto.
    const Index k = 2, nCells = 4;
    Grid grid(-1.0, 1.0, nCells);
    ScalarTestLD3 problem(scalar_config, grid);
    BOOST_TEST_REQUIRE(problem.getNumScalars() == 3);

    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(problem.getNumVars(), grid, k, problem.getNumScalars(),
                 problem.getNumAux());
    const Index n = shape.getDoF();

    N_Vector Y = N_VNew_Serial(n, ctx), dYdt = N_VClone(Y), res = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    // Perturb so the scalars are not sitting at whatever value makes G zero.
    double *ya = N_VGetArrayPointer(Y);
    for (Index i = 0; i < n; ++i)
        ya[i] += 0.05 * std::sin(1.3 * i);

    sys.residual(0.5, Y, dYdt, res);

    DGSoln Y_h(problem.getNumVars(), grid, k, N_VGetArrayPointer(Y),
               problem.getNumScalars(), problem.getNumAux());
    DGSoln dY_h(problem.getNumVars(), grid, k, N_VGetArrayPointer(dYdt),
                problem.getNumScalars(), problem.getNumAux());
    DGSoln res_h(problem.getNumVars(), grid, k, N_VGetArrayPointer(res),
                 problem.getNumScalars(), problem.getNumAux());

    for (Index j = 0; j < problem.getNumScalars(); ++j)
        BOOST_TEST(res_h.Scalar(j) == problem.ScalarGExtended(j, Y_h, dY_h, 0.5),
                   boost::test_tools::tolerance(1e-14));

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    N_VDestroy(res);
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_CASE(residual_aux_rows_are_the_projected_constraint)
{
    // The aux rows enforce G = 0 by projection: res.Aux = P_h G(Y). With a
    // linear constraint the projection is exact, so the residual coefficients
    // must equal the basis coefficients of a - c*u computed directly.
    const Index k = 3, nCells = 4;
    Grid grid(0.0, 1.0, nCells);
    AuxResidualMock problem;

    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    // Literal 0 is ambiguous between the Index and double* overloads.
    DGSoln shape(problem.getNumVars(), grid, k, Index(0), problem.getNumAux());
    const Index n = shape.getDoF();

    N_Vector Y = N_VNew_Serial(n, ctx), dYdt = N_VClone(Y), res = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);
    sys.residual(0.0, Y, dYdt, res);

    DGSoln Y_h(problem.getNumVars(), grid, k, N_VGetArrayPointer(Y), Index(0),
               problem.getNumAux());
    DGSoln res_h(problem.getNumVars(), grid, k, N_VGetArrayPointer(res), Index(0),
                 problem.getNumAux());

    auto const &basis = Y_h.getBasis();
    for (Index cell = 0; cell < nCells; ++cell)
    {
        Interval const &I = grid[cell];
        Vector nodalG(k + 1);
        for (Index j = 0; j < k + 1; ++j)
        {
            State s = Y_h.evalOnNode(cell, j);
            nodalG(j) = problem.AuxG(0, s, I.fromRef(basis.Nodes(j)), 0.0);
        }
        const Vector expected = basis.InterpolateOntoBasis(I, nodalG);
        const Vector actual = res_h.Aux(0).getCoeff(cell).second;

        BOOST_TEST((expected - actual).norm() < 1e-12,
                   "cell " << cell << ": aux residual differs by "
                           << (expected - actual).norm());
        // Not vacuous -- the mock's initial aux value does not satisfy G = 0.
        BOOST_TEST(actual.norm() > 1e-3);
    }

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    N_VDestroy(res);
    SUNContext_Free(&ctx);
}

// ------------------------------------------------------- static_residual --

BOOST_AUTO_TEST_CASE(static_residual_converts_a_physics_exception_into_a_retry)
{
    // IDA calls this through a C function pointer, so an escaping C++ exception
    // is undefined behaviour. The wrapper must catch it and return 1, which IDA
    // treats as a recoverable error and retries with a smaller step.
    const Index k = 1, nCells = 3;
    Grid grid(0.0, 1.0, nCells);

    // Build the matrices with a well-behaved problem, then swap in the throwing
    // one: initialiseMatrices itself does not evaluate SigmaFn, but
    // setInitialConditions does, so the throwing case has to be introduced
    // after the state exists.
    TestDiffusion good(diffusion_config);
    SystemSolver sys(grid, k, &good);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(good.getNumVars(), grid, k);
    const Index n = shape.getDoF();
    N_Vector Y = N_VNew_Serial(n, ctx), dYdt = N_VClone(Y), res = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    // Sanity: the healthy path returns 0.
    BOOST_TEST(static_residual(0.0, Y, dYdt, res, &sys) == 0);

    ThrowingDiffusion bad(diffusion_config);
    sys.problem = &bad;

    // The wrapper reports the exception it swallowed, which is the point -- IDA
    // is about to silently retry with a smaller step, and without this line the
    // user would never know why. Capture it rather than letting it litter a
    // passing run, and check it names the failure.
    int retval = 0;
    std::string reported;
    {
        CapturedOutput quiet;
        retval = static_residual(0.0, Y, dYdt, res, &sys);
        reported = quiet.text();
    }
    BOOST_TEST(retval == 1);
    BOOST_TEST(reported.find("deliberate physics failure") != std::string::npos,
               "static_residual should report what it caught, got: " << reported);
    BOOST_TEST(reported.find("Retrying") != std::string::npos);

    // And the exception really does escape residual() itself -- otherwise the
    // check above would pass even if the wrapper were removed.
    sys.problem = &bad;
    BOOST_CHECK_THROW(sys.residual(0.0, Y, dYdt, res), std::runtime_error);

    sys.problem = &good;

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    N_VDestroy(res);
    SUNContext_Free(&ctx);
}

// ------------------------------------------------------ getErrorWeights --

BOOST_AUTO_TEST_CASE(error_weights_follow_the_rtol_atol_definition)
{
    // ewt = 1 / (rtol |y| + atol), which is what IDA's weighted norm divides
    // by. A single-element atol applies to every variable.
    const Index k = 2, nCells = 3;
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(diffusion_config);

    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    const double rtol = 1e-4, atol = 1e-7;
    sys.setTolerances({atol}, rtol);

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(problem.getNumVars(), grid, k);
    const Index n = shape.getDoF();

    N_Vector Y = N_VNew_Serial(n, ctx), dYdt = N_VClone(Y), ewt = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    // Include negative entries: the definition takes |y|.
    double *ya = N_VGetArrayPointer(Y);
    for (Index i = 0; i < n; ++i)
        ya[i] = std::sin(0.8 * i) * (1.0 + 0.5 * i);

    BOOST_TEST(sys.getErrorWeights(Y, ewt) == 0);

    const double *w = N_VGetArrayPointer(ewt);
    for (Index i = 0; i < n; ++i)
        BOOST_TEST(w[i] == 1.0 / (rtol * std::abs(ya[i]) + atol),
                   boost::test_tools::tolerance(1e-12));

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    N_VDestroy(ewt);
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_CASE(error_weights_use_a_per_variable_atol_when_one_is_supplied)
{
    // With atol.size() == nVars each variable gets its own absolute tolerance.
    // This is the branch a multi-channel run takes and it is easy to get wrong
    // -- the sizes nVars and nCells are both small integers.
    const Index k = 1, nCells = 3, nVars = 2;
    Grid grid(0.0, 1.0, nCells);

    struct TwoVar : public TransportSystem
    {
        TwoVar() { nVars = 2; }
        Value LowerBoundary(Index, Time) const override { return 0.0; }
        Value UpperBoundary(Index, Time) const override { return 0.0; }
        bool isLowerBoundaryDirichlet(Index) const override { return true; }
        bool isUpperBoundaryDirichlet(Index) const override { return true; }
        Value SigmaFn(Index i, const State &s, Position, Time) override
        {
            return s.Derivative[i];
        }
        Value Sources(Index, const State &, Position, Time) override { return 0.0; }
        void dSigmaFn_dq(Index i, VectorRef v, const State &, Position, Time) override
        {
            v.setZero();
            v[i] = 1.0;
        }
        void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override
        {
            v.setZero();
        }
        void dSources_du(Index, VectorRef v, const State &, Position, Time) override
        {
            v.setZero();
        }
        void dSources_dq(Index, VectorRef v, const State &, Position, Time) override
        {
            v.setZero();
        }
        void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override
        {
            v.setZero();
        }
        Value InitialValue(Index i, Position x) const override
        {
            return (1.0 + i) * x * (1.0 - x);
        }
        Value InitialDerivative(Index i, Position x) const override
        {
            return (1.0 + i) * (1.0 - 2.0 * x);
        }
    } problem;

    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    const double rtol = 1e-3;
    const std::vector<double> atol{1e-5, 1e-9};
    sys.setTolerances(atol, rtol);

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(nVars, grid, k);
    const Index n = shape.getDoF();
    N_Vector Y = N_VNew_Serial(n, ctx), dYdt = N_VClone(Y), ewt = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);

    double *ya = N_VGetArrayPointer(Y);
    for (Index i = 0; i < n; ++i)
        ya[i] = 1.0 + 0.25 * i;

    sys.getErrorWeights(Y, ewt);

    DGSoln y_h(nVars, grid, k, N_VGetArrayPointer(Y));
    DGSoln w_h(nVars, grid, k, N_VGetArrayPointer(ewt));

    for (Index v = 0; v < nVars; ++v)
    {
        for (Index cell = 0; cell < nCells; ++cell)
        {
            const Vector uy = y_h.u(v).getCoeff(cell).second;
            const Vector uw = w_h.u(v).getCoeff(cell).second;
            for (Index j = 0; j < k + 1; ++j)
                BOOST_TEST(uw(j) == 1.0 / (rtol * std::abs(uy(j)) + atol[v]),
                           boost::test_tools::tolerance(1e-12));
        }
        // The lambda block uses the same per-variable tolerance.
        for (Index i = 0; i < nCells + 1; ++i)
            BOOST_TEST(w_h.lambda(v)(i) ==
                           1.0 / (rtol * std::abs(y_h.lambda(v)(i)) + atol[v]),
                       boost::test_tools::tolerance(1e-12));
    }

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    N_VDestroy(ewt);
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_CASE(error_weights_scale_the_scalar_rows_by_the_dof_count)
{
    // Global scalars are single entries competing against nVars*localDOF*nCells
    // field entries in a root-mean-square norm, so their weight carries an extra
    // sqrt(localDOF * nCells). Without it a scalar's error would be diluted to
    // invisibility and IDA would never restrict a step on its account.
    const Index k = 2, nCells = 4;
    Grid grid(-1.0, 1.0, nCells);
    ScalarTestLD3 problem(scalar_config, grid);

    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.initialiseMatrices();

    const double rtol = 1e-3, atol = 1e-6;
    sys.setTolerances({atol}, rtol);

    SUNContext ctx;
    SUNContext_Create(SUN_COMM_NULL, &ctx);

    DGSoln shape(problem.getNumVars(), grid, k, problem.getNumScalars(),
                 problem.getNumAux());
    const Index n = shape.getDoF();
    N_Vector Y = N_VNew_Serial(n, ctx), dYdt = N_VClone(Y), ewt = N_VClone(Y);
    N_VConst(0.0, Y);
    N_VConst(0.0, dYdt);
    sys.setInitialConditions(Y, dYdt);
    sys.getErrorWeights(Y, ewt);

    DGSoln y_h(problem.getNumVars(), grid, k, N_VGetArrayPointer(Y),
               problem.getNumScalars(), problem.getNumAux());
    DGSoln w_h(problem.getNumVars(), grid, k, N_VGetArrayPointer(ewt),
               problem.getNumScalars(), problem.getNumAux());

    const double localDOF = 3.0 * problem.getNumVars() * (k + 1);
    const double scale = std::sqrt(localDOF * nCells);

    for (Index s = 0; s < problem.getNumScalars(); ++s)
        BOOST_TEST(w_h.Scalar(s) ==
                       scale / (rtol * std::abs(y_h.Scalar(s)) + atol),
                   boost::test_tools::tolerance(1e-12));

    N_VDestroy(Y);
    N_VDestroy(dYdt);
    N_VDestroy(ewt);
    SUNContext_Free(&ctx);
}

BOOST_AUTO_TEST_SUITE_END()
