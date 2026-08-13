// The algebraic time derivatives: q', sigma' and phi' at t0, obtained by
// differentiating the constraints that define them.
//
// At t0 IDA leaves those blocks of dydt identically zero -- IDA_YA_YDP_INIT
// computes algebraic *values* and differential *derivatives*, so there is no y'
// for them to fetch -- which at_t0_only_the_differential_part_of_dydt_exists in
// SolverLifecycleTests.cpp pins. This file is about the vector that fills them
// in, and about the solve that produces it.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "FiniteDifferenceJacobian.hpp"
#include "PyIntegrator.hpp"
#include "SystemSolver.hpp"
#include "TestDiffusion.hpp"
#include "Types.hpp"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>

#include <cmath>
#include <numbers>
#include <string>
#include <toml.hpp>
#include <vector>

using namespace toml::literals::toml_literals;

namespace
{
const toml::value alg_config = u8R"(
    [DiffusionProblem]
    Kappa = 1.0
    Centre = 0.0
)"_toml;

constexpr Index k = 2, nCells = 4;

void configure(SystemSolver &sys, std::string const &stem)
{
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile(stem);
    sys.setOutputCadence(0.05);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-12);
    sys.setTolerances({1e-8}, 1e-6);
}

// The Frobenius-ish norm of one field across every cell, for "is this block
// populated at all" questions.
Value blockNorm(DGSoln const &soln, Index nCellsIn, char which)
{
    Value total = 0.0;
    for (Index i = 0; i < nCellsIn; ++i)
    {
        switch (which)
        {
        case 'u':
            total += soln.u(0).getCoeff(i).second.norm();
            break;
        case 'q':
            total += soln.q(0).getCoeff(i).second.norm();
            break;
        case 's':
            total += soln.sigma(0).getCoeff(i).second.norm();
            break;
        }
    }
    return total;
}

// ------------------------------------------------- fixtures with couplings --
//
// TestDiffusion is linear, has no auxiliary variables and no scalars, so on its
// own it exercises about half of the block layout. These two fill in the rest.
// Both carry explicit time dependence in more than the boundary data, because
// the right-hand side of the solve is a difference of residual() in t and a case
// whose only t is in RF would not tell a missing dSigmaHat/dt from a present one.

// One variable and one algebraic auxiliary variable:
//
//     G         = phi - u^2 - AUX_WOBBLE sin(t)     so phi = u^2 + ...
//     sigma_hat = ( 1 + phi ) q ( 1 + FLUX_DRIFT t )
//
// so the aux rows have a nonzero dG/du and dG/dphi, the sigma row depends on phi,
// and both carry an explicit d/dt.
constexpr double AUX_WOBBLE = 0.05, FLUX_DRIFT = 0.3;

class AuxDiffusion : public TransportSystem
{
public:
    AuxDiffusion()
        : TransportSystem({.variables = numberedFields(1), .aux = numberedAux(1)})
    {
    }

    Value LowerBoundary(Index, Time t) const override { return 0.1 * std::cos(t); }
    Value UpperBoundary(Index, Time t) const override { return 0.2 * std::sin(t); }

    Value SigmaFn(Index, const State &s, Position, Time t) override
    {
        return (1.0 + s.phi(0)) * s.q(0) * (1.0 + FLUX_DRIFT * t);
    }
    Value Sources(Index, const State &s, Position x, Time t) override
    {
        return std::sin(3.0 * x) * (1.0 + t) - 0.5 * s.phi(0) * s.phi(0);
    }
    Value AuxG(Index, const State &s, Position, Time t) override
    {
        return s.phi(0) - s.u(0) * s.u(0) - AUX_WOBBLE * std::sin(t);
    }
    void AuxGPrime(Index, State &out, const State &s, Position, Time) override
    {
        out.phi(0) = 1.0;
        out.u(0) = -2.0 * s.u(0);
    }

    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time t) override
    {
        v[0] = (1.0 + s.phi(0)) * (1.0 + FLUX_DRIFT * t);
    }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSigma_dPhi(Index, VectorRef v, const State &s, Position, Time t) override
    {
        v[0] = s.q(0) * (1.0 + FLUX_DRIFT * t);
    }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dPhi(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = -s.phi(0);
    }

    Value InitialValue(Index, Position x) const override
    {
        return 0.4 * std::sin(std::numbers::pi * x);
    }
    Value InitialDerivative(Index, Position x) const override
    {
        return 0.4 * std::numbers::pi * std::cos(std::numbers::pi * x);
    }
    Value InitialAuxValue(Index, Position x) const override
    {
        const double u0 = InitialValue(0, x);
        return u0 * u0;
    }
};

// One variable and one *algebraic* global scalar, coupled both ways:
//
//     G       = mu - BETA Int u dx - SCALAR_DRIFT t
//     Sources = COUPLING mu + sin(3x)
//
// so v (the scalar's effect on the fields) and w (the fields' effect on the
// constraint) are both nonzero -- with either of them zero the scalar border of
// the matrix would go unchecked -- and the constraint has an explicit d/dt.
constexpr double BETA = 0.75, COUPLING = 0.4, SCALAR_DRIFT = 0.2, KAPPA = 1.0;

class ScalarDiffusion : public TransportSystem
{
public:
    ScalarDiffusion()
        : TransportSystem({.variables = numberedFields(1),
                           .scalars = numberedScalars(1, false)})
    {
    }

    Value LowerBoundary(Index, Time t) const override { return 0.1 * std::cos(t); }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index, const State &s, Position, Time) override { return KAPPA * s.q(0); }
    Value Sources(Index, const State &s, Position x, Time) override
    {
        return COUPLING * s.scalar(0) + std::sin(3.0 * x);
    }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = KAPPA;
    }
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 0.0;
    }
    void dSources_dScalars(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = COUPLING;
    }

    static double mass(GlobalState const &y, Vector const &weights)
    {
        return ScalarHooks::integrate(y.Variable().row(0), weights);
    }

    Value ScalarG(Index, GlobalState const &yy, GlobalState const &,
                  std::vector<Position> const &, Values const &weights, Matrix const &,
                  Time t) override
    {
        return yy.Scalars()(0) - BETA * mass(yy, weights) - SCALAR_DRIFT * t;
    }
    void ScalarGPrime(GlobalStateMatrix &dG, GlobalStateMatrix &, GlobalState const &,
                      GlobalState const &, std::vector<Position> const &,
                      Values const &weights, Matrix const &, Time) override
    {
        dG[0].Variable().row(0) = -BETA * weights.transpose();
        dG[0].Scalars()(0) = 1.0;
    }

    Value InitialValue(Index, Position x) const override { return x * (1.0 - x); }
    Value InitialDerivative(Index, Position x) const override { return 1.0 - 2.0 * x; }
    Value InitialScalarValue(Index) const override { return BETA / 6.0; }
};

// An objective that depends on q and nothing else.
//
// This is the one that shows what the whole exercise bought. dG/dt for it is
// Int q' dx, and IDA leaves the q block of dydt identically zero at t0 -- so a
// gate reading IDA's derivative computes exactly 0.0 and can never reject, no
// matter how badly the objective is falling. Reading dydtComplete instead, the
// number is real.
//
// All four dgFn_d* are pure virtual on AdjointProblem, so the three this does
// not depend on are spelled out as zero rather than inherited. That is what
// makes it genuinely q-only, and therefore evidence.
class QIntegralObjective : public AdjointProblem
{
public:
    explicit QIntegralObjective(double s) : sign(s)
    {
        ng = 1;
        np = 1;
        np_boundary = 0;
    }

    using AdjointProblem::gFn;

    Value gFn(Index, const State &s, Position) const override { return sign * s.q(0); }

    Value GFn(Index, DGSoln &) const override
    {
        throw std::logic_error("GFn is not part of what this fixture exercises.");
    }
    Value dGFndp(Index, Index, DGSoln &) const override { return 0.0; }

    void dgFn_du(Index, VectorRef v, const State &, Position) override { v.setZero(); }
    void dgFn_dq(Index, VectorRef v, const State &, Position) override
    {
        v.setZero();
        v[0] = sign;
    }
    void dgFn_dsigma(Index, VectorRef v, const State &, Position) override { v.setZero(); }
    void dgFn_dphi(Index, VectorRef v, const State &, Position) override { v.setZero(); }

    void dSigmaFn_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }
    void dSources_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }

private:
    double sign;
};

// The same, depending on u alone. Its dg/dq, dg/dsigma and dg/dphi are zero, so
// completing the derivative cannot change its dG/dt -- which is what makes it
// the control for the q-only case.
class UIntegralObjective : public AdjointProblem
{
public:
    explicit UIntegralObjective(double s) : sign(s)
    {
        ng = 1;
        np = 1;
        np_boundary = 0;
    }

    using AdjointProblem::gFn;

    Value gFn(Index, const State &s, Position) const override { return sign * s.u(0); }

    Value GFn(Index, DGSoln &) const override
    {
        throw std::logic_error("GFn is not part of what this fixture exercises.");
    }
    Value dGFndp(Index, Index, DGSoln &) const override { return 0.0; }

    void dgFn_du(Index, VectorRef v, const State &, Position) override
    {
        v.setZero();
        v[0] = sign;
    }
    void dgFn_dq(Index, VectorRef v, const State &, Position) override { v.setZero(); }
    void dgFn_dsigma(Index, VectorRef v, const State &, Position) override { v.setZero(); }
    void dgFn_dphi(Index, VectorRef v, const State &, Position) override { v.setZero(); }

    void dSigmaFn_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }
    void dSources_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }

private:
    double sign;
};

// The state a Jacobian is assembled at, without going near IDA: initialiseMatrices
// plus setInitialConditions is all the assembly needs, and it keeps the fixtures
// above out of IDACalcIC's way.
struct BareSolver
{
    Grid grid;
    SystemSolver sys;
    SUNContext ctx = nullptr;
    N_Vector Y = nullptr, dYdt = nullptr;
    Index n = 0;

    BareSolver(TransportSystem &problem, Index order, Index cells, double tau,
               bool superconvergent = false)
        : grid(0.0, 1.0, cells), sys(grid, order, &problem)
    {
        sys.setTau(tau);
        sys.setSuperconvergent(superconvergent);
        sys.resetCoeffs();
        sys.initialiseMatrices();

        SUNContext_Create(SUN_COMM_NULL, &ctx);

        DGSoln shape(problem.getNumVars(), grid, order, problem.getNumScalars(),
                     problem.getNumAux());
        n = shape.getDoF();

        Y = N_VNew_Serial(n, ctx);
        dYdt = N_VClone(Y);
        N_VConst(0.0, Y);
        N_VConst(0.0, dYdt);
        sys.setInitialConditions(Y, dYdt);
    }

    ~BareSolver()
    {
        N_VDestroy(Y);
        N_VDestroy(dYdt);
        SUNContext_Free(&ctx);
    }
};

// The drift guard, as a number: the largest entry by which the assembled dF/dy
// differs from a finite difference of residual(), relative to the largest entry
// of the Jacobian itself, over the rows the residual actually defines.
//
// Also checks that the two agree about *which* rows those are. residual() leaves
// the Dirichlet trace rows untouched -- the constraint lambda = g_D(t) is imposed
// inside the linear solve -- so both matrices must be exactly zero there and
// nowhere else.
double assemblyDrift(TransportSystem &problem, Index order, Index cells, double tau,
                     double t, bool superconvergent = false)
{
    BareSolver f(problem, order, cells, tau, superconvergent);

    const Matrix Jfd = fdjac::jacobian(f.sys, f.Y, f.dYdt, t, 0.0);
    const Matrix Jasm = f.sys.assembleDenseJacobian(f.sys.y, f.sys.dydt, t, 0.0);

    BOOST_TEST(Jasm.rows() == Jfd.rows());

    const std::vector<Index> undefined = fdjac::undefinedRows(Jfd);
    BOOST_TEST(fdjac::undefinedRows(Jasm) == undefined, boost::test_tools::per_element());

    double worst = 0.0, scale = 0.0;
    Index worstRow = -1, worstCol = -1;
    for (Index i = 0; i < Jfd.rows(); ++i)
    {
        if (fdjac::isUndefined(undefined, i))
            continue;
        for (Index j = 0; j < Jfd.cols(); ++j)
        {
            scale = std::max(scale, std::abs(Jfd(i, j)));
            const double d = std::abs(Jasm(i, j) - Jfd(i, j));
            if (d > worst)
            {
                worst = d;
                worstRow = i;
                worstCol = j;
            }
        }
    }

    // |J| is reported because the drift alone cannot distinguish "the flag-on
    // assembly agrees" from "the flag never took effect": the worst-differenced
    // entry lands in the q row, which is physics-free and so bit-identical either
    // way, and the two configurations report the same drift to every digit.
    BOOST_TEST_MESSAGE("  superconvergent=" << f.sys.isSuperconvergent() << ", |J| = "
                                            << Jasm.norm() << ", worst entry (" << worstRow
                                            << ", " << worstCol << "): assembled "
                                            << Jasm(worstRow, worstCol) << " vs differenced "
                                            << Jfd(worstRow, worstCol));
    return worst / std::max(1.0, scale);
}
} // namespace

BOOST_AUTO_TEST_SUITE(algebraic_derivative_tests)

BOOST_AUTO_TEST_CASE(dydtComplete_starts_as_a_copy_of_idas_derivative)
{
    // Separate storage, seeded from IDA's. The separation is the point: writing
    // the algebraic blocks into IDA's own dYdt would change the state it takes
    // its first step from, and the symptom would be a step-size failure
    // somewhere later rather than anything pointing back here.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(alg_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "algderiv_storage");

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    // The u block is IDA's, and is not zero -- u is differential.
    BOOST_TEST(blockNorm(sys.dydtComplete, nCells, 'u') > 1e-8,
               "dydtComplete's u block is empty, so it was never seeded");

    // And it is genuinely distinct storage.
    BOOST_TEST(static_cast<const void *>(sys.dydtCompleteMem) !=
                   static_cast<const void *>(N_VGetArrayPointer(sys.dYdt)),
               "dydtComplete aliases IDA's dYdt, so writing to it would change the run");

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
}

BOOST_AUTO_TEST_CASE(the_assembled_jacobian_matches_a_finite_difference_of_the_residual)
{
    // The most important test in this file, and the reason a dense assembly is
    // affordable at all.
    //
    // This is the *third* place the HDG block layout is written down, after
    // Matrices.cpp and initializeMatricesForAdjointSolve. CLAUDE.md records what
    // that costs: the dSigma/dPhi block went missing from the adjoint assembly and
    // produced a silently wrong gradient beside a perfectly good objective, for
    // months. Nothing about the answer says a block is absent -- an omitted block
    // here would simply give the wrong q' -- so the assembly is pinned against a
    // reference that cannot drift from it: a finite difference of residual()
    // itself.
    //
    // Three cases, because a single one leaves most of the layout untouched:
    // TestDiffusion is linear with neither aux variables nor scalars, AuxDiffusion
    // adds the aux rows and the sigma/phi coupling, ScalarDiffusion adds the
    // bordered scalar row and column -- including the sign of v, which solveJacEq
    // carries inside its Woodbury elimination rather than in the stored matrix.
    for (Index order : {1, 2, 3})
    {
        TestDiffusion linear(alg_config);
        const double d = assemblyDrift(linear, order, nCells, 1.0, 0.0);
        BOOST_TEST_MESSAGE("TestDiffusion, k = " << order << ": drift " << d);
        BOOST_TEST(d < 1e-9, "the assembled dF/dy disagrees with a finite difference of "
                             "residual() at k = "
                                 << order << " (relative " << d << ")");
    }

    for (Index order : {1, 2, 3})
    {
        AuxDiffusion aux;
        const double d = assemblyDrift(aux, order, nCells, 0.75, 0.3);
        BOOST_TEST_MESSAGE("AuxDiffusion, k = " << order << ": drift " << d);
        BOOST_TEST(d < 1e-9, "the assembled dF/dy disagrees with a finite difference of "
                             "residual() with an auxiliary variable at k = "
                                 << order << " (relative " << d << ")");
    }

    for (Index order : {1, 2, 3})
    {
        ScalarDiffusion scalars;
        const double d = assemblyDrift(scalars, order, nCells, 0.75, 0.3);
        BOOST_TEST_MESSAGE("ScalarDiffusion, k = " << order << ": drift " << d);
        BOOST_TEST(d < 1e-9, "the assembled dF/dy disagrees with a finite difference of "
                             "residual() with a global scalar at k = "
                                 << order << " (relative " << d << ")");
    }

    // And with the superconvergent scheme, which takes assembleCellMatrix's other
    // branch entirely: the physics is evaluated at the k+2 star nodes with u* in
    // place of u_h, and every block gains a chain factor. The trace and scalar
    // blocks are untouched by the flag, so this is a check on the shared cell
    // assembly rather than on the surrounding scatter.
    for (Index order : {1, 2, 3})
    {
        AuxDiffusion aux;
        const double d = assemblyDrift(aux, order, nCells, 0.75, 0.3, true);
        BOOST_TEST_MESSAGE("AuxDiffusion superconvergent, k = " << order << ": drift " << d);
        BOOST_TEST(d < 1e-9, "the assembled dF/dy disagrees with a finite difference of "
                             "residual() under the superconvergent scheme at k = "
                                 << order << " (relative " << d << ")");
    }
}

BOOST_AUTO_TEST_CASE(the_u_block_round_trips_through_the_identity_row)
{
    // The u rows of the assembled matrix are overwritten with the identity and the
    // known u' put in those rows of the right-hand side, so the solve must hand u'
    // back unchanged. A free self-check on the substitution: if this fails, the
    // q, sigma and phi blocks solved for beside it are meaningless.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(alg_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "algderiv_roundtrip");

    std::vector<Vector> before;
    {
        CapturedOutput quiet;
        sys.initialize();
        for (Index i = 0; i < nCells; ++i)
            before.push_back(sys.dydtComplete.u(0).getCoeff(i).second);
        sys.computeAlgebraicTimeDerivatives();
    }

    // Not vacuous: u is differential, so this is a real derivative.
    BOOST_TEST(blockNorm(sys.dydtComplete, nCells, 'u') > 1e-8);

    for (Index i = 0; i < nCells; ++i)
    {
        const Vector after = sys.dydtComplete.u(0).getCoeff(i).second;
        for (Index j = 0; j < after.size(); ++j)
            BOOST_TEST(after(j) == before[i](j), boost::test_tools::tolerance(1e-10));
    }

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
}

BOOST_AUTO_TEST_CASE(the_algebraic_blocks_stop_being_zero)
{
    // The change, stated as a measurement. Before this call q' and sigma' are
    // identically zero -- not small, zero -- which is what makes dG/dt at t0 see
    // only its u term however much the objective depends on the others.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(alg_config);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "algderiv_nonzero");

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    BOOST_TEST(blockNorm(sys.dydtComplete, nCells, 'q') == 0.0,
               boost::test_tools::tolerance(0.0));
    BOOST_TEST(blockNorm(sys.dydtComplete, nCells, 's') == 0.0,
               boost::test_tools::tolerance(0.0));

    {
        CapturedOutput quiet;
        sys.computeAlgebraicTimeDerivatives();
    }

    BOOST_TEST(blockNorm(sys.dydtComplete, nCells, 'q') > 1e-8,
               "q' is still zero after solving for it, so the right-hand side or the "
               "solve is empty");
    BOOST_TEST(blockNorm(sys.dydtComplete, nCells, 's') > 1e-8,
               "sigma' is still zero after solving for it");

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
}

BOOST_AUTO_TEST_CASE(the_gate_sees_a_q_only_objective)
{
    // The change, stated as a measurement. g depends on q alone, so
    // dG/dt = Int q' dx -- exactly zero when read from IDA's dydt at t0, because
    // that block is empty. A gate reading it could never reject such a run
    // however badly its objective were falling.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(alg_config);
    QIntegralObjective objective(1.0);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "algderiv_qgate");
    sys.setAdjointProblem(&objective);
    sys.setObjectiveDecreaseTolerance(1e-12);

    {
        CapturedOutput quiet;
        sys.initialize();
        sys.objectiveIsDecreasing();
    }

    BOOST_TEST(sys.lastDGdt()(0) != 0.0,
               "dG/dt for a q-only objective came out exactly zero, so the gate is "
               "reading IDA's dydt rather than the completed one");

    // And what it read is the value the solve produced, not something else.
    const Value fromComplete = sys.dGdt(0, sys.y, sys.dydtComplete);
    BOOST_TEST(sys.lastDGdt()(0) == fromComplete, boost::test_tools::tolerance(1e-12));

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
}

BOOST_AUTO_TEST_CASE(a_u_only_objective_is_unchanged_by_the_completed_derivative)
{
    // The converse, and the guard against the previous test passing for the
    // wrong reason. An objective depending only on u cannot be affected by
    // filling in q', sigma' and phi' -- its dg/dq, dg/dsigma and dg/dphi are all
    // zero -- so the gate must give exactly what IDA's own derivative gives.
    //
    // If this ever differs, the solve is perturbing the u block, which the
    // round-trip test says it must not.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(alg_config);
    UIntegralObjective objective(1.0);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "algderiv_ugate");
    sys.setAdjointProblem(&objective);
    sys.setObjectiveDecreaseTolerance(1e-12);

    {
        CapturedOutput quiet;
        sys.initialize();
        sys.objectiveIsDecreasing();
    }

    const Value fromIDA = sys.dGdt(0, sys.y, sys.dydtJac);
    BOOST_TEST(sys.lastDGdt()(0) == fromIDA, boost::test_tools::tolerance(1e-10));

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
}

BOOST_AUTO_TEST_SUITE_END()
