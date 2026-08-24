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

#include <ida/ida.h>
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

// ------------------------------------------------ a manufactured solution --
//
// The only fixture here that says what the answer *should* be, rather than that
// it is nonzero, self-consistent or close to an independent estimate.
//
//     u(x, t) = (1 + t) p(x),        p(x) = 0.5 + 0.3 x - x^2
//
// p is a quadratic, so for k >= 2 the nodal basis reproduces it exactly, every
// interpolation the residual performs is exact, and the discrete solution *is*
// the manufactured one to round-off -- which is what makes the closed-form
// derivatives a legitimate reference rather than an O(h^(k+1)) approximation to
// one. The test asserts that premise before it asserts anything else.
//
// Solving u_t - kappa u_xx = S on it gives S = p(x) + 2 kappa (1 + t), and the
// derivatives follow:
//
//     u'      = p(x)
//     q'      = p'(x) = 0.3 - 2x
//     sigma'  = -kappa p'(x)              <-- note the sign
//     lambda' = p(x_face)
//
// The sign is the trap CLAUDE.md records. MaNTA stores -sigmaHat: res.sigma is
// A sigma_h + Pi(sigmaHat), so what it enforces is sigma_h = -Pi(sigmaHat), and
// the stored flux derivative is minus the physical one. Getting it backwards
// leaves a case converging at the right rate to the wrong function, so only a
// closed-form comparison like this one can catch it.
//
// Both boundaries are Dirichlet and both move in t, so the Dirichlet trace rows
// -- which the residual does not constrain at all, and which get their own
// identity row and dg_D/dt inside the solve -- are exercised rather than
// trivially zero.
constexpr double MMS_KAPPA = 0.8;

double mmsP(Position x) { return 0.5 + 0.3 * x - x * x; }
double mmsPPrime(Position x) { return 0.3 - 2.0 * x; }
double mmsU(Position x, Time t) { return (1.0 + t) * mmsP(x); }
double mmsQ(Position x, Time t) { return (1.0 + t) * mmsPPrime(x); }

class ManufacturedInTime : public TransportSystem
{
public:
    ManufacturedInTime() : TransportSystem({.variables = numberedFields(1)}) {}

    Value LowerBoundary(Index, Time t) const override { return mmsU(0.0, t); }
    Value UpperBoundary(Index, Time t) const override { return mmsU(1.0, t); }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return MMS_KAPPA * s.q(0);
    }
    // u_t - kappa u_xx on the exact solution, written out rather than
    // differenced so the test does not lean on the machinery it is testing.
    Value Sources(Index, const State &, Position x, Time t) override
    {
        return mmsP(x) + 2.0 * MMS_KAPPA * (1.0 + t);
    }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = MMS_KAPPA;
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

    Value InitialValue(Index, Position x) const override { return mmsU(x, 0.0); }
    Value InitialDerivative(Index, Position x) const override { return mmsQ(x, 0.0); }
};

// ------------------------------------------------------ autonomous, and not --
//
// Nothing in this depends on t: constant Dirichlet ends, a flux and a source that
// ignore their Time argument. residual(t + h) and residual(t - h) are then the
// same function of the same data, so their difference is zero bit for bit rather
// than 1e-9 -- which is the claim an_autonomous_case_differences_to_exactly_zero
// makes, and it is a stronger one than "small".
constexpr double AUTONOMOUS_KAPPA = 0.6;

class AutonomousDiffusion : public TransportSystem
{
public:
    AutonomousDiffusion() : TransportSystem({.variables = numberedFields(1)}) {}

    Value LowerBoundary(Index, Time) const override { return 0.3; }
    Value UpperBoundary(Index, Time) const override { return 0.1; }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return AUTONOMOUS_KAPPA * s.q(0);
    }
    Value Sources(Index, const State &, Position x, Time) override
    {
        return std::sin(3.0 * x);
    }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = AUTONOMOUS_KAPPA;
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

    Value InitialValue(Index, Position x) const override
    {
        return 0.3 - 0.2 * x + 0.5 * std::sin(std::numbers::pi * x);
    }
    Value InitialDerivative(Index, Position x) const override
    {
        return -0.2 + 0.5 * std::numbers::pi * std::cos(std::numbers::pi * x);
    }
};

// The converse: a case whose *only* explicit time dependence is in the boundary
// data, one end Neumann and one Dirichlet, so the two arrays
// updateBoundaryConditions writes in place are both live.
//
//   lower, Neumann:   L_global(node 0) = -g_N(t),  and res.lambda subtracts it,
//                     so d res.lambda(0) / dt = +dg_N/dt exactly -- a closed form
//                     to check the differencing against, with no basis functions
//                     in it.
//   upper, Dirichlet: RF_cellwise[last] carries g_D(t) into the q and u rows of
//                     that cell.
constexpr double DRIFT_KAPPA = 0.9, NEUMANN_RATE = 0.4, DIRICHLET_RATE = 0.3;

class DriftingBoundary : public TransportSystem
{
public:
    DriftingBoundary()
        : TransportSystem({.variables = numberedFields(1, BoundaryKind::Neumann,
                                                      BoundaryKind::Dirichlet)})
    {
    }

    // The imposed flux at the lower end, and the imposed value at the upper.
    Value LowerBoundary(Index, Time t) const override
    {
        return 0.35 + NEUMANN_RATE * std::sin(t);
    }
    Value UpperBoundary(Index, Time t) const override
    {
        return 0.2 + DIRICHLET_RATE * t;
    }

    // Autonomous physics, so the boundaries are the only route from t into the
    // residual and the checks below can name what they are measuring.
    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return DRIFT_KAPPA * s.q(0);
    }
    Value Sources(Index, const State &, Position, Time) override { return 0.0; }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = DRIFT_KAPPA;
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

    Value InitialValue(Index, Position x) const override { return 0.2 + 0.1 * x; }
    Value InitialDerivative(Index, Position) const override { return 0.1; }
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

// ---------------------------------------------- the two independent routes --

// This one: differentiate the constraints at t0 and read Int q' dx off the
// answer.
double dGdtFromTheConstraints()
{
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(alg_config);
    QIntegralObjective objective(1.0);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "algderiv_constraints");
    sys.setAdjointProblem(&objective);

    Value value = 0.0;
    {
        CapturedOutput quiet;
        sys.initialize();
        sys.computeAlgebraicTimeDerivatives();
        value = sys.dGdt(0, sys.y, sys.dydtComplete);
        sys.destroySundials();
    }
    return value;
}

// The other: take one IDA step and read IDA's own now-populated dYdt. This is
// what branch dgdt-gate-after-step did, and it shares no code with the route
// above -- IDA's derivative comes out of its BDF formula, not out of any
// assembly of ours.
double dGdtOneStepIn(double step, double &tReached)
{
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(alg_config);
    QIntegralObjective objective(1.0);
    SystemSolver sys(grid, k, &problem);
    configure(sys, "algderiv_onestep");
    sys.setAdjointProblem(&objective);

    Value value = 0.0;
    {
        CapturedOutput quiet;
        sys.initialize();

        // Pin the step rather than let IDA choose it: the comparison is first
        // order in the step, so it has to be a *known* number for the residual
        // between the two routes to mean anything.
        //
        // Here rather than through setInitialTimestep, which would also reach
        // IDACalcIC -- Solver.cpp hands dt0 to it as the scale for the initial
        // correction -- and the two routes must start from the same initial
        // condition or the difference measured is not the one intended.
        IDASetInitStep(sys.IDA_mem, step);

        sunrealtype tret = 0.0;
        IDASolve(sys.IDA_mem, 1.0, &tret, sys.Y, sys.dYdt, IDA_ONE_STEP);
        tReached = tret;

        // sys.dydt, not dydtComplete: the point is to use IDA's own derivative,
        // which one step in is populated in every block.
        value = sys.dGdt(0, sys.y, sys.dydt);
        sys.destroySundials();
    }
    return value;
}

// ------------------------------------------- the manufactured comparison --

// The worst absolute error in each block, against the closed forms above.
struct ManufacturedErrors
{
    double u = 0.0, q = 0.0, sigma = 0.0;             // of the derivative
    double stateU = 0.0, stateQ = 0.0, stateS = 0.0;  // of the state itself
    double lambda = 0.0;
};

ManufacturedErrors manufacturedErrors(Index order, Index cells)
{
    Grid grid(0.0, 1.0, cells);
    ManufacturedInTime problem;
    SystemSolver sys(grid, order, &problem);
    configure(sys, "algderiv_mms");

    ManufacturedErrors e;
    {
        CapturedOutput quiet;
        sys.initialize();
        sys.computeAlgebraicTimeDerivatives();
    }

    auto worst = [](double &into, double a, double b) { into = std::max(into, std::abs(a - b)); };

    // Interior sample points, three per cell: operator() has to decide which
    // interval a position belongs to, and a cell boundary is ambiguous.
    for (Index i = 0; i < cells; ++i)
    {
        Interval const &I = grid[i];
        for (Index j = 0; j < 3; ++j)
        {
            const Position x = I.x_l + (j + 0.5) / 3.0 * (I.x_u - I.x_l);

            worst(e.stateU, sys.y.u(0)(x), mmsU(x, 0.0));
            worst(e.stateQ, sys.y.q(0)(x), mmsQ(x, 0.0));
            worst(e.stateS, sys.y.sigma(0)(x), -MMS_KAPPA * mmsQ(x, 0.0));

            worst(e.u, sys.dydtComplete.u(0)(x), mmsP(x));
            worst(e.q, sys.dydtComplete.q(0)(x), mmsPPrime(x));
            worst(e.sigma, sys.dydtComplete.sigma(0)(x), -MMS_KAPPA * mmsPPrime(x));
        }
    }

    for (Index i = 0; i <= cells; ++i)
    {
        const Position x = (i < cells) ? grid[i].x_l : grid[cells - 1].x_u;
        worst(e.lambda, sys.dydtComplete.lambda(0)(i), mmsP(x));
    }

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
    return e;
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

BOOST_AUTO_TEST_CASE(it_agrees_with_the_derivative_one_ida_step_in)
{
    // The only check in this file against a *different method*. Everything else
    // here is self-consistency: the assembly against a difference of the same
    // residual, the u block against the u block that went in, the gate against
    // the vector it was pointed at.
    //
    // This one takes one IDASolve in IDA_ONE_STEP mode and reads IDA's own dYdt,
    // which one step in is populated in every block -- the route branch
    // dgdt-gate-after-step took, and the reason this fix was worth preferring to
    // it. Nothing is shared: that derivative comes out of IDA's BDF formula, this
    // one out of a dense solve against blocks we assemble.
    //
    // They differ at O(step), from two sources that both shrink with it: IDA's
    // value is a backward difference rather than a derivative, and it is the
    // derivative at t0 + step rather than at t0. So the check is a *rate* rather
    // than a fixed tolerance -- the discrepancy per unit step must be the same at
    // both steps, which is what a truncation error does and a wrong q' does not.
    //
    // Measured, at the time of writing: dG/dt = 2.4674011003 from the
    // constraints; one step to 3e-7 gives 2.4674001874 and one to 3e-8 gives
    // 2.4674009913, i.e. relative discrepancies of 3.7e-7 and 4.4e-8 for a
    // tenfold change in step.
    //
    // The step is asked for rather than commanded: IDA halves an initial step
    // that fails its error test, and on this problem it will not take more than
    // about 5e-7 whatever it is offered -- which is why the assertions below are
    // on the step it *reached* rather than on the one requested.
    const double atT0 = dGdtFromTheConstraints();
    BOOST_TEST(atT0 != 0.0, "the constraint route gave exactly zero, so there is nothing "
                            "to compare");

    double tCoarse = 0.0, tFine = 0.0;
    const double coarse = dGdtOneStepIn(3e-7, tCoarse);
    const double fine = dGdtOneStepIn(3e-8, tFine);

    const double errCoarse = std::abs(coarse - atT0);
    const double errFine = std::abs(fine - atT0);

    BOOST_TEST_MESSAGE("dG/dt from the constraints at t0: " << atT0);
    BOOST_TEST_MESSAGE("  one step to t = " << tCoarse << ": " << coarse << " (error "
                                            << errCoarse << ", relative "
                                            << errCoarse / std::abs(atT0) << ")");
    BOOST_TEST_MESSAGE("  one step to t = " << tFine << ": " << fine << " (error " << errFine
                                            << ", relative " << errFine / std::abs(atT0)
                                            << ")");
    BOOST_TEST_MESSAGE("  error ratio " << errCoarse / errFine << " for a step ratio of "
                                        << tCoarse / tFine);

    // A real separation between the two, or the rate below means nothing.
    BOOST_TEST(tCoarse > 5.0 * tFine,
               "IDA clipped the two initial steps to within a factor of "
                   << tCoarse / tFine << ", so there is no rate to measure");

    // Agreement to the step. dG/dt is O(1) here, so a relative discrepancy of
    // more than a few times the step is a disagreement rather than a truncation
    // error.
    BOOST_TEST(errCoarse / std::abs(atT0) < 10.0 * tCoarse,
               "the two routes to dG/dt disagree by "
                   << errCoarse / std::abs(atT0) << " relative, at a step of " << tCoarse);

    // And it is first order in the step actually taken, which is what says the
    // difference *is* the step. A wrong q' would leave a step-independent floor,
    // and the two rates below would then differ by the ratio of the steps rather
    // than agree.
    const double rateCoarse = errCoarse / tCoarse;
    const double rateFine = errFine / tFine;
    BOOST_TEST_MESSAGE("  discrepancy per unit step: " << rateCoarse << " and " << rateFine);
    BOOST_TEST(rateCoarse / rateFine > 0.5,
               "the disagreement per unit step is not constant (" << rateCoarse << " vs "
                                                                  << rateFine << ")");
    BOOST_TEST(rateCoarse / rateFine < 2.0,
               "the disagreement per unit step is not constant (" << rateCoarse << " vs "
                                                                  << rateFine << ")");
}

BOOST_AUTO_TEST_CASE(the_derivatives_match_a_manufactured_solution)
{
    // The only test here that checks the derivatives are *right*. Every other one
    // checks they are nonzero, self-consistent, or close to an independent
    // estimate; a uniformly wrong scale factor would pass all of them.
    //
    // u = (1 + t)(0.5 + 0.3x - x^2) is a quadratic in x, so for k >= 2 the
    // discrete solution is the exact one to round-off and the closed-form
    // derivatives are a real reference rather than an O(h^(k+1)) approximation to
    // one. The state assertions come first because that premise is what the rest
    // rests on: if the discrete solution were not exact, a derivative error and a
    // discretisation error would be indistinguishable here.
    for (Index order : {2, 3})
    {
        const ManufacturedErrors e = manufacturedErrors(order, nCells);

        BOOST_TEST_MESSAGE("k = " << order << ": state errors u " << e.stateU << ", q "
                                  << e.stateQ << ", sigma " << e.stateS);
        BOOST_TEST_MESSAGE("  derivative errors u' " << e.u << ", q' " << e.q << ", sigma' "
                                                     << e.sigma << ", lambda' " << e.lambda);

        BOOST_TEST(e.stateU < 1e-12,
                   "the discrete solution is not the manufactured one at k = "
                       << order << " (u off by " << e.stateU
                       << "), so the closed-form derivatives are not a valid reference");
        BOOST_TEST(e.stateQ < 1e-10, "q_h is off the exact q by " << e.stateQ);

        // The sign convention, pinned on the state before it is relied on for the
        // derivative: MaNTA stores -sigmaHat, so sigma_h is -kappa q_h.
        BOOST_TEST(e.stateS < 1e-10,
                   "sigma_h is not -kappa q_h at k = "
                       << order << " (off by " << e.stateS
                       << "); either the stored flux is not negated or kappa is wrong");

        // u' comes back out of the identity rows untouched, with no differencing
        // anywhere near it, so it is exact: 1e-14 measured. q', sigma' and lambda'
        // are solved for against a differenced right-hand side and sit at that
        // difference's round-off floor -- 5.6e-11, 4.5e-11 and 2.0e-12 at k = 2,
        // 8.8e-12, 7.0e-12 and 2.0e-12 at k = 3. The floor is eps^(2/3), which is
        // what timeDifferenceStep's cbrt(eps) buys; with the sqrt(eps) the design
        // specified, the k = 2 numbers were 3.4e-8, 2.7e-8 and 7.5e-10.
        //
        // sigma' is exactly -kappa q', including in the error, which is the sign
        // convention holding. Drop the minus and this reports 2.59 -- 2 kappa
        // max|p'| -- rather than 4.5e-11, so it is not a delicate check.
        BOOST_TEST(e.u < 1e-12, "u' is off its closed form by " << e.u << " at k = " << order);
        BOOST_TEST(e.q < 1e-9, "q' is off its closed form by " << e.q << " at k = " << order);
        BOOST_TEST(e.sigma < 1e-9, "sigma' is off its closed form by "
                                       << e.sigma << " at k = " << order
                                       << "; check the sign -- the stored flux is -sigmaHat");
        BOOST_TEST(e.lambda < 1e-10, "lambda' is off its closed form by "
                                         << e.lambda << " at k = " << order);
    }
}

BOOST_AUTO_TEST_CASE(an_autonomous_case_differences_to_exactly_zero)
{
    // Not "small": zero, bit for bit. residual() reaches t only through the
    // boundary data and the physics hooks; this case's boundaries are constants
    // and its hooks ignore their Time argument, so the two evaluations are the
    // same arithmetic on the same doubles and their difference is an exact 0.0.
    //
    // Asserted exactly rather than to a tolerance because that is the whole
    // claim. A differenced quantity is normally noise at 1e-11 or so -- see the
    // manufactured case -- and a tolerance of 1e-9 here would pass whether the
    // case were autonomous or merely nearly so, which is not the property being
    // stated.
    Grid grid(0.0, 1.0, nCells);
    AutonomousDiffusion problem;
    SystemSolver sys(grid, k, &problem);
    configure(sys, "algderiv_autonomous");

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    const double h = SystemSolver::timeDifferenceStep(0.0);
    Vector dFdt;
    {
        CapturedOutput quiet;
        dFdt = sys.differenceResidualInTime(0.0, h);
    }

    BOOST_TEST(dFdt.size() > 0);
    BOOST_TEST(dFdt.cwiseAbs().maxCoeff() == 0.0, boost::test_tools::tolerance(0.0));

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
}

BOOST_AUTO_TEST_CASE(a_time_dependent_boundary_reaches_the_right_hand_side)
{
    // The converse of the case above, and the check on the restore.
    //
    // DriftingBoundary's physics is autonomous and its boundaries are not, so
    // every nonzero entry of dF/dt below came from RF_cellwise or L_global -- the
    // two arrays updateBoundaryConditions writes *in place*, which is the trap
    // CLAUDE.md records: residual() leaves them at t - h, and anything reading
    // them afterwards would be reading boundary data from the wrong time with
    // nothing to say so.
    Grid grid(0.0, 1.0, nCells);
    DriftingBoundary problem;
    SystemSolver sys(grid, k, &problem);
    configure(sys, "algderiv_drifting");

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    const std::vector<Vector> rfBefore = sys.RF_cellwise;
    const Vector lBefore = sys.L_global;

    const double h = SystemSolver::timeDifferenceStep(0.0);
    Vector dFdt;
    {
        CapturedOutput quiet;
        dFdt = sys.differenceResidualInTime(0.0, h);
    }

    // Picked up at all.
    BOOST_TEST(dFdt.cwiseAbs().maxCoeff() > 1e-6,
               "a case whose boundaries move in t differenced to zero, so the explicit "
               "time dependence is not reaching the right-hand side");

    // And picked up correctly. res.lambda subtracts L_global, and for a Neumann
    // lower end L_global(node 0) is -g_N(t), so that row's explicit derivative is
    // exactly dg_N/dt -- no basis functions and no quadrature in it.
    const Index lambdaRow = static_cast<Index>(nCells) * (3 * (k + 1));
    const double expected = NEUMANN_RATE * std::cos(0.0);
    BOOST_TEST_MESSAGE("Neumann lambda row: differenced " << dFdt(lambdaRow) << ", exact "
                                                          << expected);
    BOOST_TEST(dFdt(lambdaRow) == expected, boost::test_tools::tolerance(1e-9));

    // The Dirichlet end feeds RF_cellwise, and that reaches the q row of the last
    // cell rather than a trace row -- the trace row there is not in the residual
    // at all.
    const Index lastCellQ = (static_cast<Index>(nCells) - 1) * (3 * (k + 1)) + (k + 1);
    BOOST_TEST(dFdt.segment(lastCellQ, k + 1).cwiseAbs().maxCoeff() > 1e-6,
               "the moving Dirichlet end left no trace in the q row of the cell it "
               "borders, so RF_cellwise is not being differenced");

    // Restored, bit for bit. A ScopeGuard rather than a trailing call, so that an
    // exception out of residual() cannot leave them at t - h either.
    BOOST_TEST(sys.RF_cellwise.size() == rfBefore.size());
    for (size_t i = 0; i < rfBefore.size(); ++i)
        BOOST_TEST((sys.RF_cellwise[i] - rfBefore[i]).cwiseAbs().maxCoeff() == 0.0,
                   "RF_cellwise[" << i
                                  << "] was left at t - h, so every later residual "
                                     "evaluation reads the wrong boundary data");
    BOOST_TEST((sys.L_global - lBefore).cwiseAbs().maxCoeff() == 0.0,
               "L_global was left at t - h");

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
}

BOOST_AUTO_TEST_CASE(an_armed_gate_is_given_a_consistent_initial_condition)
{
    // The gate differentiates the initial condition, so it is only as good as the
    // state initialize() left. Two paths now skip IDACalcIC -- a steady solve and
    // a restart, unless ForceConsistentIC puts it back -- and on the uncorrected
    // guess dG/dt can come out
    // with the wrong *sign*. Since the gate rejects on dGdt < -tol, that abandons
    // runs which should proceed. So an armed gate forces IDACalcIC whatever else
    // initialize() would have done.
    //
    // AuxDiffusion and ScalarDiffusion are the fixtures where it shows: their
    // initial conditions contradict their own lower Dirichlet boundary, so
    // IDACalcIC has real work to do. TestDiffusion does not -- its two states
    // agree to 3.6e-16 -- which is exactly why the gate tests in
    // SolverLifecycleTests.cpp, which use it, never noticed.
    //
    // Both halves are asserted. That the corrected value is the right one is not
    // this test's claim to make; it_agrees_with_the_derivative_one_ida_step_in is
    // the case that compares dG/dt against a different method entirely, and it
    // was that comparison -- 1.6536551 and 1.6536562 at two step sizes against
    // +1.65366 corrected and -1.76887 from the guess -- which settled it.
    auto verdict = [](auto &problem, bool steady, bool armed)
    {
        Grid grid(0.0, 1.0, nCells);
        UIntegralObjective objective(1.0);
        SystemSolver sys(grid, k, &problem);
        configure(sys, "gate_consistency");
        sys.setAdjointProblem(&objective);
        if (armed)
            sys.setObjectiveDecreaseTolerance(1e-3);
        if (steady)
        {
            sys.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
            sys.setSteadyStateTolerance(1e-10);
        }

        struct Result { bool corrected; double dGdt; } out{false, 0.0};
        {
            CapturedOutput quiet;
            sys.initialize();
            out.corrected = sys.initialConditionWasCorrected();
            sys.computeAlgebraicTimeDerivatives();
            out.dGdt = sys.dGdt(0, sys.y, sys.dydtComplete);
            sys.destroySundials();
        }
        return out;
    };

    {
        AuxDiffusion timeMarch, steadyArmed, steadyBare;
        const auto march = verdict(timeMarch, false, true);
        const auto armed = verdict(steadyArmed, true, true);
        const auto bare = verdict(steadyBare, true, false);

        BOOST_TEST_MESSAGE("AuxDiffusion, G = Int u dx: time-march " << march.dGdt
                           << ", steady+gate " << armed.dGdt
                           << ", steady without the gate " << bare.dGdt);

        BOOST_TEST(march.corrected, "a time-marching run stopped running IDACalcIC");
        BOOST_TEST(armed.corrected,
                   "the gate is armed on a steady solve and IDACalcIC was still skipped");
        BOOST_TEST(!bare.corrected,
                   "a steady solve with no gate armed ran IDACalcIC, so this case "
                   "cannot show that the gate is what forces it");

        // The gate now sees what a time-marching run would have seen.
        BOOST_TEST(armed.dGdt == march.dGdt, boost::test_tools::tolerance(1e-12));

        // Not vacuous: without the gate the two genuinely disagree, in sign.
        BOOST_TEST(armed.dGdt * bare.dGdt < 0.0,
                   "the corrected and uncorrected states give " << armed.dGdt << " and "
                       << bare.dGdt << ", which no longer straddle zero -- this fixture "
                       "can no longer tell a fixed gate from a broken one");
    }

    {
        ScalarDiffusion timeMarch, steadyArmed, steadyBare;
        const auto march = verdict(timeMarch, false, true);
        const auto armed = verdict(steadyArmed, true, true);
        const auto bare = verdict(steadyBare, true, false);

        BOOST_TEST_MESSAGE("ScalarDiffusion, G = Int u dx: time-march " << march.dGdt
                           << ", steady+gate " << armed.dGdt
                           << ", steady without the gate " << bare.dGdt);

        BOOST_TEST(armed.corrected);
        BOOST_TEST(!bare.corrected);
        BOOST_TEST(armed.dGdt == march.dGdt, boost::test_tools::tolerance(1e-12));
        BOOST_TEST(armed.dGdt * bare.dGdt < 0.0,
                   "the corrected and uncorrected states give " << armed.dGdt << " and "
                       << bare.dGdt << ", which no longer straddle zero");
    }
}

BOOST_AUTO_TEST_SUITE_END()
