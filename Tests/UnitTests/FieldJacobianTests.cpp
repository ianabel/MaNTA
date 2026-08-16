// The coupled Jacobian solve, checked the way SolveJacTests checks the
// uncoupled one: finite-difference the whole residual and require J dy = g.
//
// This is the ONLY test that can catch a wrong coupling block. A sign error in
// A1 or A2 leaves the answer correct -- the Jacobian is never assembled, so an
// error in it costs Newton speed and nothing else. The manufactured coupled run
// in CoupledResidualTests will not see it, and the regression suite will not see
// it. Hence a_sign_error_in_a1_would_be_caught below, which requires the check
// to *fail* on a perturbed A1: without that, a coupling that was never exercised
// would make the two solve tests pass vacuously.
#include <boost/test/unit_test.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "CapturedOutput.hpp"
#include "FiniteDifferenceJacobian.hpp"
#include "MMSHarness.hpp"
#include "ManufacturedFields.hpp"
#include "../../SystemSolver.hpp"
#include "../../Types.hpp"

#include <nvector/nvector_serial.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

namespace
{

using std::numbers::pi;

// ------------------------------------------------------------- physics cases --
//
// Both read the geometry slot the field models supply, and both do so
// *nonlinearly* in g. That is deliberate on two counts. A case whose flux were
// linear in g would have a dSigmaFn_dGeometry independent of the state, so A1
// could be assembled from a State with no geometry rows at all and nothing would
// notice -- which is the exact confusion setFieldModel refuses for the scalar
// path. And g enters through psi, so d(sigma_hat)/dg being a function of g is
// what makes the chain rule through dGeometry/dpsi carry real information.

// sigma_hat = g^2 q + phi/2 ,  S = g u + 0.2 q ,  G = phi - g u
//
// The aux constraint is here rather than in a separate test because A1 has three
// row blocks -- sigma, u and aux -- and only the first two are reachable without
// nAux > 0. The u block is also the only one carrying a sign of its own (the
// residual forms it as `- Pi(S)`), so exercising all three at once is what makes
// the J dy = g check a statement about the whole block rather than two thirds of
// it.
class GeometricAuxDiffusion : public TransportSystem
{
public:
    GeometricAuxDiffusion()
        : TransportSystem({.variables = {{"u", "the diffused quantity", "",
                                          BoundaryKind::Dirichlet, BoundaryKind::Dirichlet}},
                           .aux = {{"phi", "an auxiliary quantity", ""}}})
    {
    }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        const double g = s.geom(0);
        return g * g * s.q(0) + 0.5 * s.phi(0);
    }
    Value Sources(Index, const State &s, Position x, Time) override
    {
        return s.geom(0) * s.u(0) + 0.2 * s.q(0) + std::sin(3.0 * x);
    }
    Value AuxG(Index, const State &s, Position, Time) override
    {
        return s.phi(0) - s.geom(0) * s.u(0);
    }

    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = s.geom(0) * s.geom(0);
    }
    void dSigma_dPhi(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.5; }

    void dSources_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = s.geom(0);
    }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.2; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dPhi(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    void AuxGPrime(Index, State &out, const State &s, Position, Time) override
    {
        out.u(0) = -s.geom(0);
        out.q(0) = 0.0;
        out.sigma(0) = 0.0;
        out.phi(0) = 1.0;
    }

    // The three that make A1 nonzero.
    void dSigmaFn_dGeometry(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 2.0 * s.geom(0) * s.q(0);
    }
    void dSources_dGeometry(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = s.u(0);
    }
    void dAuxG_dGeometry(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = -s.u(0);
    }

    Value InitialValue(Index, Position x) const override { return std::sin(pi * x); }
    Value InitialDerivative(Index, Position x) const override { return pi * std::cos(pi * x); }
    Value InitialAuxValue(Index, Position x) const override { return 0.3 * std::sin(pi * x); }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
};

// The same physics without the auxiliary variable, so that a failure in the
// aux row block is distinguishable from a failure in the coupling itself.
class GeometricDiffusion : public TransportSystem
{
public:
    GeometricDiffusion()
        : TransportSystem({.variables = {{"u", "the diffused quantity", "",
                                          BoundaryKind::Dirichlet, BoundaryKind::Dirichlet}}})
    {
    }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        const double g = s.geom(0);
        return g * g * s.q(0);
    }
    Value Sources(Index, const State &s, Position x, Time) override
    {
        return s.geom(0) * s.u(0) + 0.2 * s.q(0) + std::sin(3.0 * x);
    }

    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = s.geom(0) * s.geom(0);
    }
    void dSources_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = s.geom(0);
    }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.2; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    void dSigmaFn_dGeometry(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 2.0 * s.geom(0) * s.q(0);
    }
    void dSources_dGeometry(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = s.u(0);
    }

    Value InitialValue(Index, Position x) const override { return std::sin(pi * x); }
    Value InitialDerivative(Index, Position x) const override { return pi * std::cos(pi * x); }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
};

// ------------------------------------------------------------------ fixtures --

// The grid the two manufactured models take and ignore. Static so the reference
// their constructor is handed outlives every model built from it.
Grid const &scratchGrid()
{
    static const Grid g(0.0, 1.0, 1);
    return g;
}

// ------------------------------------------------------------- field models --

// A2 has four slots -- sigma, q, u and aux -- and every FieldResidualPrime in
// the tree writes only the u one. ManufacturedField and ManufacturedFieldVector
// both constrain psi to an integral of u; DifferentialManufacturedField and the
// registry probe write no state derivative at all. So the other three lines of
// the A2 assembly are identically zero in every other fixture, and replacing
// them with `= 0.0` leaves the whole suite green -- exactly the
// plausible-block-nothing-can-catch failure this task exists to close, sitting
// inside the block its own guard is meant to protect.
// (a_sign_error_in_a2_would_be_caught negates whatever is *in* a2, so it
// certifies the u slot and nothing else.)
//
// This model constrains psi to an integral that reads all four:
//
//     R = psi - Int ( u + 1/2 sigma + 1/4 q + 3/4 phi ) dx
//
// so d R / d(DOF) is a nonzero multiple of the quadrature weights in each, and
// the existing J dy = g machinery reaches them with no new check.
//
// The coefficients are distinct and none is 1, so a slot sourced from the wrong
// `dR` field -- Flux where Derivative was meant, say -- is a wrong *number*
// rather than an accidentally right one.
//
// Only usable with a physics case carrying nAux >= 1: `s.phi(0)` is
// bounds-checked under DEBUG only. Hence a local fixture beside
// GeometricAuxDiffusion rather than an addition to ManufacturedFields.hpp,
// which two other test files share.
// The `strength` dial multiplies the whole integral, so the constraint is
//
//     R = psi - strength * Int ( u + 1/2 sigma + 1/4 q + 3/4 phi ) dx
//
// and B = dRdpsi = 1 is untouched by it while A2 is proportional to it. The block
// Gauss-Seidel iteration matrix M = B^-1 A2 A^-1 A1 is therefore linear in it:
// the dial is predictable, so a test can *measure* rho and assert the fixture is
// the regime it claims rather than assuming a pairing is hard enough.
class ManufacturedFieldEverySlot : public ManufacturedField
{
public:
    explicit ManufacturedFieldEverySlot(double strength_ = 1.0)
        : ManufacturedField(toml::value{}, scratchGrid()), strength(strength_)
    {
    }

    static constexpr double wSigma = 0.5, wQ = 0.25, wPhi = 0.75;

    void FieldResidual(VectorRef out, Vector const &psi, Vector const &,
                       GlobalState const &states, std::vector<Position> const &,
                       Vector const &weights, Time) override
    {
        double integral = 0.0;
        for (Index j = 0; j < weights.size(); ++j)
        {
            const State s = states[j];
            integral += weights(j) *
                        (s.u(0) + wSigma * s.sigma(0) + wQ * s.q(0) + wPhi * s.phi(0));
        }
        out(0) = psi(0) - strength * integral;
    }

    void FieldResidualPrime(GlobalStateMatrix &dR, GlobalStateMatrix &, MatrixRef dRdpsi,
                            MatrixRef, Vector const &, Vector const &, GlobalState const &,
                            std::vector<Position> const &, Vector const &weights, Time) override
    {
        dRdpsi(0, 0) = 1.0;

        // Through GlobalState's whole-matrix accessors, not dR[0][j].u(0):
        // operator[](Index) returns a State by value, so writing through it
        // compiles and modifies nothing. See ManufacturedField's own comment.
        dR[0].Variable().row(0) = -strength * weights.transpose();
        dR[0].Flux().row(0) = -strength * wSigma * weights.transpose();
        dR[0].Derivative().row(0) = -strength * wQ * weights.transpose();
        dR[0].Aux().row(0) = -strength * wPhi * weights.transpose();
    }

private:
    double strength;
};

// Everything one Jacobian evaluation needs, in one owner.
//
// Declaration order is load bearing: the destructor body frees the two
// N_Vectors, which has to happen before ~SystemSolver frees the SUNContext they
// were made in, and members are destroyed after the body in reverse order, so
// `sys` outlives them and `problem` / `grid` outlive `sys`.
class CoupledSolver
{
public:
    CoupledSolver() = default;
    CoupledSolver(CoupledSolver const &) = delete;
    CoupledSolver &operator=(CoupledSolver const &) = delete;
    CoupledSolver(CoupledSolver &&) = default;

    ~CoupledSolver()
    {
        for (N_Vector v : {Y, dYdt})
            if (v)
                N_VDestroy(v);
    }

    SystemSolver &operator*() const { return *sys; }
    SystemSolver *operator->() const { return sys.get(); }

    std::unique_ptr<TransportSystem> problem;
    std::shared_ptr<FieldModel> field;
    std::unique_ptr<Grid> grid;
    std::unique_ptr<SystemSolver> sys;
    N_Vector Y = nullptr, dYdt = nullptr;
    double cj = 0.0, t = 0.0;
};

/// A coupled solver sitting at a state, with its Jacobian blocks built.
///
/// psi is pushed off the value InitialFieldValue returns, so the state is not a
/// solution of the field row and the field column of the Jacobian is not being
/// evaluated at a degenerate point.
CoupledSolver makeCoupledSolverAtState(Index nCells, Index k,
                                       std::unique_ptr<TransportSystem> problem,
                                       std::shared_ptr<FieldModel> model,
                                       bool superconvergent = false, double tau = 0.75,
                                       double cj = 1.7)
{
    const double t = 0.0;

    CoupledSolver h;
    h.problem = std::move(problem);
    h.field = std::move(model);
    h.grid = std::make_unique<Grid>(0.0, 1.0, nCells);
    h.sys = std::make_unique<SystemSolver>(*h.grid, k, h.problem.get());
    h.sys->setTau(tau);
    h.sys->setInitialTime(t);
    h.sys->setSuperconvergent(superconvergent);
    h.sys->setFieldModel(h.field);
    h.sys->resetCoeffs();
    h.sys->initialiseMatrices();

    const size_t dof = h.sys->getSolution().getDoF();
    h.Y = N_VNew_Serial(dof, h.sys->ctx);
    h.dYdt = N_VClone(h.Y);
    N_VConst(0.0, h.Y);
    N_VConst(0.0, h.dYdt);
    h.sys->setInitialConditions(h.Y, h.dYdt);

    DGSoln yMap(h.sys->nVars, *h.grid, k, N_VGetArrayPointer(h.Y), h.sys->nScalars,
                h.sys->nAux, h.sys->getFieldDOF());
    for (Index m = 0; m < h.sys->getFieldDOF(); ++m)
        yMap.Field(m) += 0.2 + 0.05 * static_cast<double>(m);

    h.cj = cj;
    h.t = t;

    h.sys->setJacTime(t);
    h.sys->setAlpha(cj);
    h.sys->setJacEvalY(h.Y, h.dYdt);
    h.sys->updateBoundaryConditions(t);
    h.sys->updateMatricesForJacSolve();

    return h;
}

CoupledSolver singleDofFixture(Index nCells, Index k, bool superconvergent = false)
{
    return makeCoupledSolverAtState(
        nCells, k, std::make_unique<GeometricAuxDiffusion>(),
        std::make_shared<ManufacturedField>(toml::value{}, scratchGrid()), superconvergent);
}

CoupledSolver everySlotFixture(Index nCells, Index k, double strength = 1.0)
{
    return makeCoupledSolverAtState(nCells, k, std::make_unique<GeometricAuxDiffusion>(),
                                    std::make_shared<ManufacturedFieldEverySlot>(strength));
}

CoupledSolver multiDofFixture(Index nCells, Index k, double strength = 1.0)
{
    return makeCoupledSolverAtState(
        nCells, k, std::make_unique<GeometricDiffusion>(),
        std::make_shared<ManufacturedFieldVector>(toml::value{}, scratchGrid(), strength));
}

/// |lambda_max| of the block Gauss-Seidel iteration matrix M = B^-1 A2 A^-1 A1,
/// by power iteration.
///
/// With a zero right-hand side the affine map's constant term vanishes and one
/// sweep is exactly one application of M, so the ratio of successive norms
/// converges to |lambda_max|. Built from the same four calls
/// solveCoupledJacIterative makes, in the same order, rather than from a second
/// implementation that could agree with a wrong original.
double spectralRadius(CoupledSolver const &solver, int iterations = 60)
{
    const Index nField = solver->getFieldDOF();
    N_Vector work = N_VClone(solver.Y), dx = N_VClone(solver.Y);

    Vector p = Vector::Ones(nField).normalized();
    double ratio = 0.0;
    for (int it = 0; it < iterations; ++it)
    {
        N_VConst(0.0, work);
        solver->subtractA1Times(p, work);
        solver->solveTransportJac(work, dx);

        Vector r2 = Vector::Zero(nField);
        for (Index f = 0; f < nField; ++f)
            r2(f) -= N_VDotProd(solver->a2[f], dx);

        Vector next(nField);
        solver->fieldModel->solveB(next, r2);
        ratio = next.norm(); // p is a unit vector on every pass
        p = next.normalized();
    }
    N_VDestroy(work);
    N_VDestroy(dx);
    return ratio;
}

/// The largest coefficient anywhere in A1, over every cell. A zero A1 is how
/// Task 9's sign experiment came to be a statement about a zero matrix, so every
/// fixture that claims a coupling reports this.
double maxAbsA1(CoupledSolver const &solver)
{
    double m = 0.0;
    for (Matrix const &block : solver->A1_cellwise)
        m = std::max(m, block.cwiseAbs().maxCoeff());
    return m;
}

/// The components of the increment the finite-differenced Jacobian says nothing
/// about.
///
/// residual() never writes a Dirichlet trace row -- that constraint is imposed
/// inside the linear solve -- and the assembly zeroes the same trace unknown's
/// *column* wherever it appears, so J has an exactly zero row and an exactly
/// zero column there. The system has a null direction, and what the
/// factorisation returns for that component is arbitrary. Both vectors are
/// zeroed there before they are compared; the REQUIRE on the column is what
/// makes that legitimate rather than convenient, since a zero row alone would
/// only mean the component is unconstrained, not that it is irrelevant.
std::vector<Index> decoupledComponents(Matrix const &J)
{
    std::vector<Index> rows = fdjac::undefinedRows(J);
    for (Index i : rows)
        BOOST_REQUIRE_EQUAL(J.col(i).cwiseAbs().maxCoeff(), 0.0);
    return rows;
}

void zeroAt(Vector &v, std::vector<Index> const &components)
{
    for (Index i : components)
        v(i) = 0.0;
}

struct SolveReport
{
    double solutionError; ///< ||recovered - dy|| / ||dy||
    double residual;      ///< ||J recovered - g|| / ||g||, over the defined rows
    Index nDirichlet;     ///< how many components were excluded
};

/// Manufacture an increment, push it through J, and ask the exact coupled solve
/// to recover it.
SolveReport checkExactSolve(CoupledSolver &h, int trial)
{
    SystemSolver &sys = *h.sys;

    // Includes the field rows and columns: fdjac::jacobian perturbs every entry
    // of the vector IDA is handed, and psi is in that vector.
    const Matrix J = fdjac::jacobian(sys, h.Y, h.dYdt, h.t, h.cj);
    const Index n = J.rows();
    const std::vector<Index> dead = decoupledComponents(J);

    Vector dy(n);
    for (Index i = 0; i < n; ++i)
        dy(i) = std::sin(1.0 + static_cast<double>(i) * (trial + 1) * 0.7);
    zeroAt(dy, dead);

    const Vector g = J * dy;

    N_Vector gVec = N_VNew_Serial(n, sys.ctx);
    N_Vector out = N_VClone(gVec);
    std::copy(g.data(), g.data() + n, N_VGetArrayPointer(gVec));
    N_VConst(0.0, out);

    sys.solveCoupledJacExact(gVec, out);

    Vector recovered = fdjac::toVector(out);
    zeroAt(recovered, dead);

    N_VDestroy(gVec);
    N_VDestroy(out);

    return {(recovered - dy).norm() / dy.norm(),
            fdjac::relativeResidual(J, recovered, g, dead),
            static_cast<Index>(dead.size())};
}

/// The largest coefficient of one field of a DGSoln, over every cell.
double maxAbs(DGSoln::DGApprox &f, Index nCells)
{
    double m = 0.0;
    for (Index i = 0; i < nCells; ++i)
        m = std::max(m, f.getCoeff(i).second.cwiseAbs().maxCoeff());
    return m;
}

/// Configure far enough for initialize() to run, but not to write anything.
void configureQuietly(SystemSolver &sys, std::string const &stem)
{
    sys.setTau(1.0);
    sys.setInputFile(stem);
    sys.setOutputCadence(1.0);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-14);
    sys.setTolerances({1e-8}, 1e-6);
    sys.setWriteOutput(false);
    sys.setWriteDatFile(false);
}

// ------------------------------------------------------ the iterative path --
//
// A pseudo-random right-hand side, the full length of the coupled vector --
// deliberately not shaped like a residual, so agreeing on it is a statement
// about the linear solve alone.
N_Vector randomRHS(SystemSolver &sys)
{
    const Index n = static_cast<Index>(sys.getSolution().getDoF());
    N_Vector g = N_VNew_Serial(n, sys.ctx);
    double *a = N_VGetArrayPointer(g);
    for (Index i = 0; i < n; ++i)
        a[i] = std::sin(0.37 + 1.618 * static_cast<double>(i)) *
               (1.0 + 0.1 * static_cast<double>(i % 7));
    return g;
}

/// ||a - b|| / ||a||, for comparing two solves against each other rather than
/// against a finite-differenced Jacobian.
///
/// The floor is the smallest positive double, there only to keep a genuinely
/// zero `a` from dividing by zero -- not `max(.., 1.0)`. A fixed floor of 1
/// is the same shape of bug the iterative solve's own stopping criterion had
/// (Task 9 review, finding 1): it silently turns a *relative* comparison into
/// an *absolute* one for any `a` smaller than the floor, which is exactly the
/// small-correction regime this solve exists to be checked in.
double relativeDifference(N_Vector a, N_Vector b)
{
    const Vector va = fdjac::toVector(a);
    const Vector vb = fdjac::toVector(b);
    return (va - vb).norm() / std::max(va.norm(), std::numeric_limits<double>::min());
}

// A local copy of CoupledResidualTests.cpp's manufactured coupled problem --
// same exact solution, u = sin(pi x)(1+t) and psi = Int u dx = (2/pi)(1+t) --
// kept here rather than shared, the way this file's own configureQuietly
// already duplicates that file's, so an end-to-end run on the iterative path
// can be checked against the same closed form the exact path already reaches.
inline Value manufacturedCoupledSource(Position x, Time t)
{
    const double A = 1.0 + t;
    const double s = std::sin(pi * x), c = std::cos(pi * x);
    const double psi = manufacturedPsiExact(t);
    return s + A * pi * pi * s * (1.0 + 2.0 * psi * c);
}

class ManufacturedCoupledDiffusion : public TransportSystem
{
public:
    ManufacturedCoupledDiffusion()
        : TransportSystem({.variables = {{"u", "the diffused quantity", "",
                                          BoundaryKind::Dirichlet, BoundaryKind::Dirichlet}}})
    {
    }

    Value SigmaFn(Index, const State &s, Position, Time) override { return s.geom(0) * s.q(0); }
    Value Sources(Index, const State &, Position x, Time t) override
    {
        return manufacturedCoupledSource(x, t);
    }

    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = s.geom(0);
    }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    // SigmaFn reads geometry (s.geom(0) * s.q(0)), so this case has a nonzero
    // A1 block -- without this override d(sigma_hat)/dg is the TransportSystem
    // default of zero, which makes A1_cellwise identically zero for this
    // fixture and silently turns the iterative solve's end-to-end run into one
    // with no coupling to get wrong. d(g q)/dg = q; Sources and AuxG have no
    // geometry dependence, so their _dGeometry hooks stay at the base class's
    // zero default.
    void dSigmaFn_dGeometry(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = s.q(0);
    }

    Value InitialValue(Index, Position x) const override { return manufacturedU(x, 0.0); }
    Value InitialDerivative(Index, Position x) const override { return pi * std::cos(pi * x); }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
};

/// Integrate the manufactured coupled problem to tFinal under a chosen
/// FieldSolve mode, and hand back the run. This file's whole reason to exist is
/// comparing Iterative against Exact, so the mode is a parameter here rather
/// than fixed the way CoupledResidualTests.cpp's own runCoupledToTime is.
CoupledSolver runCoupledToTime(Index nCells, Index k, double tFinal,
                               std::string const &fieldSolve)
{
    CoupledSolver h;
    h.problem = std::make_unique<ManufacturedCoupledDiffusion>();
    h.field = std::make_shared<ManufacturedField>(toml::value{}, scratchGrid());
    h.grid = std::make_unique<Grid>(0.0, 1.0, nCells);
    h.sys = std::make_unique<SystemSolver>(*h.grid, k, h.problem.get());
    configureQuietly(*h.sys, "field_jac_iterative_mms_k" + std::to_string(k) + "_n" +
                                 std::to_string(nCells));
    h.sys->setOutputCadence(tFinal);
    h.sys->setFieldModel(h.field);
    h.sys->resetCoeffs();
    h.sys->setFieldSolveMode(fieldSolve == "exact" ? SystemSolver::FieldSolveMode::Exact
                                                    : SystemSolver::FieldSolveMode::Iterative);

    {
        // runSolver reports its step counts and IDACalcIC warnings; quiet the
        // same way CoupledResidualTests.cpp's runCoupledToTime does.
        CapturedOutput quiet;
        h.sys->initialize();
        h.sys->integrate(tFinal);
        h.sys->destroySundials();
    }
    return h;
}

/// L2 error of u against the manufactured solution -- the same rule
/// CoupledResidualTests.cpp's uError and MMSConvergenceTests measure with.
double uError(SystemSolver &sys, double t)
{
    return mms::l2ErrorAgainst([&](double x) { return sys.getSolution().u(0)(x); },
                               [](double x, double tt) { return manufacturedU(x, tt); },
                               sys.getSolution().getGrid(), t);
}

/// The residual of the *field* row of the coupled system for the pair a solve
/// returned:  || r2 - A2 dx - B dpsi ||  /  || r2 ||.
///
/// Not "small": **exactly zero, to roundoff**, and that is a far stronger
/// statement. It is the structural half of invariant 1 -- dpsi is accepted only
/// as B^-1(r2 - A2 dx) for the dx returned beside it, so row two holds
/// identically and row one is off by A1 . delta, i.e. by the tolerance. Returning
/// an *extrapolated* dpsi instead leaves row two off by B mu delta, which no
/// agreement-with-exact check can see -- the extrapolation is second order in
/// the increment at the point the sweep stops, so it moves the answer by far
/// less than the tolerance the comparison is made at, while destroying the
/// identity the adjoint's backward-error test is derived from. Only a residual
/// measured against roundoff separates the two.
///
/// a2 carries nothing past the cellwise transport blocks -- transposeFieldCoupling
/// throws if it ever does -- so the dot product below cannot pick up dpsi itself.
double fieldRowResidual(CoupledSolver const &solver, N_Vector rhsVec, N_Vector outVec)
{
    SystemSolver &sys = *solver;
    const Index nField = sys.getFieldDOF();
    DGSoln rhs(sys.nVars, *solver.grid, sys.k, N_VGetArrayPointer(rhsVec), sys.nScalars,
               sys.nAux, nField);
    DGSoln out(sys.nVars, *solver.grid, sys.k, N_VGetArrayPointer(outVec), sys.nScalars,
               sys.nAux, nField);

    Vector r2 = rhs.getField();
    for (Index f = 0; f < nField; ++f)
        r2(f) -= N_VDotProd(sys.a2[f], outVec);

    const Vector dpsi = out.getField();
    Vector Bdpsi = Vector::Zero(nField);
    sys.fieldModel->applyB(Bdpsi, dpsi);

    return (r2 - Bdpsi).norm() / std::max(rhs.getField().norm(),
                                          std::numeric_limits<double>::min());
}

} // namespace

BOOST_AUTO_TEST_SUITE(field_jacobian_tests)

BOOST_AUTO_TEST_CASE(the_exact_solve_inverts_a_finite_differenced_coupled_jacobian)
{
    auto solver = singleDofFixture(/*nCells=*/6, /*k=*/2);

    for (int trial = 0; trial < 3; ++trial)
    {
        const SolveReport r = checkExactSolve(solver, trial);
        BOOST_TEST_MESSAGE("single-DOF field, k = 2, nCells = 6, trial "
                           << trial << ": ||dy* - dy||/||dy|| = " << r.solutionError
                           << ", ||J dy* - g||/||g|| = " << r.residual << " ("
                           << r.nDirichlet << " Dirichlet components excluded)");
        BOOST_TEST(r.nDirichlet == 2);
        BOOST_TEST(r.solutionError < 1e-7);
        BOOST_TEST(r.residual < 1e-7);
    }
}

BOOST_AUTO_TEST_CASE(the_same_holds_for_a_multi_dof_field_block)
{
    // ManufacturedFieldVector is five coupled field unknowns with a tridiagonal
    // B and a dense dGeometry/dpsi, so the Schur complement is a real 5x5 solve
    // rather than a division.
    auto solver = multiDofFixture(/*nCells=*/6, /*k=*/2);
    BOOST_REQUIRE_EQUAL(solver->getFieldDOF(), ManufacturedFieldVector::N);

    for (int trial = 0; trial < 3; ++trial)
    {
        const SolveReport r = checkExactSolve(solver, trial);
        BOOST_TEST_MESSAGE("5-DOF field, k = 2, nCells = 6, trial "
                           << trial << ": ||dy* - dy||/||dy|| = " << r.solutionError
                           << ", ||J dy* - g||/||g|| = " << r.residual);
        BOOST_TEST(r.solutionError < 1e-7);
        BOOST_TEST(r.residual < 1e-7);
    }
}

BOOST_AUTO_TEST_CASE(the_a2_row_reaches_the_sigma_q_and_aux_slots)
{
    // Three quarters of the A2 assembly is unreached by every other fixture in
    // the tree, because every FieldResidualPrime that exists constrains psi to
    // an integral of u alone. Replacing the sigma, q and aux lines of
    // assembleFieldCoupling with `= 0.0` leaves the rest of this file and
    // coupled_residual_tests green. This is what closes that.
    const Index nCells = 6, k = 2;
    auto solver = everySlotFixture(nCells, k);

    // Structural first, so the check below cannot go quietly vacuous again the
    // way it did before: every slot of the A2 row must actually carry
    // something. The expected magnitude is the model's coefficient times a
    // quadrature weight, so this is a statement about which slot was written
    // and not only about whether anything was.
    DGSoln a2Map(solver->nVars, *solver.grid, k, N_VGetArrayPointer(solver->a2[0]),
                 solver->nScalars, solver->nAux, solver->getFieldDOF());
    const double u = maxAbs(a2Map.u(0), nCells);
    BOOST_REQUIRE_GT(u, 0.0);
    BOOST_TEST_MESSAGE("A2 row 0, largest coefficient by slot: u = "
                       << u << ", sigma = " << maxAbs(a2Map.sigma(0), nCells)
                       << ", q = " << maxAbs(a2Map.q(0), nCells)
                       << ", aux = " << maxAbs(a2Map.Aux(0), nCells));
    BOOST_TEST(maxAbs(a2Map.sigma(0), nCells) ==
                   ManufacturedFieldEverySlot::wSigma * u,
               boost::test_tools::tolerance(1e-12));
    BOOST_TEST(maxAbs(a2Map.q(0), nCells) == ManufacturedFieldEverySlot::wQ * u,
               boost::test_tools::tolerance(1e-12));
    BOOST_TEST(maxAbs(a2Map.Aux(0), nCells) == ManufacturedFieldEverySlot::wPhi * u,
               boost::test_tools::tolerance(1e-12));

    // ...and then the same J dy = g check, which now exercises all four.
    for (int trial = 0; trial < 3; ++trial)
    {
        const SolveReport r = checkExactSolve(solver, trial);
        BOOST_TEST_MESSAGE("every-slot A2, k = 2, nCells = 6, trial "
                           << trial << ": ||dy* - dy||/||dy|| = " << r.solutionError
                           << ", ||J dy* - g||/||g|| = " << r.residual);
        BOOST_TEST(r.solutionError < 1e-7);
        BOOST_TEST(r.residual < 1e-7);
    }
}

BOOST_AUTO_TEST_CASE(the_superconvergent_coupling_is_checked_the_same_way)
{
    // The star-node branch of A1: the physics is evaluated at the k+2 star nodes
    // with u* in place of u_h, and A9 replaces InterpolateOntoBasis. Nothing
    // else in the suite reaches dPhysics_dField_StarMat.
    auto solver = singleDofFixture(/*nCells=*/4, /*k=*/2, /*superconvergent=*/true);

    const SolveReport r = checkExactSolve(solver, 0);
    BOOST_TEST_MESSAGE("superconvergent, k = 2, nCells = 4: ||dy* - dy||/||dy|| = "
                       << r.solutionError << ", ||J dy* - g||/||g|| = " << r.residual);
    BOOST_TEST(r.solutionError < 1e-7);
    BOOST_TEST(r.residual < 1e-7);
}

BOOST_AUTO_TEST_CASE(a_sign_error_in_a1_would_be_caught)
{
    // Guard against the two cases above passing vacuously: perturb A1 and
    // require the check to fail. If this does not fail, the coupling is not
    // being exercised and neither of them means anything.
    //
    // The perturbation reaches A1_cellwise directly rather than through a
    // production mutator: this is a -DTEST build, where MANTA_TEST_PRIVATE has
    // widened SystemSolver's private members to public. Nothing test-only is
    // added to the shipped class.
    auto solver = singleDofFixture(/*nCells=*/6, /*k=*/2);

    const SolveReport clean = checkExactSolve(solver, 0);
    BOOST_REQUIRE(clean.solutionError < 1e-7);

    for (auto &block : solver->A1_cellwise)
    {
        BOOST_REQUIRE_GT(block.cwiseAbs().maxCoeff(), 0.0);
        block *= -1.0;
    }

    const SolveReport flipped = checkExactSolve(solver, 0);
    BOOST_TEST_MESSAGE("with A1 negated: ||dy* - dy||/||dy|| = "
                       << flipped.solutionError << " (was " << clean.solutionError << ")");
    BOOST_TEST(flipped.solutionError > 1e-3);
    BOOST_TEST(flipped.residual > 1e-3);
}

BOOST_AUTO_TEST_CASE(a_sign_error_in_a2_would_be_caught)
{
    // The same guard for the other coupling block. A2 and A1 enter the Schur
    // complement as a product, so a test that only perturbed one of them would
    // leave open the possibility that the other is never read.
    auto solver = singleDofFixture(/*nCells=*/6, /*k=*/2);

    const SolveReport clean = checkExactSolve(solver, 0);
    BOOST_REQUIRE(clean.solutionError < 1e-7);

    for (Index f = 0; f < solver->getFieldDOF(); ++f)
        N_VScale(-1.0, solver->a2[f], solver->a2[f]);

    const SolveReport flipped = checkExactSolve(solver, 0);
    BOOST_TEST_MESSAGE("with A2 negated: ||dy* - dy||/||dy|| = " << flipped.solutionError);
    BOOST_TEST(flipped.solutionError > 1e-3);
}

BOOST_AUTO_TEST_CASE(solve_jac_eq_dispatches_to_the_exact_solve)
{
    // The dispatcher, not the solve: solveJacEq must route a coupled system to
    // solveCoupledJacExact and reproduce it bit for bit. Anything left in
    // solveJacEq that touched the field block afterwards -- Task 6's
    // block-Jacobi solveB call, for instance -- would show up here and nowhere
    // else, because every other test in this file calls the exact solve
    // directly.
    auto solver = singleDofFixture(/*nCells=*/4, /*k=*/1);
    SystemSolver &sys = *solver;
    sys.setFieldSolveMode(SystemSolver::FieldSolveMode::Exact);

    const Index n = static_cast<Index>(sys.getSolution().getDoF());
    N_Vector g = N_VNew_Serial(n, sys.ctx);
    N_Vector viaDispatch = N_VClone(g), viaExact = N_VClone(g);
    double *ga = N_VGetArrayPointer(g);
    for (Index i = 0; i < n; ++i)
        ga[i] = std::cos(0.3 * static_cast<double>(i));

    sys.solveJacEq(g, viaDispatch);
    sys.solveCoupledJacExact(g, viaExact);

    const double *a = N_VGetArrayPointer(viaDispatch);
    const double *b = N_VGetArrayPointer(viaExact);
    for (Index i = 0; i < n; ++i)
        BOOST_TEST(a[i] == b[i]);

    N_VDestroy(g);
    N_VDestroy(viaDispatch);
    N_VDestroy(viaExact);
}

BOOST_AUTO_TEST_CASE(selecting_the_exact_solve_warns)
{
    // The cost of the exact path is not visible in its answers, so it has to be
    // said. Once per run, from initialize(), where nField is known -- rather
    // than from applySolverConfig, which would announce the cost of a solve that
    // will never happen on a run with no field model attached.
    auto solver = singleDofFixture(/*nCells=*/4, /*k=*/1);
    configureQuietly(*solver, "field_jac_warning");
    solver->setFieldSolveMode(SystemSolver::FieldSolveMode::Exact);

    std::string log;
    {
        CapturedOutput capture;
        solver->initialize();
        log = capture.text();
        solver->destroySundials();
    }

    BOOST_TEST(log.find("FieldSolve = exact") != std::string::npos);
    BOOST_TEST(log.find("verification") != std::string::npos);

    // ...and it has to discriminate: the default mode says something different.
    auto other = singleDofFixture(/*nCells=*/4, /*k=*/1);
    configureQuietly(*other, "field_jac_warning_default");
    BOOST_REQUIRE(other->getFieldSolveMode() == SystemSolver::FieldSolveMode::Iterative);

    std::string defaultLog;
    {
        CapturedOutput capture;
        other->initialize();
        defaultLog = capture.text();
        other->destroySundials();
    }

    BOOST_TEST(defaultLog.find("FieldSolve = exact") == std::string::npos);
    BOOST_TEST(defaultLog.find("FieldSolve = iterative") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(the_iterative_solve_agrees_with_the_exact_one)
{
    // Not to roundoff: the iterative path stops at FieldSolveTolerance, so the
    // agreement is to that tolerance and no better. What this pins is that the
    // two are solving the same system -- a sign error in either path's use of
    // A1 or A2 separates them immediately.
    auto solver = singleDofFixture(/*nCells=*/6, /*k=*/2);
    N_Vector g = randomRHS(*solver);

    N_Vector exact = N_VClone(g), iterative = N_VClone(g);
    solver->solveCoupledJacExact(g, exact);
    solver->solveCoupledJacIterative(g, iterative);

    const double diff = relativeDifference(exact, iterative);
    BOOST_TEST_MESSAGE("single-DOF field, iterative vs exact: relative difference = " << diff);
    BOOST_CHECK_SMALL(diff, 1e-6);

    N_VDestroy(g);
    N_VDestroy(exact);
    N_VDestroy(iterative);
}

BOOST_AUTO_TEST_CASE(the_iterative_solve_agrees_for_a_multi_dof_block_too)
{
    auto solver = multiDofFixture(/*nCells=*/6, /*k=*/2);
    N_Vector g = randomRHS(*solver);

    N_Vector exact = N_VClone(g), iterative = N_VClone(g);
    solver->solveCoupledJacExact(g, exact);
    solver->solveCoupledJacIterative(g, iterative);

    const double diff = relativeDifference(exact, iterative);
    BOOST_TEST_MESSAGE("5-DOF field, iterative vs exact: relative difference = " << diff);
    BOOST_CHECK_SMALL(diff, 1e-6);

    N_VDestroy(g);
    N_VDestroy(exact);
    N_VDestroy(iterative);
}

BOOST_AUTO_TEST_CASE(the_iterative_solve_agrees_at_the_scale_a_newton_step_lives_at)
{
    // The two tests above use a RHS whose converged |dpsi| happens to land
    // above 1, so `tol * max(1, |dpsi|)` and `tol * |dpsi|` coincide there and
    // neither test would have caught Task 9 review finding 1: the original
    // stopping criterion used `max(1, |dpsi|)` rather than `|dpsi|` alone, which
    // is an *absolute* magnitude test of `FieldSolveTolerance` whenever the
    // converged |dpsi| is below 1 -- exactly the regime a real Newton
    // correction lives in, where it declared convergence after the raw first
    // iterate regardless of how wrong that iterate was (measured at 38.6%
    // relative error on this same fixture before the fix). Scaling the RHS down
    // so |dpsi| is far below 1 is what exercises that regime; this is the
    // guard against reintroducing it.
    auto solver = singleDofFixture(/*nCells=*/6, /*k=*/2);
    N_Vector g = randomRHS(*solver);
    N_VScale(1e-9, g, g);

    N_Vector exact = N_VClone(g), iterative = N_VClone(g);
    solver->solveCoupledJacExact(g, exact);
    solver->solveCoupledJacIterative(g, iterative);

    // relativeDifference's floor is the smallest positive double, not 1 -- see
    // its own comment -- so this is a genuine relative check even down here.
    BOOST_REQUIRE_GT(fdjac::toVector(exact).norm(), 0.0);
    const double diff = relativeDifference(exact, iterative);
    BOOST_TEST_MESSAGE("single-DOF field, RHS scaled by 1e-9, iterative vs exact: "
                       "relative difference = "
                       << diff);
    BOOST_CHECK_SMALL(diff, 1e-6);

    N_VDestroy(g);
    N_VDestroy(exact);
    N_VDestroy(iterative);
}

BOOST_AUTO_TEST_CASE(solve_jac_eq_dispatches_to_the_iterative_solve_by_default)
{
    // The other half of solve_jac_eq_dispatches_to_the_exact_solve: with no mode
    // set explicitly, solveJacEq must reach solveCoupledJacIterative and
    // reproduce it bit for bit, not silently fall back to the exact path the
    // way it did before this task.
    auto solver = singleDofFixture(/*nCells=*/4, /*k=*/1);
    SystemSolver &sys = *solver;
    BOOST_REQUIRE(sys.getFieldSolveMode() == SystemSolver::FieldSolveMode::Iterative);

    const Index n = static_cast<Index>(sys.getSolution().getDoF());
    N_Vector g = N_VNew_Serial(n, sys.ctx);
    N_Vector viaDispatch = N_VClone(g), viaIterative = N_VClone(g);
    double *ga = N_VGetArrayPointer(g);
    for (Index i = 0; i < n; ++i)
        ga[i] = std::cos(0.3 * static_cast<double>(i));

    sys.solveJacEq(g, viaDispatch);
    sys.solveCoupledJacIterative(g, viaIterative);

    const double *a = N_VGetArrayPointer(viaDispatch);
    const double *b = N_VGetArrayPointer(viaIterative);
    for (Index i = 0; i < n; ++i)
        BOOST_TEST(a[i] == b[i]);

    N_VDestroy(g);
    N_VDestroy(viaDispatch);
    N_VDestroy(viaIterative);
}

// ------------------------------------------------------- the accelerator --

BOOST_AUTO_TEST_CASE(irons_tuck_is_exact_for_a_scalar_affine_map)
{
    // p_{k+1} = c + m p_k, fixed point c/(1-m). Two plain steps from p_0 = 0,
    // then one accelerated step must land on it exactly -- for a contraction
    // and, the point of the exercise, for a divergent map too. m = -1.611 is
    // the measured dominant eigenvalue of RichGeometricDiffusion's coupling.
    for (double m : {0.33, -1.611, 2.5})
    {
        const double c = 0.7;
        auto G = [&](const Vector &p) { return Vector::Constant(1, c + m * p(0)); };

        Vector p0 = Vector::Zero(1);
        Vector g0 = G(p0), d0 = g0 - p0;
        Vector p1 = g0;
        Vector g1 = G(p1), d1 = g1 - p1;

        Vector acc = SystemSolver::ironsTuck(g1, d1, d0);
        BOOST_CHECK_CLOSE(acc(0), c / (1.0 - m), 1e-10);
    }
}

BOOST_AUTO_TEST_CASE(irons_tuck_declines_when_the_secant_vanishes)
{
    // Equal increments mean m == 1: a translation, no fixed point, and a zero
    // denominator. The plain iterate must come back, not a NaN.
    Vector g = Vector::Constant(3, 2.0), d = Vector::Constant(3, 0.5);
    Vector out = SystemSolver::ironsTuck(g, d, d);
    BOOST_CHECK(out.isApprox(g));

    // ...and the guard is relative: increments of 1e-30 that differ by 1e-31
    // are a perfectly good secant, not a vanishing one.
    Vector small = Vector::Constant(3, 1e-30), smaller = Vector::Constant(3, 9e-31);
    Vector accelerated = SystemSolver::ironsTuck(Vector::Constant(3, 1e-30), small, smaller);
    BOOST_CHECK(accelerated.allFinite());
    BOOST_CHECK(!accelerated.isApprox(Vector::Constant(3, 1e-30)));
}

BOOST_AUTO_TEST_CASE(acceleration_converges_a_sweep_that_plainly_diverges)
{
    // First: prove the fixture is what it claims. If a later change makes the
    // coupling weak, this fails here rather than turning the test below into a
    // statement about a contraction.
    auto solver = everySlotFixture(6, 2, 2.0);
    const double rho = spectralRadius(solver);
    BOOST_TEST_MESSAGE("every-slot, strength 2: rho(M) = " << rho
                       << ", max|A1_cellwise| = " << maxAbsA1(solver));
    BOOST_REQUIRE_GT(maxAbsA1(solver), 0.0);
    BOOST_CHECK_GT(rho, 1.2);

    N_Vector g = randomRHS(*solver);
    N_Vector exact = N_VClone(g), iterative = N_VClone(g);
    solver->solveCoupledJacExact(g, exact);
    solver->solveCoupledJacIterative(g, iterative);

    // The agreement is necessary but not sufficient -- solveCoupledJacIterative
    // now escalates to solveCoupledJacExact, so this line passes with the
    // acceleration deleted. The fallback count is the assertion that matters.
    BOOST_CHECK_SMALL(relativeDifference(exact, iterative), 1e-6);
    BOOST_CHECK_EQUAL(solver->fieldSweepFallbacks, 0);
    BOOST_TEST_MESSAGE("sweeps taken: " << solver->fieldSweepIterations);
    BOOST_CHECK_LE(solver->fieldSweepIterations, 12);

    N_VDestroy(g);
    N_VDestroy(exact);
    N_VDestroy(iterative);
}

BOOST_AUTO_TEST_CASE(the_fallback_returns_the_exact_solve_bit_for_bit)
{
    // A cap of one sweep cannot converge anything that needs two, so this
    // reaches the escalation deterministically without needing a pathological
    // fixture. Bit-for-bit, not to a tolerance: the escalation does not blend
    // the sweep's last iterate with the exact answer, it discards it.
    auto solver = everySlotFixture(6, 2, 2.0);
    solver->setFieldSolveMaxSweeps(1);

    N_Vector g = randomRHS(*solver);
    N_Vector exact = N_VClone(g), iterative = N_VClone(g);
    solver->solveCoupledJacExact(g, exact);
    solver->solveCoupledJacIterative(g, iterative);

    BOOST_CHECK_EQUAL(solver->fieldSweepFallbacks, 1);
    BOOST_CHECK_EQUAL(relativeDifference(exact, iterative), 0.0);

    N_VDestroy(g);
    N_VDestroy(exact);
    N_VDestroy(iterative);
}

BOOST_AUTO_TEST_CASE(acceleration_does_not_disturb_a_contraction)
{
    // The risk in adding an extrapolation is that it destabilises the cases
    // that were already fine. rho < 1 here, and the sweep must still converge,
    // still without falling back, and in no more sweeps than the plain
    // iteration needed (2, recorded in Task 9's review).
    auto solver = everySlotFixture(6, 2, 0.25);
    const double rho = spectralRadius(solver);
    BOOST_TEST_MESSAGE("every-slot, strength 0.25: rho(M) = " << rho
                       << ", max|A1_cellwise| = " << maxAbsA1(solver));
    BOOST_CHECK_LT(rho, 1.0);

    N_Vector g = randomRHS(*solver);
    N_Vector out = N_VClone(g);
    solver->solveCoupledJacIterative(g, out);

    BOOST_CHECK_EQUAL(solver->fieldSweepFallbacks, 0);
    BOOST_TEST_MESSAGE("sweeps taken: " << solver->fieldSweepIterations);
    BOOST_CHECK_LE(solver->fieldSweepIterations, 4);

    N_VDestroy(g);
    N_VDestroy(out);
}

BOOST_AUTO_TEST_CASE(a_divergent_five_dof_coupling_is_still_solved_exactly)
{
    // The case Irons-Tuck was not expected to fix. rank(M) <= nField = 5 here --
    // A1 accumulates a different rank-one term per quadrature point, because
    // dGeometry_dpsi is a function of x -- and a rank-one secant removes one
    // eigendirection, so a spectrum with several eigenvalues outside the unit
    // circle should defeat it.
    //
    // **Measured, it does not.** At strength 10 the iteration matrix has
    // rho = 1.57 -- within 3% of the 1.611 measured on RichGeometricDiffusion --
    // and the accelerated sweep still reaches FieldSolveTolerance, in 13 to 38
    // sweeps depending on the right-hand side against a plain sweep that
    // diverges. The secant is rank one per *step*, but the steps are not the
    // same rank-one direction, so the iteration is not confined to a
    // two-dimensional subspace the way the pessimistic argument assumes. Scanned
    // over strengths from 1 to 1e5 -- rho saturating just under 2 -- and over six
    // right-hand sides, it never once failed to converge given the sweeps.
    //
    // So both halves are asserted. The guarantee, first, because it is true by
    // construction and survives any change to the accelerator: whichever way
    // solveCoupledJacIterative got there, the answer is the exact one, which is
    // what makes iterative safe as a default. Then the measurement, with a cap
    // that has room for the observed spread -- the default 20 does *not*, which
    // is why this raises it rather than pretending otherwise.
    auto solver = multiDofFixture(6, 2, 10.0);
    const double rho = spectralRadius(solver);
    BOOST_TEST_MESSAGE("five-DOF, strength 10: rho(M) = " << rho
                       << ", max|A1_cellwise| = " << maxAbsA1(solver));
    BOOST_REQUIRE_GT(maxAbsA1(solver), 0.0);
    BOOST_CHECK_GT(rho, 1.2);

    N_Vector g = randomRHS(*solver);
    N_Vector exact = N_VClone(g), iterative = N_VClone(g);
    solver->solveCoupledJacExact(g, exact);

    // ---- the guarantee, at the default cap.
    solver->solveCoupledJacIterative(g, iterative);
    BOOST_CHECK_SMALL(relativeDifference(exact, iterative), 1e-6);
    BOOST_TEST_MESSAGE("five-DOF at the default cap: fallbacks = "
                       << solver->fieldSweepFallbacks
                       << ", sweeps = " << solver->fieldSweepIterations);

    // ---- and the route, given the sweeps. A fresh fixture rather than a reset,
    // because the counters accumulate over solves and only initialize() zeroes
    // them.
    auto roomy = multiDofFixture(6, 2, 10.0);
    roomy->setFieldSolveMaxSweeps(60);
    N_Vector g2 = randomRHS(*roomy), exact2 = N_VClone(g2), iterative2 = N_VClone(g2);
    roomy->solveCoupledJacExact(g2, exact2);
    roomy->solveCoupledJacIterative(g2, iterative2);

    BOOST_TEST_MESSAGE("five-DOF at a cap of 60: fallbacks = " << roomy->fieldSweepFallbacks
                       << ", sweeps = " << roomy->fieldSweepIterations);
    BOOST_CHECK_EQUAL(roomy->fieldSweepFallbacks, 0);
    BOOST_CHECK_SMALL(relativeDifference(exact2, iterative2), 1e-6);

    N_VDestroy(g);
    N_VDestroy(exact);
    N_VDestroy(iterative);
    N_VDestroy(g2);
    N_VDestroy(exact2);
    N_VDestroy(iterative2);
}

BOOST_AUTO_TEST_CASE(the_accepted_iterate_leaves_the_field_row_exact)
{
    // Invariant 1, checked where it can actually be seen. The sweep may
    // extrapolate as much as it likes, but only the *unaccelerated* iterate may
    // be accepted, because that is the one for which
    //
    //     B dpsi + A2 dx = r2
    //
    // holds identically rather than approximately. Both stopping tests are
    // derived from that identity -- forwards it is what makes the returned pair
    // consistent to within A1 . delta, and backwards it is what makes
    // ||A2^T dz|| the *exact* backward error rather than a proxy for one.
    //
    // Accepting the accelerated iterate instead is invisible to every other test
    // in this file, and measurably so: the extrapolation at the point the sweep
    // stops is second order in the increment, so the answer moves by roughly
    // 1e-12 relative -- six orders inside the 1e-6 the agreement checks use, and
    // four orders outside roundoff. Only this residual separates them.
    for (double strength : {0.25, 2.0})
    {
        auto solver = everySlotFixture(6, 2, strength);
        N_Vector g = randomRHS(*solver);
        N_Vector out = N_VClone(g);
        solver->solveCoupledJacIterative(g, out);

        const double r = fieldRowResidual(solver, g, out);
        BOOST_TEST_MESSAGE("every-slot, strength " << strength
                           << ": field-row residual = " << r);
        BOOST_CHECK_EQUAL(solver->fieldSweepFallbacks, 0);
        BOOST_CHECK_SMALL(r, 1e-14);

        N_VDestroy(g);
        N_VDestroy(out);
    }

    // The five-DOF block, where the accelerator is genuinely approximate and the
    // increment at the stopping point is not driven to exactly zero -- which is
    // what makes this the copy that discriminates.
    auto multi = multiDofFixture(6, 2, 10.0);
    multi->setFieldSolveMaxSweeps(60);
    N_Vector g = randomRHS(*multi);
    N_Vector out = N_VClone(g);
    multi->solveCoupledJacIterative(g, out);

    const double r = fieldRowResidual(multi, g, out);
    BOOST_TEST_MESSAGE("five-DOF, strength 10: field-row residual = " << r);
    BOOST_CHECK_EQUAL(multi->fieldSweepFallbacks, 0);
    BOOST_CHECK_SMALL(r, 1e-14);

    N_VDestroy(g);
    N_VDestroy(out);
}

BOOST_AUTO_TEST_CASE(the_accelerated_sweep_is_still_scale_equivariant)
{
    // Task 9's property, re-run rather than assumed: mu is a ratio of inner
    // products of quantities that all scale with the right-hand side, so
    // Irons-Tuck is homogeneous and solve(c g) == c solve(g) should survive it.
    // The scale that broke the old criterion was 1e-9; use it again.
    auto solver = everySlotFixture(6, 2, 2.0);
    N_Vector g = randomRHS(*solver), gSmall = N_VClone(g);
    N_VScale(1e-9, g, gSmall);

    N_Vector big = N_VClone(g), small = N_VClone(g);
    solver->solveCoupledJacIterative(g, big);
    solver->solveCoupledJacIterative(gSmall, small);
    N_VScale(1e9, small, small);

    // Neither solve may have escalated: the fallback is the exact solve, which
    // is scale-equivariant by construction, so a fallback would make this a
    // statement about the LU factorisation rather than about the sweep.
    BOOST_CHECK_EQUAL(solver->fieldSweepFallbacks, 0);
    const double diff = relativeDifference(big, small);
    BOOST_TEST_MESSAGE("scale equivariance at 1e-9: relative difference " << diff);
    BOOST_CHECK_SMALL(diff, 1e-8);

    N_VDestroy(g);
    N_VDestroy(gSmall);
    N_VDestroy(big);
    N_VDestroy(small);
}

BOOST_AUTO_TEST_CASE(the_end_of_run_report_counts_the_sweeps_and_names_the_fallbacks)
{
    // The only signal a user has that the coupling is not converging, and the
    // Task 9 deferred minor this closes: the sweep used to hit its cap silently,
    // once per Jacobian solve, with nothing anywhere saying so. The run is still
    // correct -- the fallback is the exact solve -- so this is a cost report
    // rather than an error, and it has to say what to do about it.
    CoupledSolver h;
    h.problem = std::make_unique<ManufacturedCoupledDiffusion>();
    h.field = std::make_shared<ManufacturedField>(toml::value{}, scratchGrid());
    h.grid = std::make_unique<Grid>(0.0, 1.0, 8);
    h.sys = std::make_unique<SystemSolver>(*h.grid, 2, h.problem.get());
    configureQuietly(*h.sys, "field_jac_sweep_report");
    h.sys->setOutputCadence(0.1);
    h.sys->setFieldModel(h.field);
    h.sys->resetCoeffs();

    // One sweep can never satisfy a relative increment test from a zero start, so
    // every Jacobian solve escalates -- IDACalcIC's included, which is why the
    // cap is set before initialize() rather than after: the two runs below have
    // to be configured identically for their counts to be comparable.
    h.sys->setFieldSolveMaxSweeps(1);

    std::string log;
    {
        CapturedOutput capture;
        h.sys->initialize();
        h.sys->integrate(0.1);
        log = capture.text();
    }

    const auto capped = h.sys->getFieldSweepStats();
    BOOST_TEST_MESSAGE("capped run: " << capped.iterations << " sweeps over " << capped.solves
                       << " solves, " << capped.fallbacks << " fallbacks");
    BOOST_REQUIRE_GT(capped.fallbacks, 0L);
    BOOST_TEST(log.find("Coupled field sweeps") != std::string::npos);
    BOOST_TEST(log.find("exact fallbacks") != std::string::npos);
    BOOST_TEST(log.find("Raise FieldSolveMaxSweeps") != std::string::npos);

    // ...and the counts are per *run*, not cumulative. A second integration on
    // the same solver must not report the first one's sweeps as its own -- which
    // is why they are zeroed in the unconditional part of initialize() rather
    // than in initialiseMatrices(), the function the `initialised` guard skips.
    h.sys->destroySundials();
    {
        CapturedOutput capture;
        h.sys->initialize();
        h.sys->integrate(0.1);
        log = capture.text();
        h.sys->destroySundials();
    }
    const auto second = h.sys->getFieldSweepStats();
    BOOST_TEST_MESSAGE("second run: " << second.iterations << " sweeps over " << second.solves
                       << " solves, " << second.fallbacks << " fallbacks");
    BOOST_TEST(second.solves == capped.solves);
    BOOST_TEST(second.iterations == capped.iterations);
    BOOST_TEST(second.fallbacks == capped.fallbacks);

    // ...and with the cap back at its default this fixture never escalates, so
    // the warning is the cap's doing rather than something always printed.
    auto clean = runCoupledToTime(/*nCells=*/8, /*k=*/2, /*tFinal=*/0.1, "iterative");
    BOOST_TEST(clean->getFieldSweepStats().fallbacks == 0L);
}

BOOST_AUTO_TEST_CASE(a_coupled_run_on_the_iterative_path_reaches_the_manufactured_solution)
{
    // The iterative solve is an *approximation to the Jacobian*, so an
    // under-converged sweep costs Newton iterations and nothing else. This
    // checks the answer is unmoved: same tolerances as
    // CoupledResidualTests.cpp's a_coupled_run_reaches_the_manufactured_solution,
    // which runs the same problem on FieldSolveMode::Exact.
    auto solver = runCoupledToTime(/*nCells=*/16, /*k=*/3, /*tFinal=*/0.5, "iterative");

    const double eu = uError(*solver, 0.5);
    const double epsi = std::abs(solver->getSolution().Field(0) - manufacturedPsiExact(0.5));
    BOOST_TEST_MESSAGE("iterative coupled MMS at t = 0.5, k = 3, nCells = 16:  |u| error "
                       << eu << ",  |psi| error " << epsi);

    BOOST_CHECK_SMALL(eu, 1e-4);
    BOOST_CHECK_SMALL(epsi, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
