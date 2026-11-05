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
#include "../../FieldModelEquilibrium.hpp"
#include "../../SystemSolver.hpp"
#include "../../PhysicsCases/ThetaPinchDiffusion.hpp"
#include "../../PhysicsCases/ThetaPinchEquilibrium.hpp"
#include "../../Types.hpp"

#include <nvector/nvector_serial.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <string>
#include <format>
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
    explicit GeometricAuxDiffusion(GeometryDerivatives derivatives = GeometryDerivatives::Analytic)
        : TransportSystem({.variables = {{"u", "the diffused quantity", "",
                                          BoundaryKind::Dirichlet, BoundaryKind::Dirichlet}},
                           .aux = {{"phi", "an auxiliary quantity", ""}},
                           .readsGeometry = true,
                           .geometryDerivatives = derivatives})
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
                                          BoundaryKind::Dirichlet, BoundaryKind::Dirichlet}},
                           .readsGeometry = true,
                           .geometryDerivatives = GeometryDerivatives::Analytic})
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
/// The registered pair: ThetaPinchDiffusion reading ThetaPinchEquilibrium.
/// The pressure is high enough (T = 0.3, central density 1) that the field is
/// far from vacuum, so the coupling blocks are not evaluated at a degenerate
/// point; the field block's size is nCells * intervalsPerCell + 4.
CoupledSolver thetaPinchFixture(Index nCells, Index k, Index intervalsPerCell, bool superconvergent = false,
                                double cj = 1.7, std::string const &derivatives = "Analytic")
{
    using namespace toml::literals::toml_literals;
    static const toml::value base = u8R"(
        [ThetaPinchDiffusion]
        Temperature = 0.3
        D0 = 0.5
        Gamma = 2.0
    )"_toml;
    toml::value config = base;
    config.as_table().at("ThetaPinchDiffusion").as_table()["GeometryDerivatives"] = derivatives;
    const Grid grid(0.0, 1.0, nCells);
    return makeCoupledSolverAtState(
        nCells, k, std::make_unique<ThetaPinchDiffusion>(config, grid),
        std::make_shared<ThetaPinchEquilibrium>(ThetaPinchEquilibrium::Parameters{.intervalsPerCell = intervalsPerCell}, grid),
        superconvergent, /*tau=*/0.75, cj);
}

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

/// The largest coefficient anywhere in A1, over every cell. A zero A1 is how a
/// sign experiment on the coupling once came to be a statement about a zero
/// matrix -- every assertion it made held, and held vacuously -- so every
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
//
// `seed` changes the *direction*, not the magnitude: it moves the phase and the
// frequency of the sine and the offset of the modulation. That distinction is
// the point rather than a flourish. Scaling the right-hand side cannot change
// how the sweep behaves at all -- the solve is homogeneous, which
// the_accelerated_sweep_is_still_scale_equivariant asserts to eight digits at
// 1e-9 -- so a "different" right-hand side obtained by scaling takes the
// identical number of sweeps and tells you nothing. Only a genuinely different
// direction moves the count, and for five field unknowns it moves it by a
// factor of three.
//
// Seed 0 reproduces the original single-right-hand-side generator exactly, so
// every call site that predates the parameter is unchanged.
N_Vector randomRHS(SystemSolver &sys, int seed = 0)
{
    const Index n = static_cast<Index>(sys.getSolution().getDoF());
    N_Vector g = N_VNew_Serial(n, sys.ctx);
    double *a = N_VGetArrayPointer(g);
    for (Index i = 0; i < n; ++i)
        a[i] = std::sin(0.37 + 0.911 * seed + (1.618 + 0.13 * seed) * static_cast<double>(i)) *
               (1.0 + 0.1 * static_cast<double>((i + seed) % 7));
    return g;
}

/// ||a - b|| / ||a||, for comparing two solves against each other rather than
/// against a finite-differenced Jacobian.
///
/// The floor is the smallest positive double, there only to keep a genuinely
/// zero `a` from dividing by zero -- not `max(.., 1.0)`. A fixed floor of 1
/// is the same shape of bug the iterative solve's own stopping criterion had
/// before `tol * max(1, |g|)` was dropped from it: it silently turns a
/// *relative* comparison into
/// an *absolute* one for any `a` smaller than the floor, which is exactly the
/// small-correction regime this solve exists to be checked in.
double relativeDifference(N_Vector a, N_Vector b)
{
    const Vector va = fdjac::toVector(a);
    const Vector vb = fdjac::toVector(b);
    return (va - vb).norm() / std::max(va.norm(), std::numeric_limits<double>::min());
}

/// ||v|| in the error-weighted RMS norm the iterative sweep stops in: the
/// solver's own error weights at the state its Jacobian was built at.
double weightedNorm(SystemSolver &sys, N_Vector v)
{
    N_Vector yAt = N_VMake_Serial(N_VGetLength(v), sys.yJacMem, sys.ctx);
    N_Vector ewt = N_VClone(v);
    sys.getErrorWeights(yAt, ewt);
    const double out = N_VWrmsNorm(v, ewt);
    N_VDestroy(yAt);
    N_VDestroy(ewt);
    return out;
}

double weightedDifference(SystemSolver &sys, N_Vector a, N_Vector b)
{
    N_Vector d = N_VClone(a);
    N_VLinearSum(1.0, a, -1.0, b, d);
    const double out = weightedNorm(sys, d);
    N_VDestroy(d);
    return out;
}

/// What the sweep's stopping test bounds: an iterative solve within
/// FieldSolveTolerance error-weight units of the exact one, or within a
/// millionth of the increment when that is larger -- see
/// solveCoupledJacIterative.
bool withinSweepTolerance(SystemSolver &sys, N_Vector exact, N_Vector iterative)
{
    return weightedDifference(sys, exact, iterative) <=
           std::max(sys.getFieldSolveTolerance(), 1e-6 * weightedNorm(sys, exact));
}

// A local copy of CoupledResidualTests.cpp's manufactured coupled problem --
// same exact solution, u = sin(pi x)(1+t) and psi = Int u dx = (2/pi)(1+t) --
// kept here rather than shared, the way this file's own configureQuietly
// already duplicates that file's, so an end-to-end run on the iterative path
// can be checked against the same closed form the exact path already reaches.
//
// It is a copy in the strict sense: hook for hook, dSigmaFn_dGeometry included.
// The two drifted apart once, this file gaining the geometry derivative and the
// other not, which left "a local copy" describing two cases with different
// Jacobians. If you change one, change the other.
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
                                          BoundaryKind::Dirichlet, BoundaryKind::Dirichlet}},
                           .readsGeometry = true,
                           .geometryDerivatives = GeometryDerivatives::Analytic})
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
    // solveJacEq that touched the field block afterwards -- the block-Jacobi
    // solveB call it used to end with, for instance -- would show up here and nowhere
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

    // ...and it has to discriminate: the *default* mode must not warn at all.
    //
    // That is the whole point of the split. The iterative branch describes what
    // an unconfigured coupled run already does, so it is INFO -- compiled out
    // below WARNING by Logging.hpp's max_log_level, and this build carries
    // neither -DVERBOSE nor -DDEBUG. As a WARNING it fired on every coupled run
    // there is, which trains a reader to skip the level the genuine
    // fallbacks > 0 report is printed at.
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

    // Guarded by build variant, following UtilityTests.cpp's logging test. The
    // iterative branch is INFO, which Logging.hpp compiles out below WARNING --
    // so "the default run says nothing about FieldSolve" is true of a release
    // build and false of a Debug build or -DMANTA_VERBOSE=ON, both of which are
    // documented variants that no CI leg covers. Asserting the
    // release form unconditionally would have left them failing here, findable
    // only by someone already debugging something else.
#if defined(DEBUG) || defined(VERBOSE)
    BOOST_TEST(defaultLog.find("FieldSolve = iterative") != std::string::npos, defaultLog);
#else
    BOOST_TEST(defaultLog.find("FieldSolve") == std::string::npos, defaultLog);
#endif

    // Unguarded, and the assertion that actually matters: whatever the build
    // level, the default configuration must not emit a WARNING, so the genuine
    // one -- the fallback count at the end of a run -- cannot be drowned by a
    // message that fires every time.
    BOOST_TEST(defaultLog.find("WARNING") == std::string::npos, defaultLog);
}

BOOST_AUTO_TEST_CASE(the_iterative_solve_agrees_with_the_exact_one)
{
    // Not to roundoff: the iterative path stops at FieldSolveTolerance, in the
    // error-weighted norm, so the agreement is to that tolerance in that norm and
    // no better. What this pins is that the two are solving the same system -- a
    // sign error in either path's use of A1 or A2 separates them by far more.
    auto solver = singleDofFixture(/*nCells=*/6, /*k=*/2);
    N_Vector g = randomRHS(*solver);

    N_Vector exact = N_VClone(g), iterative = N_VClone(g);
    solver->solveCoupledJacExact(g, exact);
    solver->solveCoupledJacIterative(g, iterative);

    const double diff = weightedDifference(*solver, exact, iterative);
    BOOST_TEST_MESSAGE("single-DOF field, iterative vs exact: weighted difference = " << diff
                       << ", relative " << relativeDifference(exact, iterative));
    BOOST_TEST(withinSweepTolerance(*solver, exact, iterative));
    BOOST_CHECK_EQUAL(solver->fieldSweepFallbacks, 0);

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

    const double diff = weightedDifference(*solver, exact, iterative);
    BOOST_TEST_MESSAGE("5-DOF field, iterative vs exact: weighted difference = " << diff
                       << ", relative " << relativeDifference(exact, iterative));
    BOOST_TEST(withinSweepTolerance(*solver, exact, iterative));

    N_VDestroy(g);
    N_VDestroy(exact);
    N_VDestroy(iterative);
}

BOOST_AUTO_TEST_CASE(the_sweep_stops_at_an_absolute_bound_or_a_millionth_of_the_increment)
{
    // The stopping test is the change one sweep makes to the whole increment,
    // in the run's error-weighted norm, against FieldSolveTolerance or a
    // millionth of the increment, whichever is larger. The two arms are for two
    // regimes, and both are exercised here on one right-hand side scaled into
    // each: a Newton correction of about one error-weight unit, where the
    // absolute bound governs, and IDACalcIC's first step, ~4e8 units at
    // cj ~ 1e9, where 1e-4 absolute would be 1e-13 relative and twenty sweeps
    // could not reach it.
    //
    // What the norm's *shape* buys -- the reason it covers the transport
    // block -- only shows on Newton's own right-hand sides, where the old test
    // (a relative 2-norm change in psi alone) left an error in sigma and q near
    // the axis worth O(1) error-weight units and held IDA at order one for
    // twenty thousand steps. Random right-hand sides at a fixture state do not
    // concentrate there, so that is pinned by the runs themselves:
    // ThetaPinchMMSTests.cpp's the_iterative_solve_does_not_stall_ida, and
    // ThetaPinchRunTests.cpp's exact_and_iterative_field_solves_give_the_same_run.
    //
    // On the five-unknown block, because a single field unknown cannot show
    // it: Irons-Tuck is exact for a scalar affine map, so that sweep reaches
    // roundoff in three whatever the bound.
    auto probe = multiDofFixture(/*nCells=*/6, /*k=*/2);
    N_Vector g0 = randomRHS(*probe), x0 = N_VClone(g0);
    probe->solveCoupledJacExact(g0, x0);
    const double unit = 1.0 / weightedNorm(*probe, x0);
    N_VDestroy(x0);

    for (double size : {1.0, 4e8})
    {
        auto solver = multiDofFixture(/*nCells=*/6, /*k=*/2);
        N_Vector g = N_VClone(g0);
        N_VScale(size * unit, g0, g);
        N_Vector exact = N_VClone(g), iterative = N_VClone(g);
        solver->solveCoupledJacExact(g, exact);
        solver->solveCoupledJacIterative(g, iterative);

        const double diff = weightedDifference(*solver, exact, iterative);
        BOOST_TEST_MESSAGE("increment of " << weightedNorm(*solver, exact) << " error-weight units: "
                           << solver->fieldSweepIterations << " sweeps, weighted difference " << diff);
        BOOST_CHECK_EQUAL(solver->fieldSweepFallbacks, 0);
        if (size == 1.0)
            BOOST_TEST(diff <= solver->getFieldSolveTolerance());
        else
        {
            BOOST_TEST(diff > solver->getFieldSolveTolerance()); // the relative arm is the one in force
            BOOST_TEST(diff <= 1e-6 * weightedNorm(*solver, exact));
        }

        N_VDestroy(g);
        N_VDestroy(exact);
        N_VDestroy(iterative);
    }
    N_VDestroy(g0);

    // ...and an increment already below the tolerance is accepted after one
    // sweep, which is right: it moves the state by less than Newton could see.
    auto solver = singleDofFixture(/*nCells=*/6, /*k=*/2);
    N_Vector g = randomRHS(*solver);
    N_VScale(1e-9, g, g);
    N_Vector exact = N_VClone(g), iterative = N_VClone(g);
    solver->solveCoupledJacExact(g, exact);
    solver->solveCoupledJacIterative(g, iterative);
    BOOST_TEST(solver->fieldSweepIterations == 1);
    BOOST_TEST(withinSweepTolerance(*solver, exact, iterative));

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
    BOOST_TEST(withinSweepTolerance(*solver, exact, iterative));
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
    // iteration needed on this fixture (2, measured before the acceleration
    // was added).
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
    // and the accelerated sweep still reaches FieldSolveTolerance against a
    // plain sweep that diverges. The secant is rank one per *step*, but the steps are not the
    // same rank-one direction, so the iteration is not confined to a
    // two-dimensional subspace the way the pessimistic argument assumes. Scanned
    // over strengths from 1 to 1e5 -- rho saturating just under 2 -- and over six
    // right-hand sides, it never once failed to converge given the sweeps.
    //
    // So both halves are asserted. The guarantee, first, because it is true by
    // construction and survives any change to the accelerator: whichever way
    // solveCoupledJacIterative got there, the answer is the exact one, which is
    // what makes iterative safe as a default. Then the measurement.
    auto probe = multiDofFixture(6, 2, 10.0);
    const double rho = spectralRadius(probe);
    BOOST_TEST_MESSAGE("five-DOF, strength 10: rho(M) = " << rho
                       << ", max|A1_cellwise| = " << maxAbsA1(probe));
    BOOST_REQUIRE_GT(maxAbsA1(probe), 0.0);
    BOOST_CHECK_GT(rho, 1.2);

    // Six right-hand *directions*: the sweep count varies by a factor of two
    // across them, which is the whole reason one right-hand side is not a
    // measurement.
    int convergedAtDefault = 0, escalatedAtDefault = 0;
    std::string atDefault, atSixty;

    for (int seed = 0; seed < 6; ++seed)
    {
        auto solver = multiDofFixture(6, 2, 10.0);
        BOOST_REQUIRE_EQUAL(solver->getFieldSolveMaxSweeps(), 20); // the shipped default

        N_Vector g = randomRHS(*solver, seed);
        N_Vector exact = N_VClone(g), iterative = N_VClone(g);
        solver->solveCoupledJacExact(g, exact);

        // ---- the guarantee, at the shipped default cap. True by construction
        // whichever route the solve took, and the only claim here that survives
        // any change to the accelerator.
        solver->solveCoupledJacIterative(g, iterative);
        BOOST_TEST(withinSweepTolerance(*solver, exact, iterative));

        const long sweeps = solver->fieldSweepIterations;
        const long fellBack = solver->fieldSweepFallbacks;
        (fellBack > 0 ? escalatedAtDefault : convergedAtDefault)++;
        atDefault += " " + std::to_string(sweeps) + (fellBack > 0 ? "*" : "");

        // ---- and the route, given room. Same solver, so the counts are read as
        // deltas: only initialize() zeroes them, and these fixtures never call it.
        solver->setFieldSolveMaxSweeps(60);
        solver->solveCoupledJacIterative(g, iterative);
        BOOST_CHECK_EQUAL(solver->fieldSweepFallbacks - fellBack, 0);
        BOOST_TEST(withinSweepTolerance(*solver, exact, iterative));
        atSixty += " " + std::to_string(solver->fieldSweepIterations - sweeps);

        N_VDestroy(g);
        N_VDestroy(exact);
        N_VDestroy(iterative);
    }

    BOOST_TEST_MESSAGE("five-DOF sweeps by right-hand side, cap 20 (* = escalated):"
                       << atDefault << "\n  the same at cap 60:" << atSixty);

    // The accelerator copes with five divergent field unknowns -- every
    // right-hand side converges once there is room for it...
    BOOST_CHECK_EQUAL(escalatedAtDefault + convergedAtDefault, 6);

    // ...but FieldSolveMaxSweeps = 20 is **not enough** for all of them, and
    // that is a statement about the shipped default rather than about this
    // fixture: 10-23 sweeps, one of six over the cap. Asserted as a tripwire,
    // both ways, and recorded with the TODO entry that holds the same
    // measurement, so that a change to the accelerator, the stopping test or
    // the default has to update both. Read it beside MMSFieldTests.cpp's few
    // sweeps per Jacobian solve over whole integrations: a *random* right-hand
    // side is the hard case, while Newton's are small, smooth corrections about
    // a nearby state, so neither says the cap is wrong in a real run.
    BOOST_TEST_MESSAGE("of six right-hand sides, " << escalatedAtDefault
                       << " needed more than the default cap of 20");
    BOOST_CHECK_GT(convergedAtDefault, 0);
    BOOST_CHECK_GT(escalatedAtDefault, 0);
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

BOOST_AUTO_TEST_CASE(the_end_of_run_report_counts_the_sweeps_and_names_the_fallbacks)
{
    // The only signal a user has that the coupling is not converging, and the
    // gap this closes: the sweep used to hit its cap silently,
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

    // One sweep is accepted only when the whole increment is already below the
    // tolerance, so the Jacobian solves of the transient escalate -- IDACalcIC's
    // included, which is why the
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

// The first registered field model, on the same checks the manufactured ones
// get. Its A2 reaches the transport through the pressure's aux column only, and
// its A1 through all four geometry slots, so this is the fixture that runs the
// aux half of A2 and a multi-slot A1 together.
BOOST_AUTO_TEST_CASE(the_theta_pinch_exact_solve_inverts_its_finite_differenced_jacobian)
{
    for (bool superconvergent : {false, true})
    {
        BOOST_TEST_CONTEXT("superconvergent = " << superconvergent)
        {
            auto solver = thetaPinchFixture(/*nCells=*/5, /*k=*/3, /*intervalsPerCell=*/1, superconvergent);

            // Not vacuous: both coupling blocks carry something.
            BOOST_REQUIRE_GT(maxAbsA1(solver), 1e-3);
            DGSoln a2Map(solver->nVars, *solver.grid, 3, N_VGetArrayPointer(solver->a2[0]),
                         solver->nScalars, solver->nAux, solver->getFieldDOF());
            BOOST_REQUIRE_GT(maxAbs(a2Map.Aux(0), 5), 1e-3);

            for (int trial = 0; trial < 2; ++trial)
            {
                const SolveReport r = checkExactSolve(solver, trial);
                BOOST_TEST_MESSAGE("theta pinch, k = 3, nCells = 5, one interval per cell, trial "
                                   << trial << ": ||dy* - dy||/||dy|| = " << r.solutionError
                                   << ", ||J dy* - g||/||g|| = " << r.residual << " ("
                                   << r.nDirichlet << " Dirichlet components excluded)");
                BOOST_TEST(r.nDirichlet == 1);
                BOOST_TEST(r.solutionError < 1e-6);
                BOOST_TEST(r.residual < 1e-7);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(the_theta_pinch_iterative_solve_agrees_with_the_exact_one)
{
    auto solver = thetaPinchFixture(/*nCells=*/5, /*k=*/3, /*intervalsPerCell=*/2);
    N_Vector g = randomRHS(*solver);
    N_Vector exact = N_VClone(g), iterative = N_VClone(g);
    solver->solveCoupledJacExact(g, exact);
    solver->solveCoupledJacIterative(g, iterative);

    const double diff = weightedDifference(*solver, exact, iterative);
    BOOST_TEST_MESSAGE("theta pinch, two intervals per cell, iterative vs exact: weighted difference = "
                       << diff << ", relative " << relativeDifference(exact, iterative));
    BOOST_TEST(withinSweepTolerance(*solver, exact, iterative));
    BOOST_CHECK_EQUAL(solver->fieldSweepFallbacks, 0);

    N_VDestroy(g);
    N_VDestroy(exact);
    N_VDestroy(iterative);
}



// The theta pinch's block Gauss-Seidel iteration contracts at every cj,
// steady (cj = 0, where the adjoint always runs) to stiff, and at every field
// resolution: measured 0.10 to 0.19. So a sweep on it that fails to converge is
// a defect in the sweep, not a property of the coupling -- which is how the
// alignment guard in ironsTuck was found.
BOOST_AUTO_TEST_CASE(the_theta_pinch_sweep_contracts_at_every_cj)
{
    for (double cj : {0.0, 1.7, 100.0, 1e4})
        for (Index m : {1, 4})
        {
            auto solver = thetaPinchFixture(6, 3, m, false, cj);
            const double rho = spectralRadius(solver, 200);
            BOOST_TEST_MESSAGE("cj = " << cj << ", " << m << " intervals per cell: rho = " << rho);
            BOOST_TEST(rho < 0.3);
        }
}

// ---------------------------------------- geometryDerivatives = Differenced --

/// GeometricAuxDiffusion counting what a Jacobian build asks of it: batched
/// value evaluations, and calls of the three geometry hooks.
class CountingGeometricAux : public GeometricAuxDiffusion
{
public:
    explicit CountingGeometricAux(GeometryDerivatives d) : GeometricAuxDiffusion(d) {}

    PhysicsOutput ComputePhysics(GlobalState const &s, std::vector<Position> const &x, Time t) override
    {
        ++batched;
        return GeometricAuxDiffusion::ComputePhysics(s, x, t);
    }
    void dSigmaFn_dGeometry(Index i, VectorRef v, const State &s, Position x, Time t) override
    {
        ++hooks;
        GeometricAuxDiffusion::dSigmaFn_dGeometry(i, v, s, x, t);
    }
    void dSources_dGeometry(Index i, VectorRef v, const State &s, Position x, Time t) override
    {
        ++hooks;
        GeometricAuxDiffusion::dSources_dGeometry(i, v, s, x, t);
    }
    void dAuxG_dGeometry(Index i, VectorRef v, const State &s, Position x, Time t) override
    {
        ++hooks;
        GeometricAuxDiffusion::dAuxG_dGeometry(i, v, s, x, t);
    }

    int batched = 0, hooks = 0;
};

double relativeA1Difference(CoupledSolver const &a, CoupledSolver const &b)
{
    double d = 0.0;
    for (size_t i = 0; i < a->A1_cellwise.size(); ++i)
        d = std::max(d, (a->A1_cellwise[i] - b->A1_cellwise[i]).cwiseAbs().maxCoeff());
    return d / maxAbsA1(a);
}

// The solver's forward difference of the case's values reproduces the case's
// own derivatives, for every row block of A1 (GeometricAuxDiffusion reaches the
// sigma, u and aux rows) and on both node sets.
BOOST_AUTO_TEST_CASE(a_differenced_a1_matches_the_analytic_one)
{
    for (bool superconvergent : {false, true})
    {
        BOOST_TEST_CONTEXT("superconvergent = " << superconvergent)
        {
            auto field = std::make_shared<ManufacturedFieldEverySlot>(1.0);
            auto analytic = makeCoupledSolverAtState(
                6, 2, std::make_unique<GeometricAuxDiffusion>(GeometryDerivatives::Analytic), field,
                superconvergent);
            auto differenced = makeCoupledSolverAtState(
                6, 2, std::make_unique<GeometricAuxDiffusion>(GeometryDerivatives::Differenced),
                std::make_shared<ManufacturedFieldEverySlot>(1.0), superconvergent);
            BOOST_REQUIRE_GT(maxAbsA1(analytic), 1e-3);
            const double manufactured = relativeA1Difference(analytic, differenced);

            auto pinchA = thetaPinchFixture(5, 3, 1, superconvergent, 1.7, "Analytic");
            auto pinchD = thetaPinchFixture(5, 3, 1, superconvergent, 1.7, "Differenced");
            BOOST_REQUIRE_GT(maxAbsA1(pinchA), 1e-3);
            const double pinch = relativeA1Difference(pinchA, pinchD);

            BOOST_TEST_MESSAGE("differenced A1 against analytic, relative: manufactured "
                               << manufactured << ", theta pinch " << pinch);
            BOOST_TEST(manufactured < 1e-6);
            BOOST_TEST(pinch < 1e-6);
        }
    }
}

BOOST_AUTO_TEST_CASE(the_exact_solve_inverts_the_jacobian_with_a_differenced_a1)
{
    auto solver = thetaPinchFixture(/*nCells=*/5, /*k=*/3, /*intervalsPerCell=*/1, false, 1.7, "Differenced");
    for (int trial = 0; trial < 2; ++trial)
    {
        const SolveReport r = checkExactSolve(solver, trial);
        BOOST_TEST_MESSAGE("theta pinch, differenced A1, trial "
                           << trial << ": ||dy* - dy||/||dy|| = " << r.solutionError
                           << ", ||J dy* - g||/||g|| = " << r.residual);
        BOOST_TEST(r.solutionError < 1e-5);
        BOOST_TEST(r.residual < 1e-6);
    }
}

// What a build costs, exactly, under each declaration -- the evaluation plan's
// FieldCoupling site, and the calls actually made. Differenced makes nGeometry + 1
// batched value calls and never reaches a geometry hook; Analytic the reverse.
// And the unperturbed call comes last, so the case's source cache, which the
// output reads, holds the sources at the state rather than at a shifted geometry.
BOOST_AUTO_TEST_CASE(a_build_spends_what_its_geometry_derivatives_declare)
{
    for (GeometryDerivatives d : {GeometryDerivatives::Analytic, GeometryDerivatives::Differenced})
    {
        const bool differenced = d == GeometryDerivatives::Differenced;
        BOOST_TEST_CONTEXT((differenced ? "Differenced" : "Analytic"))
        {
            auto problem = std::make_unique<CountingGeometricAux>(d);
            CountingGeometricAux &p = *problem;
            auto h = makeCoupledSolverAtState(6, 2, std::move(problem),
                                              std::make_shared<ManufacturedFieldEverySlot>(1.0));
            const Index nGeom = h->getGeometrySlots();
            const Index nNodes = 6 * 3;
            BOOST_REQUIRE_GE(nGeom, 1);

            p.batched = p.hooks = 0;
            h->updateMatricesForJacSolve();
            if (differenced)
            {
                BOOST_TEST(p.batched == nGeom + 1);
                BOOST_TEST(p.hooks == 0);
            }
            else
            {
                BOOST_TEST(p.batched == 0);
                BOOST_TEST(p.hooks == 3 * nNodes); // sigma and S for one variable, G for one aux
            }

            const EvaluationPlan plan = h->evaluationPlan();
            const auto site = std::find_if(plan.sites.begin(), plan.sites.end(), [](EvaluationSite const &s)
                                           { return s.kind == EvaluationKind::FieldCoupling; });
            BOOST_REQUIRE(site != plan.sites.end());
            BOOST_TEST((site->cadence == EvaluationCadence::PerJacobianBuild));
            BOOST_TEST((site->entry ==
                        (differenced ? EvaluationEntry::ComputePhysics : EvaluationEntry::Pointwise)));
            BOOST_TEST(site->calls == (differenced ? nGeom + 1 : 1));

            if (differenced)
            {
                const Values cached = p.getSourceCache(0)->get();
                auto nodes = h->physicsNodesAt(h->yJac, h->t);
                const PhysicsOutput atState = p.GeometricAuxDiffusion::ComputePhysics(nodes.states, nodes.points, h->t);
                BOOST_TEST((cached - atState[1][0]).cwiseAbs().maxCoeff() == 0.0,
                           boost::test_tools::tolerance(0.0));
                // Not vacuous: the sources do move with the geometry.
                nodes.states.GeometryMatrix().row(0).array() += 1e-3;
                const PhysicsOutput shifted = p.GeometricAuxDiffusion::ComputePhysics(nodes.states, nodes.points, h->t);
                BOOST_TEST((shifted[1][0] - atState[1][0]).cwiseAbs().maxCoeff() > 1e-6);
            }
        }
    }
}

// ------------------------------------------- tier 1: the reduced coupling --
//
// The field model run as an equilibrium (FieldModelEquilibrium) and coupled
// through its response, FieldCoupling = "reduced". The residual is tier 0's --
// geometry from an equilibrium solved at each transport state -- so the
// Jacobian the reduced solve inverts is checked against a finite difference of
// *that* residual, which re-solves the equilibrium at every perturbation, and
// against tier 2's exact solve with psi eliminated.

/// A tier-1 solver at a state, its Jacobian blocks built. No psi to push off
/// anything: the equilibrium is solved for the state.
/// FieldModelEquilibrium with its transposed response switched off: what an
/// adjoint that never applied G'^T would compute.
class BlindTransposeEquilibrium : public FieldModelEquilibrium
{
public:
    using FieldModelEquilibrium::FieldModelEquilibrium;
    void applyResponseTranspose(GlobalState &, Equilibrium const &, GlobalState const &, Matrix const &,
                                std::vector<Position> const &, Vector const &, Time,
                                std::vector<Position> const &) override
    {
    }
};

CoupledSolver makeReducedSolverAtState(Index nCells, Index k, std::unique_ptr<TransportSystem> problem,
                                       std::shared_ptr<FieldModel> model, bool differencedResponse,
                                       bool superconvergent = false, double cj = 1.7, bool blindTranspose = false)
{
    CoupledSolver h;
    h.problem = std::move(problem);
    h.field = std::move(model);
    h.grid = std::make_unique<Grid>(0.0, 1.0, nCells);
    h.sys = std::make_unique<SystemSolver>(*h.grid, k, h.problem.get());
    h.sys->setTau(0.75);
    h.sys->setInitialTime(0.0);
    h.sys->setSuperconvergent(superconvergent);
    h.sys->setEquilibriumModel(blindTranspose
                                   ? std::make_shared<BlindTransposeEquilibrium>(h.field, differencedResponse)
                                   : std::make_shared<FieldModelEquilibrium>(h.field, differencedResponse),
                               SystemSolver::EquilibriumCoupling::Reduced);
    h.sys->resetCoeffs();
    h.sys->initialiseMatrices();

    const size_t dof = h.sys->getSolution().getDoF();
    h.Y = N_VNew_Serial(dof, h.sys->ctx);
    h.dYdt = N_VClone(h.Y);
    N_VConst(0.0, h.Y);
    N_VConst(0.0, h.dYdt);
    h.sys->setInitialConditions(h.Y, h.dYdt);

    h.cj = cj;
    h.t = 0.0;
    h.sys->setJacTime(h.t);
    h.sys->setAlpha(cj);
    h.sys->setJacEvalY(h.Y, h.dYdt);
    h.sys->updateBoundaryConditions(h.t);
    h.sys->updateMatricesForJacSolve();
    return h;
}

CoupledSolver reducedThetaPinch(Index nCells, Index k, bool differencedResponse, bool superconvergent = false,
                                Index intervalsPerCell = 1, bool blindTranspose = false)
{
    using namespace toml::literals::toml_literals;
    static const toml::value config = u8R"(
        [ThetaPinchDiffusion]
        Temperature = 0.3
        D0 = 0.5
        Gamma = 2.0
    )"_toml;
    const Grid grid(0.0, 1.0, nCells);
    return makeReducedSolverAtState(
        nCells, k, std::make_unique<ThetaPinchDiffusion>(config, grid),
        std::make_shared<ThetaPinchEquilibrium>(ThetaPinchEquilibrium::Parameters{.intervalsPerCell = intervalsPerCell},
                                                grid),
        differencedResponse, superconvergent, /*cj=*/1.7, blindTranspose);
}

/// checkExactSolve for tier 1: the reduced exact solve against a finite
/// difference of the tier-0 residual.
SolveReport checkReducedSolve(CoupledSolver &h, int trial)
{
    SystemSolver &sys = *h.sys;
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
    sys.setFieldSolveMode(SystemSolver::FieldSolveMode::Exact);
    sys.solveJacEq(gVec, out);

    Vector recovered = fdjac::toVector(out);
    zeroAt(recovered, dead);
    N_VDestroy(gVec);
    N_VDestroy(out);
    return {(recovered - dy).norm() / dy.norm(), fdjac::relativeResidual(J, recovered, g, dead),
            static_cast<Index>(dead.size())};
}

// The model's own response is good to the finite difference's own error, as
// tier 2's exact solve is: 1e-8 in the solution, 6e-10 in the residual. The
// differenced response is a forward difference of the equilibrium at a step of
// sqrt(eps), good to 3e-7 relative (the_adapters_response_is_the_derivative_of_
// its_solve), and the check sees exactly that: 2e-6 and 1e-7 at worst.
BOOST_AUTO_TEST_CASE(the_reduced_exact_solve_inverts_the_finite_differenced_tier_zero_jacobian)
{
    for (bool differenced : {false, true})
    {
        const double solutionBound = differenced ? 1e-5 : 1e-6, residualBound = differenced ? 1e-6 : 1e-7;
        BOOST_TEST_CONTEXT("differenced response = " << differenced)
        {
            auto manufactured = makeReducedSolverAtState(4, 2, std::make_unique<GeometricAuxDiffusion>(),
                                                         std::make_shared<ManufacturedFieldEverySlot>(),
                                                         differenced);
            const SolveReport m = checkReducedSolve(manufactured, 0);
            BOOST_TEST_MESSAGE("manufactured, every slot: ||dy* - dy||/||dy|| = "
                               << m.solutionError << ", ||J dy* - g||/||g|| = " << m.residual);
            BOOST_TEST(m.solutionError < solutionBound);
            BOOST_TEST(m.residual < residualBound);

            for (bool superconvergent : {false, true})
            {
                auto pinch = reducedThetaPinch(5, 3, differenced, superconvergent);
                for (int trial = 0; trial < 2; ++trial)
                {
                    const SolveReport r = checkReducedSolve(pinch, trial);
                    BOOST_TEST_MESSAGE("theta pinch, superconvergent = "
                                       << superconvergent << ", trial " << trial << ": ||dy* - dy||/||dy|| = "
                                       << r.solutionError << ", ||J dy* - g||/||g|| = " << r.residual);
                    BOOST_TEST(r.nDirichlet == 1);
                    BOOST_TEST(r.solutionError < solutionBound);
                    BOOST_TEST(r.residual < residualBound);
                }
            }
        }
    }
}

// The coupling is not vacuous: dropping A_g (tier 0's Jacobian) misses the
// finite-differenced one by orders more than the reduced solve does.
BOOST_AUTO_TEST_CASE(the_reduced_check_sees_a_missing_coupling)
{
    auto pinch = reducedThetaPinch(5, 3, false);
    SystemSolver &sys = *pinch.sys;
    const SolveReport coupled = checkReducedSolve(pinch, 0);
    for (Matrix &Ag : sys.reducedBlocks.Ag)
        Ag.setZero();
    const SolveReport blind = checkReducedSolve(pinch, 0);
    BOOST_TEST_MESSAGE("theta pinch: residual " << coupled.residual << " coupled, " << blind.residual
                                                << " with A_g zeroed");
    BOOST_TEST(blind.residual > 1e3 * coupled.residual);
}

// With the adapter's own response, tier 1 *is* tier 2 with psi eliminated: the
// same transport increment as tier 2's exact solve of the same right-hand side
// with zero field rows, at psi on its equilibrium.
BOOST_AUTO_TEST_CASE(the_reduced_exact_solve_is_tier_two_with_psi_eliminated)
{
    for (bool superconvergent : {false, true})
    {
        BOOST_TEST_CONTEXT("superconvergent = " << superconvergent)
        {
            auto reduced = reducedThetaPinch(5, 3, false, superconvergent);
            auto coupled = thetaPinchFixture(5, 3, 1, superconvergent);

            // Put tier 2 at tier 1's state, psi included.
            SystemSolver &r = *reduced.sys, &c = *coupled.sys;
            const Index nTransport = static_cast<Index>(r.getSolution().getDoF());
            std::copy(N_VGetArrayPointer(reduced.Y), N_VGetArrayPointer(reduced.Y) + nTransport,
                      N_VGetArrayPointer(coupled.Y));
            DGSoln yMap(c.nVars, *coupled.grid, 3, N_VGetArrayPointer(coupled.Y), c.nScalars, c.nAux,
                        c.getFieldDOF());
            yMap.getField() = FieldModelEquilibrium::psi(*r.reducedBlocks.eq);
            c.setJacEvalY(coupled.Y, coupled.dYdt);
            c.updateMatricesForJacSolve();

            N_Vector g1 = randomRHS(r), x1 = N_VClone(g1);
            N_Vector g2 = N_VNew_Serial(c.getSolution().getDoF(), c.ctx), x2 = N_VClone(g2);
            N_VConst(0.0, g2);
            std::copy(N_VGetArrayPointer(g1), N_VGetArrayPointer(g1) + nTransport, N_VGetArrayPointer(g2));

            r.setFieldSolveMode(SystemSolver::FieldSolveMode::Exact);
            r.solveJacEq(g1, x1);
            c.solveCoupledJacExact(g2, x2);

            const Vector a = fdjac::toVector(x1), b = fdjac::toVector(x2).head(nTransport);
            const double diff = (a - b).norm() / a.norm();
            BOOST_TEST_MESSAGE("tier 1 exact against tier 2 exact, transport block: " << diff);
            BOOST_TEST(diff < 1e-10);

            for (N_Vector v : {g1, x1, g2, x2})
                N_VDestroy(v);
        }
    }
}

BOOST_AUTO_TEST_CASE(the_reduced_iterative_and_krylov_solves_agree_with_the_exact_one)
{
    for (bool differenced : {false, true})
        for (Index intervals : {1, 2})
        {
            auto pinch = reducedThetaPinch(5, 3, differenced, false, intervals);
            SystemSolver &sys = *pinch.sys;
            N_Vector g = randomRHS(sys);
            N_Vector exact = N_VClone(g), iterative = N_VClone(g), krylov = N_VClone(g);
            sys.setFieldSolveMode(SystemSolver::FieldSolveMode::Exact);
            sys.solveJacEq(g, exact);
            sys.setFieldSolveMode(SystemSolver::FieldSolveMode::Iterative);
            sys.solveJacEq(g, iterative);
            sys.setFieldSolveMode(SystemSolver::FieldSolveMode::Krylov);
            sys.solveJacEq(g, krylov);

            const auto stats = sys.getFieldSweepStats();
            BOOST_TEST_MESSAGE("differenced = " << differenced << ", " << intervals
                                                << " intervals per cell: iterative "
                                                << weightedDifference(sys, exact, iterative) << ", krylov "
                                                << weightedDifference(sys, exact, krylov) << " from exact; "
                                                << stats.iterations << " sweeps, " << stats.krylovIterations
                                                << " GMRES iterations");
            BOOST_TEST(withinSweepTolerance(sys, exact, iterative));
            BOOST_TEST(withinSweepTolerance(sys, exact, krylov));
            BOOST_TEST(stats.fallbacks == 0);
            BOOST_TEST(stats.krylovSolves == 1);
            for (N_Vector v : {g, exact, iterative, krylov})
                N_VDestroy(v);
        }
}

// Krylov at tier 2 is the same code on dpsi, and converges where the plain
// sweep diverges: the five-unknown coupling at a strength where rho > 1.
BOOST_AUTO_TEST_CASE(the_tier_two_krylov_solve_agrees_with_the_exact_one)
{
    for (double strength : {1.0, 40.0})
    {
        auto solver = multiDofFixture(4, 2, strength);
        SystemSolver &sys = *solver.sys;
        N_Vector g = randomRHS(sys);
        N_Vector exact = N_VClone(g), krylov = N_VClone(g);
        sys.solveCoupledJacExact(g, exact);
        sys.setFieldSolveMode(SystemSolver::FieldSolveMode::Krylov);
        sys.solveJacEq(g, krylov);
        const auto stats = sys.getFieldSweepStats();
        BOOST_TEST_MESSAGE("strength " << strength << ": rho " << spectralRadius(solver) << ", krylov "
                                       << weightedDifference(sys, exact, krylov) << " from exact in "
                                       << stats.krylovIterations << " iterations");
        BOOST_TEST(withinSweepTolerance(sys, exact, krylov));
        BOOST_TEST(stats.fallbacks == 0);
        // The field row: dpsi is consistent with dx, as the sweep leaves it.
        BOOST_TEST(fieldRowResidual(solver, g, krylov) < 1e-12);
        for (N_Vector v : {g, exact, krylov})
            N_VDestroy(v);
    }
}

// The adapter's exact response against the base class's difference of Solve,
// on one direction at the theta pinch's state.
BOOST_AUTO_TEST_CASE(the_adapters_response_is_the_derivative_of_its_solve)
{
    auto pinch = reducedThetaPinch(5, 3, false);
    SystemSolver &sys = *pinch.sys;
    auto &b = sys.reducedBlocks;
    FieldModelEquilibrium exact(pinch.field, false), differenced(pinch.field, true);

    GlobalState dp = *b.states;
    dp.Variable().setConstant(0.0);
    for (Eigen::Index j = 0; j < dp.Variable().cols(); ++j)
        dp.Variable()(0, j) = std::sin(0.3 + 1.1 * static_cast<double>(j));
    dp.Aux() = 0.3 * dp.Variable();

    const Eigen::Index nPoints = static_cast<Eigen::Index>(b.physicsPoints.size());
    Matrix dgExact(sys.getGeometrySlots(), nPoints), dgDiff(sys.getGeometrySlots(), nPoints);
    exact.applyResponse(dgExact, *b.eq, *b.states, dp, b.inputPoints, b.weights, b.t, b.physicsPoints);
    differenced.applyResponse(dgDiff, *b.eq, *b.states, dp, b.inputPoints, b.weights, b.t, b.physicsPoints);
    const double diff = (dgExact - dgDiff).norm() / dgExact.norm();
    BOOST_TEST_MESSAGE("theta pinch response, exact against differenced: " << diff << " relative; |dg| "
                                                                          << dgExact.norm());
    BOOST_TEST(dgExact.norm() > 1e-3);
    BOOST_TEST(diff < 1e-6);
}

BOOST_AUTO_TEST_CASE(a_reduced_equilibrium_needs_the_cases_geometry_derivatives)
{
    using namespace toml::literals::toml_literals;
    const toml::value config = u8R"(
        [ThetaPinchDiffusion]
        Temperature = 0.3
        GeometryDerivatives = "None"
    )"_toml;
    const Grid grid(0.0, 1.0, 4);
    ThetaPinchDiffusion problem(config, grid);
    auto model = std::make_shared<FieldModelEquilibrium>(
        std::make_shared<ThetaPinchEquilibrium>(ThetaPinchEquilibrium::Parameters{}, grid));
    SystemSolver sys(grid, 2, &problem);
    try
    {
        sys.setEquilibriumModel(model, SystemSolver::EquilibriumCoupling::Reduced);
        BOOST_FAIL("a reduced equilibrium was attached to a case with no geometry derivatives");
    }
    catch (std::logic_error const &e)
    {
        BOOST_TEST(std::string(e.what()).find("geometryDerivatives is None") != std::string::npos);
    }
    BOOST_CHECK_NO_THROW(sys.setEquilibriumModel(model, SystemSolver::EquilibriumCoupling::Consistent));
}

// ---------------------------------------------- tier 1: the reduced adjoint --
//
// The adjoint twin of the checks above: assembled from the build's record at
// alpha = 0 and required to invert J_red^T, the transpose of a finite
// difference of the tier-0 residual at cj = 0.

/// The adjoint state as one vector in the solution vector's ordering, cells
/// then lambda (then psi at tier 2).
Vector adjointVector(SystemSolver const &sys)
{
    const Index nCells = static_cast<Index>(sys.nCells), cellDOF = static_cast<Index>(sys.localDOF);
    const Index nLambda = sys.adjoint_lambdas.size(), nField = sys.getFieldDOF();
    Vector z(nCells * cellDOF + nLambda + nField);
    for (Index i = 0; i < nCells; ++i)
        z.segment(i * cellDOF, cellDOF) = sys.adjoint_squ[i];
    z.segment(nCells * cellDOF, nLambda) = sys.adjoint_lambdas;
    if (nField > 0)
        z.tail(nField) = sys.adjoint_field;
    return z;
}

/// A cellwise right-hand side, deliberately not shaped like an objective's.
std::vector<Vector> randomCellRHS(SystemSolver const &sys, int trial)
{
    const Index cellDOF = static_cast<Index>(sys.localDOF);
    std::vector<Vector> out(sys.nCells, Vector(cellDOF));
    for (Index i = 0; i < static_cast<Index>(sys.nCells); ++i)
        for (Index j = 0; j < cellDOF; ++j)
            out[i](j) = std::sin(0.4 + (trial + 1) * 0.9 * static_cast<double>(i * cellDOF + j));
    return out;
}

/// ||J_red^T z - g|| / ||g|| over the rows the residual defines, solved by
/// `mode`, with no geometry share in the right-hand side.
double reducedAdjointResidual(CoupledSolver &h, SystemSolver::FieldSolveMode mode, int trial)
{
    SystemSolver &sys = *h.sys;
    const Matrix J = fdjac::jacobian(sys, h.Y, h.dYdt, h.t, 0.0);
    const std::vector<Index> dead = fdjac::undefinedRows(J);
    sys.assembleAdjointOperator(sys.jacobianDerivatives);

    sys.G_y = randomCellRHS(sys, trial);
    sys.G_field.resize(0);
    sys.G_geometry.resize(0);
    Vector g = Vector::Zero(J.rows());
    for (Index i = 0; i < static_cast<Index>(sys.nCells); ++i)
        g.segment(i * sys.localDOF, sys.localDOF) = sys.G_y[i];

    sys.setFieldSolveMode(mode);
    sys.solveAdjointState();
    return fdjac::relativeResidual(Matrix(J.transpose()), adjointVector(sys), g, dead);
}

BOOST_AUTO_TEST_CASE(the_reduced_adjoint_inverts_the_transpose_of_the_tier_zero_jacobian)
{
    using Mode = SystemSolver::FieldSolveMode;
    for (bool differenced : {false, true})
        for (bool superconvergent : {false, true})
        {
            // 1e-7, field_adjoint_tests' bound for tier 2's transpose check:
            // measured 2e-8 to 3e-8 with the model's transpose, in every mode
            // alike, which is the finite difference's own error. The assembled
            // transpose is good to 1e-7 (the_assembled_transpose_is_the_
            // transpose_of_the_response), and the check sees 5e-7.
            const double bound = differenced ? 1e-6 : 1e-7;
            BOOST_TEST_CONTEXT("differenced response = " << differenced << ", superconvergent = " << superconvergent)
            {
                auto pinch = reducedThetaPinch(5, 3, differenced, superconvergent);
                for (Mode mode : {Mode::Exact, Mode::Iterative, Mode::Krylov})
                {
                    const double r = reducedAdjointResidual(pinch, mode, 0);
                    const auto stats = pinch->getFieldSweepStats();
                    BOOST_TEST_MESSAGE("mode " << static_cast<int>(mode) << ": ||J^T z - g||/||g|| = " << r
                                               << "; adjoint sweeps " << stats.adjointSweeps << ", GMRES "
                                               << stats.adjointKrylovIterations << ", fell back "
                                               << stats.adjointFellBack);
                    BOOST_TEST(r < bound);
                    BOOST_TEST(!stats.adjointFellBack);
                }
            }
        }
}

// The vacuity guards, at the operator. A_g^T zeroed, or a model whose
// transposed response does nothing, is tier 0's adjoint, which misses the
// transpose by orders more than the reduced solve does.
BOOST_AUTO_TEST_CASE(the_reduced_adjoint_check_sees_a_missing_transpose)
{
    using Mode = SystemSolver::FieldSolveMode;
    auto pinch = reducedThetaPinch(5, 3, false);
    const double good = reducedAdjointResidual(pinch, Mode::Exact, 0);

    SystemSolver &sys = *pinch.sys;
    for (Matrix &block : sys.Ag_transpose_cellwise)
    {
        BOOST_REQUIRE_GT(block.cwiseAbs().maxCoeff(), 0.0);
        block.setZero();
    }
    sys.solveAdjointState();
    const Matrix J = fdjac::jacobian(sys, pinch.Y, pinch.dYdt, pinch.t, 0.0);
    Vector g = Vector::Zero(J.rows());
    for (Index i = 0; i < static_cast<Index>(sys.nCells); ++i)
        g.segment(i * sys.localDOF, sys.localDOF) = sys.G_y[i];
    const double noAg = fdjac::relativeResidual(Matrix(J.transpose()), adjointVector(sys), g, fdjac::undefinedRows(J));

    auto blind = reducedThetaPinch(5, 3, false, false, 1, /*blindTranspose=*/true);
    const double noResponse = reducedAdjointResidual(blind, Mode::Exact, 0);

    BOOST_TEST_MESSAGE("||J^T z - g||/||g||: " << good << " coupled, " << noAg << " with A_g^T zeroed, "
                                               << noResponse << " with no transposed response");
    BOOST_TEST(noAg > 1e3 * good);
    BOOST_TEST(noResponse > 1e3 * good);
}

// With the adapter's own response, tier 1's adjoint is tier 2's with psi
// eliminated: the same transport block of z, for the same G_y and a geometry
// share that tier 2 sees as G_psi = sum_j (dGeometry/dpsi_j)^T G_geometry_j.
// And tier 2's own three adjoint solves agree, GMRES among them.
BOOST_AUTO_TEST_CASE(the_reduced_adjoint_is_tier_twos_with_psi_eliminated)
{
    using Mode = SystemSolver::FieldSolveMode;
    for (bool superconvergent : {false, true})
    {
        BOOST_TEST_CONTEXT("superconvergent = " << superconvergent)
        {
            auto reduced = reducedThetaPinch(5, 3, false, superconvergent);
            auto coupled = thetaPinchFixture(5, 3, 1, superconvergent);
            SystemSolver &r = *reduced.sys, &c = *coupled.sys;

            const Index nTransport = static_cast<Index>(r.getSolution().getDoF());
            std::copy(N_VGetArrayPointer(reduced.Y), N_VGetArrayPointer(reduced.Y) + nTransport,
                      N_VGetArrayPointer(coupled.Y));
            DGSoln yMap(c.nVars, *coupled.grid, 3, N_VGetArrayPointer(coupled.Y), c.nScalars, c.nAux,
                        c.getFieldDOF());
            const Vector psi = FieldModelEquilibrium::psi(*r.reducedBlocks.eq);
            yMap.getField() = psi;
            c.setJacEvalY(coupled.Y, coupled.dYdt);
            c.updateMatricesForJacSolve();

            r.assembleAdjointOperator(r.jacobianDerivatives);
            c.assembleAdjointOperator(c.jacobianDerivatives);

            // The same right-hand side, both ways.
            const std::vector<Vector> Gy = randomCellRHS(r, 1);
            auto const &points = r.reducedBlocks.physicsPoints;
            const Index nGeom = r.getGeometrySlots();
            Vector Ggeom(nGeom * static_cast<Index>(points.size()));
            for (Index j = 0; j < Ggeom.size(); ++j)
                Ggeom(j) = std::cos(0.3 + 0.77 * static_cast<double>(j));
            Vector Gpsi = Vector::Zero(c.getFieldDOF());
            Matrix dgdpsi(nGeom, c.getFieldDOF());
            for (size_t j = 0; j < points.size(); ++j)
            {
                dgdpsi.setZero();
                c.getFieldModel()->dGeometry_dpsi(dgdpsi, psi, points[j], r.reducedBlocks.t);
                Gpsi += dgdpsi.transpose() * Ggeom.segment(static_cast<Index>(j) * nGeom, nGeom);
            }

            c.G_y = Gy;
            c.G_field = Gpsi;
            c.setFieldSolveMode(Mode::Exact);
            c.solveAdjointState();
            const Vector want = adjointVector(c).head(nTransport);

            for (Mode mode : {Mode::Krylov, Mode::Iterative})
            {
                c.G_y = Gy;
                c.G_field = Gpsi;
                c.setFieldSolveMode(mode);
                c.solveAdjointState();
                const double diff = (adjointVector(c).head(nTransport) - want).norm() / want.norm();
                BOOST_TEST_MESSAGE("tier 2, mode " << static_cast<int>(mode) << " against exact: " << diff);
                BOOST_TEST(diff < 1e-7);
            }

            for (Mode mode : {Mode::Exact, Mode::Krylov, Mode::Iterative})
            {
                r.G_y = Gy;
                r.G_geometry = Ggeom;
                r.setFieldSolveMode(mode);
                r.solveAdjointState();
                const double diff = (adjointVector(r) - want).norm() / want.norm();
                BOOST_TEST_MESSAGE("tier 1, mode " << static_cast<int>(mode) << " against tier 2 exact: " << diff);
                BOOST_TEST(diff < (mode == Mode::Exact ? 1e-12 : 1e-7));
            }
        }
    }
}

// The default transposed response is the assembled G': its transpose against
// the adapter's exact one, on one sensitivity, and built once per equilibrium.
BOOST_AUTO_TEST_CASE(the_assembled_transpose_is_the_transpose_of_the_response)
{
    auto pinch = reducedThetaPinch(5, 3, false);
    SystemSolver &sys = *pinch.sys;
    auto &b = sys.reducedBlocks;
    FieldModelEquilibrium exact(pinch.field, false), differenced(pinch.field, true);

    const Eigen::Index nPoints = static_cast<Eigen::Index>(b.physicsPoints.size());
    Matrix dg(sys.getGeometrySlots(), nPoints);
    for (Eigen::Index j = 0; j < dg.size(); ++j)
        dg.data()[j] = std::sin(0.2 + 1.3 * static_cast<double>(j));

    GlobalState dpExact = b.states->zeroedLike(), dpDiff = b.states->zeroedLike();
    exact.applyResponseTranspose(dpExact, *b.eq, *b.states, dg, b.inputPoints, b.weights, b.t, b.physicsPoints);
    const long before = exact.newtonIterations() + differenced.newtonIterations();
    differenced.applyResponseTranspose(dpDiff, *b.eq, *b.states, dg, b.inputPoints, b.weights, b.t, b.physicsPoints);
    const long assembling = differenced.newtonIterations();
    differenced.applyResponseTranspose(dpDiff, *b.eq, *b.states, dg, b.inputPoints, b.weights, b.t, b.physicsPoints);
    BOOST_TEST(differenced.newtonIterations() == assembling);
    BOOST_TEST(before == 0);

    auto all = [](GlobalState const &s)
    {
        Vector out(s.Variable().size() + s.Derivative().size() + s.Flux().size() + s.Aux().size());
        out << Eigen::Map<const Vector>(s.Variable().data(), s.Variable().size()),
            Eigen::Map<const Vector>(s.Derivative().data(), s.Derivative().size()),
            Eigen::Map<const Vector>(s.Flux().data(), s.Flux().size()),
            Eigen::Map<const Vector>(s.Aux().data(), s.Aux().size());
        return out;
    };
    const Vector a = all(dpExact), d = all(dpDiff);
    const double diff = (a - d).norm() / a.norm();
    BOOST_TEST_MESSAGE("G'^T dg, exact against assembled: " << diff << " relative; |G'^T dg| " << a.norm()
                                                           << "; assembling took " << assembling
                                                           << " Newton iterations");
    BOOST_TEST(a.norm() > 1e-3);
    BOOST_TEST(diff < 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()
