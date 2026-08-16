// The adjoint through the field coupling.
//
// This is the one place in the coupling where a missing block is *silently
// wrong* rather than merely slow. Everywhere else the Jacobian is never
// assembled and IDA tolerates an inexact linear solve, so a wrong A1 or A2
// costs Newton iterations and the answer is unmoved. Here
// initializeMatricesForAdjointSolve stores the forward blocks' transpose and
// that transpose *is* the operator being inverted: a coupling block present
// forwards and absent here gives a wrong gradient beside a perfectly good G,
// with nothing failing. The dSigma/dPhi block was missing for exactly that
// reason and cost nothing visible until python/Tests/test_adjoint_aux.py was
// written.
//
// So every gradient below is checked twice -- against a finite difference of
// the objective, computed by re-running the solver and reusing nothing of the
// adjoint, and (for the closed-form fixture) against the analytic dG/dp of a
// problem whose coupled steady state is exact in the discrete space. And the
// vacuity guard zeroes each transposed block in turn and *requires* the
// finite-difference check to fail, because a gradient test on an objective
// that never sees the coupling passes for the wrong reason.
#include <boost/test/unit_test.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "CapturedOutput.hpp"
#include "FiniteDifferenceJacobian.hpp"
#include "ManufacturedFields.hpp"
#include "../../AdjointProblem.hpp"
#include "../../NetCDFIO.hpp"
#include "../../SystemSolver.hpp"
#include "../../Types.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using std::numbers::pi;

// The parameter vector every fixture here differentiates with respect to.
constexpr Index P_KAPPA = 0, P_S0 = 1;
constexpr Index NP = 2;

// The grid the manufactured field models take and ignore. Static so the
// reference their constructor is handed outlives every model built from it.
Grid const &scratchGrid()
{
    static const Grid g(0.0, 1.0, 1);
    return g;
}

// ------------------------------------------------------ the closed-form case --
//
// The coupled problem
//
//     -( kappa u' )' = s0 g(x; psi),   g = 1 + psi x,   u(0) = u(1) = 0
//     psi            = Int_0^1 u dx
//
// has the exact steady solution
//
//     u(x) = (s0/kappa) [ (x - x^2)/2 + psi (x - x^3)/6 ],
//     psi  = 2 s0 / (24 kappa - s0)
//
// and psi is a genuine fixed point -- it is not obtainable without solving the
// coupling. Two properties make it worth the arithmetic rather than another
// finite-difference-only fixture:
//
//  * u is a cubic, sigma_hat = kappa u' a quadratic and S = s0(1 + psi x)
//    linear, so at k = 3 every interpolation the residual performs is exact,
//    the discrete solution *is* the analytic one, and the interpolatory
//    quadrature G = sum_m w_m u_m is exactly Int u dx. The closed form is then
//    a legitimate reference rather than an approximation, and no tolerance has
//    to be chosen to make anything pass.
//
//  * with kappa = 0.5 and s0 = 4 the feedback gain s0/(24 kappa) is 1/3, so
//    freezing psi -- which is what a zero A1^T or A2^T amounts to -- moves the
//    gradient by 33%. The vacuity guard therefore fails by a mile rather than
//    by a hair.
constexpr double KAPPA0 = 0.5, S0_0 = 4.0;

double exactPsi(double kappa, double s0) { return 2.0 * s0 / (24.0 * kappa - s0); }

/// G = Int u dx, which for this problem is psi itself.
double exactG(double kappa, double s0) { return exactPsi(kappa, s0); }

Vector exactGradient(double kappa, double s0)
{
    const double d = 24.0 * kappa - s0;
    Vector g(NP);
    g(P_KAPPA) = -48.0 * s0 / (d * d);
    g(P_S0) = 48.0 * kappa / (d * d);
    return g;
}

/// The gradient of the *same* objective with psi frozen at its fixed-point
/// value, i.e. the answer a zero A1 or A2 gives:
///
///     G_frozen(p) = (s0/kappa) ( 1/12 + psi/24 ),   psi held fixed
///
/// Exactly 2/3 of exactGradient in both components at (KAPPA0, S0_0). This is
/// what the vacuity guard measures against: "the gradient moved" is a much
/// weaker statement than "the gradient moved to precisely the value dropping the
/// coupling predicts".
Vector frozenPsiGradient(double kappa, double s0)
{
    const double c = 1.0 / 12.0 + exactPsi(kappa, s0) / 24.0;
    Vector g(NP);
    g(P_KAPPA) = -s0 * c / (kappa * kappa);
    g(P_S0) = c / kappa;
    return g;
}

/// sigma_hat = kappa q,  S = s0 g. Geometry enters the source alone, which is
/// what keeps the exact solution polynomial: a geometry-dependent diffusivity
/// would make it a quadrature of 1/g and the closed form would go.
class ClosedFormCoupledDiffusion : public TransportSystem
{
public:
    explicit ClosedFormCoupledDiffusion(Vector p)
        : TransportSystem({.variables = {{"u", "the diffused quantity", "",
                                          BoundaryKind::Dirichlet, BoundaryKind::Dirichlet}}}),
          params(std::move(p))
    {
    }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return params(P_KAPPA) * s.q(0);
    }
    Value Sources(Index, const State &s, Position, Time) override
    {
        return params(P_S0) * s.geom(0);
    }

    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = params(P_KAPPA);
    }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    /// The whole of A1 for this case: dS/dg = s0, in the u row block.
    void dSources_dGeometry(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = params(P_S0);
    }

    Value InitialValue(Index, Position) const override { return 0.0; }
    Value InitialDerivative(Index, Position) const override { return 0.0; }
    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Vector params;
};

/// ManufacturedField's constraint (psi = Int u dx) with a *linear* geometry
/// shape, so the manufactured source stays in P_k and the closed form above is
/// exact. ManufacturedField's own cos(pi x) would not be.
class LinearGeometryField : public ManufacturedField
{
public:
    LinearGeometryField() : ManufacturedField(toml::value{}, scratchGrid()) {}

    void Geometry(VectorRef out, Vector const &psi, Position x, Time) override
    {
        out(0) = 1.0 + psi(0) * x;
    }
    void dGeometry_dpsi(MatrixRef out, Vector const &, Position x, Time) override
    {
        out(0, 0) = x;
    }
    /// Zero rather than ManufacturedField's (2/pi): u starts at zero here, and a
    /// psi inconsistent with it is only work for IDACalcIC.
    void InitialFieldValue(VectorRef out) override { out(0) = 0.0; }
};

/// LinearGeometryField whose transposed field solve fails partway through the
/// adjoint sweep.
///
/// The only way left to reach Solver.cpp's exception_ptr guard, and it has to be
/// kept reachable: solveCoupledAdjointIterative no longer throws on
/// non-convergence -- it escalates to the exact transposed solve -- but
/// solveCoupledAdjointExact can still throw out of the field model, and the
/// guard is what stops a netCDF file dying with it.
///
/// The *second* call, not the first, because solveBTranspose is reached only
/// from the iterative adjoint sweep -- the forward solve uses solveB and the
/// exact adjoint uses applyBTranspose -- so the forward run completes untouched
/// and it is the adjoint alone that fails. Failing on the second call rather
/// than the first also means the escalation has genuinely been entered rather
/// than the very first sweep having been refused.
class FailingTransposeField : public LinearGeometryField
{
public:
    void solveBTranspose(VectorRef out, Vector const &rhs) const override
    {
        if (++calls >= 2)
            throw std::runtime_error("the field model refuses to solve B^T");
        LinearGeometryField::solveBTranspose(out, rhs);
    }

    mutable int calls = 0;
};

/// The same constraint made *differential*:
///
///     R = dpsi/dt + psi - Int u dx
///
/// so dRdpsi = 1 and dRddpsidt = 1, the steady state is the same fixed point as
/// LinearGeometryField's, and the closed-form G and dG/dp above apply unchanged.
/// A controlled comparison, not a second problem.
///
/// This exists for one line of the solver. initializeMatricesForAdjointSolve
/// re-assembles the field coupling at **alpha = 0** rather than reusing the
/// forward solve's blocks, because the M it builds beside them omits
/// assembleCellMatrix's alpha-weighted mass term and so is the *steady*
/// Jacobian. B = dRdpsi + alpha dRddpsidt is where that bites: for an algebraic
/// DOF dRddpsidt is zero and alpha is multiplied by nothing, so the choice is
/// unobservable -- and every other field model in the tree, including every one
/// this file's other tests use, is algebraic. Passing IDA's last cj instead of
/// zero leaves the whole suite green and puts this fixture's gradient 33% out,
/// beside a G correct to seven figures. Which is the failure this file exists to
/// prevent, sitting on the subtlest decision in the code it tests.
class DifferentialLinearGeometryField : public FieldModel
{
public:
    DifferentialLinearGeometryField() : FieldModel(buildSpec()) {}

    static FieldModelSpec buildSpec()
    {
        FieldModelSpec s;
        s.dofs = {{"psi", "a differential manufactured field unknown", "1",
                   /* differential */ true}};
        s.geometry = {{"g", "metric factor multiplying the source", "1"}};
        s.label = "x";
        return s;
    }

    void FieldResidual(VectorRef out, Vector const &psi, Vector const &dpsidt,
                       GlobalState const &states, std::vector<Position> const &,
                       Vector const &weights, Time) override
    {
        double integral = 0.0;
        for (Index j = 0; j < weights.size(); ++j)
            integral += weights(j) * states[j].u(0);
        out(0) = dpsidt(0) + psi(0) - integral;
    }

    void Geometry(VectorRef out, Vector const &psi, Position x, Time) override
    {
        out(0) = 1.0 + psi(0) * x;
    }
    void dGeometry_dpsi(MatrixRef out, Vector const &, Position x, Time) override
    {
        out(0, 0) = x;
    }

    void FieldResidualPrime(GlobalStateMatrix &dR, GlobalStateMatrix &, MatrixRef dRdpsi,
                            MatrixRef dRddpsidt, Vector const &, Vector const &,
                            GlobalState const &, std::vector<Position> const &,
                            Vector const &weights, Time) override
    {
        dRdpsi(0, 0) = 1.0;
        // The block that makes alpha observable, and the one initialize()
        // requires a differential DOF to have.
        dRddpsidt(0, 0) = 1.0;
        dR[0].Variable().row(0) = -weights.transpose();
    }

    /// Consistent with u = 0 at t = 0: dpsi/dt = -psi + Int u dx is then zero, so
    /// IDACalcIC -- which holds a differential *value* fixed -- has nothing to
    /// fight.
    void InitialFieldValue(VectorRef out) override { out(0) = 0.0; }
};

// ------------------------------------------------------------ the rich case --
//
// Geometry in the flux, the source *and* the auxiliary constraint, so all three
// row blocks of A1 are nonzero -- the closed-form case above reaches only the u
// row. Checked against finite differences alone; there is no closed form for it
// and inventing a tolerance to pretend otherwise would be worse than not
// trying.
//
//     sigma_hat = kappa g^2 q + 0.5 phi
//     S         = s0 sin(3x) - 0.5 g u + 0.2 q
//     G_aux     = phi - g u
class RichGeometricDiffusion : public TransportSystem
{
public:
    explicit RichGeometricDiffusion(Vector p)
        : TransportSystem({.variables = {{"u", "the diffused quantity", "",
                                          BoundaryKind::Dirichlet, BoundaryKind::Dirichlet}},
                           .aux = {{"phi", "an auxiliary quantity", ""}}}),
          params(std::move(p))
    {
    }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        const double g = s.geom(0);
        return params(P_KAPPA) * g * g * s.q(0) + 0.5 * s.phi(0);
    }
    Value Sources(Index, const State &s, Position x, Time) override
    {
        return params(P_S0) * std::sin(3.0 * x) - 0.5 * s.geom(0) * s.u(0) + 0.2 * s.q(0);
    }
    Value AuxG(Index, const State &s, Position, Time) override
    {
        return s.phi(0) - s.geom(0) * s.u(0);
    }

    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = params(P_KAPPA) * s.geom(0) * s.geom(0);
    }
    void dSigma_dPhi(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.5; }

    void dSources_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = -0.5 * s.geom(0);
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

    // The three that make all of A1 nonzero.
    void dSigmaFn_dGeometry(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 2.0 * params(P_KAPPA) * s.geom(0) * s.q(0);
    }
    void dSources_dGeometry(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = -0.5 * s.u(0);
    }
    void dAuxG_dGeometry(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = -s.u(0);
    }

    Value InitialValue(Index, Position x) const override { return std::sin(pi * x); }
    Value InitialDerivative(Index, Position x) const override { return pi * std::cos(pi * x); }
    Value InitialAuxValue(Index, Position x) const override { return std::sin(pi * x); }
    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Vector params;
};

/// A2's four slots, not just the u one. Every FieldResidualPrime in the tree
/// constrains psi to an integral of u alone, so three quarters of the A2
/// transpose would be unexercised without this -- and A2^T is only ever
/// *applied*, never printed, so a slot sourced from the wrong field would be a
/// wrong number with nothing to say so.
///
///     R = psi - Int ( u + 1/2 sigma + 1/4 q + 3/4 phi ) dx
class EverySlotField : public ManufacturedField
{
public:
    EverySlotField() : ManufacturedField(toml::value{}, scratchGrid()) {}

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
        out(0) = psi(0) - integral;
    }

    void FieldResidualPrime(GlobalStateMatrix &dR, GlobalStateMatrix &, MatrixRef dRdpsi,
                            MatrixRef, Vector const &, Vector const &, GlobalState const &,
                            std::vector<Position> const &, Vector const &weights, Time) override
    {
        dRdpsi(0, 0) = 1.0;

        // Through GlobalState's whole-matrix accessors: operator[](Index)
        // returns a State by value, so writing through it modifies nothing.
        dR[0].Variable().row(0) = -weights.transpose();
        dR[0].Flux().row(0) = -wSigma * weights.transpose();
        dR[0].Derivative().row(0) = -wQ * weights.transpose();
        dR[0].Aux().row(0) = -wPhi * weights.transpose();
    }

    void InitialFieldValue(VectorRef out) override { out(0) = 0.0; }
};

// ------------------------------------------------------------- the objective --

/// G = Int I_h[g] dx by the basis's own interpolatory weights -- the same
/// quadrature initializeMatricesForAdjointSolve differentiates to build G_y, so
/// the reported objective and the reported gradient are exactly a function and
/// its derivative. PyAdjointProblem::GFn is written this way for the same
/// reason; this is its C++ twin.
Value integrateOnNodes(AdjointProblem const &ap, Index gIndex, DGSoln &Y)
{
    const GlobalState states = Y.evalOnNodes();
    const std::vector<Position> points = Y.getPoints();
    const Values g = ap.gFn(gIndex, states, points);

    const Index order = static_cast<Index>(Y.getBasis().Order());
    Value out = 0.0;
    for (size_t i = 0; i < Y.getGrid().getNCells(); ++i)
    {
        const Interval &I = Y.getGrid()[static_cast<Index>(i)];
        const Vector weights = Y.getBasis().getIntegrationWeights(I);
        out += g.segment(static_cast<Index>(i) * (order + 1), order + 1).dot(weights);
    }
    return out;
}

/// Shared by both adjoint problems: the parameters enter through the physics
/// alone, so the *explicit* dG/dp is zero and the entire gradient has to come
/// out of the adjoint solve. That is what makes these tests statements about
/// the adjoint rather than about dGFndp.
class CoupledAdjointBase : public AdjointProblem
{
public:
    CoupledAdjointBase()
    {
        ng = 1;
        np = NP;
        np_boundary = 0;
    }

    Value GFn(Index gIndex, DGSoln &Y) const override { return integrateOnNodes(*this, gIndex, Y); }
    Value dGFndp(Index, Index, DGSoln &) const override { return 0.0; }

    void dgFn_dq(Index, VectorRef v, const State &, Position) override { v[0] = 0.0; }
    void dgFn_dsigma(Index, VectorRef v, const State &, Position) override { v[0] = 0.0; }
    void dgFn_dphi(Index, VectorRef v, const State &, Position) override
    {
        // Sized nAux, which is zero for the closed-form case and one for the
        // rich one. Written elementwise so both are correct.
        for (Index i = 0; i < v.size(); ++i)
            v[i] = 0.0;
    }
};

/// G = Int u dx.
class ClosedFormAdjoint : public CoupledAdjointBase
{
public:
    Value gFn(Index, const State &s, Position) const override { return s.u(0); }
    void dgFn_du(Index, VectorRef v, const State &, Position) override { v[0] = 1.0; }

    /// sigma_hat = kappa q, so d(sigma_hat)/d kappa = q and nothing depends on s0.
    void dSigmaFn_dp(Index, Index pIndex, Value &out, const State &s, Position) override
    {
        out = (pIndex == P_KAPPA) ? s.q(0) : 0.0;
    }
    /// S = s0 g, so dS/d s0 = g. **This reads geometry**, which is what makes
    /// the evaluateGeometry call in computeAdjointGradients load bearing:
    /// without it this hook sees a zero-length geometry vector, and the gradient
    /// with respect to s0 is wrong by exactly the psi-dependent part of g.
    void dSources_dp(Index, Index pIndex, Value &out, const State &s, Position) override
    {
        out = (pIndex == P_S0) ? s.geom(0) : 0.0;
    }
};

/// G = Int 1/2 u^2 dx.
class RichAdjoint : public CoupledAdjointBase
{
public:
    Value gFn(Index, const State &s, Position) const override { return 0.5 * s.u(0) * s.u(0); }
    void dgFn_du(Index, VectorRef v, const State &s, Position) override { v[0] = s.u(0); }

    void dSigmaFn_dp(Index, Index pIndex, Value &out, const State &s, Position) override
    {
        out = (pIndex == P_KAPPA) ? s.geom(0) * s.geom(0) * s.q(0) : 0.0;
    }
    void dSources_dp(Index, Index pIndex, Value &out, const State &, Position x) override
    {
        out = (pIndex == P_S0) ? std::sin(3.0 * x) : 0.0;
    }
    void dAux_dp(Index, Index, Value &out, const State &, Position) override { out = 0.0; }
};

// ----------------------------------------------------------------- the runs --

/// Everything one coupled adjoint run owns. Declaration order is load bearing:
/// members are destroyed in reverse, so `sys` goes first and the objects it
/// holds raw pointers to outlive it.
class AdjointRun
{
public:
    AdjointRun() = default;
    AdjointRun(AdjointRun const &) = delete;
    AdjointRun &operator=(AdjointRun const &) = delete;
    AdjointRun(AdjointRun &&) = default;

    ~AdjointRun()
    {
        if (sys)
            sys->destroySundials();
    }

    SystemSolver &operator*() const { return *sys; }
    SystemSolver *operator->() const { return sys.get(); }

    /// The objective at the state the run finished in. Reads yJac, which owns
    /// its memory and so survives destroySundials().
    double objective() const { return adjoint->GFn(0, sys->yJac); }

    /// dG/dp as computeAdjointGradients left it.
    Vector gradient() const { return sys->G_p.row(0).transpose(); }

    std::unique_ptr<TransportSystem> problem;
    std::unique_ptr<AdjointProblem> adjoint;
    std::shared_ptr<FieldModel> field;
    std::unique_ptr<Grid> grid;
    std::unique_ptr<SystemSolver> sys;
};

// ------------------------------------------- the transpose, checked directly --
//
// The end-to-end gradient checks below say the whole chain is right. This says
// which *link* is: the adjoint solve applies J^T, so finite-difference J at
// cj = 0 and require J^T z = g. It is the adjoint twin of
// field_jacobian_tests' "the exact solve inverts a finite-differenced coupled
// Jacobian", and it is what localises a failure to the operator rather than to
// F_p or to a state that is not quite steady.

/// A coupled solver sitting at a state with its adjoint matrices built, and the
/// N_Vectors that state lives in.
class AdjointStateFixture
{
public:
    AdjointStateFixture() = default;
    AdjointStateFixture(AdjointStateFixture const &) = delete;
    AdjointStateFixture(AdjointStateFixture &&) = default;

    ~AdjointStateFixture()
    {
        for (N_Vector v : {Y, dYdt})
            if (v)
                N_VDestroy(v);
    }

    SystemSolver *operator->() const { return sys.get(); }

    std::unique_ptr<TransportSystem> problem;
    std::unique_ptr<AdjointProblem> adjoint;
    std::shared_ptr<FieldModel> field;
    std::unique_ptr<Grid> grid;
    std::unique_ptr<SystemSolver> sys;
    N_Vector Y = nullptr, dYdt = nullptr;
    double t = 0.0;
};

AdjointStateFixture makeAdjointStateFixture(std::unique_ptr<TransportSystem> problem,
                                            std::unique_ptr<AdjointProblem> adjoint,
                                            std::shared_ptr<FieldModel> field, Index nCells,
                                            Index k, bool superconvergent = false)
{
    AdjointStateFixture h;
    h.problem = std::move(problem);
    h.adjoint = std::move(adjoint);
    h.field = std::move(field);
    h.grid = std::make_unique<Grid>(0.0, 1.0, nCells);
    h.sys = std::make_unique<SystemSolver>(*h.grid, k, h.problem.get());
    h.sys->setTau(0.75);
    h.sys->setInitialTime(0.0);
    h.sys->setSuperconvergent(superconvergent);
    h.sys->setAdjointProblem(h.adjoint.get());
    h.sys->setFieldModel(h.field);
    h.sys->resetCoeffs();
    h.sys->initialiseMatrices();

    const size_t dof = h.sys->getSolution().getDoF();
    h.Y = N_VNew_Serial(dof, h.sys->ctx);
    h.dYdt = N_VClone(h.Y);
    N_VConst(0.0, h.Y);
    N_VConst(0.0, h.dYdt);
    h.sys->setInitialConditions(h.Y, h.dYdt);

    // Push psi off the value InitialFieldValue returned, so the field column of
    // the Jacobian is not evaluated at a degenerate point.
    DGSoln yMap(h.sys->nVars, *h.grid, k, N_VGetArrayPointer(h.Y), h.sys->nScalars,
                h.sys->nAux, h.sys->getFieldDOF());
    for (Index m = 0; m < h.sys->getFieldDOF(); ++m)
        yMap.Field(m) += 0.2 + 0.05 * static_cast<double>(m);

    h.sys->setJacTime(0.0);
    // cj = 0: the adjoint operator is the *steady* Jacobian's transpose --
    // initializeMatricesForAdjointSolve builds M from A/B/D directly and never
    // adds assembleCellMatrix's alpha-weighted mass term.
    h.sys->setAlpha(0.0);
    h.sys->setJacEvalY(h.Y, h.dYdt);
    h.sys->updateBoundaryConditions(0.0);
    h.sys->initializeMatricesForAdjointSolve();
    return h;
}

/// Assemble the adjoint state into one full-length vector, in the solution
/// vector's own ordering: [ sigma | q | u | aux ] per cell, then all of lambda,
/// then psi. adjoint_lambdas is already variable-major over (nCells + 1), which
/// is the lambda block's own layout.
Vector adjointStateVector(SystemSolver const &sys, Index nCells, Index localDOF)
{
    const Index nLambda = sys.adjoint_lambdas.size();
    const Index nField = sys.getFieldDOF();
    Vector z(nCells * localDOF + nLambda + nField);
    for (Index i = 0; i < nCells; ++i)
        z.segment(i * localDOF, localDOF) = sys.adjoint_squ[i];
    z.segment(nCells * localDOF, nLambda) = sys.adjoint_lambdas;
    if (nField > 0)
        z.tail(nField) = sys.adjoint_field;
    return z;
}

/// ||J^T z - g|| / ||g||, over the rows the residual actually defines.
double checkAdjointTranspose(AdjointStateFixture &h, int trial)
{
    SystemSolver &sys = *h.sys;
    const Index nCells = static_cast<Index>(sys.nCells);
    const Index localDOF = static_cast<Index>(sys.localDOF);

    const Matrix J = fdjac::jacobian(sys, h.Y, h.dYdt, h.t, 0.0);
    const std::vector<Index> dead = fdjac::undefinedRows(J);
    for (Index i : dead)
        BOOST_REQUIRE_EQUAL(J.col(i).cwiseAbs().maxCoeff(), 0.0);

    // An arbitrary right-hand side, deliberately not shaped like a dg: the
    // objective's own G_y would exercise only whichever blocks it happens to
    // reach.
    Vector g = Vector::Zero(J.rows());
    for (Index i = 0; i < nCells; ++i)
    {
        Vector cell(localDOF);
        for (Index j = 0; j < localDOF; ++j)
            cell(j) = std::sin(0.4 + (trial + 1) * 0.9 * static_cast<double>(i * localDOF + j));
        sys.G_y[i] = cell;
        g.segment(i * localDOF, localDOF) = cell;
    }
    // The field row of the adjoint right-hand side. Zero in production -- there
    // is no dg/dgeometry hook -- but the solve must invert the transpose for a
    // general right-hand side, not only for the one the objective supplies.
    for (Index f = 0; f < sys.getFieldDOF(); ++f)
        sys.G_field(f) = std::cos(0.3 + 1.7 * static_cast<double>(f) + trial);
    if (sys.getFieldDOF() > 0)
        g.tail(sys.getFieldDOF()) = sys.G_field;

    sys.solveAdjointState(0);

    const Vector z = adjointStateVector(sys, nCells, localDOF);
    return fdjac::relativeResidual(Matrix(J.transpose()), z, g, dead);
}

/// Configure far enough to run, and to write nothing.
void configureQuietly(SystemSolver &sys, std::string const &stem)
{
    sys.setTau(1.0);
    sys.setInputFile(stem);
    sys.setOutputCadence(1.0);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-14);
    // Tight, so that the finite-difference reference below is limited by its own
    // step size rather than by how far from steady the state is: the adjoint
    // state method assumes F(y, p) = 0, and a loose integration is the one error
    // a gradient check cannot tell from a wrong transpose.
    sys.setTolerances({1e-12}, 1e-11);
    sys.setWriteOutput(false);
    sys.setWriteDatFile(false);
}

AdjointRun buildRun(std::unique_ptr<TransportSystem> problem,
                    std::unique_ptr<AdjointProblem> adjoint,
                    std::shared_ptr<FieldModel> field, Index nCells, Index k,
                    std::string const &stem, SystemSolver::FieldSolveMode mode,
                    bool solveAdjoint = true, bool superconvergent = false)
{
    AdjointRun h;
    h.problem = std::move(problem);
    h.adjoint = std::move(adjoint);
    h.field = std::move(field);
    h.grid = std::make_unique<Grid>(0.0, 1.0, nCells);
    h.sys = std::make_unique<SystemSolver>(*h.grid, k, h.problem.get());
    configureQuietly(*h.sys, stem);
    h.sys->setOutputCadence(1.0);
    h.sys->setSuperconvergent(superconvergent);
    h.sys->setFieldModel(h.field);
    h.sys->setFieldSolveMode(mode);
    h.sys->setAdjointProblem(h.adjoint.get());
    h.sys->setSolveAdjoint(solveAdjoint);
    h.sys->resetCoeffs();
    return h;
}

/// Long enough that the diffusive transient (t ~ 1/(kappa pi^2) ~ 0.2) is
/// thoroughly dead. The adjoint state method assumes F(y, p) = 0, which only
/// holds once du/dt has decayed.
constexpr double T_FINAL = 20.0;

void integrateQuietly(AdjointRun &h, double tFinal = T_FINAL)
{
    CapturedOutput quiet;
    h->initialize();
    h->integrate(tFinal);
}

AdjointRun closedFormRun(Vector const &p, Index nCells, Index k, std::string const &stem,
                         SystemSolver::FieldSolveMode mode =
                             SystemSolver::FieldSolveMode::Iterative,
                         bool solveAdjoint = true)
{
    return buildRun(std::make_unique<ClosedFormCoupledDiffusion>(p),
                    std::make_unique<ClosedFormAdjoint>(),
                    std::make_shared<LinearGeometryField>(), nCells, k, stem, mode,
                    solveAdjoint);
}

AdjointRun richRun(Vector const &p, Index nCells, Index k, std::string const &stem,
                   SystemSolver::FieldSolveMode mode = SystemSolver::FieldSolveMode::Exact)
{
    return buildRun(std::make_unique<RichGeometricDiffusion>(p),
                    std::make_unique<RichAdjoint>(), std::make_shared<EverySlotField>(),
                    nCells, k, stem, mode);
}

/// FieldSolve = exact by default, so that the finite-difference references built
/// from this factory are independent of the sweep under test. Five field
/// unknowns whose geometry interpolant reaches every cell is the hardest coupled
/// fixture here: before the Irons-Tuck acceleration and the escalation to the
/// exact solve, the transposed sweep could not reach
/// FieldSolveTolerance within the default twenty sweeps and the iterative
/// adjoint threw, which is what
/// a_five_dof_adjoint_reaches_the_same_gradient_either_way used to pin. It now
/// escalates instead, and that test says so.
AdjointRun multiDofRun(Vector const &p, Index nCells, Index k, std::string const &stem,
                       SystemSolver::FieldSolveMode mode = SystemSolver::FieldSolveMode::Exact)
{
    return buildRun(std::make_unique<ClosedFormCoupledDiffusion>(p),
                    std::make_unique<ClosedFormAdjoint>(),
                    std::make_shared<ManufacturedFieldVector>(toml::value{}, scratchGrid()),
                    nCells, k, stem, mode);
}

/// The closed-form problem again, with the field DOF declared *differential*.
/// Same fixed point, same closed-form G and dG/dp; the only thing that changes is
/// that B = dRdpsi + alpha dRddpsidt now depends on alpha, which is what makes
/// initializeMatricesForAdjointSolve's choice of alpha = 0 observable at all.
AdjointRun differentialRun(Vector const &p, Index nCells, Index k, std::string const &stem,
                           SystemSolver::FieldSolveMode mode =
                               SystemSolver::FieldSolveMode::Iterative)
{
    return buildRun(std::make_unique<ClosedFormCoupledDiffusion>(p),
                    std::make_unique<ClosedFormAdjoint>(),
                    std::make_shared<DifferentialLinearGeometryField>(), nCells, k, stem, mode);
}

using RunFactory = AdjointRun (*)(Vector const &, Index, Index, std::string const &);

AdjointRun closedFormFactory(Vector const &p, Index nCells, Index k, std::string const &stem)
{
    return closedFormRun(p, nCells, k, stem);
}
AdjointRun richFactory(Vector const &p, Index nCells, Index k, std::string const &stem)
{
    return richRun(p, nCells, k, stem);
}
AdjointRun differentialFactory(Vector const &p, Index nCells, Index k, std::string const &stem)
{
    return differentialRun(p, nCells, k, stem);
}

/// Central differences of the objective, computed by re-running the solver.
/// Nothing about the adjoint is reused.
Vector finiteDifferenceGradient(RunFactory factory, Vector const &p0, Index nCells, Index k,
                                std::string const &stem, double hRel = 1e-5,
                                double tFinal = T_FINAL)
{
    Vector fd(NP);
    for (Index i = 0; i < NP; ++i)
    {
        const double h = hRel * std::abs(p0(i));
        Vector plus = p0, minus = p0;
        plus(i) += h;
        minus(i) -= h;

        AdjointRun a = factory(plus, nCells, k, stem + "_p" + std::to_string(i));
        integrateQuietly(a, tFinal);
        AdjointRun b = factory(minus, nCells, k, stem + "_m" + std::to_string(i));
        integrateQuietly(b, tFinal);

        fd(i) = (a.objective() - b.objective()) / (2.0 * h);
    }
    return fd;
}

/// The residual of the *field* row of the transposed coupled system for the
/// adjoint state a solve left behind:
///
///     || G_psi - A1^T z_x - B^T z_psi ||   relative to the largest term in it.
///
/// Not "small": **exactly zero, to roundoff**, and that is the whole of
/// invariant 1. solveCoupledAdjointIterative's stopping test is a relative
/// *backward error* rather
/// than a proxy for one precisely because this row holds identically -- the
/// residual of the pair returned is then A2^T (z_psi^{n+1} - z_psi^n) in row one
/// and nothing in row two, which is what makes ||A2^T dz|| the exact backward
/// error. That derivation survives the Irons-Tuck acceleration if and only if
/// the iterate *accepted* is the unaccelerated solveBTranspose output; returning
/// an extrapolated z_psi instead leaves this row off by B^T mu delta and turns
/// an exact residual into an estimate, with nothing downstream to say so.
///
/// The gradient itself barely moves -- the extrapolation at the stopping point
/// is second order in the increment -- so no gradient check separates the two.
/// This does.
double adjointFieldRowResidual(SystemSolver &sys)
{
    Vector r = sys.G_field;
    for (Index i = 0; i < static_cast<Index>(sys.nCells); ++i)
        r -= sys.A1_transpose_cellwise[i] * sys.adjoint_squ[i];

    Vector BTz = Vector::Zero(sys.getFieldDOF());
    sys.getFieldModel()->applyBTranspose(BTz, sys.adjoint_field);

    const double scale = std::max({sys.G_field.norm(), (sys.G_field - r).norm(), BTz.norm(),
                                   std::numeric_limits<double>::min()});
    return (r - BTz).norm() / scale;
}

double relativeError(Vector const &a, Vector const &b) { return (a - b).norm() / b.norm(); }

Vector baseParameters() { return (Vector(NP) << KAPPA0, S0_0).finished(); }

} // namespace

BOOST_AUTO_TEST_SUITE(field_adjoint_tests)

BOOST_AUTO_TEST_CASE(a_coupled_run_no_longer_refuses_to_solve_its_adjoint)
{
    // runAdjointSolve threw outright on any attached field model until this
    // task -- deliberately, because the adjoint matrices carried neither
    // geometry nor a transpose of the coupling. The refusal named this task; it
    // is gone, and this is what says so.
    auto run = closedFormRun(baseParameters(), /*nCells=*/4, /*k=*/3, "field_adjoint_refusal");
    BOOST_CHECK_NO_THROW(integrateQuietly(run));

    BOOST_TEST(run->G_p.rows() == 1);
    BOOST_TEST(run->G_p.cols() == NP);
    BOOST_TEST(run.gradient().norm() > 0.0);
}

BOOST_AUTO_TEST_CASE(the_objective_matches_its_closed_form)
{
    // Before trusting a gradient, check the thing being differentiated. At k = 3
    // the discrete solution *is* the analytic one -- u is a cubic, the flux a
    // quadratic and the source linear -- so this is an equality, not a
    // convergence statement, and it pins the coupled steady state as well as the
    // quadrature.
    const Vector p = baseParameters();
    auto run = closedFormRun(p, /*nCells=*/4, /*k=*/3, "field_adjoint_G");
    integrateQuietly(run);

    const double G = run.objective();
    const double psi = run->getSolution().Field(0);
    BOOST_TEST_MESSAGE("closed-form coupled steady state: G = "
                       << G << " (exact " << exactG(KAPPA0, S0_0) << "), psi = " << psi
                       << " (exact " << exactPsi(KAPPA0, S0_0) << ")");

    BOOST_TEST(G == exactG(KAPPA0, S0_0), boost::test_tools::tolerance(1e-8));
    BOOST_TEST(psi == exactPsi(KAPPA0, S0_0), boost::test_tools::tolerance(1e-8));
}

BOOST_AUTO_TEST_CASE(the_gradient_matches_the_closed_form_and_a_finite_difference)
{
    // The load-bearing test. Both references are independent of the adjoint: the
    // closed form is analytic, and the finite difference re-runs the solver.
    const Vector p0 = baseParameters();
    const Index nCells = 4, k = 3;

    auto run = closedFormRun(p0, nCells, k, "field_adjoint_grad");
    integrateQuietly(run);
    const Vector adjointGrad = run.gradient();

    const Vector analytic = exactGradient(KAPPA0, S0_0);
    const Vector fd =
        finiteDifferenceGradient(&closedFormFactory, p0, nCells, k, "field_adjoint_grad_fd");

    BOOST_TEST_MESSAGE("closed-form coupled gradient:\n  adjoint          = "
                       << adjointGrad.transpose() << "\n  closed form      = "
                       << analytic.transpose() << "\n  finite difference = "
                       << fd.transpose() << "\n  ||adj - exact||/||exact|| = "
                       << relativeError(adjointGrad, analytic)
                       << ", ||adj - fd||/||fd|| = " << relativeError(adjointGrad, fd));

    BOOST_TEST(relativeError(adjointGrad, analytic) < 1e-7);
    BOOST_TEST(relativeError(adjointGrad, fd) < 1e-6);
}

BOOST_AUTO_TEST_CASE(the_uncoupled_gradient_is_a_third_wrong_here)
{
    // Why the fixture is worth its arithmetic: it quantifies what the coupling
    // is worth, so the vacuity guard below has a number to be measured against
    // rather than an assertion that something changed. Freezing psi -- which is
    // what a zero A1^T amounts to -- gives the gradient of
    //
    //     G_frozen(p) = (s0/kappa) ( 1/12 + psi/24 )   with psi held fixed
    //
    // and at these parameters that is 2/3 of the true derivative in both
    // components.
    const Vector frozen = frozenPsiGradient(KAPPA0, S0_0);
    const Vector truth = exactGradient(KAPPA0, S0_0);
    BOOST_TEST_MESSAGE("frozen-psi gradient " << frozen.transpose() << " against the true "
                                              << truth.transpose() << ": relative error "
                                              << relativeError(frozen, truth));
    BOOST_TEST(relativeError(frozen, truth) > 0.3);
}

BOOST_AUTO_TEST_CASE(dropping_a_transposed_coupling_block_makes_the_gradient_wrong)
{
    // The vacuity guard. Without it, the gradient checks above would pass on an
    // objective that never saw the coupling -- which is exactly how a zero-A1
    // fixture once made "the sweep converged" trivially true, in the forward
    // solve's own tests. maxAbsA1 in field_jacobian_tests exists for the same
    // reason.
    //
    // Reached through MANTA_TEST_PRIVATE, as in field_jacobian_tests: this is a
    // -DTEST build, so nothing test-only is added to SystemSolver.
    const Vector p0 = baseParameters();
    const Index nCells = 4, k = 3;

    auto run = closedFormRun(p0, nCells, k, "field_adjoint_guard");
    integrateQuietly(run);
    const Vector good = run.gradient();
    BOOST_REQUIRE(relativeError(good, exactGradient(KAPPA0, S0_0)) < 1e-7);

    // What dropping the coupling *should* give, in closed form.
    const Vector frozen = frozenPsiGradient(KAPPA0, S0_0);

    // ---- A1^T
    for (Matrix &block : run->A1_transpose_cellwise)
    {
        BOOST_REQUIRE_GT(block.cwiseAbs().maxCoeff(), 0.0);
        block.setZero();
    }
    run->solveAdjointState(0);
    run->computeAdjointGradients();
    const Vector noA1 = run.gradient();
    BOOST_TEST_MESSAGE("with A1^T zeroed: " << noA1.transpose() << " against " << good.transpose()
                                            << ", relative error " << relativeError(noA1, good)
                                            << "; against the frozen-psi closed form "
                                            << relativeError(noA1, frozen));
    BOOST_TEST(relativeError(noA1, good) > 1e-2);

    // ...and not merely wrong: wrong by exactly the amount freezing psi
    // predicts. That is what says the coupling contributes what the mathematics
    // says it should, rather than merely contributing something.
    BOOST_TEST(relativeError(noA1, frozen) < 1e-7);

    // ...restored, which also says initializeMatricesForAdjointSolve rebuilds
    // rather than grows the containers it appends to.
    run->initializeMatricesForAdjointSolve();
    run->solveAdjointState(0);
    run->computeAdjointGradients();
    BOOST_TEST(relativeError(run.gradient(), good) < 1e-12);

    // ---- A2^T. A1 and A2 enter the Schur complement as a product, so zeroing
    // only one leaves open the possibility that the other is never read.
    for (Matrix &block : run->A2_transpose_cellwise)
    {
        BOOST_REQUIRE_GT(block.cwiseAbs().maxCoeff(), 0.0);
        block.setZero();
    }
    run->solveAdjointState(0);
    run->computeAdjointGradients();
    const Vector noA2 = run.gradient();
    BOOST_TEST_MESSAGE("with A2^T zeroed: " << noA2.transpose() << ", relative error "
                                            << relativeError(noA2, good)
                                            << "; against the frozen-psi closed form "
                                            << relativeError(noA2, frozen));
    BOOST_TEST(relativeError(noA2, good) > 1e-2);
    BOOST_TEST(relativeError(noA2, frozen) < 1e-7);

    run->initializeMatricesForAdjointSolve();
    run->solveAdjointState(0);
    run->computeAdjointGradients();
    BOOST_TEST(relativeError(run.gradient(), good) < 1e-12);
}

BOOST_AUTO_TEST_CASE(the_exact_and_iterative_adjoint_solves_agree)
{
    // Not to roundoff: the iterative path stops at FieldSolveTolerance. What
    // this pins is that the two are solving the same transposed system -- a sign
    // error in either path's use of A1^T or A2^T separates them at once.
    const Vector p0 = baseParameters();
    const Index nCells = 4, k = 3;

    auto iterative = closedFormRun(p0, nCells, k, "field_adjoint_iter",
                                   SystemSolver::FieldSolveMode::Iterative);
    integrateQuietly(iterative);

    auto exact = closedFormRun(p0, nCells, k, "field_adjoint_exact",
                               SystemSolver::FieldSolveMode::Exact);
    integrateQuietly(exact);

    const double diff = relativeError(iterative.gradient(), exact.gradient());
    BOOST_TEST_MESSAGE("iterative vs exact adjoint gradient: relative difference " << diff);
    BOOST_TEST(diff < 1e-7);
}

BOOST_AUTO_TEST_CASE(an_exhausted_adjoint_sweep_still_gives_the_right_gradient)
{
    // The replacement for the throw this used to assert, and the reason the
    // throw could go. One sweep cannot converge, the escalation to the exact
    // transposed Schur solve fires, and the gradient is the exact one -- which
    // is the whole point of escalating rather than failing: the caller loses
    // time, never accuracy. Escalating is strictly stronger than refusing.
    //
    // The run itself is done with solveAdjoint off, so the cap below is imposed
    // on the *adjoint* alone and not on the forward solve that produced the
    // state -- otherwise this would be a test of whether IDA copes with a
    // one-sweep Jacobian.
    const Vector p0 = baseParameters();
    auto run = closedFormRun(p0, /*nCells=*/4, /*k=*/3, "field_adjoint_escalate",
                             SystemSolver::FieldSolveMode::Iterative, /*solveAdjoint=*/false);
    integrateQuietly(run);

    run->setSolveAdjoint(true);
    run->setFieldSolveMaxAdjointSweeps(1);
    run->setFieldSolveTolerance(1e-14);

    std::string log;
    {
        CapturedOutput capture;
        BOOST_CHECK_NO_THROW(run->runAdjointSolve());
        log = capture.text();
    }
    BOOST_TEST_MESSAGE("escalation: " << log);

    // The count, not the agreement, is the assertion that matters: the gradient
    // check below passes whether or not the escalation ever ran.
    BOOST_TEST(run->getFieldSweepStats().adjointFellBack);
    BOOST_TEST(relativeError(run.gradient(), exactGradient(KAPPA0, S0_0)) < 1e-7);

    // The warning has to be able to show the numbers it is about. std::to_string
    // is fixed to six decimals, so it would print this tolerance as "0.000000"
    // and a residual of 3.7e-5 as "0.000037" -- a convergence report with the
    // convergence redacted.
    BOOST_TEST(log.find("1e-14") != std::string::npos);
    BOOST_TEST(log.find("FieldSolveMaxAdjointSweeps") != std::string::npos);

    // ...and it is the cap that did it, not something that always escalates.
    // A fresh run rather than a second runAdjointSolve on this one, because
    // adjointFellBack is a per-*run* flag and only initialize() clears it.
    auto again = closedFormRun(p0, /*nCells=*/4, /*k=*/3, "field_adjoint_escalate_room",
                               SystemSolver::FieldSolveMode::Iterative, /*solveAdjoint=*/false);
    integrateQuietly(again);
    again->setSolveAdjoint(true);
    again->setFieldSolveTolerance(1e-14);
    {
        CapturedOutput quiet;
        BOOST_CHECK_NO_THROW(again->runAdjointSolve());
    }
    BOOST_TEST(!again->getFieldSweepStats().adjointFellBack);
    BOOST_TEST_MESSAGE("at the default adjoint cap: "
                       << again->getFieldSweepStats().adjointSweeps << " sweeps");
    BOOST_TEST(relativeError(again.gradient(), exactGradient(KAPPA0, S0_0)) < 1e-7);
}

BOOST_AUTO_TEST_CASE(the_adjoint_sweep_converges_on_the_rich_fixture)
{
    // RichGeometricDiffusion is where rho = 1.611 was measured, at cj = 0, which
    // is where every adjoint solve runs. Before this task the plain sweep
    // diverged there: the forward run could not be integrated on
    // FieldSolve = iterative at all -- IDA cut h to the floor over five attempts
    // at the first step, reporting Newton update norms of order 1e10 -- and the
    // adjoint threw. Both halves are checked here, and the fallback counts are
    // what say the acceleration did the work rather than the escalation.
    const Vector p0 = (Vector(NP) << 1.0, 1.0).finished();
    const Index nCells = 6, k = 2;

    auto run = richRun(p0, nCells, k, "field_adjoint_rich_iter",
                       SystemSolver::FieldSolveMode::Iterative);
    BOOST_CHECK_NO_THROW(integrateQuietly(run));

    const auto stats = run->getFieldSweepStats();
    BOOST_TEST_MESSAGE("rich fixture on iterative: " << stats.iterations
                       << " forward sweeps over " << stats.solves << " solves ("
                       << stats.fallbacks << " forward fallbacks), " << stats.adjointSweeps
                       << " adjoint sweeps, adjointFellBack = " << stats.adjointFellBack);
    BOOST_TEST(!stats.adjointFellBack);
    BOOST_TEST(stats.fallbacks == 0L);

    // The finite-difference reference runs on FieldSolve = exact -- richFactory's
    // default -- so it is independent of everything under test here.
    const Vector fd =
        finiteDifferenceGradient(&richFactory, p0, nCells, k, "field_adjoint_rich_iter_fd");
    BOOST_TEST_MESSAGE("rich iterative gradient " << run.gradient().transpose()
                       << " against finite difference " << fd.transpose() << ", relative error "
                       << relativeError(run.gradient(), fd));
    BOOST_TEST(relativeError(run.gradient(), fd) < 1e-5);
}

BOOST_AUTO_TEST_CASE(a_failing_field_model_does_not_destroy_the_runs_output)
{
    // runAdjointSolve is called from integrate() *before* finaliseDiagnostics,
    // nc_output.Close() and WriteRestartFile, and runSolver's catch(...)
    // rethrows. So a throw out of the adjoint would have thrown away the netCDF
    // and the restart file of a run that had integrated perfectly -- the
    // gradient is the optional half of a run, and the solution is not.
    //
    // integrate() logs the failure where it happens, finishes the output, and
    // rethrows. This checks that the restart file is written and complete and
    // that the exception still reaches the caller. It deliberately says nothing
    // about the .nc; see the comment on the assertions below for the measurement
    // behind that.
    //
    // **The forcing condition changed with this task and the test had to be
    // re-armed rather than deleted.** It used to force a refusal with
    // setFieldSolveMaxSweeps(1) and a tolerance of 1e-14 -- but that now
    // escalates to the exact transposed solve instead of throwing, so the test
    // would have passed while asserting nothing at all. The guard is still
    // needed, because solveCoupledAdjointExact can throw out of the field model,
    // so the mechanism is now a field model that refuses to solve B^T.
    const std::string stem = "field_adjoint_output_survives";
    const std::filesystem::path nc = stem + ".nc";
    const std::filesystem::path restart = stem + ".restart.nc";

    // Removed first, so a file left by an earlier run cannot make this pass
    // without anything having been written.
    std::filesystem::remove(nc);
    std::filesystem::remove(restart);

    auto field = std::make_shared<FailingTransposeField>();
    auto run = buildRun(std::make_unique<ClosedFormCoupledDiffusion>(baseParameters()),
                        std::make_unique<ClosedFormAdjoint>(), field, /*nCells=*/4, /*k=*/3, stem,
                        SystemSolver::FieldSolveMode::Iterative);
    run->setWriteOutput(true);
    run->setOutputCadence(5.0);

    bool threw = false;
    std::string message;
    {
        CapturedOutput quiet;
        run->initialize();
        try
        {
            run->integrate(T_FINAL);
        }
        catch (std::runtime_error const &e)
        {
            threw = true;
            message = e.what();
        }
    }

    BOOST_TEST(threw);
    BOOST_TEST_MESSAGE("failure reached the caller: " << message);
    // The forward run reached solveB, never solveBTranspose, so the failure is
    // the adjoint's alone -- and it took more than one sweep to get there, which
    // is what says the sweep was running rather than being refused outright.
    BOOST_TEST(field->calls >= 2);

    // **There are deliberately no assertions about the .nc, and that is the
    // honest answer to a review finding rather than a weaker one.**
    //
    // The finding was that the netCDF assertions this test used to carry were
    // vacuous, and the fix attempted
    // here first was to read the file back. That is no better, and it was
    // measured rather than argued: with the guard in Solver.cpp replaced by a
    // bare `throw;`, so that finaliseDiagnostics, nc_output.Close() and
    // WriteRestartFile are all skipped, the .nc still opens for read and is
    // *indistinguishable* from the guarded one through every netCDF-visible
    // property --
    //
    //     2 dims, 3 top-level vars (nVariables, t, x), 2 groups
    //     Grid: CellBoundaries[5] Index[5] PolyOrder[]
    //     u:    q[5,11] sigma[5,11] u[5,11] u_star[5,11]
    //     t = { 0, 5, 10, 15, 20 },  last u slice max|u| = 1.4999999999999989
    //
    // identical on both sides, because netCDF flushes each record as
    // WriteTimeslice writes it and nothing is appended to the file after the
    // adjoint solve. The only difference is the file's size on disk -- 26099
    // against 28715 bytes -- which is HDF5 bookkeeping written at close and
    // carries no readable content, so no content assertion can reach it.
    //
    // A test dressed up to look discriminating when it cannot be is worse than
    // one that admits its scope. The restart file is written by an explicit call
    // *after* the guard, so it is the thing that actually distinguishes the two,
    // and it carries the test alone.
    //
    // REQUIRE, not TEST: file_size() throws filesystem_error on a missing file,
    // so a non-fatal existence check here aborts the case on the very mutation
    // this test exists to catch, taking every later assertion with it.
    BOOST_REQUIRE(std::filesystem::exists(restart));
    BOOST_TEST(std::filesystem::file_size(restart) > 0u);
    {
        netCDF::NcFile f(restart.string(), netCDF::NcFile::FileMode::read);
        netCDF::NcVar Y = f.getGroup("RestartData").getVar("Y");
        BOOST_REQUIRE(!Y.isNull());
        // The whole DOF vector, psi included: a restart file written from the
        // uncoupled length formula would be one entry short and would read back
        // as consistent, which is why the length is what is checked.
        BOOST_TEST(Y.getDim(0).getSize() == run->getSolution().getDoF());
    }

    // Swept here rather than left for `make clean_data`, because the output
    // lands in whatever directory the binary was launched from, and
    // CLEAN_DATA_DIRS deliberately excludes Tests/UnitTests -- its .nc files are
    // tracked test *inputs*. Running one test from that directory is the
    // documented way to run one test, so this must not leave litter there.
    std::filesystem::remove(nc);
    std::filesystem::remove(restart);
}

BOOST_AUTO_TEST_CASE(a_five_dof_adjoint_reaches_the_same_gradient_either_way)
{
    // What this test used to assert was the refusal: the five-unknown field
    // block, whose geometry interpolant couples every cell to every field
    // unknown, could not reach FieldSolveTolerance within the default twenty
    // sweeps, and the iterative adjoint threw rather than returning a wrong
    // gradient. Rewritten rather than deleted, because the same forcing
    // condition now has to produce a correct gradient instead of an exception.
    //
    // The comparison is against the *exact* transposed Schur solve on the very
    // same state -- one integration, two adjoint solves -- so nothing about the
    // forward run or the finite-difference step can enter the difference.
    const Vector p0 = baseParameters();
    auto run = multiDofRun(p0, /*nCells=*/4, /*k=*/3, "field_adjoint_nocontract",
                           SystemSolver::FieldSolveMode::Exact);
    integrateQuietly(run);
    BOOST_REQUIRE_EQUAL(run->getFieldDOF(), ManufacturedFieldVector::N);
    BOOST_REQUIRE_EQUAL(run->getFieldSolveMaxAdjointSweeps(), 100);

    {
        CapturedOutput quiet;
        run->runAdjointSolve();
    }
    const Vector exactGrad = run.gradient();
    BOOST_REQUIRE_GT(exactGrad.norm(), 0.0);

    {
        CapturedOutput quiet;
        run->setFieldSolveMode(SystemSolver::FieldSolveMode::Iterative);
        BOOST_CHECK_NO_THROW(run->runAdjointSolve());
    }
    const auto stats = run->getFieldSweepStats();
    BOOST_TEST_MESSAGE("five-DOF adjoint on iterative: " << stats.adjointSweeps
                       << " sweeps, adjointFellBack = " << stats.adjointFellBack);

    // The count first: with the escalation in place, agreement alone would pass
    // even if the sweep never converged, because the fallback *is* the exact
    // solve this is being compared against.
    BOOST_TEST(!stats.adjointFellBack);
    BOOST_TEST(relativeError(run.gradient(), exactGrad) < 1e-7);

    // ...and the escalation is still reachable and still correct, forced by a
    // cap of one on a fresh run -- the same state cannot be reused, because
    // adjointFellBack is per-run.
    auto capped = multiDofRun(p0, /*nCells=*/4, /*k=*/3, "field_adjoint_nocontract_capped",
                              SystemSolver::FieldSolveMode::Exact);
    integrateQuietly(capped);
    capped->setFieldSolveMode(SystemSolver::FieldSolveMode::Iterative);
    capped->setFieldSolveMaxAdjointSweeps(1);
    {
        CapturedOutput quiet;
        BOOST_CHECK_NO_THROW(capped->runAdjointSolve());
    }
    BOOST_TEST(capped->getFieldSweepStats().adjointFellBack);
    BOOST_TEST(relativeError(capped.gradient(), exactGrad) < 1e-7);
}

BOOST_AUTO_TEST_CASE(the_accepted_adjoint_iterate_leaves_the_field_row_exact)
{
    // Invariant 1, backwards, where it is load bearing rather than tidy: the
    // sweep's stopping test is an *exact* backward error only because row two of
    // the transposed system holds identically, and it does so only when the
    // iterate accepted is the one solveBTranspose produced against this sweep's
    // z_x. Accepting the extrapolation instead moves the gradient by less than
    // any check here can see -- measured, the rich fixture's gradient does not
    // change in a single one of its sixteen digits -- while quietly demoting the
    // stopping test to a proxy.
    //
    // Five field unknowns, because that is where the accelerator is genuinely
    // approximate. With nField == 1 Irons-Tuck lands on the fixed point exactly,
    // the increment at the stopping sweep is driven to *zero*, and the
    // extrapolation is then identically the plain iterate -- so a single-DOF
    // fixture cannot tell the two apart at all.
    const Vector p0 = baseParameters();
    auto run = multiDofRun(p0, /*nCells=*/4, /*k=*/3, "field_adjoint_rowtwo",
                           SystemSolver::FieldSolveMode::Exact);
    integrateQuietly(run);
    BOOST_REQUIRE_EQUAL(run->getFieldDOF(), ManufacturedFieldVector::N);

    {
        CapturedOutput quiet;
        run->setFieldSolveMode(SystemSolver::FieldSolveMode::Iterative);
        run->runAdjointSolve();
    }
    const double r = adjointFieldRowResidual(*run);
    BOOST_TEST_MESSAGE("five-DOF adjoint, transposed field-row residual = " << r);

    // The sweep, not the escalation: the exact transposed solve satisfies this
    // row by construction, so a fallback would make the check vacuous.
    BOOST_TEST(!run->getFieldSweepStats().adjointFellBack);
    BOOST_CHECK_SMALL(r, 1e-14);
}

BOOST_AUTO_TEST_CASE(the_gradient_matches_finite_differences_through_every_a1_row_block)
{
    // The closed-form case reaches the u row of A1 alone, because its geometry
    // enters the source and nothing else. This one has geometry in the flux, the
    // source and the auxiliary constraint, and an A2 row reading all four slots.
    const Vector p0 = (Vector(NP) << 1.0, 1.0).finished();
    const Index nCells = 6, k = 2;

    auto run = richRun(p0, nCells, k, "field_adjoint_rich");
    integrateQuietly(run);

    // Structural first: all three row blocks of A1^T must actually carry
    // something, or "the gradient agrees" is a statement about the u row again.
    const Index nodes = k + 1;
    double maxSigma = 0.0, maxU = 0.0, maxAux = 0.0;
    for (Matrix const &block : run->A1_transpose_cellwise)
    {
        maxSigma = std::max(maxSigma, block.middleCols(0, nodes).cwiseAbs().maxCoeff());
        maxU = std::max(maxU, block.middleCols(2 * nodes, nodes).cwiseAbs().maxCoeff());
        maxAux = std::max(maxAux, block.middleCols(3 * nodes, nodes).cwiseAbs().maxCoeff());
    }
    BOOST_TEST_MESSAGE("A1^T row-block magnitudes: sigma = " << maxSigma << ", u = " << maxU
                                                             << ", aux = " << maxAux);
    BOOST_REQUIRE_GT(maxSigma, 0.0);
    BOOST_REQUIRE_GT(maxU, 0.0);
    BOOST_REQUIRE_GT(maxAux, 0.0);

    const Vector adjointGrad = run.gradient();
    const Vector fd = finiteDifferenceGradient(&richFactory, p0, nCells, k, "field_adjoint_rich_fd");

    BOOST_TEST_MESSAGE("rich coupled gradient:\n  adjoint           = "
                       << adjointGrad.transpose() << "\n  finite difference = " << fd.transpose()
                       << "\n  relative error = " << relativeError(adjointGrad, fd));
    BOOST_TEST(relativeError(adjointGrad, fd) < 1e-5);
}

BOOST_AUTO_TEST_CASE(a_differential_field_dof_is_differentiated_at_alpha_zero)
{
    // The one line of the solver nothing else here can reach.
    //
    // initializeMatricesForAdjointSolve calls assembleFieldCoupling with an
    // explicit 0.0 rather than with `alpha`, because the M it builds beside the
    // field blocks omits assembleCellMatrix's alpha-weighted mass term and so is
    // the *steady* Jacobian. B = dRdpsi + alpha dRddpsidt is where that choice
    // becomes visible -- and only for a *differential* field DOF, because an
    // algebraic one has dRddpsidt identically zero and multiplies alpha by
    // nothing.
    //
    // Every other field model in this file, and every field model on any adjoint
    // path in the tree, is algebraic. So passing IDA's last cj instead of zero
    // leaves the entire suite green and puts this fixture's gradient 33% out,
    // beside a G correct to seven figures. This fixture is that mutation's only
    // detector, which is the whole reason it is here: same closed form, one
    // declaration changed.
    const Vector p0 = baseParameters();
    const Index nCells = 4, k = 3;

    // Longer than T_FINAL, and the reason is the fixture rather than the
    // tolerance. Making psi differential gives the problem a genuine slow mode:
    // dpsi/dt = -psi + Int u dx, and Int u dx itself carries psi with gain
    // s0/(24 kappa) = 1/3, so the linearised rate is 2/3 rather than 1. At
    // T_FINAL = 20 that leaves exp(-2*20/3) = 1.6e-6 of transient, which is
    // exactly the 1.8e-6 the adjoint and the closed form then disagree by -- an
    // error in the *state*, visible in G to the same figure, not in the adjoint.
    // The adjoint state method assumes F(y, p) = 0, so the honest fix is to make
    // that true rather than to widen the tolerance around its being false; at
    // t = 40 the residual transient is exp(-26.7) = 2.6e-12.
    const double tSteady = 40.0;

    for (auto mode : {SystemSolver::FieldSolveMode::Iterative,
                      SystemSolver::FieldSolveMode::Exact})
    {
        const bool exact = mode == SystemSolver::FieldSolveMode::Exact;
        auto run = differentialRun(p0, nCells, k,
                                   std::string("field_adjoint_diff_") +
                                       (exact ? "exact" : "iter"),
                                   mode);
        BOOST_REQUIRE(run->getFieldModel()->isFieldDOFDifferential(0));
        integrateQuietly(run, tSteady);

        const Vector adjointGrad = run.gradient();
        const Vector analytic = exactGradient(KAPPA0, S0_0);
        BOOST_TEST_MESSAGE("differential field DOF, FieldSolve = "
                           << (exact ? "exact" : "iterative") << ": G = " << run.objective()
                           << " (exact " << exactG(KAPPA0, S0_0) << "), gradient "
                           << adjointGrad.transpose() << " against " << analytic.transpose()
                           << ", relative error " << relativeError(adjointGrad, analytic));

        // Six orders of magnitude of headroom against the 33% that passing
        // `alpha` instead of 0.0 produces, so this is not a tolerance chosen to
        // pass.
        BOOST_TEST(relativeError(adjointGrad, analytic) < 1e-7);
    }

    // ...and against finite differences of the same runs, which assume nothing
    // about the steady state at all.
    auto base = differentialRun(p0, nCells, k, "field_adjoint_diff_base");
    integrateQuietly(base, tSteady);
    const Vector fd = finiteDifferenceGradient(&differentialFactory, p0, nCells, k,
                                               "field_adjoint_diff_fd", 1e-5, tSteady);
    BOOST_TEST_MESSAGE("differential field DOF, adjoint " << base.gradient().transpose()
                                                          << " vs finite difference "
                                                          << fd.transpose() << ", relative error "
                                                          << relativeError(base.gradient(), fd));
    BOOST_TEST(relativeError(base.gradient(), fd) < 1e-6);
}

BOOST_AUTO_TEST_CASE(the_adjoint_solve_inverts_the_transpose_of_the_jacobian)
{
    // The operator, on its own. Every other test here goes through a run, an
    // objective and an F_p; this one finite-differences the coupled residual,
    // transposes it, and requires the adjoint solve to invert exactly that. A
    // failure localises to the transposes rather than to anything downstream.
    for (int trial = 0; trial < 3; ++trial)
    {
        auto h = makeAdjointStateFixture(
            std::make_unique<ClosedFormCoupledDiffusion>(baseParameters()),
            std::make_unique<ClosedFormAdjoint>(), std::make_shared<LinearGeometryField>(),
            /*nCells=*/6, /*k=*/2);

        const double r = checkAdjointTranspose(h, trial);
        BOOST_TEST_MESSAGE("single-DOF field, k = 2, nCells = 6, trial " << trial
                                                                        << ": ||J^T z - g||/||g|| = " << r);
        BOOST_TEST(r < 1e-7);
    }
}

BOOST_AUTO_TEST_CASE(the_transpose_check_reaches_every_coupled_block)
{
    // The same check on the rich fixture -- nAux > 0, geometry in the flux, the
    // source and the auxiliary constraint, and an A2 row reading all four slots
    // -- and on a five-unknown field block, where B^T is a genuine transposed
    // solve rather than a division.
    for (int trial = 0; trial < 2; ++trial)
    {
        auto rich = makeAdjointStateFixture(
            std::make_unique<RichGeometricDiffusion>((Vector(NP) << 1.0, 1.0).finished()),
            std::make_unique<RichAdjoint>(), std::make_shared<EverySlotField>(),
            /*nCells=*/5, /*k=*/2);
        rich->setFieldSolveMode(SystemSolver::FieldSolveMode::Exact);
        const double r = checkAdjointTranspose(rich, trial);
        BOOST_TEST_MESSAGE("rich fixture, trial " << trial << ": ||J^T z - g||/||g|| = " << r);
        BOOST_TEST(r < 1e-7);

        auto multi = makeAdjointStateFixture(
            std::make_unique<ClosedFormCoupledDiffusion>(baseParameters()),
            std::make_unique<ClosedFormAdjoint>(),
            std::make_shared<ManufacturedFieldVector>(toml::value{}, scratchGrid()),
            /*nCells=*/5, /*k=*/2);
        multi->setFieldSolveMode(SystemSolver::FieldSolveMode::Exact);
        BOOST_REQUIRE_EQUAL(multi->getFieldDOF(), ManufacturedFieldVector::N);
        const double rm = checkAdjointTranspose(multi, trial);
        BOOST_TEST_MESSAGE("5-DOF field, trial " << trial << ": ||J^T z - g||/||g|| = " << rm);
        BOOST_TEST(rm < 1e-7);
    }
}

BOOST_AUTO_TEST_CASE(the_superconvergent_transpose_is_checked_the_same_way)
{
    // With Superconvergent = true the physics is evaluated at the k+2 star nodes
    // with u* in place of u_h, A9 replaces InterpolateOntoBasis, and every block
    // gains a chain factor -- including A1, through dPhysics_dField_StarMat, and
    // the adjoint's own M through accumulateStarBlocks. field_jacobian_tests
    // covers the forward side of that (the_superconvergent_coupling_is_checked
    // _the_same_way); this is the transposed half, which nothing reached.
    //
    // The state fixture is the right vehicle rather than an end-to-end run: it
    // asks about the operator directly, which is what the star chain changes.
    for (int trial = 0; trial < 2; ++trial)
    {
        auto h = makeAdjointStateFixture(
            std::make_unique<ClosedFormCoupledDiffusion>(baseParameters()),
            std::make_unique<ClosedFormAdjoint>(), std::make_shared<LinearGeometryField>(),
            /*nCells=*/5, /*k=*/2, /*superconvergent=*/true);
        BOOST_REQUIRE(h->isSuperconvergent());
        const double r = checkAdjointTranspose(h, trial);
        BOOST_TEST_MESSAGE("superconvergent single-DOF field, trial "
                           << trial << ": ||J^T z - g||/||g|| = " << r);
        BOOST_TEST(r < 1e-7);

        auto rich = makeAdjointStateFixture(
            std::make_unique<RichGeometricDiffusion>((Vector(NP) << 1.0, 1.0).finished()),
            std::make_unique<RichAdjoint>(), std::make_shared<EverySlotField>(),
            /*nCells=*/5, /*k=*/2, /*superconvergent=*/true);
        rich->setFieldSolveMode(SystemSolver::FieldSolveMode::Exact);
        const double rr = checkAdjointTranspose(rich, trial);
        BOOST_TEST_MESSAGE("superconvergent rich fixture, trial "
                           << trial << ": ||J^T z - g||/||g|| = " << rr);
        BOOST_TEST(rr < 1e-7);
    }
}

BOOST_AUTO_TEST_CASE(the_same_holds_for_a_multi_dof_field_block)
{
    // ManufacturedFieldVector is five coupled field unknowns with a tridiagonal
    // B and a dense dGeometry/dpsi, so B^T is a real transposed solve rather
    // than a division and the Schur complement is 5x5.
    const Vector p0 = baseParameters();
    const Index nCells = 4, k = 3;

    auto run = multiDofRun(p0, nCells, k, "field_adjoint_multi");
    BOOST_REQUIRE_EQUAL(run->getFieldDOF(), ManufacturedFieldVector::N);
    integrateQuietly(run);

    const Vector adjointGrad = run.gradient();

    Vector fd(NP);
    for (Index i = 0; i < NP; ++i)
    {
        const double h = 1e-5 * std::abs(p0(i));
        Vector plus = p0, minus = p0;
        plus(i) += h;
        minus(i) -= h;

        auto a = multiDofRun(plus, nCells, k, "field_adjoint_multi_p" + std::to_string(i));
        integrateQuietly(a);
        auto b = multiDofRun(minus, nCells, k, "field_adjoint_multi_m" + std::to_string(i));
        integrateQuietly(b);
        fd(i) = (a.objective() - b.objective()) / (2.0 * h);
    }

    BOOST_TEST_MESSAGE("5-DOF field gradient:\n  adjoint           = "
                       << adjointGrad.transpose() << "\n  finite difference = " << fd.transpose()
                       << "\n  relative error = " << relativeError(adjointGrad, fd));
    BOOST_TEST(relativeError(adjointGrad, fd) < 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
