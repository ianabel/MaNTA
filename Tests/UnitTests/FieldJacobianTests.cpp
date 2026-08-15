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
#include "ManufacturedFields.hpp"
#include "../../SystemSolver.hpp"
#include "../../Types.hpp"

#include <nvector/nvector_serial.h>

#include <algorithm>
#include <cmath>
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

CoupledSolver multiDofFixture(Index nCells, Index k)
{
    return makeCoupledSolverAtState(
        nCells, k, std::make_unique<GeometricDiffusion>(),
        std::make_shared<ManufacturedFieldVector>(toml::value{}, scratchGrid()));
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

BOOST_AUTO_TEST_SUITE_END()
