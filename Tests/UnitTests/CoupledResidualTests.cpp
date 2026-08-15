// The coupled residual: field rows in the residual vector, geometry evaluated
// at the physics nodes and reaching the physics through State::geom.
//
// Four things are pinned here and each has a distinct failure mode.
//
//  * The field rows land in the field block and nowhere else. Getting a column
//    index wrong in this layout is the most common way to break the solver
//    silently.
//  * Geometry reaches a physics case's SigmaFn. Without this the coupling is
//    one-way and every subsequent test would still pass on a decoupled problem.
//  * A field DOF declared differential whose residual carries no d/dt is
//    refused at initialisation. Left to IDA it is IDA_LINESEARCH_FAIL (-13),
//    a message about the linesearch for a defect in the declaration -- which is
//    what kept python-physics/mirror-plasma's voltage controller from ever
//    starting.
//  * A coupled run reaches a manufactured solution in both u and psi. That is
//    the end-to-end check, and the only one that would notice psi being carried
//    along without ever being solved for.
#include <boost/test/unit_test.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "CapturedOutput.hpp"
#include "MMSHarness.hpp"
#include "ManufacturedFields.hpp"
#include "../../SystemSolver.hpp"
#include "../../Types.hpp"

#include <nvector/nvector_serial.h>

#include <cmath>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

// Defined in SystemSolver.cpp. IDA is handed this rather than the member, and
// the catch-and-return-1 inside it is what a_field_model_that_throws exercises.
int static_residual(sunrealtype tres, N_Vector Y, N_Vector dydt, N_Vector resval,
                    void *user_data);

namespace
{

using std::numbers::pi;

// ------------------------------------------------------------- physics cases --

// Linear diffusion that does *not* read geometry. Deliberately: it is what lets
// the layout test below say "the field row moved and nothing else did", which is
// only a statement about the layout if the transport rows have no way of seeing
// psi at all.
class PlainDiffusion : public TransportSystem
{
public:
    PlainDiffusion()
        : TransportSystem({.variables = {{"u", "the diffused quantity", "",
                                          BoundaryKind::Dirichlet, BoundaryKind::Dirichlet}}})
    {
    }

    Value SigmaFn(Index, const State &s, Position, Time) override { return s.q(0); }
    Value Sources(Index, const State &, Position, Time) override { return 0.0; }

    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 1.0; }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    Value InitialValue(Index, Position x) const override { return x * (1.0 - x); }
    Value InitialDerivative(Index, Position x) const override { return 1.0 - 2.0 * x; }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
};

// Records the geometry it was handed, node by node, and uses it.
//
// The batched hook is overridden rather than the pointwise one because the
// default batched SigmaFn is a `#pragma omp parallel for` over the pointwise
// version: under an OMP build, appending to a shared vector from inside it would
// be a data race and the recorded order would be nondeterministic.
class GeometryProbeCase : public TransportSystem
{
public:
    GeometryProbeCase()
        : TransportSystem({.variables = {{"u", "the diffused quantity", "",
                                          BoundaryKind::Dirichlet, BoundaryKind::Dirichlet}}})
    {
    }

    static constexpr double kappa = 2.0;

    Values SigmaFn(Index, GlobalState const &states, std::vector<Position> const &abscissae,
                   Time) override
    {
        seen.clear();
        points.clear();
        Values out(states.size());
        for (size_t j = 0; j < states.size(); ++j)
        {
            const State s = states[j];
            seen.push_back(s.geom(0));
            points.push_back(abscissae[j]);
            out(static_cast<Eigen::Index>(j)) = s.geom(0) * kappa * s.q(0);
        }
        return out;
    }

    // Reached only if the batched override above is bypassed; kept consistent
    // with it so the two cannot disagree.
    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return s.geom(0) * kappa * s.q(0);
    }
    Value Sources(Index, const State &, Position, Time) override { return 0.0; }

    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = s.geom(0) * kappa;
    }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }

    Value InitialValue(Index, Position x) const override { return x * (1.0 - x); }
    Value InitialDerivative(Index, Position x) const override { return 1.0 - 2.0 * x; }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    static std::vector<double> const &lastGeometry() { return seen; }
    static std::vector<Position> const &lastPoints() { return points; }

private:
    static inline std::vector<double> seen;
    static inline std::vector<Position> points;
};

// The manufactured coupled problem.
//
//     u_t - d_x[ g(x; psi) u_x ] = S ,   g = 1 + psi c(x) ,   R = psi - Int u dx
//
// with the exact solution u = sin(pi x)(1 + t), which vanishes at both ends for
// every t and so is consistent with the homogeneous Dirichlet conditions, and
// psi = Int u dx = (2/pi)(1 + t).
//
// The source is compensated against u_exact and psi_exact, never against the
// state the hook is handed. A compensation written against the discrete state
// would be an exact row operation -- residual() evaluates the hooks on the same
// states at the same abscissae and pushes them through the same projection -- so
// it would cancel identically and the test would silently measure an uncoupled
// problem. The same reasoning as in ManufacturedFields.hpp, one level up.
//
// Note the sign. The stored sigma is -sigma_hat, so what is integrated is
// u_t - d_x[sigma_hat] = S; deriving S from u_t + d_x[sigma_hat] gives an
// anti-diffusion equation that still converges, at the right rate, to a
// different function.
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

    Value InitialValue(Index, Position x) const override { return manufacturedU(x, 0.0); }
    Value InitialDerivative(Index, Position x) const override
    {
        return pi * std::cos(pi * x);
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
};

// -------------------------------------------------------------- field models --

// A grid for the two field models that take one and ignore it. Static so the
// reference the FieldModel constructor is handed outlives every model built
// from it, which is what lets these have default constructors.
Grid const &scratchGrid()
{
    static const Grid g(0.0, 1.0, 1);
    return g;
}

// ManufacturedField with its one DOF declared differential and its residual
// left alone -- so dR/d(dpsi/dt) is identically zero. The misdeclaration
// initialize() must refuse.
class BadlyDeclaredDifferentialField : public ManufacturedField
{
public:
    BadlyDeclaredDifferentialField() : ManufacturedField(toml::value{}, scratchGrid())
    {
        // `spec` is protected on FieldModel, and mutating it here rather than
        // writing a second copy of the model is deliberate: the point of the
        // fixture is that the *only* difference from a working model is the
        // declaration.
        spec.dofs[0].differential = true;
    }
};

// The same psi(t) = (2/pi)(1 + t) that ManufacturedField constrains
// algebraically, expressed instead as the differential row
//
//     R = dpsi/dt - 2/pi ,     psi(0) = 2/pi
//
// so `differential = true` is the *correct* declaration and dR/d(dpsi/dt) = 1.
//
// This exists because the `id` vector is the one thing the brief singled out
// that nothing else reaches: BadlyDeclaredDifferentialField is refused before
// IDASetId is called, and ManufacturedField is algebraic, so
// `isDifferential.Field(f) = 1.0` -- writing the wrong slot, writing Scalar(f),
// or the loop not running at all -- would pass the whole suite. Marking psi
// algebraic here is not a small error either: IDA_YA_YDP_INIT would then solve
// for its *value* against a row with dR/dpsi = 0, which is irreducible, and the
// run dies in the linesearch.
//
// A controlled comparison rather than a separate problem: same exact solution,
// same geometry, same transport case and the same discretisation as the
// algebraic end-to-end test, so the two answers are directly comparable.
class DifferentialManufacturedField : public FieldModel
{
public:
    DifferentialManufacturedField() : FieldModel(buildSpec()) {}

    static FieldModelSpec buildSpec()
    {
        FieldModelSpec s;
        s.dofs = {{"psi", "the manufactured field unknown", "1", /* differential */ true}};
        s.geometry = {{"g", "metric factor multiplying the diffusivity", "1"}};
        s.label = "x";
        return s;
    }

    void FieldResidual(VectorRef out, Vector const &, Vector const &dpsidt,
                       GlobalState const &, std::vector<Position> const &, Vector const &,
                       Time) override
    {
        out(0) = dpsidt(0) - 2.0 / pi;
    }

    void Geometry(VectorRef out, Vector const &psi, Position x, Time) override
    {
        out(0) = 1.0 + psi(0) * manufacturedC(x);
    }

    void dGeometry_dpsi(MatrixRef out, Vector const &, Position x, Time) override
    {
        out(0, 0) = manufacturedC(x);
    }

    // dR/dpsi is identically zero; the whole block is dR/d(dpsi/dt), which is
    // what makes B = alpha and what the refusal in initialize() checks for.
    void FieldResidualPrime(GlobalStateMatrix &, GlobalStateMatrix &, MatrixRef,
                            MatrixRef dRddpsidt, Vector const &, Vector const &,
                            GlobalState const &, std::vector<Position> const &,
                            Vector const &, Time) override
    {
        dRddpsidt(0, 0) = 1.0;
    }

    void InitialFieldValue(VectorRef out) override { out(0) = manufacturedPsiExact(0.0); }
};

// Refuses to be evaluated. A model that cannot evaluate at the state it is given
// -- no x-point, a boundary that has left the domain -- is required to throw,
// and static_residual is required to turn that into IDA's recoverable-error
// return rather than letting it out.
class ThrowingField : public ManufacturedField
{
public:
    ThrowingField() : ManufacturedField(toml::value{}, scratchGrid()) {}

    void FieldResidual(VectorRef, Vector const &, Vector const &, GlobalState const &,
                       std::vector<Position> const &, Vector const &, Time) override
    {
        throw std::runtime_error("no equilibrium at this state");
    }
};

// ------------------------------------------------------------------ fixtures --

// Owns everything one run needs and dereferences to the solver.
//
// The physics case, the field model and the grid all have to outlive the
// SystemSolver -- it holds a raw pointer to the first and a copy of the third,
// which its DGSoln members then reference -- so declaration order here is
// load bearing: members are destroyed in reverse, and `sys` has to go first.
class SolverHandle
{
public:
    SolverHandle() = default;
    SolverHandle(SolverHandle const &) = delete;
    SolverHandle &operator=(SolverHandle const &) = delete;
    SolverHandle(SolverHandle &&) = default;
    SolverHandle &operator=(SolverHandle &&) = default;

    ~SolverHandle()
    {
        // Idempotent, and safe with no preceding initialize(). It is here for
        // the test that expects initialize() to *throw*: everything IDA had
        // allocated by then is still allocated.
        if (sys)
        {
            CapturedOutput quiet;
            sys->destroySundials();
        }
    }

    SystemSolver &operator*() const { return *sys; }
    SystemSolver *operator->() const { return sys.get(); }
    SystemSolver *get() const { return sys.get(); }

    std::unique_ptr<TransportSystem> problem;
    std::shared_ptr<FieldModel> field;
    std::unique_ptr<Grid> grid;
    std::unique_ptr<SystemSolver> sys;
};

// A handle plus the three vectors a bare residual evaluation needs. The
// destructor frees them before the handle's does, which matters: N_VDestroy has
// to happen before ~SystemSolver frees the SUNContext they were made in.
struct CoupledFixture
{
    SolverHandle solver;
    N_Vector y = nullptr, dydt = nullptr, res = nullptr;

    CoupledFixture() = default;
    CoupledFixture(CoupledFixture const &) = delete;
    CoupledFixture(CoupledFixture &&) = default;

    ~CoupledFixture()
    {
        for (N_Vector v : {y, dydt, res})
            if (v)
                N_VDestroy(v);
    }
};

/// A DGSoln view over an N_Vector, shaped the way this solver's vector is.
DGSoln mapSoln(SystemSolver &sys, N_Vector v)
{
    DGSoln out(sys.nVars, sys.grid, sys.k, sys.nScalars, sys.nAux, sys.getFieldDOF());
    out.Map(N_VGetArrayPointer(v));
    return out;
}

/// Configure far enough to run, but not to write anything.
void configureQuietly(SystemSolver &sys, std::string const &stem)
{
    sys.setTau(1.0);
    sys.setInputFile(stem);
    sys.setOutputCadence(1.0);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-14);
    sys.setTolerances({1e-10}, 1e-8);
    sys.setWriteOutput(false);
    sys.setWriteDatFile(false);
}

/// A solver with `problem` and `model` attached, configured but not initialised.
SolverHandle makeSolverWithModel(std::unique_ptr<TransportSystem> problem,
                                 std::shared_ptr<FieldModel> model, Index nCells, Index k,
                                 std::string const &stem)
{
    SolverHandle h;
    h.problem = std::move(problem);
    h.field = std::move(model);
    h.grid = std::make_unique<Grid>(0.0, 1.0, nCells);
    h.sys = std::make_unique<SystemSolver>(*h.grid, k, h.problem.get());
    configureQuietly(*h.sys, stem);
    h.sys->setFieldModel(h.field);
    h.sys->resetCoeffs();
    return h;
}

/// The overload the differential-declaration test uses: a plain physics case,
/// whichever model is under suspicion, and a shape small enough to be quick.
SolverHandle makeSolverWithModel(std::shared_ptr<FieldModel> model)
{
    return makeSolverWithModel(std::make_unique<PlainDiffusion>(), std::move(model), 4, 2,
                               "coupled_declaration");
}

/// A solver ready for a bare residual evaluation: matrices built, three vectors
/// of the right length allocated, no IDA anywhere.
CoupledFixture makeFixture(std::unique_ptr<TransportSystem> problem,
                           std::shared_ptr<FieldModel> model, Index nCells, Index k,
                           std::string const &stem)
{
    CoupledFixture f;
    f.solver = makeSolverWithModel(std::move(problem), std::move(model), nCells, k, stem);
    f.solver->initialiseMatrices();

    const size_t dof = f.solver->getSolution().getDoF();
    f.y = N_VNew_Serial(dof, f.solver->ctx);
    f.dydt = N_VClone(f.y);
    f.res = N_VClone(f.y);
    N_VConst(0.0, f.y);
    N_VConst(0.0, f.dydt);
    N_VConst(0.0, f.res);
    return f;
}

CoupledFixture makeCoupledSolver(Index nCells, Index k)
{
    return makeFixture(std::make_unique<PlainDiffusion>(),
                       std::make_shared<ManufacturedField>(toml::value{}, scratchGrid()),
                       nCells, k, "coupled_layout");
}

CoupledFixture makeCoupledSolverWithProbe(Index nCells, Index k)
{
    return makeFixture(std::make_unique<GeometryProbeCase>(),
                       std::make_shared<ManufacturedField>(toml::value{}, scratchGrid()),
                       nCells, k, "coupled_probe");
}

CoupledFixture makeCoupledSolverWithThrowingField(Index nCells, Index k)
{
    return makeFixture(std::make_unique<PlainDiffusion>(), std::make_shared<ThrowingField>(),
                       nCells, k, "coupled_throwing");
}

/// Integrate the manufactured coupled problem to tFinal and hand back the run.
///
/// The field model is a parameter so the algebraic and differential statements
/// of the same psi(t) can be run at the same discretisation and their answers
/// compared directly.
SolverHandle runCoupledToTime(Index nCells, Index k, double tFinal,
                              std::shared_ptr<FieldModel> model,
                              std::string const &tag = "")
{
    SolverHandle h = makeSolverWithModel(
        std::make_unique<ManufacturedCoupledDiffusion>(), std::move(model), nCells, k,
        "coupled_mms" + tag + "_k" + std::to_string(k) + "_n" + std::to_string(nCells));
    h.sys->setOutputCadence(tFinal);

    {
        // runSolver reports its step counts and IDACalcIC warnings, and with no
        // coupling blocks in the Jacobian there are a lot of them.
        CapturedOutput quiet;
        h.sys->initialize();
        h.sys->integrate(tFinal);
        h.sys->destroySundials();
    }
    return h;
}

SolverHandle runCoupledToTime(Index nCells, Index k, double tFinal)
{
    return runCoupledToTime(nCells, k, tFinal,
                            std::make_shared<ManufacturedField>(toml::value{}, scratchGrid()));
}

/// L2 error of u against the manufactured solution, by a quadrature independent
/// of the basis's own -- the same rule MMSConvergenceTests measures with.
double uError(SystemSolver &sys, double t)
{
    return mms::l2ErrorAgainst([&](double x) { return sys.getSolution().u(0)(x); },
                               [](double x, double tt) { return manufacturedU(x, tt); },
                               sys.getSolution().getGrid(), t);
}

/// The whole residual vector, copied out so two evaluations can be differenced.
Vector copyOf(N_Vector v)
{
    return Eigen::Map<const Vector>(N_VGetArrayPointer(v), N_VGetLength(v));
}

} // namespace

BOOST_AUTO_TEST_SUITE(coupled_residual_tests)

BOOST_AUTO_TEST_CASE(the_field_rows_appear_in_the_field_block)
{
    // Evaluate the residual at a state whose psi is wrong by a known amount and
    // check the field row equals that amount.
    //
    // u = x(1 - x) rather than the manufactured sin(pi x), deliberately: it lies
    // in P_k for every k >= 2, so the interpolatory quadrature the field residual
    // contracts against reproduces Int u dx = 1/6 exactly and the expected answer
    // is a number rather than a number plus a discretisation error. Against
    // sin(pi x) at k = 2, nCells = 8 that error is ~1e-6 -- small, and quite large
    // enough to force a tolerance loose enough to hide a real defect.
    auto fixture = makeCoupledSolver(/*nCells=*/8, /*k=*/2);
    SystemSolver &solver = *fixture.solver;

    DGSoln yMap = mapSoln(solver, fixture.y);
    yMap.AssignU([](Index, Position x) { return x * (1.0 - x); });
    yMap.AssignQ([](Index, Position x) { return 1.0 - 2.0 * x; });
    yMap.EvaluateLambda();
    yMap.Field(0) = 1.0 / 6.0 + 0.25;

    solver.residual(0.0, fixture.y, fixture.dydt, fixture.res);

    DGSoln resMap = mapSoln(solver, fixture.res);
    BOOST_CHECK_CLOSE(resMap.Field(0), 0.25, 1e-9);

    // ...and nowhere else. PlainDiffusion does not read geometry, so moving psi
    // may only move the field row; anything else is a layout error, which is the
    // failure this whole block exists to catch and the one nothing downstream
    // would report.
    const Vector before = copyOf(fixture.res);
    yMap.Field(0) += 1.0;
    solver.residual(0.0, fixture.y, fixture.dydt, fixture.res);
    const Vector diff = copyOf(fixture.res) - before;

    const Eigen::Index fieldRow = diff.size() - 1;
    BOOST_CHECK_CLOSE(diff(fieldRow), 1.0, 1e-9);
    BOOST_CHECK_SMALL(diff.head(fieldRow).cwiseAbs().maxCoeff(), 1e-14);
}

BOOST_AUTO_TEST_CASE(geometry_reaches_the_physics_case)
{
    // GeometryProbeCase records the geometry it was handed at each node. With
    // psi set to a known value, g(x) = 1 + psi c(x) is a closed form.
    auto fixture = makeCoupledSolverWithProbe(/*nCells=*/4, /*k=*/1);
    SystemSolver &solver = *fixture.solver;

    DGSoln yMap = mapSoln(solver, fixture.y);
    yMap.Field(0) = 0.5;

    solver.residual(0.0, fixture.y, fixture.dydt, fixture.res);

    auto const &seen = GeometryProbeCase::lastGeometry();
    auto const &points = GeometryProbeCase::lastPoints();
    BOOST_REQUIRE_EQUAL(seen.size(), points.size());
    BOOST_REQUIRE_GT(seen.size(), 0u);
    for (size_t j = 0; j < seen.size(); ++j)
        BOOST_CHECK_CLOSE(seen[j], 1.0 + 0.5 * manufacturedC(points[j]), 1e-10);

    // A constant geometry would pass the loop above for the one node where
    // c(x) happens to vanish, and nowhere else; check the values actually vary,
    // so the assertion is about a function of x rather than about a number.
    BOOST_CHECK_GT(*std::max_element(seen.begin(), seen.end()) -
                       *std::min_element(seen.begin(), seen.end()),
                   0.1);
}

BOOST_AUTO_TEST_CASE(a_differential_field_dof_with_no_time_derivative_is_refused)
{
    // The same check ScalarGPrime's dGdot needs: call FieldResidualPrime and
    // require the dRddpsidt row to be nonzero for every DOF declared
    // differential.
    auto solver = makeSolverWithModel(std::make_shared<BadlyDeclaredDifferentialField>());
    CapturedOutput quiet;
    BOOST_CHECK_THROW(solver->initialize(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(a_correctly_declared_algebraic_field_dof_is_not_refused)
{
    // The other half of the refusal above: it has to discriminate. The identical
    // model with `differential = false` must initialise.
    auto solver = makeSolverWithModel(
        std::make_shared<ManufacturedField>(toml::value{}, scratchGrid()));
    CapturedOutput quiet;
    BOOST_CHECK_NO_THROW(solver->initialize());
}

BOOST_AUTO_TEST_CASE(a_legitimately_differential_field_dof_is_not_refused)
{
    // The refusal must also let through a DOF that is differential *and* says
    // so with a residual carrying d/dt. Without this the check could be an
    // unconditional throw on `differential == true` and nothing would notice.
    auto solver = makeSolverWithModel(std::make_shared<DifferentialManufacturedField>());
    CapturedOutput quiet;
    BOOST_CHECK_NO_THROW(solver->initialize());
}

BOOST_AUTO_TEST_CASE(a_differential_field_dof_reaches_the_id_vector)
{
    // The `id` vector IDASetId receives, read back directly.
    //
    // This is the one line the brief singled out and the only thing that reads
    // it: writing the wrong slot, writing Scalar(f) instead, or not running the
    // loop at all is invisible everywhere else, because IDA answers a wrong id
    // with a different *initialisation problem* rather than an error.
    auto differential = makeSolverWithModel(std::make_shared<DifferentialManufacturedField>());
    {
        CapturedOutput quiet;
        differential->initialize();
    }
    DGSoln idMap = mapSoln(*differential, differential->id);
    BOOST_CHECK_EQUAL(idMap.Field(0), 1.0);

    // ...and an algebraic DOF must leave it zero, or IDA_YA_YDP_INIT would be
    // asked for a derivative it has no equation for.
    auto algebraic = makeSolverWithModel(
        std::make_shared<ManufacturedField>(toml::value{}, scratchGrid()));
    {
        CapturedOutput quiet;
        algebraic->initialize();
    }
    DGSoln algebraicId = mapSoln(*algebraic, algebraic->id);
    BOOST_CHECK_EQUAL(algebraicId.Field(0), 0.0);
}

BOOST_AUTO_TEST_CASE(a_field_model_that_throws_is_a_recoverable_error)
{
    // static_residual catches and returns 1, which IDA treats as recoverable
    // and retries with a smaller step. Throwing out of the residual would abort
    // a run a shorter step would have survived.
    auto fixture = makeCoupledSolverWithThrowingField(4, 1);
    CapturedOutput quiet;
    const int retval =
        static_residual(0.0, fixture.y, fixture.dydt, fixture.res, fixture.solver.get());
    BOOST_CHECK_EQUAL(retval, 1);
}

BOOST_AUTO_TEST_CASE(a_coupled_run_reaches_the_manufactured_solution)
{
    // The end-to-end check: integrate to t = 0.5 and compare both u and psi
    // against their closed forms.
    //
    // Slow relative to the others, and expected to be: the Jacobian carries the
    // field model's own block but neither coupling block, so IDA is running a
    // block-Jacobi Newton and pays for it in iterations.
    auto solver = runCoupledToTime(/*nCells=*/16, /*k=*/3, /*tFinal=*/0.5);

    const double eu = uError(*solver, 0.5);
    const double epsi = std::abs(solver->getSolution().Field(0) - manufacturedPsiExact(0.5));
    BOOST_TEST_MESSAGE("coupled MMS at t = 0.5, k = 3, nCells = 16:  |u| error "
                       << eu << ",  |psi| error " << epsi);

    BOOST_CHECK_SMALL(eu, 1e-4);
    BOOST_CHECK_SMALL(epsi, 1e-5);
}

BOOST_AUTO_TEST_CASE(a_differential_field_dof_integrates_to_the_same_solution)
{
    // The same run with psi stated as an ODE instead of an algebraic identity.
    // Same exact solution, same discretisation, same bounds -- so this is a
    // controlled comparison against the case above rather than a second problem
    // with tolerances of its own.
    //
    // What it exercises that nothing else does: the `id` slot in the integration
    // rather than in isolation, IDACalcIC's differential branch (it solves for
    // dpsi/dt here, not for psi), the field entry of getErrorWeights -- psi is
    // now in IDA's local error test, and a zero weight there is a division by
    // zero rather than a loose tolerance -- and B = alpha rather than B = 1, so
    // solveB is applied to a matrix that changes with the step size.
    auto solver = runCoupledToTime(/*nCells=*/16, /*k=*/3, /*tFinal=*/0.5,
                                   std::make_shared<DifferentialManufacturedField>(), "_diff");

    const double eu = uError(*solver, 0.5);
    const double epsi = std::abs(solver->getSolution().Field(0) - manufacturedPsiExact(0.5));
    BOOST_TEST_MESSAGE("differential-psi coupled MMS at t = 0.5, k = 3, nCells = 16:  |u| error "
                       << eu << ",  |psi| error " << epsi);

    BOOST_CHECK_SMALL(eu, 1e-4);
    BOOST_CHECK_SMALL(epsi, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
