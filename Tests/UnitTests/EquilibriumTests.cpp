// Tier 0 of the field coupling (docs/superpowers/specs/2026-10-23-field-coupling-tiers-design.md):
// an equilibrium evaluated as a black box, held by the solver as a cache rather
// than as unknowns. What these pin is the cache -- where it is refreshed, where
// it is reused, where it is dropped -- and that a run built on it converges to
// the coupled problem's solution. ThetaPinchRunTests.cpp runs the registered
// model the same way.

#include <boost/test/unit_test.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "CapturedOutput.hpp"
#include "MMSHarness.hpp"
#include "ManufacturedFields.hpp"
#include "../../EquilibriumModel.hpp"
#include "../../FieldModelEquilibrium.hpp"
#include "../../SystemSolver.hpp"

#include <nvector/nvector_serial.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace mms;

namespace
{

/// ManufacturedField as a native tier-0 model: psi = Int u dx, by the weights
/// it is handed, and g = 1 + psi c(x). It records what it was asked, which is
/// what lets the cache be tested from the model's side.
class IntegralEquilibrium : public EquilibriumModel
{
public:
    IntegralEquilibrium()
        : EquilibriumModel({.geometry = {{"g", "metric factor multiplying the diffusivity", "1"}},
                            .label = "x"})
    {
    }

    struct Answer : public Equilibrium
    {
        double psi = 0.0;
    };

    std::unique_ptr<Equilibrium> Solve(GlobalState const &states, std::vector<Position> const &,
                                       Vector const &weights, Time, Equilibrium const *warm) override
    {
        warmWasNull.push_back(warm == nullptr);
        auto a = std::make_unique<Answer>();
        for (Index j = 0; j < weights.size(); ++j)
            a->psi += weights(j) * states[j].u(0);
        return a;
    }

    void Geometry(MatrixRef g, Equilibrium const &e, std::vector<Position> const &points) override
    {
        const double psi = dynamic_cast<Answer const &>(e).psi;
        for (size_t j = 0; j < points.size(); ++j)
            g(0, static_cast<Eigen::Index>(j)) = 1.0 + psi * manufacturedC(points[j]);
    }

    void resetForRun() override { ++resets; }

    // The interface the manufactured transport case reads its source from.
    Vector fieldExact(Time t) const { return Vector::Constant(1, manufacturedPsiExact(t)); }
    Value geometryExact(Position x, Time t) const { return 1.0 + manufacturedPsiExact(t) * manufacturedC(x); }
    Value dGeometryExact_dx(Position x, Time t) const
    {
        return -manufacturedPsiExact(t) * M_PI * std::sin(M_PI * x);
    }

    std::vector<bool> warmWasNull;
    int resets = 0;
};

/// ManufacturedField with its one DOF declared differential, which tier 0
/// cannot express.
struct DifferentialField : public ManufacturedField
{
    DifferentialField() { spec.dofs[0].differential = true; }
};

/// The manufactured coupled problem on a solver of its own, configured to run.
template <typename Model>
struct Tier0Run
{
    std::unique_ptr<Grid> grid;
    std::shared_ptr<Model> model;
    std::unique_ptr<ManufacturedGeometricDiffusion<Model>> problem;
    std::unique_ptr<SystemSolver> sys;
};

template <typename Model>
Tier0Run<Model> makeRun(Index nCells, Index k, std::shared_ptr<Model> model,
                        std::shared_ptr<EquilibriumModel> equilibrium)
{
    Tier0Run<Model> r;
    r.grid = std::make_unique<Grid>(0.0, 1.0, nCells);
    r.model = std::move(model);
    r.problem = std::make_unique<ManufacturedGeometricDiffusion<Model>>(*r.model);
    r.sys = std::make_unique<SystemSolver>(*r.grid, k, r.problem.get());
    SystemSolver &sys = *r.sys;
    sys.setTau(1.0);
    sys.setEquilibriumModel(std::move(equilibrium));
    sys.resetCoeffs();
    sys.setInputFile("tier0");
    sys.setOutputCadence(0.5);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-14);
    sys.setTolerances({1e-10}, 1e-8);
    sys.setWriteOutput(false);
    sys.setWriteDatFile(false);
    return r;
}

Tier0Run<IntegralEquilibrium> makeNativeRun(Index nCells = 16, Index k = 3)
{
    auto model = std::make_shared<IntegralEquilibrium>();
    return makeRun<IntegralEquilibrium>(nCells, k, model, model);
}

Tier0Run<ManufacturedField> makeAdaptedRun(Index nCells = 16, Index k = 3)
{
    auto model = std::make_shared<ManufacturedField>();
    return makeRun<ManufacturedField>(nCells, k, model, std::make_shared<FieldModelEquilibrium>(model));
}

void runQuietly(SystemSolver &sys, double tFinal)
{
    CapturedOutput quiet;
    sys.runSolver(tFinal);
}

} // namespace

BOOST_AUTO_TEST_SUITE(equilibrium_tests)

BOOST_AUTO_TEST_CASE(an_equilibrium_adds_no_unknowns)
{
    // The field is a cache, not an unknown: the layout, the error weights and
    // IDA's Newton norms are exactly the uncoupled run's.
    auto r = makeAdaptedRun(8, 2);
    BOOST_CHECK_EQUAL(r.sys->getFieldDOF(), 0);
    BOOST_CHECK_EQUAL(r.sys->getGeometrySlots(), 1);
    BOOST_CHECK_EQUAL(r.sys->getSolution().getDoF(), DGSoln::getDoF(1, 8, 2, 0, 0, 0));
}

BOOST_AUTO_TEST_CASE(a_tier_zero_run_reaches_the_coupled_solution)
{
    // The coupled run differentiates through the field and this one does not;
    // both are solving the same DAE, so at the same discretisation they agree
    // to the integration tolerance and carry the same discretisation error.
    const CoupledErrors coupled = solveCoupledOnce<ManufacturedField>(
        3, 16, 0.5, false, SystemSolver::FieldSolveMode::Exact, {.absolute = 1e-10, .relative = 1e-8});

    auto native = makeNativeRun();
    auto adapted = makeAdaptedRun();
    runQuietly(*native.sys, 0.5);
    runQuietly(*adapted.sys, 0.5);
    const double eNative = l2Error(*native.sys, *native.grid, 0.5);
    const double eAdapted = l2Error(*adapted.sys, *adapted.grid, 0.5);

    BOOST_TEST_MESSAGE("u error at t = 0.5, k = 3, 16 cells: coupled " << coupled.u << ", native tier 0 "
                                                                       << eNative << ", adapted tier 0 "
                                                                       << eAdapted);
    BOOST_TEST(eNative == coupled.u, boost::test_tools::tolerance(1e-3));
    BOOST_TEST(eAdapted == coupled.u, boost::test_tools::tolerance(1e-3));
    BOOST_TEST(coupled.u < 1e-4);

    // And the adapter's Newton reaches the native model's closed form.
    const double gap = l2ErrorAgainst([&](double x) { return adapted.sys->getSolution().u(0)(x); },
                                           [&](double x, double) { return native.sys->getSolution().u(0)(x); },
                                           *native.grid, 0.5);
    BOOST_TEST(gap < 1e-9);
}

BOOST_AUTO_TEST_CASE(the_equilibrium_is_solved_once_per_distinct_input)
{
    // A residual evaluated again at the same state is what the cache saves --
    // KINSOL's merit function, a Jacobian build at the last residual's state --
    // and any other state is a different residual.
    auto r = makeNativeRun(4, 2);
    SystemSolver &sys = *r.sys;
    sys.initialiseMatrices();

    const size_t dof = sys.getSolution().getDoF();
    N_Vector y = N_VNew_Serial(dof, sys.ctx), dydt = N_VClone(y), res = N_VClone(y);
    N_VConst(0.0, dydt);
    DGSoln Y(1, *r.grid, 2, N_VGetArrayPointer(y), 0, 0, 0);
    Y.AssignU([](Index, Position x) { return std::sin(M_PI * x); });
    Y.AssignQ([](Index, Position x) { return M_PI * std::cos(M_PI * x); });
    for (Index i = 0; i <= 4; ++i)
        Y.lambda(0)[i] = 0.0;

    sys.residual(0.1, y, dydt, res);
    BOOST_CHECK_EQUAL(sys.getEquilibriumSolves(), 1);
    sys.residual(0.1, y, dydt, res);
    BOOST_CHECK_EQUAL(sys.getEquilibriumSolves(), 1);

    // A trace value is not something the model is handed, so it is not part of
    // the key either.
    Y.lambda(0)[2] += 1e-3;
    sys.residual(0.1, y, dydt, res);
    BOOST_CHECK_EQUAL(sys.getEquilibriumSolves(), 1);

    // A change in a cell value is, however small.
    Y.u(0).getCoeff(1).second(0) += 1e-14;
    sys.residual(0.1, y, dydt, res);
    BOOST_CHECK_EQUAL(sys.getEquilibriumSolves(), 2);

    // And so is a change of time.
    sys.residual(0.2, y, dydt, res);
    BOOST_CHECK_EQUAL(sys.getEquilibriumSolves(), 3);

    // Every solve after the first is warm-started from the last.
    BOOST_REQUIRE_EQUAL(r.model->warmWasNull.size(), 3u);
    BOOST_TEST(r.model->warmWasNull[0]);
    BOOST_TEST(!r.model->warmWasNull[1]);
    BOOST_TEST(!r.model->warmWasNull[2]);

    for (N_Vector v : {y, dydt, res})
        N_VDestroy(v);
}

BOOST_AUTO_TEST_CASE(each_run_starts_from_a_cold_equilibrium)
{
    // The RF_cellwise trap: initialize() is where a run's cache is dropped, in
    // its unconditional part, so a reused solver's first Solve is cold like a
    // fresh one's -- and the model is told, through resetForRun.
    auto r = makeNativeRun(8, 2);
    runQuietly(*r.sys, 0.5);
    const size_t firstRun = r.model->warmWasNull.size();
    runQuietly(*r.sys, 0.5);

    BOOST_CHECK_EQUAL(r.model->resets, 2);
    BOOST_REQUIRE_GT(firstRun, 1u);
    BOOST_REQUIRE_GT(r.model->warmWasNull.size(), firstRun);
    BOOST_TEST(r.model->warmWasNull[0]);
    BOOST_TEST(r.model->warmWasNull[firstRun]);
    long cold = 0;
    for (bool b : r.model->warmWasNull)
        cold += b ? 1 : 0;
    BOOST_CHECK_EQUAL(cold, 2);
}

BOOST_AUTO_TEST_CASE(a_reused_tier_zero_solver_matches_a_fresh_one_bit_for_bit)
{
    // Through the adapter. Its Newton starts from the warm psi, but the
    // manufactured field's row is linear and lands on the same bits from any
    // start, so this does not see a warm start leaking across runs -- measured:
    // it passes with initialize()'s cache reset deleted. The theta pinch's
    // nonlinear rows do, and a_reused_tier_zero_theta_pinch_solver_matches_a_
    // fresh_one_bit_for_bit is the guard for that; this one pins everything else
    // a reused tier-0 run carries.
    auto fresh = makeAdaptedRun(8, 2);
    auto reused = makeAdaptedRun(8, 2);
    runQuietly(*fresh.sys, 0.5);
    runQuietly(*reused.sys, 0.5);
    runQuietly(*reused.sys, 0.5);
    for (double x : {0.1, 0.3, 0.5, 0.7, 0.9})
        BOOST_TEST(fresh.sys->getSolution().u(0)(x) == reused.sys->getSolution().u(0)(x),
                   boost::test_tools::tolerance(0.0));
}

BOOST_AUTO_TEST_CASE(an_adjoint_is_refused)
{
    // No derivatives through the field, and a frozen-field gradient is wrong by
    // the field's share. The objective steady test needs an adjoint too.
    auto r = makeNativeRun(4, 2);
    r.sys->setSteadyStateTolerance(1e-6);
    r.sys->setSteadyMode(SystemSolver::SteadyMode::Newton);
    r.sys->setSteadyTest(SystemSolver::SteadyTest::Objective);
    CapturedOutput quiet;
    BOOST_CHECK_THROW(r.sys->initialize(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(a_solver_has_one_source_of_geometry)
{
    Grid grid(0.0, 1.0, 4);
    auto field = std::make_shared<ManufacturedField>();
    ManufacturedGeometricDiffusion<ManufacturedField> problem(*field);

    SystemSolver coupled(grid, 2, &problem);
    coupled.setFieldModel(field);
    BOOST_CHECK_THROW(coupled.setEquilibriumModel(std::make_shared<FieldModelEquilibrium>(field)),
                      std::logic_error);

    SystemSolver tier0(grid, 2, &problem);
    tier0.setEquilibriumModel(std::make_shared<FieldModelEquilibrium>(field));
    BOOST_CHECK_THROW(tier0.setFieldModel(field), std::logic_error);
    // Neither refusal moved anything.
    BOOST_CHECK_EQUAL(tier0.getFieldDOF(), 0);
    BOOST_CHECK_EQUAL(coupled.getFieldDOF(), 1);

    // Detaching one makes room for the other.
    tier0.setEquilibriumModel(nullptr);
    BOOST_CHECK_EQUAL(tier0.getGeometrySlots(), 0);
    BOOST_CHECK_NO_THROW(tier0.setFieldModel(field));
}

BOOST_AUTO_TEST_CASE(an_equilibrium_needs_a_case_that_reads_geometry)
{
    // The same tier -1 refusal as a field model's.
    class Blind : public TransportSystem
    {
    public:
        Blind() : TransportSystem({.variables = {{"u", "", "", BoundaryKind::Dirichlet, BoundaryKind::Dirichlet}}}) {}
        Value SigmaFn(Index, const State &s, Position, Time) override { return s.q(0); }
        Value Sources(Index, const State &, Position, Time) override { return 0.0; }
        void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
        void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 1.0; }
        void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
        void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
        void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
        Value InitialValue(Index, Position) const override { return 0.0; }
        Value InitialDerivative(Index, Position) const override { return 0.0; }
        Value LowerBoundary(Index, Time) const override { return 0.0; }
        Value UpperBoundary(Index, Time) const override { return 0.0; }
    } blind;
    Grid grid(0.0, 1.0, 4);
    SystemSolver sys(grid, 2, &blind);
    BOOST_CHECK_THROW(sys.setEquilibriumModel(std::make_shared<IntegralEquilibrium>()), std::logic_error);
}

BOOST_AUTO_TEST_CASE(a_field_with_dynamics_cannot_be_an_equilibrium)
{
    // A differential DOF is not an instantaneous function of the transport
    // state, which is all tier 0 can express.
    BOOST_CHECK_THROW(FieldModelEquilibrium(std::make_shared<DifferentialField>()), std::invalid_argument);
    BOOST_CHECK_NO_THROW(FieldModelEquilibrium(std::make_shared<ManufacturedField>()));
}

BOOST_AUTO_TEST_SUITE_END()
