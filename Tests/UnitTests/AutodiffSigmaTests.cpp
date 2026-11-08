// What the autodiff layer hands a case: the physical flux, and the geometry.
//
// Two properties of AutodiffTransportSystem that nothing in the tree exercised,
// because no case used them:
//
//  * **`sigma` in Source and GFunc is sigma_hat**, the flux Flux returned. The
//    solver stores sigma_h = -Pi(sigma_hat) (CLAUDE.md, "a sign convention, not
//    an identity"), and State::sigma is that stored value, so the layer negates
//    it on the way in -- and negates d/d(sigma) on the way out, because autodiff
//    then differentiates with respect to sigma_hat while the Jacobian wants
//    d/d(stored sigma). Getting the first wrong is a source of the wrong sign,
//    which converges at the right rate to the wrong function; getting only the
//    second wrong costs Newton iterations and nothing else. So each half has its
//    own test: a manufactured solve for the value, a finite-differenced
//    Jacobian for the derivative.
//
//  * **MMS_Source differentiates the flux the solver integrates**: the
//    geometry-aware Flux, at the case's manufactured geometry (MMS_Geometry),
//    with the chain through dg/dx. The geometry-blind Flux is the pure virtual
//    every case must write to compile, so a geometry-dependent case has some
//    stand-in there; a manufactured source built from the stand-in converges at
//    the right rate to the wrong function. The checks are a closed form, point
//    by point, and a coupled run whose forcing is MMS_Source itself.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "FiniteDifferenceJacobian.hpp"
#include "MMSHarness.hpp"
#include "ManufacturedFields.hpp"
#include "PhysicsCases/AutodiffTransportSystem.hpp"
#include "SystemSolver.hpp"
#include "Types.hpp"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>

#include <cmath>
#include <memory>
#include <numbers>
#include <string>
#include <toml.hpp>
#include <vector>

using namespace mms;
using std::numbers::pi;

namespace
{

toml::value const &emptyConfig()
{
    static const toml::value cfg = toml::table{};
    return cfg;
}

Grid const &unitGrid()
{
    static const Grid g(0.0, 1.0, 4);
    return g;
}

/// u_t - u_xx for u = sin(pi x)(1 + t).
inline double diffusionSource(double x, double t)
{
    return std::sin(pi * x) * (1.0 + pi * pi * (1.0 + t));
}

/*
    u_t - u_xx = S with sigma_hat = q and

        S = diffusionSource(x, t) + C (sigma - u_x,exact)

    where `sigma` is what Source is handed. If that is sigma_hat the added term
    vanishes at the exact solution, and the closed form is the answer. If it is
    the stored sigma, -sigma_hat, the term is -2 C u_x there and the run solves
    an advection-diffusion problem instead -- an O(C) error that refinement does
    not remove.

    C is a template parameter so the same class is its own control at C = 0.
*/
template <double C>
class SigmaReadingSource : public AutodiffTransportSystem
{
public:
    SigmaReadingSource() : AutodiffTransportSystem(emptyConfig(), unitGrid(), buildSpec()) {}

    static SystemSpec buildSpec()
    {
        return {.variables = {{"u", "the diffused quantity", "", BoundaryKind::Dirichlet,
                               BoundaryKind::Dirichlet}}};
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
    Value InitialValue(Index, Position x) const override { return exactSolution(x, 0.0); }
    Value InitialDerivative(Index, Position x) const override { return exactDerivative(x, 0.0); }

private:
    Real Flux(Index i, RealVector, RealVector q, Real, Time) override { return q(i); }

    Real Source(Index, RealVector, RealVector, RealVector sigma, RealVector, Real x,
                Time t) override
    {
        const double xv = x.val;
        return diffusionSource(xv, t) + C * (sigma(0) - exactDerivative(xv, t));
    }
};

using SigmaCoupled = SigmaReadingSource<0.8>;
using SigmaUncoupled = SigmaReadingSource<0.0>;

/// An auxiliary variable equal to the flux: G = phi - sigma, which is zero when
/// phi = sigma_hat.
class SigmaReadingAux : public AutodiffTransportSystem
{
public:
    SigmaReadingAux() : AutodiffTransportSystem(emptyConfig(), unitGrid(), buildSpec()) {}

    static SystemSpec buildSpec()
    {
        return {.variables = {{"u", "", "", BoundaryKind::Dirichlet, BoundaryKind::Dirichlet}},
                .aux = {{"phi", "the flux, as an auxiliary", ""}}};
    }

private:
    Real Flux(Index i, RealVector, RealVector q, Real, Time) override { return q(i); }
    Real GFunc(Index, RealVector, RealVector, RealVector sigma, RealVector phi, Position,
               Time) override
    {
        return phi(0) - sigma(0);
    }
    Value InitialAuxValue(Index, Position, Time) const override { return 0.0; }
};

// ---------------------------------------------------------------------------
// The manufactured-source cases.
// ---------------------------------------------------------------------------

/*
    sigma_hat = kappa g u_x, with g the manufactured field's geometry, no
    physical source, and the forcing MMS_Source. The geometry-blind Flux is the
    stand-in a geometry-dependent case writes to compile -- kappa u_x, the flux
    at g = 1 -- and a manufactured source differentiated from it is wrong.

    Its manufactured solution and geometry are ManufacturedField's, so
    MMS_Source has to equal mms::coupledSource for that model: the source the
    hand-written coupled study (MMSFieldTests.cpp) derived independently.
*/
class GeometricMMS : public AutodiffTransportSystem
{
public:
    GeometricMMS() : AutodiffTransportSystem(emptyConfig(), unitGrid(), buildSpec())
    {
        useMMS = true;
    }

    static SystemSpec buildSpec()
    {
        return {.variables = {{"u", "the diffused quantity", "", BoundaryKind::Dirichlet,
                               BoundaryKind::Dirichlet}},
                .readsGeometry = true,
                .geometryDerivatives = GeometryDerivatives::Analytic};
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
    Value InitialValue(Index, Position x) const override { return exactSolution(x, 0.0); }
    Value InitialDerivative(Index, Position x) const override { return exactDerivative(x, 0.0); }

    Value manufacturedSource(Index i, Position x, Time t) { return MMS_Source(i, x, t); }

    /// The physical source, which is zero, plus the manufactured forcing --
    /// added at this level rather than inside Source, because MMS_Source
    /// evaluates Source to find out what the physics already supplies.
    Value Sources(Index i, const State &s, Position x, Time t) override
    {
        return AutodiffTransportSystem::Sources(i, s, x, t) + MMS_Source(i, x, t);
    }

protected:
    Real2nd MMS_Solution(Index, Real2nd x, Real2nd t) override
    {
        return sin(pi * x) * (1.0 + t);
    }

    Real2ndVector MMS_Geometry(Real2nd x, Real2nd t) override
    {
        Real2ndVector g(1);
        g(0) = 1.0 + (2.0 / pi) * (1.0 + t) * cos(pi * x);
        return g;
    }

private:
    Real Flux(Index i, RealVector, RealVector q, Real, Time) override
    {
        return coupledKappa * q(i);
    }
    Real Flux(Index i, RealVector, RealVector q, RealVector geom, Real, Time) override
    {
        return coupledKappa * geom(0) * q(i);
    }
};

/*
    Two variables whose sources read each other's flux, so that MMS_Source has
    to put each variable's own flux in its slot of `sigma`:

        sigma_hat_i = (1 + i) q_i,   S_i = MMS forcing + sigma(1 - i),

    with u_0 = u_1 = sin(pi x)(1 + t). The closed form is

        S_i = u_t - (1 + i) u_xx - (2 - i) u_x.
*/
class CrossFluxMMS : public AutodiffTransportSystem
{
public:
    CrossFluxMMS() : AutodiffTransportSystem(emptyConfig(), unitGrid(), buildSpec())
    {
        useMMS = true;
    }

    static SystemSpec buildSpec() { return {.variables = numberedFields(2)}; }

    Value manufacturedSource(Index i, Position x, Time t) { return MMS_Source(i, x, t); }

protected:
    Real2nd MMS_Solution(Index, Real2nd x, Real2nd t) override
    {
        return sin(pi * x) * (1.0 + t);
    }

private:
    Real Flux(Index i, RealVector, RealVector q, Real, Time) override
    {
        return (1.0 + static_cast<double>(i)) * q(i);
    }
    Real Source(Index i, RealVector, RealVector, RealVector sigma, RealVector, Real, Time) override
    {
        return sigma(1 - i);
    }
};

double crossFluxSource(Index i, double x, double t)
{
    const double A = 1.0 + t;
    const double ut = std::sin(pi * x);
    const double ux = A * pi * std::cos(pi * x);
    const double uxx = -A * pi * pi * std::sin(pi * x);
    return ut - (1.0 + i) * uxx - (2.0 - i) * ux;
}

// ---------------------------------------------------------------------------
// The finite-differenced Jacobian, at a state off the manifold.
// ---------------------------------------------------------------------------

/// A solver on `Problem`, sitting at its initial condition pushed off the
/// manifold so the stored sigma is not -sigma_hat there, with its Jacobian
/// blocks built at alpha.
template <class Problem>
struct JacobianFixture
{
    Grid grid{0.0, 1.0, 4};
    Problem problem;
    SystemSolver sys{grid, 2, &problem};
    SUNContext ctx{};
    N_Vector Y{}, dYdt{};
    Index n{};
    double alpha;

    explicit JacobianFixture(double alpha_) : alpha(alpha_)
    {
        sys.setTau(1.0);
        sys.resetCoeffs();
        sys.initialiseMatrices();

        SUNContext_Create(SUN_COMM_NULL, &ctx);
        n = DGSoln(problem.getNumVars(), grid, 2).getDoF();
        Y = N_VNew_Serial(n, ctx);
        dYdt = N_VClone(Y);
        N_VConst(0.0, Y);
        N_VConst(0.0, dYdt);
        sys.setInitialConditions(Y, dYdt);

        double *y = N_VGetArrayPointer(Y);
        for (Index i = 0; i < n; ++i)
            y[i] += 0.17 * std::sin(1.3 + 0.7 * i);

        sys.setJacTime(0.0);
        sys.setAlpha(alpha);
        sys.setJacEvalY(Y, dYdt);
        sys.updateBoundaryConditions(0.0);
        sys.updateMatricesForJacSolve();
    }

    ~JacobianFixture()
    {
        N_VDestroy(Y);
        N_VDestroy(dYdt);
        SUNContext_Free(&ctx);
    }

    /// The worst ||J dy - g|| / ||g|| over three right-hand sides.
    double worstSolveResidual()
    {
        const Matrix J = fdjac::jacobian(sys, Y, dYdt, 0.0, alpha);
        const std::vector<Index> skip = fdjac::undefinedRows(J);

        double worst = 0.0;
        for (int trial = 0; trial < 3; ++trial)
        {
            N_Vector g = N_VNew_Serial(n, ctx);
            N_Vector dy = N_VClone(g);
            double *ga = N_VGetArrayPointer(g);
            for (Index i = 0; i < n; ++i)
                ga[i] = std::sin(1.0 + i * (trial + 1) * 0.7);

            const Vector gVec = fdjac::toVector(g);
            sys.solveHDGJac(g, dy);
            worst = std::max(worst, fdjac::relativeResidual(J, fdjac::toVector(dy), gVec, skip));

            N_VDestroy(g);
            N_VDestroy(dy);
        }
        return worst;
    }
};

/// One coupled run of GeometricMMS against ManufacturedField's geometry, and
/// the L2 error of u at tFinal.
double geometricMMSError(Index k, Index nCells, double tFinal)
{
    Grid grid(0.0, 1.0, nCells);
    auto model = std::make_shared<ManufacturedField>();
    GeometricMMS problem;

    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.setFieldModel(model);
    sys.setFieldSolveMode(SystemSolver::FieldSolveMode::Exact);
    sys.resetCoeffs();
    sys.setInputFile("autodiff_geometric_mms");
    sys.setOutputCadence(tFinal);
    sys.setNOutput(2);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-14);
    Tolerances tol;
    sys.setTolerances({tol.absolute}, tol.relative);
    sys.setWriteOutput(false);
    sys.setWriteDatFile(false);

    {
        CapturedOutput quiet;
        sys.runSolver(tFinal);
    }
    return l2Error(sys, grid, tFinal);
}

const std::vector<double> kSampleX = {0.07, 0.13, 0.37, 0.5, 0.62, 0.81, 0.94};
const std::vector<double> kSampleT = {0.0, 0.1, 0.25};

} // namespace

BOOST_AUTO_TEST_SUITE(autodiff_sigma_tests)

// ---------------------------------------------------------------------------
// sigma is sigma_hat.
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(source_sees_the_physical_flux_not_the_stored_one)
{
    // Pointwise, before any solver: a stored sigma of -0.3 is a physical flux
    // of +0.3, and that is what Source is handed.
    SigmaCoupled problem;
    const double x = 0.31, t = 0.2;

    State s(1);
    s.sigma(0) = -0.3;
    const double expected = diffusionSource(x, t) + 0.8 * (0.3 - exactDerivative(x, t));
    BOOST_TEST(problem.Sources(0, s, x, t) == expected, boost::test_tools::tolerance(1e-14));
}

BOOST_AUTO_TEST_CASE(dsources_dsigma_is_the_derivative_in_the_stored_sigma)
{
    // The compensating negation. The Jacobian's sigma column is d/d(stored
    // sigma), so this has to agree with a difference of Sources in the stored
    // value -- -C here, not the +C autodiff takes with respect to sigma_hat.
    SigmaCoupled problem;
    const double x = 0.31, t = 0.2, h = 1e-6;

    auto at = [&](double stored)
    {
        State s(1);
        s.sigma(0) = stored;
        return problem.Sources(0, s, x, t);
    };
    const double fd = (at(-0.3 + h) - at(-0.3 - h)) / (2.0 * h);

    State s(1);
    s.sigma(0) = -0.3;
    Vector d = Vector::Zero(1);
    problem.dSources_dsigma(0, d, s, x, t);

    BOOST_TEST(d(0) == fd, boost::test_tools::tolerance(1e-8));
    BOOST_TEST(d(0) == -0.8, boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(the_auxiliary_function_sees_the_physical_flux_too)
{
    // GFunc on the same footing as Source: phi = 0.3 and a stored sigma of
    // -0.3 satisfy phi = sigma_hat, and d/d(stored sigma) of phi - sigma_hat
    // is +1.
    SigmaReadingAux problem;
    const double x = 0.4, t = 0.0, h = 1e-6;

    auto at = [&](double stored)
    {
        State s(1, 0, 1);
        s.sigma(0) = stored;
        s.phi(0) = 0.3;
        return problem.AuxG(0, s, x, t);
    };
    BOOST_TEST(at(-0.3) == 0.0, boost::test_tools::tolerance(1e-14));

    State s(1, 0, 1);
    s.sigma(0) = -0.3;
    s.phi(0) = 0.3;
    State out(1, 0, 1);
    problem.AuxGPrime(0, out, s, x, t);

    const double fd = (at(-0.3 + h) - at(-0.3 - h)) / (2.0 * h);
    BOOST_TEST(out.sigma(0) == fd, boost::test_tools::tolerance(1e-8));
    BOOST_TEST(out.sigma(0) == 1.0, boost::test_tools::tolerance(1e-12));
    BOOST_TEST(out.phi(0) == 1.0, boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(a_source_reading_sigma_solves_the_manufactured_problem)
{
    // The closed form. With sigma = sigma_hat the added term cancels at the
    // exact solution, so the coupled and uncoupled cases converge to the same
    // function; with the stored sigma the coupled one would solve
    // u_t - u_xx + 1.6 u_x = ..., an O(1) error in u.
    const Index k = 2, n = 16;
    const Errors coupled = solveAndMeasureBoth<SigmaCoupled>(k, n, 0.1);
    const Errors uncoupled = solveAndMeasureBoth<SigmaUncoupled>(k, n, 0.1);

    BOOST_TEST_MESSAGE("source reading sigma: u error " << coupled.u << ", control "
                                                        << uncoupled.u);
    BOOST_TEST(coupled.u < 1e-3);
    BOOST_TEST(coupled.u < 10.0 * uncoupled.u);
}

BOOST_AUTO_TEST_CASE(the_jacobian_carries_dsources_dsigma_with_the_right_sign)
{
    // The derivative half, which the solve above cannot see: a wrong sign on
    // dS/dsigma slows Newton and leaves the answer alone. Steady (alpha = 0),
    // because the term is not weighted by alpha.
    JacobianFixture<SigmaCoupled> f(0.0);
    const double worst = f.worstSolveResidual();
    BOOST_TEST_MESSAGE("sigma-reading source: worst ||J dy - g|| / ||g|| = " << worst);
    BOOST_TEST(worst < 1e-6);
}

// ---------------------------------------------------------------------------
// MMS_Source differentiates the flux the solver integrates.
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(the_manufactured_source_uses_the_geometry_aware_flux)
{
    // Against the coupled study's own hand-derived source for this geometry.
    // The geometry-blind stand-in gives the g = 1 source instead, and dropping
    // the dg/dx chain term leaves out kappa A pi g' cos(pi x); either is out by
    // O(1) at most of these points.
    GeometricMMS problem;
    const ManufacturedField model;

    for (double t : kSampleT)
        for (double x : kSampleX)
            BOOST_TEST(problem.manufacturedSource(0, x, t) == coupledSource(model, x, t),
                       boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(each_variable_s_flux_goes_in_its_own_slot)
{
    // sigma(j) is variable j's flux, not variable i's.
    CrossFluxMMS problem;
    for (Index i : {0, 1})
        for (double t : kSampleT)
            for (double x : kSampleX)
                BOOST_TEST(problem.manufacturedSource(i, x, t) == crossFluxSource(i, x, t),
                           boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(a_coupled_run_forced_by_mms_source_converges_to_the_manufactured_solution)
{
    // End to end: the transport case's whole forcing is MMS_Source, the
    // geometry comes from the attached field model, and u converges at k + 1
    // to the manufactured solution -- with the same error, at the same
    // resolution, as the coupled study's hand-written case, which solves the
    // same problem with the source derived by hand. A source built against the
    // stand-in flux is O(1) wrong and converges to something else.
    const Index k = 2;
    const double tFinal = 0.25;
    const double e8 = geometricMMSError(k, 8, tFinal);
    const double e16 = geometricMMSError(k, 16, tFinal);
    const double order = localOrder(8, 16, e8, e16);
    const double byHand =
        solveCoupledOnce<ManufacturedField>(k, 16, tFinal, false, SystemSolver::FieldSolveMode::Exact).u;

    BOOST_TEST_MESSAGE("MMS_Source with geometry, k = 2: e(8) = "
                       << e8 << ", e(16) = " << e16 << ", order " << order
                       << "; the hand-derived case at n = 16: " << byHand);
    BOOST_TEST(order > k + 0.8);
    BOOST_TEST(e16 == byHand, boost::test_tools::tolerance(1e-8));
}

BOOST_AUTO_TEST_SUITE_END()
