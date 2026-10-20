#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "../../SystemSolver.hpp"
#include "../../Types.hpp"

#include <netcdf>
#include <nvector/nvector_serial.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

/*
    The explicit time derivative hooks (dSigmaFn_dt, dAuxG_dt, dLowerBoundary_dt,
    ...), the solver's dF/dt at fixed state assembled from them, and what it is
    for: the algebraic components of y' at a projected restart, solved from the
    differentiated constraints G_z z' = -G_t - G_u u'.
 */

namespace
{

/// A diffusion whose formulas depend on t explicitly everywhere they can:
///
///     sigma_hat = kappa(t) q,   kappa = 1 + sin(t) / 2
///     S         = 1 + x - phi
///     G         = phi - (1 + t) u^2
///     a Neumann lower end with data 0.2 t, a Dirichlet upper end at cos(t) / 2.
class TimeDependentAuxDiffusion : public TransportSystem
{
public:
    TimeDependentAuxDiffusion()
        : TransportSystem({.variables = {{"u", "the diffused quantity", "", BoundaryKind::Neumann,
                                          BoundaryKind::Dirichlet}},
                           .aux = {{"phi", "(1 + t) u^2", ""}},
                           .regrid = RegridPolicy::InPlace})
    {
    }

    static double kappa(Time t) { return 1.0 + 0.5 * std::sin(t); }

    Value SigmaFn(Index, const State &s, Position, Time t) override { return kappa(t) * s.q(0); }
    Value Sources(Index, const State &s, Position x, Time) override { return 1.0 + x - s.phi(0); }
    Value AuxG(Index, const State &s, Position, Time t) override
    {
        return s.phi(0) - (1.0 + t) * s.u(0) * s.u(0);
    }

    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time t) override { v[0] = kappa(t); }
    void dSigma_dPhi(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override { v[0] = 0.0; }
    void dSources_dPhi(Index, VectorRef v, const State &, Position, Time) override { v[0] = -1.0; }
    void AuxGPrime(Index, State &out, const State &s, Position, Time t) override
    {
        out.u(0) = -2.0 * (1.0 + t) * s.u(0);
        out.phi(0) = 1.0;
    }

    Value dSigmaFn_dt(Index, const State &s, Position, Time t) override { return 0.5 * std::cos(t) * s.q(0); }
    Value dAuxG_dt(Index, const State &s, Position, Time) override { return -s.u(0) * s.u(0); }

    Value LowerBoundary(Index, Time t) const override { return 0.2 * t; }
    Value UpperBoundary(Index, Time t) const override { return 0.5 * std::cos(t); }
    Value dLowerBoundary_dt(Index, Time) const override { return 0.2; }
    Value dUpperBoundary_dt(Index, Time t) const override { return -0.5 * std::sin(t); }

    Value InitialValue(Index, Position x) const override { return 0.5 + 0.3 * (1.0 - x * x); }
    Value InitialDerivative(Index, Position x) const override { return -0.6 * x; }
    Value InitialAuxValue(Index, Position x) const override
    {
        const double u = 0.5 + 0.3 * (1.0 - x * x);
        return u * u;
    }
};

void configure(SystemSolver &sys, std::string const &stem, double cadence, double t0)
{
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile(stem);
    sys.setOutputCadence(cadence);
    sys.setNOutput(11);
    sys.setInitialTime(t0);
    sys.setMinStepSize(1e-12);
    sys.setTolerances({1e-8}, 1e-6);
    sys.setWriteDatFile(false);
}

void removeOutput(std::string const &stem)
{
    for (const char *ext : {".nc", ".restart.nc", ".dat"})
        std::filesystem::remove(stem + ext);
}

struct RestartFile
{
    std::vector<Position> cellBoundaries;
    Index order = 0;
    std::vector<double> Y, dYdt;
};

RestartFile readRestart(std::string const &stem)
{
    RestartFile f;
    netCDF::NcFile file(stem + ".restart.nc", netCDF::NcFile::FileMode::read);
    netCDF::NcGroup grid = file.getGroup("Grid");
    f.cellBoundaries.resize(grid.getDim("Index").getSize());
    grid.getVar("CellBoundaries").getVar(f.cellBoundaries.data());
    grid.getVar("PolyOrder").getVar(&f.order);
    netCDF::NcGroup data = file.getGroup("RestartData");
    const size_t n = data.getDim("nDOF").getSize();
    f.Y.resize(n);
    f.dYdt.resize(n);
    data.getVar("Y").getVar(f.Y.data());
    data.getVar("dYdt").getVar(f.dYdt.data());
    return f;
}

} // namespace

BOOST_AUTO_TEST_SUITE(explicit_time_derivative_tests)

// dF/dt at fixed (Y, Y'), assembled from the hooks, against a central
// difference of the residual in t alone. Every row: the flux and aux rows
// through the physics hooks, the q and u rows through the Dirichlet datum in
// RF_cellwise, and the trace row through the Neumann datum in L_global.
BOOST_AUTO_TEST_CASE(the_explicit_time_derivative_is_the_residuals)
{
    for (bool superconvergent : {false, true})
    {
        BOOST_TEST_CONTEXT("superconvergent = " << superconvergent)
        {
            Grid grid(0.0, 1.0, 4);
            TimeDependentAuxDiffusion problem;
            SystemSolver sys(grid, 3, &problem);
            configure(sys, "explicit_dt_assembly", 0.1, 0.0);
            sys.setWriteOutput(false);
            sys.setSuperconvergent(superconvergent);
            {
                CapturedOutput quiet;
                sys.initialize();
            }

            const double t = 0.37, eps = 1e-5;
            N_Vector Fp = N_VClone(sys.Y), Fm = N_VClone(sys.Y), dFdt = N_VClone(sys.Y);
            sys.residual(t + eps, sys.Y, sys.dYdt, Fp);
            sys.residual(t - eps, sys.Y, sys.dYdt, Fm);
            N_VConst(0.0, dFdt);
            DGSoln Yh(sys.nVars, grid, 3, N_VGetArrayPointer(sys.Y), 0, sys.nAux);
            DGSoln dYh(sys.nVars, grid, 3, N_VGetArrayPointer(sys.dYdt), 0, sys.nAux);
            DGSoln out(sys.nVars, grid, 3, N_VGetArrayPointer(dFdt), 0, sys.nAux);
            sys.explicitTimeDerivative(Yh, dYh, t, out);

            double const *p = N_VGetArrayPointer(Fp), *m = N_VGetArrayPointer(Fm), *a = N_VGetArrayPointer(dFdt);
            double worst = 0.0, scale = 0.0, norm2 = 0.0;
            for (sunindextype i = 0; i < N_VGetLength(Fp); ++i)
            {
                const double fd = (p[i] - m[i]) / (2.0 * eps);
                worst = std::max(worst, std::abs(a[i] - fd));
                scale = std::max(scale, std::abs(fd));
                norm2 += a[i] * a[i];
            }
            BOOST_TEST_MESSAGE("|dF/dt - central difference| = " << worst << " against |dF/dt| = " << scale
                                                                 << ", ||dF/dt||_2 = " << std::sqrt(norm2));
            BOOST_TEST(scale > 0.1); // not vacuous
            BOOST_TEST(worst < 1e-8 * scale);

            for (N_Vector v : {Fp, Fm, dFdt})
                N_VDestroy(v);
            CapturedOutput quiet;
            sys.destroySundials();
        }
    }
}

// A restart onto a different degree is projected, and IDACalcIC then makes its
// values consistent but leaves the algebraic components of y' at zero, because
// IDA_YA_YDP_INIT computes algebraic values and differential derivatives only.
// They are now solved from G_z z' = -G_t - G_u u', exactly. So the algebraic
// rows' total derivative along (1, y') vanishes, which it does not with z' = 0.
BOOST_AUTO_TEST_CASE(a_projected_restart_starts_from_consistent_algebraic_derivatives)
{
    const std::string stem = "explicit_dt_restart";
    const double T_RESTART = 0.25;
    {
        Grid grid(0.0, 1.0, 4);
        TimeDependentAuxDiffusion problem;
        SystemSolver sys(grid, 3, &problem);
        configure(sys, stem, T_RESTART, 0.0);
        sys.setWriteOutput(true);
        CapturedOutput quiet;
        sys.runSolver(T_RESTART);
    }
    const RestartFile file = readRestart(stem);
    Grid grid(file.cellBoundaries);
    TimeDependentAuxDiffusion problem;
    problem.setRestartValues(file.Y, file.dYdt, grid, file.order);

    SystemSolver sys(grid, file.order + 1, &problem); // projected, not copied
    configure(sys, stem + "_resumed", T_RESTART / 5.0, T_RESTART);
    sys.setWriteOutput(false);
    {
        CapturedOutput quiet;
        sys.initialize();
    }
    BOOST_REQUIRE(sys.initialConditionWasCorrected());
    BOOST_TEST_MESSAGE("refinements " << sys.consistentDerivativeRefinements << ", final defect "
                                      << sys.consistentDerivativeDefect);
    BOOST_TEST(sys.consistentDerivativeRefinements <= 2);
    BOOST_TEST(sys.consistentDerivativeDefect <= 1e-14);

    // d/dt of the algebraic rows along (1, y'): [F(t + e, Y + e Y') - F(t - e, Y - e Y')] / 2e,
    // with Y' held fixed (the algebraic rows do not read it).
    auto alongTrajectory = [&](N_Vector dY)
    {
        const double e = 1e-6;
        N_Vector Yp = N_VClone(sys.Y), Ym = N_VClone(sys.Y), Fp = N_VClone(sys.Y), Fm = N_VClone(sys.Y);
        N_VLinearSum(1.0, sys.Y, e, dY, Yp);
        N_VLinearSum(1.0, sys.Y, -e, dY, Ym);
        sys.residual(T_RESTART + e, Yp, dY, Fp);
        sys.residual(T_RESTART - e, Ym, dY, Fm);
        double const *p = N_VGetArrayPointer(Fp), *m = N_VGetArrayPointer(Fm), *isDiff = N_VGetArrayPointer(sys.id);
        double worst = 0.0;
        for (sunindextype i = 0; i < N_VGetLength(Fp); ++i)
            if (isDiff[i] == 0.0)
                worst = std::max(worst, std::abs(p[i] - m[i]) / (2.0 * e));
        for (N_Vector v : {Yp, Ym, Fp, Fm})
            N_VDestroy(v);
        return worst;
    };
    const double consistent = alongTrajectory(sys.dYdt);
    N_Vector differentialOnly = N_VClone(sys.dYdt);
    N_VProd(sys.dYdt, sys.id, differentialOnly);
    const double zeroed = alongTrajectory(differentialOnly);
    N_VDestroy(differentialOnly);
    BOOST_TEST_MESSAGE("d/dt of the algebraic rows along y': " << consistent << " solved, " << zeroed
                                                               << " with z' = 0");
    BOOST_TEST(zeroed > 1e-2);
    BOOST_TEST(consistent < 1e-6 * zeroed);

    // ...and the resumed run gets past its first step, which at this cadence it
    // used not to.
    {
        CapturedOutput quiet;
        BOOST_CHECK_NO_THROW(sys.integrate(2.0 * T_RESTART));
        sys.destroySundials();
    }
    removeOutput(stem);
    removeOutput(stem + "_resumed");
}

BOOST_AUTO_TEST_SUITE_END()
