// Where the spectrum of the semi-discrete operator lies, and what the flux has
// to do to put it there.
//
// The choice of a BDF integration over a stiffly accurate collocation method
// rests on the stiffness here being *parabolic*. Dahlquist's second barrier
// caps an A-stable linear multistep method at order two, so BDF of order three
// to five are only A(alpha)-stable, with
//
//     order  3      4      5
//     alpha  86.03  73.35  51.84   degrees
//
// and an eigenvalue further than alpha from the negative real axis is outside
// the wedge. Radau IIA is L-stable at every order and would not care. So the
// argument is only as good as the claim that this operator's spectrum stays
// near the negative real axis -- which is a property of the *flux*, not of the
// discretisation, and is what this file measures.
//
// The quantity. Linearising F(Y, Ydot, t) = 0 about a state gives
// M Ydot + K Y = 0 with M = dF/dYdot and K = dF/dY, so a mode Y = v exp(lambda
// t) satisfies (K + lambda M) v = 0. That is to say: **lambda is exactly a
// value of the coefficient c at which the Jacobian the solver already
// assembles, J(c) = K + cM, is singular.** The spectrum is therefore not an
// extra object -- it is the set of coefficients this solver must never be
// handed, and the finite ones are the time scales of the problem.
//
// M is singular, since only u and the differential scalars carry a time
// derivative, so the pencil has infinite eigenvalues as well. For a regular
// index-one pencil there are exactly rank(M) finite ones, which is a check on
// the index worth making rather than assuming.
//
// Two practical points. The Dirichlet trace unknowns are removed first: the
// residual does not write those rows -- their constraint is imposed inside the
// linear solve -- so leaving them in makes the pencil *singular* rather than
// merely rank-deficient, and no eigenvalue of it means anything. Deleting the
// row and the column is the right reduction because the value is prescribed
// data rather than an unknown. And K and M are finite-differenced from the
// residual, so the whole calculation is independent of the assembled Jacobian
// and of the derivative hooks a case supplies.

#include <boost/test/unit_test.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "CapturedOutput.hpp"
#include "FiniteDifferenceJacobian.hpp"
#include "SystemSolver.hpp"
#include "TestDiffusion.hpp"
#include "Types.hpp"

#include <sundials/sundials_context.h>
#include <nvector/nvector_serial.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <string>
#include <toml.hpp>
#include <vector>

using namespace toml::literals::toml_literals;

namespace
{

const toml::value empty_config = u8R"(
    [DiffusionProblem]
    Kappa = 1.0
)"_toml;

// --- fluxes ---------------------------------------------------------------
//
// The stored sigma is -sigma_hat, so the equation integrated is
// u_t - d_x[sighat(u, q, x, t)] = S. With sighat = kappa q that is
// u_t - kappa u_xx = S: diffusion for kappa > 0, and the backward heat
// equation for kappa < 0. Every fixture below is a statement about the sign and
// size of d(sighat)/dq.

// The four derivative hooks are pure virtual, and none of these fixtures has a
// source that depends on the state. Zeroing them once here keeps each flux
// below to the one function that distinguishes it -- and note that the spectrum
// does not read any of them: K and M are finite-differenced from the residual,
// so these exist only so the solver can build its matrices at all.
class ZeroDerivatives : public TransportSystem
{
public:
    using TransportSystem::TransportSystem;

    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v.setZero();
    };
    void dSources_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v.setZero();
    };
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v.setZero();
    };
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override
    {
        v.setZero();
    };
};

// sighat = kappa q. Constant coefficient, so the spectrum is the discrete
// Dirichlet Laplacian's and is known in closed form.
class LinearFlux : public ZeroDerivatives
{
public:
    explicit LinearFlux(double kappa_)
        : ZeroDerivatives({.variables = numberedFields(1)}), kappa(kappa_) {};

    Value LowerBoundary(Index, Time) const override { return 0.0; };
    Value UpperBoundary(Index, Time) const override { return 0.0; };

    Value SigmaFn(Index, const State &s, Position, Time) override { return kappa * s.q(0); };
    Value Sources(Index, const State &, Position, Time) override { return 1.0; };
    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = kappa;
    };
    Value InitialValue(Index, Position x) const override { return x * (1.0 - x); };
    Value InitialDerivative(Index, Position x) const override { return 1.0 - 2.0 * x; };

    double kappa;
};

// sighat = kappa q - a u. The second term is an advection at speed a:
// u_t + a u_x - kappa u_xx = S. The cell Peclet number a*h/kappa is the
// parameter that tilts the spectrum off the real axis.
class AdvectiveFlux : public ZeroDerivatives
{
public:
    AdvectiveFlux(double kappa_, double a_)
        : ZeroDerivatives({.variables = numberedFields(1)}), kappa(kappa_), a(a_) {};

    Value LowerBoundary(Index, Time) const override { return 0.0; };
    Value UpperBoundary(Index, Time) const override { return 0.0; };

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return kappa * s.q(0) - a * s.u(0);
    };
    Value Sources(Index, const State &, Position, Time) override { return 1.0; };
    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = kappa;
    };
    void dSigmaFn_du(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = -a;
    };
    Value InitialValue(Index, Position x) const override { return x * (1.0 - x); };
    Value InitialDerivative(Index, Position x) const override { return 1.0 - 2.0 * x; };

    double kappa, a;
};

// sighat = (1 + u^2) q -- nonlinear, still strictly diffusive.
class NonlinearFlux : public ZeroDerivatives
{
public:
    NonlinearFlux() : ZeroDerivatives({.variables = numberedFields(1)}) {};

    Value LowerBoundary(Index, Time) const override { return 1.0; };
    Value UpperBoundary(Index, Time) const override { return 0.0; };

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return (1.0 + s.u(0) * s.u(0)) * s.q(0);
    };
    Value Sources(Index, const State &, Position, Time) override { return 1.0; };
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 1.0 + s.u(0) * s.u(0);
    };
    void dSigmaFn_du(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 2.0 * s.u(0) * s.q(0);
    };
    Value InitialValue(Index, Position x) const override { return 1.0 - x; };
    Value InitialDerivative(Index, Position) const override { return -1.0; };
};

// sighat_j = D_jk q_k with D constant. The multi-variable case, and the one
// that decides what the assumption has to say.
//
// If D is diagonalisable, D = P diag(mu) P^-1, then setting w = P^-1 u
// decouples the system into scalar problems with diffusivities mu_k -- the
// stabilisation tau is a scalar and commutes with P, and both ends are
// Dirichlet for every variable, so nothing recouples them. The discrete
// spectrum is therefore exactly {-nu_n mu_k}, the single-variable discrete
// Dirichlet-Laplacian eigenvalues nu_n > 0 scaled by the eigenvalues of D, and
//
//     max_lambda |arg(-lambda)|  =  max_k |arg mu_k| ,
//
// independent of the mesh, the degree, and n. The angle of the spectrum *is*
// the angle of the spectrum of D.
class MatrixFlux : public ZeroDerivatives
{
public:
    explicit MatrixFlux(Matrix const &D_)
        : ZeroDerivatives({.variables = numberedFields(D_.rows())}), D(D_) {};

    Value LowerBoundary(Index, Time) const override { return 0.0; };
    Value UpperBoundary(Index, Time) const override { return 0.0; };

    Value SigmaFn(Index i, const State &s, Position, Time) override
    {
        double out = 0.0;
        for (Index j = 0; j < D.cols(); ++j)
            out += D(i, j) * s.q(j);
        return out;
    };
    Value Sources(Index, const State &, Position, Time) override { return 1.0; };
    void dSigmaFn_dq(Index i, VectorRef v, const State &, Position, Time) override
    {
        for (Index j = 0; j < D.cols(); ++j)
            v[j] = D(i, j);
    };
    Value InitialValue(Index, Position x) const override { return x * (1.0 - x); };
    Value InitialDerivative(Index, Position x) const override { return 1.0 - 2.0 * x; };

    Matrix D;
};

// Jardin's critical-gradient diffusivity, the stiff case the transport
// literature is built around: chi jumps from chi0 to chi0 + kappa(|q| - qc)
// above a critical gradient. Still diffusive -- d(sighat)/dq is chi0 below the
// threshold and chi0 + 2 kappa(|q| - qc) above it, both positive -- but by a
// factor that changes by orders of magnitude across the domain.
class CriticalGradientFlux : public ZeroDerivatives
{
public:
    CriticalGradientFlux(double chi0_, double kappa_, double qc_)
        : ZeroDerivatives({.variables = numberedFields(1)}),
          chi0(chi0_), kappa(kappa_), qc(qc_) {};

    Value LowerBoundary(Index, Time) const override { return 1.0; };
    Value UpperBoundary(Index, Time) const override { return 0.0; };

    double chi(double q) const
    {
        const double g = std::abs(q);
        return g > qc ? chi0 + kappa * (g - qc) : chi0;
    }
    double dchi_dq(double q) const
    {
        const double g = std::abs(q);
        return g > qc ? kappa * (q > 0 ? 1.0 : -1.0) : 0.0;
    }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return chi(s.q(0)) * s.q(0);
    };
    Value Sources(Index, const State &, Position, Time) override { return 1.0; };
    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = chi(s.q(0)) + dchi_dq(s.q(0)) * s.q(0);
    };
    Value InitialValue(Index, Position x) const override { return 1.0 - x; };
    Value InitialDerivative(Index, Position) const override { return -1.0; };

    double chi0, kappa, qc;
};

// --- the calculation ------------------------------------------------------

struct Spectrum
{
    std::vector<std::complex<double>> finite;
    Index unknowns = 0;        // after the Dirichlet traces are removed
    Index differentialDOF = 0; // rank(M), i.e. how many finite eigenvalues to expect
    double maxAngleDeg = 0.0;  // furthest any eigenvalue lies from the negative real axis
    double maxRealPart = -std::numeric_limits<double>::infinity();
    double stiffness = 0.0;    // |lambda|_max / |lambda|_min
};

// Delete the rows and columns the residual never writes, from both matrices at
// once, so the reduced pencil is over the genuine unknowns.
Matrix removeRowsAndCols(Matrix const &A, std::vector<Index> const &drop)
{
    std::vector<Index> keep;
    for (Index i = 0; i < A.rows(); ++i)
        if (!fdjac::isUndefined(drop, i))
            keep.push_back(i);

    Matrix out(keep.size(), keep.size());
    for (size_t i = 0; i < keep.size(); ++i)
        for (size_t j = 0; j < keep.size(); ++j)
            out(i, j) = A(keep[i], keep[j]);
    return out;
}

Spectrum spectrumOf(SystemSolver &sys, N_Vector Y, N_Vector dYdt, double t)
{
    // K = dF/dY at cj = 0; J(1) - K isolates dF/dYdot. fdjac perturbs Y[j] by h
    // and Ydot[j] by cj*h together, which is IDA's convention and is what makes
    // the difference come out as the mass matrix rather than as something else.
    const Matrix K = fdjac::jacobian(sys, Y, dYdt, t, 0.0);
    const Matrix M = fdjac::jacobian(sys, Y, dYdt, t, 1.0) - K;

    const auto drop = fdjac::undefinedRows(K);
    const Matrix Kr = removeRowsAndCols(K, drop);
    const Matrix Mr = removeRowsAndCols(M, drop);

    Spectrum out;
    out.unknowns = Kr.rows();
    out.differentialDOF = Eigen::FullPivLU<Matrix>(Mr).rank();

    // A v = lambda B v with A = K and B = -M is (K + lambda M) v = 0, so the
    // eigenvalues come out as lambda with no further sign work.
    Eigen::GeneralizedEigenSolver<Matrix> ges;
    ges.compute(Kr, -Mr);

    const auto alphas = ges.alphas();
    const auto betas = ges.betas();
    double maxMag = 0.0, minMag = std::numeric_limits<double>::infinity();
    for (Index i = 0; i < alphas.size(); ++i)
    {
        // An infinite eigenvalue is beta = 0, and beta is only ever *nearly*
        // zero in floating point. Scaling the test by the largest |beta| makes
        // it independent of how the pencil happens to be normalised.
        if (std::abs(betas(i)) <= 1e-10 * betas.cwiseAbs().maxCoeff())
            continue;
        const std::complex<double> lambda = alphas(i) / betas(i);
        out.finite.push_back(lambda);
        maxMag = std::max(maxMag, std::abs(lambda));
        minMag = std::min(minMag, std::abs(lambda));
        out.maxRealPart = std::max(out.maxRealPart, lambda.real());
    }

    // The angle from the negative real axis, |arg(-lambda)|, which is the
    // quantity A(alpha)-stability bounds. Measured only on eigenvalues large
    // enough for the angle to mean something: the finite-difference Jacobian is
    // accurate to about 1e-10 relative, so the argument of an eigenvalue eight
    // orders below the largest is noise.
    for (auto const &lambda : out.finite)
        if (std::abs(lambda) > 1e-8 * maxMag)
            out.maxAngleDeg = std::max(
                out.maxAngleDeg,
                std::abs(std::arg(-lambda)) * 180.0 / std::numbers::pi);

    out.stiffness = minMag > 0.0 ? maxMag / minMag : 0.0;
    return out;
}

// The lighter of the two setup paths: no IDA, no output files. initialiseMatrices
// and updateBoundaryConditions are what residual() needs, and setInitialConditions
// gives a state to linearise about.
struct Fixture
{
    Fixture(TransportSystem &problem, Index k, Index nCells, double tau = 1.0)
        : grid(0.0, 1.0, nCells), sys(grid, k, &problem)
    {
        sys.setTau(tau);
        sys.resetCoeffs();
        sys.initialiseMatrices();
        SUNContext_Create(SUN_COMM_NULL, &ctx);
        DGSoln shape(problem.getNumVars(), grid, k);
        Y = N_VNew_Serial(shape.getDoF(), ctx);
        dYdt = N_VClone(Y);
        N_VConst(0.0, Y);
        N_VConst(0.0, dYdt);
        sys.setInitialConditions(Y, dYdt);
        sys.updateBoundaryConditions(0.0);
    }
    ~Fixture()
    {
        N_VDestroy(Y);
        N_VDestroy(dYdt);
        SUNContext_Free(&ctx);
    }
    Spectrum spectrum() { return spectrumOf(sys, Y, dYdt, 0.0); }

    Grid grid;
    SystemSolver sys;
    SUNContext ctx = nullptr;
    N_Vector Y = nullptr, dYdt = nullptr;
};

void report(std::string const &what, Spectrum const &s)
{
    BOOST_TEST_MESSAGE(std::format(
        "  {:<34} {:>3} unknowns, {:>3} finite, angle {:>7.3f} deg, "
        "max Re {:>11.3e}, stiffness {:>9.2e}",
        what, s.unknowns, s.finite.size(), s.maxAngleDeg, s.maxRealPart,
        s.stiffness));
}

} // namespace

BOOST_AUTO_TEST_SUITE(spectrum_tests)

BOOST_AUTO_TEST_CASE(the_pencil_has_one_finite_eigenvalue_per_differential_unknown)
{
    // The index-one statement, counted rather than asserted. rank(M) is the
    // number of unknowns carrying a time derivative -- the nodal values of u --
    // and a regular index-one pencil has exactly that many finite eigenvalues.
    // A count short of it would mean the algebraic block had gone singular; a
    // count over it is not possible for this M.
    for (Index k : {1, 2, 3})
        for (Index nCells : {3, 6})
        {
            LinearFlux problem(1.0);
            Fixture f(problem, k, nCells);
            const auto s = f.spectrum();
            report(std::format("kappa q, k={} cells={}", k, nCells), s);
            BOOST_TEST(s.differentialDOF == nCells * (k + 1));
            BOOST_TEST(static_cast<Index>(s.finite.size()) == s.differentialDOF);
        }
}

BOOST_AUTO_TEST_CASE(a_constant_diffusivity_reproduces_the_continuum_eigenvalues)
{
    // Validation of the whole calculation against something known. For
    // u_t = kappa u_xx on (0,1) with Dirichlet ends the eigenvalues are
    // -kappa n^2 pi^2. If the pencil, the sign convention or the Dirichlet
    // reduction were wrong, this is what would say so.
    //
    // Only the low modes are resolved, which is ordinary: a discretisation with
    // N degrees of freedom does not approximate N eigenvalues, and the rest of
    // this spectrum is real, negative and meaningless. The count that is
    // resolved rises with the mesh and with tau, and the extreme eigenvalue
    // scales as h^-2 -- the parabolic scaling, and the reason an implicit
    // method is wanted here at all.
    LinearFlux problem(1.0);
    Fixture f(problem, 4, 12);
    const auto s = f.spectrum();
    report("kappa q, k=4 cells=12", s);

    std::vector<double> real;
    for (auto const &lambda : s.finite)
    {
        BOOST_TEST(std::abs(lambda.imag()) < 1e-6 * std::abs(lambda),
                   "eigenvalue " << lambda.real() << " + " << lambda.imag()
                   << "i is not real");
        real.push_back(lambda.real());
    }
    std::sort(real.begin(), real.end(), std::greater<double>());

    const double pi2 = std::numbers::pi * std::numbers::pi;
    for (int n = 1; n <= 3; ++n)
    {
        const double exact = -static_cast<double>(n * n) * pi2;
        const double rel = std::abs(real[n - 1] - exact) / std::abs(exact);
        BOOST_TEST_MESSAGE(std::format("    n={}  computed {:>14.8f}  exact {:>14.8f}"
                                       "  rel {:.2e}",
                                       n, real[n - 1], exact, rel));
        BOOST_TEST(rel < 1e-6);
    }

    // The extreme eigenvalue goes as h^-2. Halving h should multiply it by four,
    // and does to better than a percent -- which is the statement that the
    // stiffness of this operator is the stiffness of a diffusion operator, with
    // no faster time scale hidden in the hybridisation.
    double previous = 0.0;
    for (Index nc : {6, 12, 24})
    {
        LinearFlux pr(1.0);
        Fixture ff(pr, 4, nc);
        const auto sp = ff.spectrum();
        double extreme = 0.0;
        for (auto const &l : sp.finite)
            extreme = std::min(extreme, l.real());
        if (previous != 0.0)
        {
            const double ratio = extreme / previous;
            BOOST_TEST_MESSAGE(std::format("    cells={:<3} extreme {:>14.2f}"
                                           "  ratio {:.4f}",
                                           nc, extreme, ratio));
            BOOST_TEST(std::abs(ratio - 4.0) < 0.04);
        }
        previous = extreme;
    }
}

BOOST_AUTO_TEST_CASE(a_diffusive_flux_puts_the_spectrum_on_the_negative_real_axis)
{
    // The claim the integrator choice rests on. Three fluxes spanning what a
    // transport model does -- constant, nonlinear in u, and a critical-gradient
    // chi whose derivative jumps by two orders of magnitude across the domain --
    // and in every case the spectrum is real, negative, and so at angle zero.
    //
    // What they have in common is the only thing that matters: d(sighat)/dq > 0
    // everywhere. That is the assumption, and it is what "locally at least
    // mildly diffusive" means.
    {
        LinearFlux problem(1.0);
        Fixture f(problem, 3, 8);
        const auto s = f.spectrum();
        report("kappa q", s);
        BOOST_TEST(s.maxAngleDeg < 1e-4);
        BOOST_TEST(s.maxRealPart < 0.0);
    }
    {
        NonlinearFlux problem;
        Fixture f(problem, 3, 8);
        const auto s = f.spectrum();
        report("(1 + u^2) q", s);
        BOOST_TEST(s.maxAngleDeg < 1e-4);
        BOOST_TEST(s.maxRealPart < 0.0);
    }
    {
        CriticalGradientFlux problem(0.1, 10.0, 0.5);
        Fixture f(problem, 3, 8);
        const auto s = f.spectrum();
        report("chi(q) q, critical gradient", s);
        BOOST_TEST(s.maxAngleDeg < 1e-4);
        BOOST_TEST(s.maxRealPart < 0.0);
    }
}

BOOST_AUTO_TEST_CASE(advection_leaves_the_spectrum_alone_until_the_cell_peclet_number_does_not)
{
    // The V_j term. Adding -a u to the flux makes the operator non-self-adjoint:
    // u_t + a u_x - kappa u_xx = S. One might expect that to tilt the spectrum
    // immediately, and it does not, for a reason worth knowing. On a bounded
    // interval with Dirichlet ends the constant-coefficient advection-diffusion
    // operator is *symmetrisable*: the gauge transformation v = u exp(-ax/2kappa)
    // turns it into -kappa d_xx + a^2/4kappa, which is self-adjoint. Its
    // spectrum is therefore real, at -(kappa n^2 pi^2 + a^2/4kappa).
    //
    // What the discretisation does to that similarity is the question, since the
    // transformation is increasingly ill-conditioned as a/kappa grows. The sweep
    // below answers it: the discrete spectrum stays real while the *cell* Peclet
    // number a h / kappa is of order one or less, tilts once it is not, and at a
    // cell Peclet number of order a hundred crosses into the right half plane
    // altogether -- an under-resolution instability of the discretisation, not a
    // property of the equation, and the usual reason to resolve or upwind an
    // advection-dominated layer.
    const double kappa = 1.0;
    const Index k = 3, nCells = 8;
    const double h = 1.0 / nCells;

    for (double a : {1.0, 8.0, 32.0, 100.0, 1000.0})
    {
        AdvectiveFlux problem(kappa, a);
        Fixture f(problem, k, nCells);
        const auto s = f.spectrum();
        report(std::format("kappa q - a u, Pe_cell={:.3g}", a * h / kappa), s);
    }

    // At a cell Peclet number of one the spectrum is still real to the accuracy
    // of the finite-difference Jacobian, so nothing about the BDF wedge is in
    // question for a transport problem, which is diffusion-dominated by
    // construction.
    {
        AdvectiveFlux problem(kappa, 8.0);
        Fixture f(problem, k, nCells);
        const auto s = f.spectrum();
        BOOST_TEST(s.maxAngleDeg < 1e-4);
        BOOST_TEST(s.maxRealPart < 0.0);
    }

    // And the failure is under-resolution rather than the method: it is the cell
    // Peclet number that matters, so refining at fixed a recovers a real
    // spectrum from one that had tilted.
    {
        AdvectiveFlux coarse(kappa, 100.0);
        Fixture fc(coarse, k, 8);
        const auto sc = fc.spectrum();
        BOOST_TEST(sc.maxAngleDeg > 1.0);

        AdvectiveFlux fine(kappa, 100.0);
        Fixture ff(fine, k, 128);
        const auto sf = ff.spectrum();
        report("kappa q - 100 u, 128 cells", sf);
        BOOST_TEST(sf.maxAngleDeg < sc.maxAngleDeg);
        BOOST_TEST(sf.maxRealPart < 0.0);
    }
}

BOOST_AUTO_TEST_CASE(the_spectrums_angle_is_the_angle_of_the_spectrum_of_D)
{
    // The multi-variable statement, and the sharp one. Writing the flux as
    //
    //     sigma_j ~= V_j(u, q) + D_jk(u, q) dq_k ,   D_jk = d(sighat_j)/d(q_k)
    //
    // the assumption that matters is not that D is positive definite but that
    // its spectrum lies in a *sector* about the positive real axis:
    //
    //     max_k |arg mu_k(D)|  <=  alpha ,
    //
    // with alpha the A(alpha) wedge of the highest BDF order in use. The two
    // are not the same condition, and the gap between them is exactly what the
    // rows below measure. A real matrix whose symmetric part is positive
    // definite has every eigenvalue in the open right half plane -- so the
    // problem is well posed and the semi-discrete spectrum is in the left half
    // plane -- while arg mu may still approach 90 degrees, which is outside
    // every wedge above order two.
    //
    // D = [[1, b], [-b, 1]] has eigenvalues 1 +- ib and symmetric part the
    // identity, so it is positive definite for every b and its spectrum leaves
    // the sector as fast as b grows. It is not a contrivance: Onsager-Casimir
    // reciprocity in a magnetic field, L_ij(B) = L_ji(-B), permits exactly this
    // antisymmetric part, and a gyro-like cross-transport term produces it.
    const double wedgeBDF3 = 86.03, wedgeBDF4 = 73.35, wedgeBDF5 = 51.84;

    // Symmetric positive definite: eigenvalues real positive, angle zero. The
    // off-diagonal coupling is strong -- the eigenvalues are 0.5 and 2.5 -- so
    // this is not a diagonal matrix in disguise.
    {
        Matrix D(2, 2);
        D << 1.5, 1.0, 1.0, 1.5;
        MatrixFlux problem(D);
        Fixture f(problem, 3, 6);
        const auto s = f.spectrum();
        report("D symmetric, spec 0.5 and 2.5", s);
        BOOST_TEST(s.maxAngleDeg < 1e-4);
        BOOST_TEST(s.maxRealPart < 0.0);
    }

    // Non-symmetric, positive definite, and the angle tracks arg(1 + ib)
    // exactly -- which is the claim above, checked rather than argued.
    for (double b : {0.2, 1.0, 3.0, 10.0})
    {
        Matrix D(2, 2);
        D << 1.0, b, -b, 1.0;
        MatrixFlux problem(D);
        Fixture f(problem, 3, 6);
        const auto s = f.spectrum();
        const double predicted = std::atan(b) * 180.0 / std::numbers::pi;
        report(std::format("D = [[1,{0}],[-{0},1]], predicted {1:.2f} deg", b, predicted), s);

        // Still well posed at every b: the symmetric part is the identity.
        BOOST_TEST(s.maxRealPart < 0.0);
        BOOST_TEST(std::abs(s.maxAngleDeg - predicted) < 1e-3,
                   "angle " << s.maxAngleDeg << " against arg(1 + " << b
                   << "i) = " << predicted);
    }

    // What that costs, order by order. b = 1 is inside every wedge; b = 3
    // leaves BDF5's; b = 10 leaves BDF4's as well. A positive definite D is
    // therefore not sufficient on its own, and this is the row that says so.
    {
        Matrix D(2, 2);
        D << 1.0, 3.0, -3.0, 1.0;
        MatrixFlux problem(D);
        Fixture f(problem, 3, 6);
        const auto s = f.spectrum();
        BOOST_TEST(s.maxAngleDeg > wedgeBDF5);
        BOOST_TEST(s.maxAngleDeg < wedgeBDF4);
    }
    {
        Matrix D(2, 2);
        D << 1.0, 10.0, -10.0, 1.0;
        MatrixFlux problem(D);
        Fixture f(problem, 3, 6);
        const auto s = f.spectrum();
        BOOST_TEST(s.maxAngleDeg > wedgeBDF4);
        BOOST_TEST(s.maxAngleDeg < wedgeBDF3);
    }
}

BOOST_AUTO_TEST_CASE(a_symmetrisable_D_is_the_condition_that_is_actually_checkable)
{
    // The sector condition is a statement about eigenvalues, which a physics
    // case cannot readily check. A sufficient condition it can is
    // symmetrisability: if D = S A with S symmetric positive definite and A
    // symmetric, then D is similar to S^{1/2} A S^{1/2}, which is symmetric, so
    // spec(D) is real. Positive as well when A is positive definite, and the
    // angle is then zero whatever the coupling.
    //
    // This is the usual structure of a transport matrix -- an Onsager matrix
    // times a metric -- which is why the assumption is mild in practice even
    // though it is not implied by positive definiteness.
    Matrix S(2, 2), A(2, 2);
    S << 2.0, 0.7, 0.7, 1.0;   // symmetric positive definite
    A << 1.0, -0.4, -0.4, 3.0; // symmetric positive definite
    const Matrix D = S * A;

    BOOST_TEST_MESSAGE(std::format("    D = [[{:.3f}, {:.3f}], [{:.3f}, {:.3f}]],"
                                   " non-symmetric by {:.3f}",
                                   D(0, 0), D(0, 1), D(1, 0), D(1, 1),
                                   std::abs(D(0, 1) - D(1, 0))));

    MatrixFlux problem(D);
    Fixture f(problem, 3, 6);
    const auto s = f.spectrum();
    report("D = S A, both symmetric positive definite", s);
    BOOST_TEST(s.maxAngleDeg < 1e-4);
    BOOST_TEST(s.maxRealPart < 0.0);
}

BOOST_AUTO_TEST_CASE(an_antidiffusive_flux_is_what_the_assumption_excludes)
{
    // Not vacuous: reverse the sign of d(sighat)/dq and the eigenvalues cross
    // into the right half plane. The equation is then the backward heat
    // equation, which no A-stable method integrates because there is nothing
    // stable to integrate -- the ill-posedness is the physics, not the scheme.
    // This is the case the paper's assumption rules out, and it says why the
    // assumption has to be stated rather than assumed.
    LinearFlux problem(-1.0);
    Fixture f(problem, 3, 8);
    const auto s = f.spectrum();
    report("-kappa q (anti-diffusive)", s);
    BOOST_TEST(s.maxRealPart > 0.0);
    BOOST_TEST(s.maxAngleDeg > 179.0);
}

BOOST_AUTO_TEST_SUITE_END()
