// Order-of-accuracy tests for the couplings: auxiliary variables and global
// scalars, with the superconvergent flag off and on.
//
// The flag-on Jacobian assembles star blocks for both -- Sigma_phi, Sphi and the
// aux constraint rows at SystemSolver.cpp:688-734, dSources_dScalars_StarMat at
// :783 -- and until this file existed nothing measured whether the observed
// order survived either. That gap mattered because the Jacobian is never
// assembled: an error in those blocks costs Newton iterations rather than
// accuracy, so the regression suite cannot see it and an order study is the only
// instrument that reaches it.
//
// Two things to know before reading the cases.
//
// FIRST: the scalars never see u*. ScalarG and ScalarGPrime are evaluated on
// Y_h.evalOnNodes() -- the element nodes, with u_h -- in both the residual
// (SystemSolver.cpp:1194-1201) and the Jacobian (SystemSolver.cpp:815-818),
// regardless of the flag, by design (SystemSolver.hpp:359-363: "the scalars do
// not enter the postprocessing"). So no scalar can superconverge here, and this
// file does not assert that one does. What it asserts is that coupling a scalar
// in does not cost u_h its k+1 or u* its k+2.
//
// SECOND, and much easier to get wrong: a compensating source term written
// against the *discrete* state can be an exact row operation on the residual,
// and then it changes nothing at all. ManufacturedReaction's `f - F(u)` device
// works because F(u) appears nowhere else. If the same device is used against an
// aux variable -- Sources = S(x,t) - G, with G the aux constraint -- then since
// residual() evaluates Sources and AuxG on the same states at the same abscissae
// and pushes both through the same projectOntoTestSpace (SystemSolver.cpp:1104),
// linearity gives
//
//     S_cellwise = P(S) - res.Aux
//
// exactly, so adding the term is precisely `res.u += res.Aux`. The solution set
// is unchanged at every h, in both modes, and the study silently measures the
// uncoupled problem. ManufacturedAux below therefore compensates against
// uExact(x,t), which is a known function of x and t and so cannot cancel against
// any residual row.

#include <boost/test/unit_test.hpp>

#include "MMSHarness.hpp"
#include "PyIntegrator.hpp"
#include "ScalarDerivativeCheck.hpp"

#include <array>
#include <cmath>
#include <format>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace
{

using namespace mms;
using scalarcheck::checkScalarDerivative;

// ------------------------------------------------------------ aux variable --

// The nonlinear-flux problem of MMSConvergenceTests.cpp, with the (1 + u^2)
// factor routed through an algebraic auxiliary variable:
//
//     G         = phi - u^2                     so phi = u^2
//     sigma_hat = (1 + phi) q
//     Sources   = nonlinearFluxSource(x, t) + ( uExact(x,t)^4 - phi^2 )
//
// The trailing term vanishes at the exact solution, where phi = u^2 = uExact^2
// and so phi^2 = uExact^4. It is written against uExact rather than against u
// deliberately -- see the note at the top of this file -- and it is squared
// rather than linear so that dSources_dPhi = -2 phi is state dependent, which is
// the only thing that exercises the non-constant path through the phi column.
//
// Worth knowing what the flux coupling alone would and would not show. With the
// flag off, res.Aux = M * G(nodes) with M nonsingular, so G vanishes at every
// node; the basis is nodal, so phi_h's coefficients *are* u_h(node)^2 and
// sigma_hat = (1 + phi_h) q_h equals (1 + u_h^2) q_h node for node. The aux
// routing is then exactly the direct problem. With the flag on it is not: phi is
// sampled at the star nodes through V while u is replaced by u*, and the
// constraint becomes a projection rather than an interpolation. The source term
// is what makes the two differ in both modes.
class ManufacturedAux : public TransportSystem
{
public:
    ManufacturedAux()
        : TransportSystem({.variables = numberedFields(1), .aux = numberedAux(1)})
    {
    }

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index, const State &s, Position, Time) override
    {
        return (1.0 + s.phi(0)) * s.q(0);
    }

    Value Sources(Index, const State &s, Position x, Time t) override
    {
        const double ue = exactSolution(x, t);
        const double ue4 = ue * ue * ue * ue;
        return nonlinearFluxSource(x, t) + (ue4 - s.phi(0) * s.phi(0));
    }

    // G = phi - u^2. AuxGPrime reports dG with respect to every field at once,
    // into a State that arrives zeroed, so only the two nonzero entries are
    // written -- and the u it differentiates is the *incoming* state's, not the
    // out-parameter's.
    Value AuxG(Index, const State &s, Position, Time) override
    {
        return s.phi(0) - s.u(0) * s.u(0);
    }

    void AuxGPrime(Index, State &out, const State &s, Position, Time) override
    {
        out.phi(0) = 1.0;
        out.u(0) = -2.0 * s.u(0);
    }

    // The hook is dSigma_dPhi, not dSigmaFn_dPhi: a case that spells it the
    // second way overrides nothing, and TransportSystem::dSigma_dPhi throws at
    // the first Jacobian assembly.
    void dSigma_dPhi(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = s.q(0);
    }
    void dSources_dPhi(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = -2.0 * s.phi(0);
    }

    void dSigmaFn_dq(Index, VectorRef v, const State &s, Position, Time) override
    {
        v[0] = 1.0 + s.phi(0);
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

    Value InitialValue(Index, Position x) const override { return exactSolution(x, 0.0); }
    Value InitialDerivative(Index, Position x) const override
    {
        return exactDerivative(x, 0.0);
    }
    Value InitialAuxValue(Index, Position x) const override
    {
        const double u0 = exactSolution(x, 0.0);
        return u0 * u0;
    }

    /// The L2 error of the aux field against its own exact value, u_exact^2.
    static double extraError(SystemSolver &sys, Grid const &grid, double t)
    {
        return l2ErrorAgainst([&](double x) { return sys.yJac.Aux(0)(x); },
                              [](double x, double tt)
                              {
                                  const double ue = exactSolution(x, tt);
                                  return ue * ue;
                              },
                              grid, t);
    }
};

// ----------------------------------------------------------------- scalars --

/// The manufactured forcing for sigma_hat = q, i.e. u_t - u_xx = S.
double linearSource(double x, double t)
{
    return std::sin(pi * x) * (1.0 + pi * pi * (1.0 + t));
}

// One global scalar constrained to a functional of the solution, coupled back
// into the source:
//
//     sigma_hat = q
//     Sources   = linearSource(x, t) + ( muExact(t) - mu ) sin(pi x)
//
// The compensating term vanishes at the exact solution and is *not* a row
// operation -- the scalar row is a single global equation while the source term
// is cell-local, so it genuinely perturbs the answer. Without it the v vector of
// the bordered solve would be identically zero and the elimination would go
// untested.
//
// muExact is pure virtual rather than a shared constant on purpose. The obvious
// way to write the differential variant is to derive from the algebraic one and
// override only the constraint, which silently inherits the wrong muExact and
// leaves the source wrong by an O(1) amount -- the manufactured solution is then
// simply not the solution, and the rate collapses to zero. Making Sources depend
// on a virtual that each case must supply removes the possibility.
class ManufacturedScalarBase : public TransportSystem
{
public:
    explicit ManufacturedScalarBase(bool differential)
        : TransportSystem({.variables = numberedFields(1),
                           .scalars = numberedScalars(1, differential)})
    {
    }

    /// The exact scalar trajectory. Each concrete case has its own.
    virtual double muExact(double t) const = 0;

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index, const State &s, Position, Time) override { return s.q(0); }

    Value Sources(Index, const State &s, Position x, Time t) override
    {
        return linearSource(x, t) + (muExact(t) - s.scalar(0)) * std::sin(pi * x);
    }

    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override
    {
        v[0] = 1.0;
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

    // The leading Index is the variable; the out-vector is indexed by scalar.
    // Both are 1 here, so a confusion between them would not show -- write it
    // deliberately anyway.
    void dSources_dScalars(Index, VectorRef v, const State &, Position x, Time) override
    {
        v[0] = -std::sin(pi * x);
    }

    /// Int_0^1 u dx on the framework's node quadrature. Using the weights the
    /// hook is handed rather than a rule of its own is not a convenience: the
    /// derivative of `weights . u` with respect to u_j is exactly weights(j),
    /// which is what ScalarGPrime reports.
    static double mass(GlobalState const &y, Vector const &weights)
    {
        return ScalarHooks::integrate(y.Variable().row(0), weights);
    }

    Value InitialValue(Index, Position x) const override { return exactSolution(x, 0.0); }
    Value InitialDerivative(Index, Position x) const override
    {
        return exactDerivative(x, 0.0);
    }
};

// G = mu - Int u dx, algebraic. Int_0^1 sin(pi x)(1+t) dx = 2(1+t)/pi.
class ManufacturedScalarAlgebraic : public ManufacturedScalarBase
{
public:
    ManufacturedScalarAlgebraic() : ManufacturedScalarBase(false) {}

    static double mu(double t) { return 2.0 * (1.0 + t) / pi; }
    double muExact(double t) const override { return mu(t); }

    Value ScalarG(Index, GlobalState const &y, GlobalState const &,
                  std::vector<Position> const &, Values const &weights, Matrix const &,
                  Time) override
    {
        return y.Scalars()(0) - mass(y, weights);
    }

    void ScalarGPrime(GlobalStateMatrix &dG, GlobalStateMatrix &, GlobalState const &,
                      GlobalState const &, std::vector<Position> const &,
                      Values const &weights, Matrix const &, Time) override
    {
        dG[0].Variable().row(0) = -weights.transpose();
        dG[0].Scalars()(0) = 1.0;
    }

    Value InitialScalarValue(Index) const override { return mu(0.0); }

    static double extraError(SystemSolver &sys, Grid const &, double t)
    {
        return std::abs(sys.yJac.Scalar(0) - mu(t));
    }
};

// G = dmu/dt - Int u dx, differential. Integrating 2(1+t)/pi from 0 with
// mu(0) = 0 gives mu(t) = (2t + t^2)/pi.
//
// The `true` below is load bearing. Left at the default, three things go wrong
// at once and none of them says so: the id vector leaves mu algebraic,
// InitialScalarDerivative is never consulted, and the bordered system's N
// becomes dG/dmu = 0 -- singular, because ScalarGPrime correctly writes nothing
// into dG's scalar block.
class ManufacturedScalarDifferential : public ManufacturedScalarBase
{
    // Its own, not a process-wide one; see Integrator::Cache. mutable because
    // InitialScalarDerivative is const and filling a memo is not a change.
    mutable Integrator::Cache integrator;

public:
    ManufacturedScalarDifferential() : ManufacturedScalarBase(true) {}

    static double mu(double t) { return (2.0 * t + t * t) / pi; }
    double muExact(double t) const override { return mu(t); }

    Value ScalarG(Index, GlobalState const &y, GlobalState const &ydot,
                  std::vector<Position> const &, Values const &weights, Matrix const &,
                  Time) override
    {
        return ydot.Scalars()(0) - mass(y, weights);
    }

    void ScalarGPrime(GlobalStateMatrix &dG, GlobalStateMatrix &dGdot, GlobalState const &,
                      GlobalState const &, std::vector<Position> const &,
                      Values const &weights, Matrix const &, Time) override
    {
        dG[0].Variable().row(0) = -weights.transpose();
        dGdot[0].Scalars()(0) = 1.0;
    }

    Value InitialScalarValue(Index) const override { return mu(0.0); }

    // Unlike the algebraic case this value is binding: a differential scalar has
    // id = 1, so IDACalcIC holds mu(0) fixed and solves for mu'(0) rather than
    // repairing it. Sample the discrete solution rather than returning 2/pi, so
    // the initial state is consistent with the constraint the residual enforces.
    Value InitialScalarDerivative(Index, const DGSoln &y, const DGSoln &) const override
    {
        return mass(y.evalOnNodes(),
                    integrator.integrationWeights(y.getBasis(), y.getGrid()));
    }

    static double extraError(SystemSolver &sys, Grid const &, double t)
    {
        return std::abs(sys.yJac.Scalar(0) - mu(t));
    }
};

// A fixture that builds a consistent-ish state to hand to checkScalarDerivative.
// The values need not solve anything -- the check finite-differences the case's
// own constraint against its own reported derivative -- but they must not be
// zero, or a derivative proportional to the state would look correct.
struct ScalarDerivativeFixture
{
    std::vector<double> yMem, dydtMem;
    Grid grid;
    DGSoln y, dydt;

    ScalarDerivativeFixture(Index k, Index nCells)
        : grid(0.0, 1.0, nCells), y(1, grid, k, 1, 0), dydt(1, grid, k, 1, 0)
    {
        yMem.assign(y.getDoF(), 0.0);
        dydtMem.assign(dydt.getDoF(), 0.0);
        y.Map(yMem.data());
        dydt.Map(dydtMem.data());

        y.zeroCoeffs();
        dydt.zeroCoeffs();

        y.AssignU([](Index, Position x) { return exactSolution(x, 0.3); });
        y.AssignQ([](Index, Position x) { return exactDerivative(x, 0.3); });

        // AssignSigma wants a physics hook, not a function of x, so fill the
        // flux coefficients directly. They only have to be nonzero: a dG/dsigma
        // that is wrongly proportional to sigma reads as correct against a zero
        // flux, and these constraints genuinely have no sigma dependence, which
        // is exactly the claim worth testing.
        for (Index cell = 0; cell < static_cast<Index>(grid.getNCells()); ++cell)
            for (Index l = 0; l < k + 1; ++l)
                y.sigma(0).getCoeff(cell).second(l) = 0.3 + 0.1 * (cell + l);

        y.Scalar(0) = 0.7;
        dydt.Scalar(0) = 0.4;
    }
};

// ---------------------------------------------- two of everything at once --

// Every study above has one aux variable or one scalar, never two and never
// both, so an indexing error that only appears at the *second* entry of either
// -- a phi(0) for a phi(1), a scalar row written into the wrong slot, a block
// sized nVars where it should be nAux -- reaches none of them. This case has two
// variables, two aux variables and two scalars, one algebraic and one
// differential, every one of them coupled to something other than itself:
//
//     u0 = sin(pi x)(1 + t)                  u1 = e^x sin(pi x)(1 + 2t) / 2
//
//     G_phi0 = phi0 - u0 u1                  G_phi1 = phi1 - u1^2 - u0/2
//     sigma_hat0 = (1 + phi1) q0             sigma_hat1 = (2 + phi0) q1
//
//     G_mu0 = mu0 - Int (u0 + x^2 u1)        algebraic
//     G_mu1 = dmu1/dt - Int x u1 - mu0       differential, and reads mu0
//
//     S0 = F0 + (phi1E^2 - phi1^2) + (mu0E - mu0) sin(pi x) + (mu1E - mu1) x / 2
//     S1 = F1 + (phi0E^2 - phi0^2) + (mu1E - mu1) x^2 sin(pi x)
//
// with F_i the forcing of u_i's equation at the exact solution and E marking an
// exact value. Each compensating term vanishes at the exact solution and is
// written against it, never against the discrete state -- see the note at the
// top of this file.
//
// It is deliberately asymmetric, which is the lesson of
// test_adjoint_combinations.py: a transposition is invisible to a problem
// symmetric enough to make the transposed block equal the original. u1 carries
// e^x, so nothing is symmetric about x = 1/2; the scalars weight u by 1, x and
// x^2; each flux reads the *other* variable's aux; and N = dG/dmu is triangular,
// mu1's row reading mu0 and not the reverse.
class ManufacturedCoupledPair : public TransportSystem
{
public:
    ManufacturedCoupledPair()
        : TransportSystem({.variables = {{"u0", "", ""}, {"u1", "", ""}},
                           .scalars = {{"mu0", "algebraic", "", false},
                                       {"mu1", "differential", "", true}},
                           .aux = numberedAux(2)})
    {
    }

    // ---- the exact solution, and what the forcing is derived from --------

    static double u0(double x, double t) { return std::sin(pi * x) * (1.0 + t); }
    static double u0x(double x, double t) { return pi * std::cos(pi * x) * (1.0 + t); }
    static double u0xx(double x, double t) { return -pi * pi * u0(x, t); }
    static double u0t(double x, double) { return std::sin(pi * x); }

    // Halved so the diffusivities, 1 + phi1 and 2 + phi0, stay below about 3.
    static constexpr double A1 = 0.5;
    static double u1(double x, double t)
    {
        return A1 * std::exp(x) * std::sin(pi * x) * (1.0 + 2.0 * t);
    }
    static double u1x(double x, double t)
    {
        return A1 * std::exp(x) * (std::sin(pi * x) + pi * std::cos(pi * x)) * (1.0 + 2.0 * t);
    }
    static double u1xx(double x, double t)
    {
        return A1 * std::exp(x) *
               ((1.0 - pi * pi) * std::sin(pi * x) + 2.0 * pi * std::cos(pi * x)) *
               (1.0 + 2.0 * t);
    }
    static double u1t(double x, double) { return 2.0 * A1 * std::exp(x) * std::sin(pi * x); }

    static double phi0(double x, double t) { return u0(x, t) * u1(x, t); }
    static double phi0x(double x, double t)
    {
        return u0x(x, t) * u1(x, t) + u0(x, t) * u1x(x, t);
    }
    static double phi1(double x, double t) { return u1(x, t) * u1(x, t) + 0.5 * u0(x, t); }
    static double phi1x(double x, double t)
    {
        return 2.0 * u1(x, t) * u1x(x, t) + 0.5 * u0x(x, t);
    }

    // u_t - d_x[ sigma_hat ] at the exact solution.
    static double forcing0(double x, double t)
    {
        return u0t(x, t) - (phi1x(x, t) * u0x(x, t) + (1.0 + phi1(x, t)) * u0xx(x, t));
    }
    static double forcing1(double x, double t)
    {
        return u1t(x, t) - (phi0x(x, t) * u1x(x, t) + (2.0 + phi0(x, t)) * u1xx(x, t));
    }

    // Int_0^1 x^n e^x sin(pi x) dx, n = 1, 2, by quadrature: exact to round-off,
    // and simpler to trust than the closed forms.
    static double moment(int n)
    {
        boost::math::quadrature::gauss<double, 30> gauss;
        return gauss.integrate(
            [n](double x) { return std::pow(x, n) * std::exp(x) * std::sin(pi * x); }, 0.0,
            1.0);
    }

    // mu0 = Int u0 + x^2 u1 = 2(1+t)/pi + A1 (1+2t) M2.
    static double mu0(double t)
    {
        return 2.0 * (1.0 + t) / pi + A1 * (1.0 + 2.0 * t) * moment(2);
    }
    // dmu1/dt = Int x u1 + mu0 = A1 (1+2t) M1 + mu0(t), from mu1(0) = 0.
    static double mu1(double t)
    {
        return A1 * (t + t * t) * moment(1) + 2.0 * (t + 0.5 * t * t) / pi +
               A1 * (t + t * t) * moment(2);
    }

    // ---- the case -------------------------------------------------------

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }

    Value SigmaFn(Index i, const State &s, Position, Time) override
    {
        return i == 0 ? (1.0 + s.phi(1)) * s.q(0) : (2.0 + s.phi(0)) * s.q(1);
    }

    Value Sources(Index i, const State &s, Position x, Time t) override
    {
        if (i == 0)
        {
            const double p1 = phi1(x, t);
            return forcing0(x, t) + (p1 * p1 - s.phi(1) * s.phi(1)) +
                   (mu0(t) - s.scalar(0)) * std::sin(pi * x) +
                   (mu1(t) - s.scalar(1)) * 0.5 * x;
        }
        const double p0 = phi0(x, t);
        return forcing1(x, t) + (p0 * p0 - s.phi(0) * s.phi(0)) +
               (mu1(t) - s.scalar(1)) * x * x * std::sin(pi * x);
    }

    Value AuxG(Index a, const State &s, Position, Time) override
    {
        return a == 0 ? s.phi(0) - s.u(0) * s.u(1)
                      : s.phi(1) - s.u(1) * s.u(1) - 0.5 * s.u(0);
    }

    void AuxGPrime(Index a, State &out, const State &s, Position, Time) override
    {
        if (a == 0)
        {
            out.phi(0) = 1.0;
            out.u(0) = -s.u(1);
            out.u(1) = -s.u(0);
        }
        else
        {
            out.phi(1) = 1.0;
            out.u(0) = -0.5;
            out.u(1) = -2.0 * s.u(1);
        }
    }

    void dSigmaFn_dq(Index i, VectorRef v, const State &s, Position, Time) override
    {
        if (i == 0)
            v[0] = 1.0 + s.phi(1);
        else
            v[1] = 2.0 + s.phi(0);
    }
    void dSigmaFn_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSigma_dPhi(Index i, VectorRef v, const State &s, Position, Time) override
    {
        if (i == 0)
            v[1] = s.q(0);
        else
            v[0] = s.q(1);
    }

    void dSources_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dPhi(Index i, VectorRef v, const State &s, Position, Time) override
    {
        if (i == 0)
            v[1] = -2.0 * s.phi(1);
        else
            v[0] = -2.0 * s.phi(0);
    }
    // Indexed by scalar; the leading Index is the variable.
    void dSources_dScalars(Index i, VectorRef v, const State &, Position x, Time) override
    {
        if (i == 0)
        {
            v[0] = -std::sin(pi * x);
            v[1] = -0.5 * x;
        }
        else
            v[1] = -x * x * std::sin(pi * x);
    }

    // The scalar constraints, on the framework's node quadrature (see
    // ManufacturedScalarBase::mass for why it must be that one).
    static double weighted(GlobalState const &y, Index var, std::vector<Position> const &x,
                           Values const &w, int power)
    {
        double total = 0.0;
        for (size_t j = 0; j < x.size(); ++j)
            total += w(j) * std::pow(x[j], power) * y.Variable()(var, j);
        return total;
    }

    Value ScalarG(Index s, GlobalState const &y, GlobalState const &ydot,
                  std::vector<Position> const &x, Values const &w, Matrix const &,
                  Time) override
    {
        if (s == 0)
            return y.Scalars()(0) - weighted(y, 0, x, w, 0) - weighted(y, 1, x, w, 2);
        return ydot.Scalars()(1) - weighted(y, 1, x, w, 1) - y.Scalars()(0);
    }

    void ScalarGPrime(GlobalStateMatrix &dG, GlobalStateMatrix &dGdot, GlobalState const &,
                      GlobalState const &, std::vector<Position> const &x, Values const &w,
                      Matrix const &, Time) override
    {
        for (size_t j = 0; j < x.size(); ++j)
        {
            dG[0].Variable()(0, j) = -w(j);
            dG[0].Variable()(1, j) = -w(j) * x[j] * x[j];
            dG[1].Variable()(1, j) = -w(j) * x[j];
        }
        dG[0].Scalars()(0) = 1.0;
        dG[1].Scalars()(0) = -1.0;
        dGdot[1].Scalars()(1) = 1.0;
    }

    Value InitialValue(Index i, Position x) const override
    {
        return i == 0 ? u0(x, 0.0) : u1(x, 0.0);
    }
    Value InitialDerivative(Index i, Position x) const override
    {
        return i == 0 ? u0x(x, 0.0) : u1x(x, 0.0);
    }
    Value InitialAuxValue(Index a, Position x) const override
    {
        return a == 0 ? phi0(x, 0.0) : phi1(x, 0.0);
    }
    Value InitialScalarValue(Index s) const override { return s == 0 ? mu0(0.0) : mu1(0.0); }
    // Only a guess: mu1 is differential, so IDACalcIC solves for its derivative.
    Value InitialScalarDerivative(Index, const DGSoln &, const DGSoln &) const override
    {
        return A1 * moment(1) + mu0(0.0);
    }
};

/// Every error the pair case has, at the final time: L2 for the fields, and
/// signed for the scalars, whose errors cross zero on the way down.
struct PairErrors
{
    std::array<double, 8> e;
    static constexpr std::array<const char *, 8> names = {"u0",   "u1",   "u0*", "u1*",
                                                          "phi0", "phi1", "mu0", "mu1"};
};

PairErrors measurePair(SystemSolver &sys, Grid const &grid, double t)
{
    using P = ManufacturedCoupledPair;
    auto field = [&](auto f, auto exact) { return l2ErrorAgainst(f, exact, grid, t); };
    return {{field([&](double x) { return sys.yJac.u(0)(x); }, P::u0),
             field([&](double x) { return sys.yJac.u(1)(x); }, P::u1),
             field([&](double x) { return sys.getPostprocessor()->uStar(0)(x); }, P::u0),
             field([&](double x) { return sys.getPostprocessor()->uStar(1)(x); }, P::u1),
             field([&](double x) { return sys.yJac.Aux(0)(x); }, P::phi0),
             field([&](double x) { return sys.yJac.Aux(1)(x); }, P::phi1),
             (sys.yJac.Scalar(0) - P::mu0(t)),
             (sys.yJac.Scalar(1) - P::mu1(t))}};
}

} // namespace

BOOST_AUTO_TEST_SUITE(mms_aux_scalar_tests)

// ------------------------------------------------- the algebra, before the solver --

BOOST_AUTO_TEST_CASE(the_aux_problem_is_consistent_with_the_exact_solution)
{
    // Three separate claims, each of which would otherwise be discovered as a
    // converged-but-wrong answer: the constraint is satisfied by phi = u^2, the
    // compensating source term vanishes there, and what is left is the same PDE
    // the nonlinear-flux case solves.
    ManufacturedAux problem;
    const double t = 0.37, h = 1e-5;

    auto flux = [&](double x)
    {
        const double u = exactSolution(x, t);
        return (1.0 + u * u) * exactDerivative(x, t);
    };

    for (double x : {0.13, 0.5, 0.81})
    {
        const double ue = exactSolution(x, t);

        State s(1, 0, 1);
        s.u(0) = ue;
        s.q(0) = exactDerivative(x, t);
        s.phi(0) = ue * ue;

        BOOST_TEST(problem.AuxG(0, s, x, t) == 0.0,
                   boost::test_tools::tolerance(1e-14));

        // The source reduces to the nonlinear-flux forcing at the exact solution.
        BOOST_TEST(problem.Sources(0, s, x, t) == nonlinearFluxSource(x, t),
                   boost::test_tools::tolerance(1e-12));

        // ...and the flux really is the one that forcing was derived for.
        BOOST_TEST(problem.SigmaFn(0, s, x, t) == flux(x),
                   boost::test_tools::tolerance(1e-12));

        // ...which together must satisfy u_t - d_x[sigma_hat] = S.
        const double dFluxdx = (flux(x + h) - flux(x - h)) / (2.0 * h);
        const double dudt = (exactSolution(x, t + h) - exactSolution(x, t - h)) / (2.0 * h);
        BOOST_TEST(dudt - dFluxdx == problem.Sources(0, s, x, t),
                   boost::test_tools::tolerance(1e-5));
    }
}

BOOST_AUTO_TEST_CASE(the_aux_case_reports_its_own_derivatives_correctly)
{
    // No solver: central-difference the case's own hooks against the derivatives
    // it advertises. An order study cannot do this -- the Jacobian is never
    // assembled, so a wrong entry costs Newton iterations and nothing else --
    // and this fails with a specific name attached when one is wrong.
    ManufacturedAux problem;
    const double t = 0.37, h = 1e-6;

    for (double x : {0.13, 0.5, 0.81})
    {
        const double u = exactSolution(x, t), q = exactDerivative(x, t);
        const double phi = u * u;

        auto stateAt = [&](double uu, double qq, double pp)
        {
            State s(1, 0, 1);
            s.u(0) = uu;
            s.q(0) = qq;
            s.phi(0) = pp;
            return s;
        };
        const State s = stateAt(u, q, phi);

        auto central = [&](auto &&f, double du, double dq, double dp)
        {
            return (f(stateAt(u + du, q + dq, phi + dp)) -
                    f(stateAt(u - du, q - dq, phi - dp))) /
                   (2.0 * h);
        };
        auto sigma = [&](State ss) { return problem.SigmaFn(0, ss, x, t); };
        auto source = [&](State ss) { return problem.Sources(0, ss, x, t); };
        auto auxg = [&](State ss) { return problem.AuxG(0, ss, x, t); };

        Vector v(1);

        v.setZero();
        problem.dSigmaFn_dq(0, v, s, x, t);
        BOOST_TEST(v[0] == central(sigma, 0.0, h, 0.0),
                   boost::test_tools::tolerance(1e-6));

        v.setZero();
        problem.dSigma_dPhi(0, v, s, x, t);
        BOOST_TEST(v[0] == central(sigma, 0.0, 0.0, h),
                   boost::test_tools::tolerance(1e-6));

        v.setZero();
        problem.dSources_dPhi(0, v, s, x, t);
        BOOST_TEST(v[0] == central(source, 0.0, 0.0, h),
                   boost::test_tools::tolerance(1e-6));

        // AuxGPrime reports every block at once, into a zeroed State.
        State dG(1, 0, 1);
        problem.AuxGPrime(0, dG, s, x, t);
        BOOST_TEST(dG.u(0) == central(auxg, h, 0.0, 0.0),
                   boost::test_tools::tolerance(1e-6));
        BOOST_TEST(dG.phi(0) == central(auxg, 0.0, 0.0, h),
                   boost::test_tools::tolerance(1e-6));
        BOOST_TEST(dG.q(0) == 0.0, boost::test_tools::tolerance(1e-12));
    }
}

BOOST_AUTO_TEST_CASE(the_scalar_trajectories_match_their_closed_forms)
{
    // Int_0^1 sin(pi x)(1+t) dx = 2(1+t)/pi, and its integral from 0 is
    // (2t + t^2)/pi. Both by quadrature, against the closed forms the cases use.
    boost::math::quadrature::gauss<double, 30> gauss;

    for (double t : {0.0, 0.25, 1.0})
    {
        const double integral =
            gauss.integrate([&](double x) { return exactSolution(x, t); }, 0.0, 1.0);
        BOOST_TEST(ManufacturedScalarAlgebraic::mu(t) == integral,
                   boost::test_tools::tolerance(1e-12));
    }

    // The differential one: d/dt mu must be the same integral.
    const double t = 0.37, h = 1e-6;
    const double dmudt = (ManufacturedScalarDifferential::mu(t + h) -
                          ManufacturedScalarDifferential::mu(t - h)) /
                         (2.0 * h);
    BOOST_TEST(dmudt == ManufacturedScalarAlgebraic::mu(t),
               boost::test_tools::tolerance(1e-8));
    BOOST_TEST(ManufacturedScalarDifferential::mu(0.0) == 0.0,
               boost::test_tools::tolerance(1e-15));

    // And the compensating source term really does vanish on the exact scalar,
    // leaving the plain linear forcing.
    for (double x : {0.13, 0.5, 0.81})
    {
        ManufacturedScalarAlgebraic algebraic;
        State s(1, 1, 0);
        s.scalar(0) = ManufacturedScalarAlgebraic::mu(t);
        BOOST_TEST(algebraic.Sources(0, s, x, t) == linearSource(x, t),
                   boost::test_tools::tolerance(1e-12));

        ManufacturedScalarDifferential differential;
        State sd(1, 1, 0);
        sd.scalar(0) = ManufacturedScalarDifferential::mu(t);
        BOOST_TEST(differential.Sources(0, sd, x, t) == linearSource(x, t),
                   boost::test_tools::tolerance(1e-12));
    }
}

BOOST_AUTO_TEST_CASE(the_scalar_cases_report_their_own_derivatives_correctly)
{
    // checkScalarDerivative finite-differences ScalarG against ScalarGPrime for
    // every degree of freedom. Run it before trusting an order study built on
    // these cases: a wrong scalar Jacobian only slows Newton down, so the study
    // would still converge and still look right.
    for (Index k : {1, 2, 3})
    {
        ScalarDerivativeFixture algebraicFixture(k, 4);
        ManufacturedScalarAlgebraic algebraic;
        const double worstAlgebraic =
            checkScalarDerivative(algebraic, algebraicFixture.y, algebraicFixture.dydt,
                                  algebraicFixture.grid, k, 0.3, 1e-8);

        ScalarDerivativeFixture differentialFixture(k, 4);
        ManufacturedScalarDifferential differential;
        const double worstDifferential = checkScalarDerivative(
            differential, differentialFixture.y, differentialFixture.dydt,
            differentialFixture.grid, k, 0.3, 1e-8);

        BOOST_TEST_MESSAGE("k = " << k << ": worst scalar derivative error, algebraic "
                                  << worstAlgebraic << ", differential "
                                  << worstDifferential);
    }
}

// ------------------------------------------------------------ order studies --

BOOST_AUTO_TEST_CASE(the_order_survives_an_auxiliary_variable)
{
    // Measured:
    //
    //     k = 1:  off u 1.85 u* 2.69  |  on u 1.89 u* 3.18  |  phi off 1.91 on 2.00
    //     k = 2:  off u 2.89 u* 4.12  |  on u 2.89 u* 4.59  |  phi off 2.78 on 2.98
    //
    // u* reaches k+2 with the flag on and phi holds k+1, which is the answer the
    // TODO wanted: routing the flux nonlinearity through an algebraic auxiliary
    // variable costs the scheme nothing.
    //
    // phi cannot do better than k+1 and is not expected to. It is a P_k field
    // whatever it is constrained to equal, so even though u* is k+2 accurate,
    // interpolating u*^2 back into P_k caps the result at the space's own
    // approximation order.
    //
    // The flag-off u* column is again a fit through a falling rate -- 3.46e-2,
    // 3.18e-3, 4.85e-4, 1.29e-4, i.e. ratios 10.9, 6.6, 3.8 -- the same transient
    // superconvergence the nonlinear-flux case shows, which is unsurprising since
    // this is that problem with the (1 + u^2) routed through phi.
    const double tFinal = 0.25;

    for (auto const &c : std::vector<std::pair<Index, std::vector<Index>>>{
             {1, {4, 8, 16, 32}}, {2, {4, 8, 16}}})
    {
        const Rates r = measureRates<ManufacturedAux>(c.first, c.second, tFinal);
        BOOST_TEST_MESSAGE("aux, " + report(c.first, r, "phi"));

        BOOST_TEST(r.uOff > c.first + 1 - 0.2,
                   "k = " << c.first << ": u lost its rate with the flag off ("
                          << r.uOff << ")");
        BOOST_TEST(r.uOn > c.first + 1 - 0.2,
                   "k = " << c.first << ": u lost its rate with the flag on ("
                          << r.uOn << ")");
        BOOST_TEST(r.starOn > c.first + 2 - 0.35,
                   "k = " << c.first << ": u* did not reach k+2 = " << c.first + 2
                          << " with the flag on (observed " << r.starOn << ")");
        BOOST_TEST(r.extraOn > c.first + 1 - 0.35,
                   "k = " << c.first << ": phi lost its rate with the flag on ("
                          << r.extraOn << ")");
    }
}

BOOST_AUTO_TEST_CASE(the_order_survives_an_algebraic_scalar)
{
    // Measured:
    //
    //     k = 1:  off u 1.96 u* 2.16  |  on u 1.96 u* 3.08  |  mu off 2.15 on 3.09
    //     k = 2:  off u 2.97 u* 4.07  |  on u 2.97 u* 4.03  |  mu off 4.04 on 4.78
    //
    // The u and u* columns reproduce the linear-diffusion case of
    // MMSConvergenceTests.cpp to two decimal places, which is the headline: the
    // scalar coupling costs the field nothing, flag off or on.
    //
    // The mu column is the surprise, and it is worth being careful about what it
    // does and does not say. mu superconverges -- k+1 with the flag off, k+2 with
    // it on (4.38e-3, 4.93e-4, 5.85e-5, 7.13e-6: ratios 8.9, 8.4, 8.2, i.e. 2^3)
    // -- *even though the scalar constraint never sees u\**. ScalarG is evaluated
    // on u_h at the element nodes in both modes. So this is not the postprocessing
    // leaking into the scalar; it is that mu is a linear functional of u_h, the
    // flag changes what u_h is, and the functional error of the flag-on solution
    // is an order better than its L2 error even though the two solutions have the
    // same L2 rate.
    //
    // Do not read the k = 2 mu figures as k+3. Those errors reach 1e-8, within
    // about an order of the 1e-9 relative integration tolerance, so the fit there
    // is sitting on the temporal noise floor. Only the k = 1 column is clean
    // enough to support a claim, which is why the assertion below is k+1 and the
    // k+2 observation is recorded here rather than enforced.
    const double tFinal = 0.25;

    for (auto const &c : std::vector<std::pair<Index, std::vector<Index>>>{
             {1, {4, 8, 16, 32}}, {2, {4, 8, 16}}})
    {
        const Rates r = measureRates<ManufacturedScalarAlgebraic>(c.first, c.second, tFinal);
        BOOST_TEST_MESSAGE("algebraic scalar, " + report(c.first, r, "mu"));

        BOOST_TEST(r.uOff > c.first + 1 - 0.2,
                   "k = " << c.first << ": u lost its rate with the flag off ("
                          << r.uOff << ")");
        BOOST_TEST(r.uOn > c.first + 1 - 0.2,
                   "k = " << c.first << ": u lost its rate with the flag on ("
                          << r.uOn << ")");
        BOOST_TEST(r.starOn > c.first + 2 - 0.35,
                   "k = " << c.first << ": u* did not reach k+2 = " << c.first + 2
                          << " with the flag on (observed " << r.starOn << ")");
        BOOST_TEST(r.extraOn > c.first + 1 - 0.2,
                   "k = " << c.first << ": mu lost its rate with the flag on ("
                          << r.extraOn << ")");
    }
}

BOOST_AUTO_TEST_CASE(the_order_survives_a_differential_scalar)
{
    // Measured:
    //
    //     k = 1:  off u 1.96 u* 2.17  |  on u 1.96 u* 3.08  |  mu off 2.26 on 2.96
    //     k = 2:  off u 2.97 u* 4.07  |  on u 2.97 u* 4.03  |  mu off 4.16 on 5.38
    //
    // Within noise of the algebraic case, which is the point: making the scalar
    // differential changes N from 1 to alpha in the bordered solve and brings
    // InitialScalarDerivative into play, and neither disturbs the observed order.
    // The k = 2 mu column is again at the temporal noise floor (3.9e-9) and the
    // 5.38 there should not be read as a rate.
    const double tFinal = 0.25;

    for (auto const &c : std::vector<std::pair<Index, std::vector<Index>>>{
             {1, {4, 8, 16, 32}}, {2, {4, 8, 16}}})
    {
        const Rates r =
            measureRates<ManufacturedScalarDifferential>(c.first, c.second, tFinal);
        BOOST_TEST_MESSAGE("differential scalar, " + report(c.first, r, "mu"));

        BOOST_TEST(r.uOff > c.first + 1 - 0.2,
                   "k = " << c.first << ": u lost its rate with the flag off ("
                          << r.uOff << ")");
        BOOST_TEST(r.uOn > c.first + 1 - 0.2,
                   "k = " << c.first << ": u lost its rate with the flag on ("
                          << r.uOn << ")");
        BOOST_TEST(r.starOn > c.first + 2 - 0.35,
                   "k = " << c.first << ": u* did not reach k+2 = " << c.first + 2
                          << " with the flag on (observed " << r.starOn << ")");
        BOOST_TEST(r.extraOn > c.first + 1 - 0.2,
                   "k = " << c.first << ": mu lost its rate with the flag on ("
                          << r.extraOn << ")");
    }
}


BOOST_AUTO_TEST_CASE(the_pair_problem_is_consistent_with_the_exact_solution)
{
    // The manufactured solution has to be the solution before an order study of
    // it means anything; a slip in any of the hand-derived pieces converges, at
    // the right rate, to the wrong function.
    using P = ManufacturedCoupledPair;
    P problem;
    const double t = 0.37, h = 1e-5;

    auto exactState = [&](double x, double tt)
    {
        State s(2, 2, 2);
        s.u(0) = P::u0(x, tt);
        s.u(1) = P::u1(x, tt);
        s.q(0) = P::u0x(x, tt);
        s.q(1) = P::u1x(x, tt);
        s.phi(0) = P::phi0(x, tt);
        s.phi(1) = P::phi1(x, tt);
        s.scalar(0) = P::mu0(tt);
        s.scalar(1) = P::mu1(tt);
        return s;
    };

    for (double x : {0.13, 0.5, 0.81})
    {
        const State s = exactState(x, t);
        for (Index a : {0, 1})
            BOOST_TEST(problem.AuxG(a, s, x, t) == 0.0, boost::test_tools::tolerance(1e-14));

        BOOST_TEST(problem.Sources(0, s, x, t) == P::forcing0(x, t),
                   boost::test_tools::tolerance(1e-12));
        BOOST_TEST(problem.Sources(1, s, x, t) == P::forcing1(x, t),
                   boost::test_tools::tolerance(1e-12));

        // u_t - d_x[sigma_hat] = S, each piece differenced from the exact
        // solution independently of the hand-written derivatives.
        auto exactU = [&](Index i, double xx, double tt)
        { return i == 0 ? P::u0(xx, tt) : P::u1(xx, tt); };
        for (Index i : {0, 1})
        {
            auto flux = [&](double xx)
            {
                State ss = exactState(xx, t);
                ss.q(i) = (exactU(i, xx + h, t) - exactU(i, xx - h, t)) / (2.0 * h);
                return problem.SigmaFn(i, ss, xx, t);
            };
            const double dFluxdx = (flux(x + h) - flux(x - h)) / (2.0 * h);
            const double dudt = (exactU(i, x, t + h) - exactU(i, x, t - h)) / (2.0 * h);
            BOOST_TEST(dudt - dFluxdx == problem.Sources(i, s, x, t),
                       boost::test_tools::tolerance(1e-4));
        }
    }

    // The scalar trajectories, by an independent quadrature.
    boost::math::quadrature::gauss<double, 30> gauss;
    for (double tt : {0.0, 0.25, 1.0})
    {
        const double m0 = gauss.integrate(
            [&](double x) { return P::u0(x, tt) + x * x * P::u1(x, tt); }, 0.0, 1.0);
        BOOST_TEST(P::mu0(tt) == m0, boost::test_tools::tolerance(1e-12));

        const double m1 =
            gauss.integrate([&](double x) { return x * P::u1(x, tt); }, 0.0, 1.0);
        const double dmu1 = (P::mu1(tt + 1e-6) - P::mu1(tt - 1e-6)) / 2e-6;
        BOOST_TEST(dmu1 == m1 + P::mu0(tt), boost::test_tools::tolerance(1e-8));
    }
    BOOST_TEST(P::mu1(0.0) == 0.0, boost::test_tools::tolerance(1e-15));
}

BOOST_AUTO_TEST_CASE(the_pair_case_reports_its_own_derivatives_correctly)
{
    // Every pointwise derivative hook, against a central difference of the hook
    // it differentiates, in every component of the state -- so an entry written
    // into the wrong slot shows up as a wrong slot here, with its name, rather
    // than as slow Newton convergence in the study below.
    using P = ManufacturedCoupledPair;
    P problem;
    const double t = 0.37, h = 1e-6;

    for (double x : {0.13, 0.5, 0.81})
    {
        State base(2, 2, 2);
        base.u(0) = P::u0(x, t) + 0.1;
        base.u(1) = P::u1(x, t) - 0.2;
        base.q(0) = P::u0x(x, t) + 0.3;
        base.q(1) = P::u1x(x, t) - 0.1;
        base.phi(0) = P::phi0(x, t) + 0.2;
        base.phi(1) = P::phi1(x, t) - 0.3;
        base.scalar(0) = P::mu0(t) + 0.1;
        base.scalar(1) = P::mu1(t) - 0.2;

        enum Part { U, Q, Phi, Mu };
        auto component = [](State &s, Part p, Index j) -> double &
        {
            switch (p)
            {
            case U: return s.u(j);
            case Q: return s.q(j);
            case Phi: return s.phi(j);
            default: return s.scalar(j);
            }
        };
        auto central = [&](auto &&f, Part p, Index j)
        {
            State plus = base, minus = base;
            component(plus, p, j) += h;
            component(minus, p, j) -= h;
            return (f(plus) - f(minus)) / (2.0 * h);
        };

        for (Index i : {0, 1})
        {
            auto sigma = [&](State const &s) { return problem.SigmaFn(i, s, x, t); };
            auto source = [&](State const &s) { return problem.Sources(i, s, x, t); };
            auto auxg = [&](State const &s) { return problem.AuxG(i, s, x, t); };

            struct Hook
            {
                const char *name;
                std::function<void(VectorRef)> fill;
                std::function<double(State const &)> f;
                Part part;
            };
            const std::vector<Hook> hooks = {
                {"dSigmaFn_du", [&](VectorRef v) { problem.dSigmaFn_du(i, v, base, x, t); }, sigma, U},
                {"dSigmaFn_dq", [&](VectorRef v) { problem.dSigmaFn_dq(i, v, base, x, t); }, sigma, Q},
                {"dSigma_dPhi", [&](VectorRef v) { problem.dSigma_dPhi(i, v, base, x, t); }, sigma, Phi},
                {"dSources_du", [&](VectorRef v) { problem.dSources_du(i, v, base, x, t); }, source, U},
                {"dSources_dq", [&](VectorRef v) { problem.dSources_dq(i, v, base, x, t); }, source, Q},
                {"dSources_dPhi", [&](VectorRef v) { problem.dSources_dPhi(i, v, base, x, t); }, source, Phi},
                {"dSources_dScalars", [&](VectorRef v) { problem.dSources_dScalars(i, v, base, x, t); }, source, Mu},
            };
            for (auto const &hook : hooks)
            {
                Vector v = Vector::Zero(2);
                hook.fill(v);
                for (Index j : {0, 1})
                    BOOST_TEST(v[j] == central(hook.f, hook.part, j),
                               boost::test_tools::tolerance(1e-6)
                                   << hook.name << " of variable " << i << ", entry " << j
                                   << ", x = " << x);
            }

            State dG(2, 2, 2);
            problem.AuxGPrime(i, dG, base, x, t);
            for (Index j : {0, 1})
            {
                BOOST_TEST(dG.u(j) == central(auxg, U, j), boost::test_tools::tolerance(1e-6)
                                                               << "AuxGPrime " << i << " u" << j);
                BOOST_TEST(dG.q(j) == central(auxg, Q, j), boost::test_tools::tolerance(1e-6)
                                                               << "AuxGPrime " << i << " q" << j);
                BOOST_TEST(dG.phi(j) == central(auxg, Phi, j),
                           boost::test_tools::tolerance(1e-6) << "AuxGPrime " << i << " phi" << j);
            }
        }
    }

    // ...and the scalar rows, through the same check every scalar case uses,
    // which also requires each row to be one IDACalcIC can move.
    for (Index k : {1, 2, 3})
    {
        Grid grid(0.0, 1.0, 4);
        std::vector<double> yMem, dydtMem;
        DGSoln y(2, grid, k, 2, 2), dydt(2, grid, k, 2, 2);
        yMem.assign(y.getDoF(), 0.0);
        dydtMem.assign(dydt.getDoF(), 0.0);
        y.Map(yMem.data());
        dydt.Map(dydtMem.data());
        y.zeroCoeffs();
        dydt.zeroCoeffs();
        y.AssignU([](Index i, Position x)
                  { return i == 0 ? P::u0(x, 0.3) : P::u1(x, 0.3); });
        y.AssignQ([](Index i, Position x)
                  { return i == 0 ? P::u0x(x, 0.3) : P::u1x(x, 0.3); });
        for (Index v : {0, 1})
            for (Index cell = 0; cell < static_cast<Index>(grid.getNCells()); ++cell)
                for (Index l = 0; l < k + 1; ++l)
                {
                    y.sigma(v).getCoeff(cell).second(l) = 0.3 + 0.1 * (cell + l) + v;
                    y.Aux(v).getCoeff(cell).second(l) = 0.2 - 0.05 * (cell + l) + v;
                }
        y.Scalar(0) = 0.7;
        y.Scalar(1) = -0.4;
        dydt.Scalar(0) = 0.4;
        dydt.Scalar(1) = 0.9;

        P problem2;
        const double worst = checkScalarDerivative(problem2, y, dydt, grid, k, 0.3, 1e-8);
        BOOST_TEST_MESSAGE("pair, k = " << k << ": worst scalar derivative error " << worst);
    }
}

BOOST_AUTO_TEST_CASE(the_order_survives_two_aux_variables_and_two_scalars)
{
    // Measured, as the order between the two finest grids:
    //
    //            flag  u0   u1   u0*  u1*  phi0 phi1 mu0  mu1
    //     k = 1  off   1.98 1.99 2.03 1.72 1.98 1.98 1.94 1.90
    //            on    1.99 1.99 2.95 2.98 2.00 2.00 1.86 1.77
    //     k = 2  off   2.97 2.97 3.99 4.03 2.96 2.96 2.47 1.11
    //            on    2.98 2.98 4.41 4.12 3.00 3.00 5.93 5.16
    //
    // Every column the single-coupling studies above measure, at the same
    // rates, for both variables and both aux fields -- so the second entry of
    // each block is indexed as the first is. The scalars are asserted with the
    // flag on only, as above: with it off at k = 2 both cross zero between the
    // two finest grids, and a local order through a crossing means nothing.
    // mu1 at k = 1 is still climbing (1.40, then 1.77) at n = 128, which is
    // why the scalars are held to k+1 - 0.35 rather than to the fields' 0.2.
    // The flag-off u* columns are the transient superconvergence the
    // single-coupling studies record, and are not asserted for that reason.
    const double tFinal = 0.25;

    for (auto const &c : std::vector<std::pair<Index, std::vector<Index>>>{
             {1, {8, 16, 32, 64, 128}}, {2, {4, 8, 16, 32}}})
    {
        const Index k = c.first;
        auto const &cells = c.second;
        for (bool sc : {false, true})
        {
            std::vector<PairErrors> errs;
            for (Index n : cells)
                errs.push_back(solveAndMeasureWith<ManufacturedCoupledPair>(
                    k, n, tFinal, sc, {}, measurePair));

            std::string line = std::format("pair, k = {}, flag {}:", k, sc ? "on " : "off");
            // The order between the two finest grids, not a fit: the scalar
            // errors change sign once each on the way down (mu0 with the flag
            // on at k = 1 goes +2.1e-4, -3.6e-5, -2.1e-5, -6.6e-6, -1.8e-6),
            // and a fit through a zero crossing reads as a lost rate.
            const size_t last = cells.size() - 1;
            std::array<double, 8> order{};
            for (size_t m = 0; m < 8; ++m)
            {
                order[m] = localOrder(cells[last - 1], cells[last],
                                      std::abs(errs[last - 1].e[m]), std::abs(errs[last].e[m]));
                line += std::format("  {} {:.2f}", PairErrors::names[m], order[m]);
            }
            for (size_t i = 0; i < cells.size(); ++i)
            {
                line += std::format("\n      n={:<3}", cells[i]);
                for (size_t m = 0; m < 8; ++m)
                    line += std::format(" {} {:+.2e}", PairErrors::names[m], errs[i].e[m]);
            }
            BOOST_TEST_MESSAGE(line);

            auto expect = [&](size_t m, double rate, double slack)
            {
                BOOST_TEST(order[m] > rate - slack,
                           "k = " << k << ", flag " << (sc ? "on" : "off") << ": "
                                  << PairErrors::names[m] << " converged at " << order[m]
                                  << " against " << rate);
            };
            for (size_t m : {0, 1, 4, 5}) // u0, u1, phi0, phi1
                expect(m, k + 1, 0.2);
            if (sc)
            {
                expect(2, k + 2, 0.35); // u0*
                expect(3, k + 2, 0.35); // u1*
                expect(6, k + 1, 0.35); // mu0
                expect(7, k + 1, 0.35); // mu1
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
