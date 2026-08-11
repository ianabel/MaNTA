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

#include <cmath>
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
                    Integrator::getIntegrationWeights(y.getBasis(), y.getGrid()));
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

BOOST_AUTO_TEST_SUITE_END()
