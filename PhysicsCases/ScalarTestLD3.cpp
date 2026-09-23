
#include "ScalarTestLD3.hpp"
#include "Logging.hpp"
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <cmath>
#include <numbers>

/*
	Linear Diffusion test case with a coupled scalar.

	du         d^2 u
	-- - Kappa ----- = J S( x )
	dt          dx^2

	where J is chosen to enforce constant total mass M of u i.e.

	d   /1      dM
   --  |   u = --  = 0
	dt  /-1     dt

	and

	S( x ) = A exp( -( x/ alpha )^2 ) ; with A^-1 = alpha * sqrt( pi ) * Erf[ 1/alpha ] so S has unit mass

	The explicit equation for J is

	J_exact = [ - Kappa du/dx ]_( x = 1 ) - [ - Kappa du/dx ]_( x = -1 )

	but we use a PID controller on top of that to keep M constant:

	E = M(t=0) - M
	J = gamma * E + gamma_d * dE/dt + gamma_I * Int_0^t ( E(t') dt' ) + J_exact

    to handle the integral term, we write

	J = gamma * E + gamma_d * dE/dt + gamma_I * I + J_exact
    dI/dt = E

	and treat I as a third scalar

 */

// Needed to register the class
REGISTER_PHYSICS_IMPL(ScalarTestLD3);

ScalarTestLD3::ScalarTestLD3(toml::value const &config, Grid const &)
	// E and I are differential -- G_0 and G_2 depend explicitly on dE/dt and
	// dI/dt -- while J is algebraic. This was isScalarDifferential(s).
	: TransportSystem({.variables = {{"u", "the diffused quantity", ""}},
					   .scalars = {{"E", "mass error, M0 - M", "", true},
								   {"J", "source strength from the PID controller", "", false},
								   {"I", "integral of the error", "", true}}})
{
	// Construct your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if (config.count("DiffusionProblem") != 1)
		throw std::invalid_argument("There should be a [DiffusionProblem] section if you are using the ScalarTestLD3 physics model.");

	auto const &DiffConfig = config.at("DiffusionProblem");

	kappa = toml::find_or(DiffConfig, "Kappa", 1.0);
	alpha = toml::find_or(DiffConfig, "alpha", 0.2);
	beta = toml::find_or(DiffConfig, "beta", 1.0);
	gamma = toml::find_or(DiffConfig, "gamma", 1.0);
	gamma_d = toml::find_or(DiffConfig, "gamma_d", 0.0);
	gamma_I = toml::find_or(DiffConfig, "gamma_I", 0.0);
	u0 = toml::find_or(DiffConfig, "u0", 0.1);

	M0 = 2 * u0 + 4 * beta / std::numbers::pi;
	// Was an unconditional std::cerr, which printed once per construction --
	// eight times in a unit-test run, and once per regression case. INFO is
	// compiled out of a release build and still available with VERBOSE.
	logmsg<LOG_LEVEL::INFO>("ScalarTestLD3 target mass M0 = {}", M0);
}

// Dirichlet Boundary Condition
Value ScalarTestLD3::LowerBoundary(Index, Time) const
{
	return u0;
}

Value ScalarTestLD3::UpperBoundary(Index, Time) const
{
	return u0;
}

Value ScalarTestLD3::SigmaFn(Index i, const State &s, Position x, Time)
{
	return kappa * s.q(i);
}

Value ScalarTestLD3::ScaledSource(Position x) const
{
	double Ainv = alpha * std::sqrt(std::numbers::pi) * std::erf(1.0 / alpha);
	return exp(-(x / alpha) * (x / alpha)) / Ainv;
}

Value ScalarTestLD3::Sources(Index i, const State &s, Position x, Time)
{
	if (i == 0)
	{
		double J = s.scalar(1);
		return J * ScaledSource(x) + 0.5 * std::cos(std::numbers::pi * x);
	}
	else if (i == 1)
	{
		return ScaledSource(x);
	}

	throw std::logic_error("Index out of range");
}

void ScalarTestLD3::dSigmaFn_dq(Index i, VectorRef v, const State &, Position, Time)
{
	v[i] = kappa;
};

void ScalarTestLD3::dSigmaFn_du(Index i, VectorRef v, const State &, Position, Time)
{
	v[i] = 0.0;
};

void ScalarTestLD3::dSources_du(Index i, VectorRef v, const State &, Position, Time)
{
	v[i] = 0.0;
};

void ScalarTestLD3::dSources_dq(Index i, VectorRef v, const State &, Position, Time)
{
	v[i] = 0.0;
};

void ScalarTestLD3::dSources_dsigma(Index i, VectorRef v, const State &, Position, Time)
{
	v[i] = 0.0;
};

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value ScalarTestLD3::InitialValue(Index, Position x) const
{
	return u0 + beta * std::cos(std::numbers::pi * x / 2.0);
}

Value ScalarTestLD3::InitialDerivative(Index, Position x) const
{
	return -(beta * std::numbers::pi / 2.0) * std::sin(std::numbers::pi * x / 2.0);
}

Value ScalarTestLD3::ScalarG(Index s, GlobalState const &y, GlobalState const &ydot,
                             std::vector<Position> const &, Values const &weights,
                             Matrix const &phiBoundary, Time)
{
    using namespace ScalarHooks;

    const double dEdt = ydot.Scalars()(0);
    const double dIdt = ydot.Scalars()(2);
    const double E = y.Scalars()(0);
    const double J = y.Scalars()(1);
    const double I = y.Scalars()(2);

    if (s == 0)
    {
        // E = (M0 - M) => G_0 = E - (M - M0).
        //
        // The mass is the framework's quadrature of the nodal values. It used
        // to be a global adaptive Kronrod rule applied to y.u(0) cell by cell,
        // which is not a smooth function of the coefficients -- the
        // finite-difference reference in ScalarJacobianTests disagreed with the
        // exact Int phi by 8% at k = 4 on 16 cells.
        const double M = integrate(y.Variable().row(0), weights);
        return E - (M0 - M);
    }
    else if (s == 1)
    {
        // J = gamma E + gamma_d dE/dt + gamma_I I + [ sigma(+1) - sigma(-1) ]
        const double sigmaUpper = boundaryValue(y.Flux().row(0), phiBoundary, Upper);
        const double sigmaLower = boundaryValue(y.Flux().row(0), phiBoundary, Lower);
        return J - gamma * E - gamma_d * dEdt - gamma_I * I - (sigmaUpper - sigmaLower);
    }
    else if (s == 2)
    {
        // dI/dt = E <=> I = Int_0^t E
        return dIdt - E;
    }
    else
    {
        throw std::logic_error("scalar index > nScalars");
    }
}

void ScalarTestLD3::ScalarGPrime(GlobalStateMatrix &dG, GlobalStateMatrix &dGdot,
                                 GlobalState const &, GlobalState const &,
                                 std::vector<Position> const &, Values const &weights,
                                 Matrix const &phiBoundary, Time)
{
    using namespace ScalarHooks;

    // All three scalars at once. dG[s] and dGdot[s] arrive zeroed.

    // G_0 = E - M0 + M, so dG_0/du_j = +dM/du_j = +weights(j). This was
    // negated, and the sign is not confined to the scalar rows: w enters
    // solveJacEq's bordered elimination, so getting it wrong corrupts the whole
    // solve. A wrong Jacobian only slows Newton, so the run still converged to
    // the right answer and no regression case could see it.
    dG[0].Variable().row(0) = weights.transpose();
    dG[0].Scalars()(0) = 1.0; // dG_0/dE

    // G_1: dG_1/dsigma = -[ delta(x - 1) - delta(x + 1) ], which as a
    // derivative with respect to the flux degrees of freedom is the boundary
    // basis values at either end.
    addBoundaryDerivative(dG[1].Flux().row(0), phiBoundary, Upper, -1.0);
    addBoundaryDerivative(dG[1].Flux().row(0), phiBoundary, Lower, +1.0);
    dG[1].Scalars()(0) = -gamma;
    dG[1].Scalars()(1) = 1.0;
    // dG_1/dI. G_1 carries a -gamma_I * I term that was never differentiated;
    // latent only because gamma_I defaults to 0.
    dG[1].Scalars()(2) = -gamma_I;
    dGdot[1].Scalars()(0) = -gamma_d;

    // G_2 = dI/dt - E
    dG[2].Scalars()(0) = -1.0;
    dGdot[2].Scalars()(2) = 1.0;
}

void ScalarTestLD3::dSources_dScalars(Index i, VectorRef v, const State &, Position x, Time)
{
	// setZero() first: the i == 0 branch used to assign v[0] and v[1] and leave
	// v[2] (dS/dI) alone. dSources_dScalars_Mat hands this an uninitialised
	// Eigen vector, so that entry was whatever was in the buffer -- undefined
	// behaviour, and it put a garbage column into the scalar coupling matrix v,
	// which corrupts the *field* rows of the bordered solve.
	v.setZero();
	if (i == 0)
		v[1] = ScaledSource(x); // S_0 depends on the scalars only through J
}

double ScalarTestLD3::Mass( DGSoln const &y, Index var )
{
	double total = 0.0;
	Grid const &grid = y.getGrid();
	for ( Index i = 0; i < static_cast<Index>( grid.getNCells() ); ++i )
		total += boost::math::quadrature::gauss_kronrod<double, 31>::integrate(
			[ & ]( double x ){ return y.u( var )( x ); }, grid[ i ].x_l, grid[ i ].x_u );
	return total;
}

Value ScalarTestLD3::InitialScalarValue(Index s) const
{
	// Our job to make sure this is consistent!
	if (s == 0) // E
		return 0;
	else if (s == 1) // J
		return -kappa * ( InitialDerivative( 0, 1 ) - InitialDerivative( 0, -1 ) );
    else if (s == 2) // I
        return 0.0;
	else
		throw std::logic_error("scalar index > nScalars");
}

Value ScalarTestLD3::InitialScalarDerivative(Index s, const DGSoln &y, const DGSoln &dydt) const
{
	// Our job to make sure this is consistent!
	if( s == 0 ) // dE/dt at t=0
    {
        double Mdot = Mass( dydt, 0 );
        return Mdot;
    } else if ( s == 2 ) {
        double E = y.Scalar(0);
        return E; // dI/dt = E
    } else
		throw std::logic_error("Initial derivative called for algebraic (non-differential) scalar");
}

void ScalarTestLD3::initialiseDiagnostics(NetCDFIO &nc)
{
	nc.AddTimeSeries("Mass", "Integral of the solution over the domain", "", M0);
}

void ScalarTestLD3::writeDiagnostics(DGSoln const &y, double, NetCDFIO &nc, size_t tIndex)
{
	double mass = Mass( y, 0 );
	nc.AppendToTimeSeries("Mass", mass, tIndex);
}
