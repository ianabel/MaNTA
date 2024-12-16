
#include "ScalarTestLD3.hpp"
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

	to handle the integral term, we add on

	dI/dt = E

	and treat I as a third scalar

 */

// Needed to register the class
REGISTER_PHYSICS_IMPL(ScalarTestLD3);

ScalarTestLD3::ScalarTestLD3(toml::value const &config, Grid const &)
{
	// Always set nVars in a derived constructor
	nVars = 2;
	nScalars = 2;

	// Construst your problem from user-specified config
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
	u0 = toml::find_or(DiffConfig, "u0", 0.1);

	M0 = 2 * u0 + 4 * beta / std::numbers::pi;
	std::cerr << "M0 : " << M0 << std::endl;
}

// Dirichlet Boundary Conditon
Value ScalarTestLD3::LowerBoundary(Index, Time) const
{
	return u0;
}

Value ScalarTestLD3::UpperBoundary(Index, Time) const
{
	return u0;
}

bool ScalarTestLD3::isLowerBoundaryDirichlet(Index) const { return true; };
bool ScalarTestLD3::isUpperBoundaryDirichlet(Index) const { return true; };

Value ScalarTestLD3::SigmaFn(Index i, const State &s, Position x, Time)
{
	return kappa * s.Derivative[i];
}

Value ScalarTestLD3::ScaledSource(Position x) const
{
	double Ainv = alpha * std::sqrt(std::numbers::pi) * std::erf(1.0 / alpha);
	return exp(-(x / alpha) * (x / alpha)) / Ainv;
}

Value ScalarTestLD3::Sources(Index i, const State &s, Position x, Time)
{
	double J = 0;
	if (i == 0)
	{
		double J = s.Scalars[1];
		return J * ScaledSource(x) + 0.5 * std::cos(std::numbers::pi * x);
	}
	else if (i == 1)
	{
		return ScaledSource(x);
	}

	throw std::logic_error("Index out of range");
}

void ScalarTestLD3::dSigmaFn_dq(Index i, Values &v, const State &, Position, Time)
{
	v[i] = kappa;
};

void ScalarTestLD3::dSigmaFn_du(Index i, Values &v, const State &, Position, Time)
{
	v[i] = 0.0;
};

void ScalarTestLD3::dSources_du(Index i, Values &v, const State &, Position, Time)
{
	v[i] = 0.0;
};

void ScalarTestLD3::dSources_dq(Index i, Values &v, const State &, Position, Time)
{
	v[i] = 0.0;
};

void ScalarTestLD3::dSources_dsigma(Index i, Values &v, const State &, Position, Time)
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

bool ScalarTestLD3::isScalarDifferential(Index s)
{
	if (s == 0)
		return true; // E is differential, as we depend on dE/dt expliticly
	else
		return false; // J is not differential
}

Value ScalarTestLD3::ScalarGExtended(Index s, const DGSoln &y, const DGSoln &dydt, Time)
{
	double dEdt = dydt.Scalar(0);
	double E = y.Scalar(0);
	double J = y.Scalar(1);
	if (s == 0)
	{
		// E = (M0 - M)
		// => G_0 = E - (M-M0)
		double M = boost::math::quadrature::gauss_kronrod<double, 31>::integrate([&](double x)
																				 { return y.u(0)(x); }, -1, 1);
		return E - (M0 - M);
	}
	else if (s == 1)
	{
		// J = gamma * E + gamma_d * dE/dt + [ sigma(x = +1) - sigma(x = -1) ]
		// => G_1 = J - gamma * E - gamma_d * dE/dt - [ sigma(x = +1) - sigma(x = -1) ]
		return J - gamma * E - gamma_d * dEdt - (y.sigma(0)(1) - y.sigma(0)(-1));
	}
	else
	{
		throw std::logic_error("scalar index > nScalars");
	}
}

void ScalarTestLD3::ScalarGPrimeExtended(Index scalarIndex, State &s, State &out_dt, const DGSoln &y, const DGSoln &dydt, std::function<double(double)> P, Interval I, Time)
{
	s.zero();
	out_dt.zero();
	if (scalarIndex == 0)
	{
		s.Flux[0] = 0.0;	   // d G_0 / d sigma
		s.Derivative[0] = 0.0; // d G_0 / d (u')
		// dG_0 / du = - dM/du (as functional derivative, taken as an inner product with P)
		double P_mass = boost::math::quadrature::gauss_kronrod<double, 31>::integrate(P, I.x_l, I.x_u);
		s.Variable[0] = -P_mass;
		s.Scalars[0] = 1.0; // dG_0/dE
		s.Scalars[1] = 0.0; // dG_0/dJ
	}
	else if (scalarIndex == 1)
	{
		// dG_1 / d sigma = -[ delta(x-1) - delta(x + 1) ] ;
		// return as functional derivative acting on P
		s.Flux[0] = 0.0;
		if (abs(I.x_u - 1) < 1e-9)
			s.Flux[0] -= P(I.x_u);
		if (abs(I.x_l + 1) < 1e-9)
			s.Flux[0] += P(I.x_l);
		// dG_1/dE
		s.Scalars[0] = -gamma;
		// dG_1/dJ
		s.Scalars[1] = 1.0;
		out_dt.Scalars[0] = -gamma_d;
	}
	else
	{
		throw std::logic_error("scalar index > nScalars");
	}
}

void ScalarTestLD3::dSources_dScalars(Index i, Values &v, const State &, Position x, Time)
{
	if (i == 0)
	{
		v[0] = 0.0;
		v[1] = ScaledSource(x);
	}
	else
		v.setZero();
}

Value ScalarTestLD3::InitialScalarValue(Index s) const
{
	// Our job to make sure this is consistent!
	if (s == 0) // E
		return 0;
	else if (s == 1) // J
		return -kappa * (InitialDerivative(0, 1) - InitialDerivative(0, -1));
	else
		throw std::logic_error("scalar index > nScalars");
}

Value ScalarTestLD3::InitialScalarDerivative(Index s, const DGSoln &y, const DGSoln &dydt) const
{
	// Our job to make sure this is consistent!
	if (s == 0) // dE/dt at t=0
	{
		double Mdot = boost::math::quadrature::gauss_kronrod<double, 31>::integrate([&](double x)
																					{ return dydt.u(0)(x); }, -1, 1);
		return Mdot;
	}
	else
		throw std::logic_error("Initial derivative called for algebraic (non-differential) scalar");
}

void ScalarTestLD3::initialiseDiagnostics(NetCDFIO &nc)
{
	nc.AddTimeSeries("Mass", "Integral of the solution over the domain", "", M0);
}

void ScalarTestLD3::writeDiagnostics(DGSoln const &y, double, NetCDFIO &nc, size_t tIndex)
{
	double mass = boost::math::quadrature::gauss_kronrod<double, 31>::integrate([&](double x)
																				{ return y.u(0)(x); }, -1, 1);
	nc.AppendToTimeSeries("Mass", mass, tIndex);
}
