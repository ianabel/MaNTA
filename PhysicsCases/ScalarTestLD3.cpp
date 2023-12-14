
#include "ScalarTestLD3.hpp"
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <cmath>
#include <numbers>

/*
	Linear Diffusion test case with a coupled scalar.

	du         d^2 u
	-- - Kappa ----- = J S( x )
	dt          dx^2

	where J is chosen to enforce constant total mass of u i.e.

	d   /1
   --  |   u  = 0
	dt  /-1

	and 

	S( x ) = A exp( -( x/ alpha )^2 ) ; with A^-1 = alpha * sqrt( pi ) * Erf[ 1/alpha ] so S has unit mass

	The explicit equation for J is

	J = [ - Kappa du/dx ]_( x = 1 ) - [ - Kappa du/dx ]_( x = -1 )

 */

// Needed to register the class
REGISTER_PHYSICS_IMPL(ScalarTestLD3);

ScalarTestLD3::ScalarTestLD3(toml::value const &config)
{
	// Always set nVars in a derived constructor
	nVars = 1;
	nScalars = 1;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if (config.count("DiffusionProblem") != 1)
		throw std::invalid_argument("There should be a [DiffusionProblem] section if you are using the ScalarTestLD3 physics model.");

	auto const &DiffConfig = config.at("DiffusionProblem");

	kappa = toml::find_or(DiffConfig, "Kappa", 1.0);
	alpha = toml::find_or(DiffConfig, "alpha", 0.2);
	beta = toml::find_or(DiffConfig, "beta", 0.2);
	u0 = toml::find_or(DiffConfig, "u0", 0.1);

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

Value ScalarTestLD3::SigmaFn(Index, const State &s, Position x, Time)
{
	return kappa * s.Derivative[0];
}

Value ScalarTestLD3::ScaledSource( Position x ) const
{
	double Ainv = alpha * std::sqrt( std::numbers::pi ) * std::erf( 1.0/alpha );
	return exp( -( x/alpha )*( x/alpha ) )/Ainv;
}

Value ScalarTestLD3::Sources(Index, const State &s, Position x, Time)
{
	double J = s.Scalars[ 0 ];

	return J * ScaledSource( x );
}

void ScalarTestLD3::dSigmaFn_dq(Index, Values &v, const State &, Position, Time)
{
	v[0] = kappa;
};

void ScalarTestLD3::dSigmaFn_du(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void ScalarTestLD3::dSources_du(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void ScalarTestLD3::dSources_dq(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void ScalarTestLD3::dSources_dsigma(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value ScalarTestLD3::InitialValue(Index, Position x) const
{
	return u0 + beta*std::cos( std::numbers::pi * x / 2.0 );
}

Value ScalarTestLD3::InitialDerivative(Index, Position x) const
{
	return -( beta * std::numbers::pi / 2.0 )*std::sin( std::numbers::pi * x / 2.0 );
}

Value ScalarTestLD3::ScalarG( Index, const DGSoln & y, Time )
{
	// J = sigma(x = +1) - sigma(x = -1)

	return y.Scalar( 0 ) - ( y.sigma( 0 )( 1 ) - y.sigma( 0 )( -1 ) );
}

void ScalarTestLD3::ScalarGPrime( Index, State &s, const DGSoln &y, std::function<double( double )> P, Interval I, Time )
{
	s.Flux[ 0 ] = 0.0;
	if ( abs( I.x_u - 1 ) < 1e-8 )
		s.Flux[ 0 ] -= P( I.x_u );
	if ( abs( I.x_l + 1 ) < 1e-8 )
		s.Flux[ 0 ] += P( I.x_l );
	s.Derivative[ 0 ] = 0.0;
	s.Variable[ 0 ] = 0.0;
	s.Scalars[ 0 ] = 1.0;
}

void ScalarTestLD3::dSources_dScalars( Index, Values &v, const State &, Position x, Time )
{
	v[ 0 ] = ScaledSource( x );
}

Value ScalarTestLD3::InitialScalarValue( Index s ) const
{
	// Our job to make sure this is consistent!
	return -kappa * ( InitialDerivative( 0, 1 ) - InitialDerivative( 0, -1 ) );
}

void ScalarTestLD3::initialiseDiagnostics( NetCDFIO &nc )
{
	nc.AddTimeSeries( "Mass", "Integral of the solution over the domain", "", 2*u0 + 4*beta/std::numbers::pi );
}

void ScalarTestLD3::writeDiagnostics( DGSoln const& y, double, NetCDFIO &nc, size_t tIndex )
{
	double mass = boost::math::quadrature::gauss_kronrod<double, 31>::integrate( [ & ]( double x ){ return y.u( 0 )( x );}, -1, 1 );
	nc.AppendToTimeSeries( "Mass", mass, tIndex );
}


