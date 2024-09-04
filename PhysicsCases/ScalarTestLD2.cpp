
#include "ScalarTestLD2.hpp"
#include <boost/math/quadrature/gauss_kronrod.hpp>

/*
	Linear Diffusion test case with a coupled scalar.

	du         d^2 u
	-- - Kappa ----- = J S( x )
	dt          dx^2

	where

       /1
	J = |  u( x, t ) dx
       /0

	and

	S( x ) = beta*( 1 - x/alpha ) for  0 <= x <= alpha and 0 otherwise.

 */

 /*
	Steady state solution as computed by mathematica

	u(x) = u(1) + (alpha*beta/2)*J*(1-x) + J H[alpha-x] * beta (x - alpha)^3 / (6 alpha)

=	with H[x] the Heaviside function.

	J = (24 u(1) )/( 24 - 6 alpha beta + alpha^3 beta )
	
	and so for u(1) = .1, beta = 10, alpha = 0.2:

	u(x) = 0.1 + J*(1-x) + (x>0.2) ? 0 : (25 J/3) * (x - 0.2)^3

	and

	J = 0.198675

	*/

// Needed to register the class
REGISTER_PHYSICS_IMPL(ScalarTestLD2);

ScalarTestLD2::ScalarTestLD2(toml::value const &config, Grid const&)
{
	// Always set nVars in a derived constructor
	nVars = 1;
	nScalars = 1;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if (config.count("DiffusionProblem") != 1)
		throw std::invalid_argument("There should be a [DiffusionProblem] section if you are using the ScalarTestLD2 physics model.");

	auto const &DiffConfig = config.at("DiffusionProblem");

	kappa = toml::find_or(DiffConfig, "Kappa", 1.0);
	alpha = toml::find_or(DiffConfig, "alpha", 0.2);
	beta = toml::find_or(DiffConfig, "beta", 10.0);
	u0 = toml::find_or(DiffConfig, "u0", 0.1);

}

// Dirichlet Boundary Conditon
Value ScalarTestLD2::LowerBoundary(Index, Time) const
{
	return 0.0;
}

Value ScalarTestLD2::UpperBoundary(Index, Time) const
{
	return u0;
}

bool ScalarTestLD2::isLowerBoundaryDirichlet(Index) const { return false; };
bool ScalarTestLD2::isUpperBoundaryDirichlet(Index) const { return true; };

Value ScalarTestLD2::SigmaFn(Index, const State &s, Position x, Time)
{
	return kappa * s.Derivative[0];
}

Value ScalarTestLD2::ScaledSource( Position x ) const
{
	if ( x > alpha )
		return 0;
	else 
		return beta*( 1.0 - x/alpha );
}

Value ScalarTestLD2::Sources(Index i, const State &s, Position x, Time)
{
	double J = s.Scalars[ 0 ];

	return ( J ) * ScaledSource( x );
}

void ScalarTestLD2::dSigmaFn_dq(Index, Values &v, const State &, Position, Time)
{
	v[0] = kappa;
};

void ScalarTestLD2::dSigmaFn_du(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void ScalarTestLD2::dSources_du(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void ScalarTestLD2::dSources_dq(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void ScalarTestLD2::dSources_dsigma(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value ScalarTestLD2::InitialValue(Index, Position x) const
{
	return u0;
}

Value ScalarTestLD2::InitialDerivative(Index, Position x) const
{
	return 0.0;
}

Value ScalarTestLD2::ScalarG( Index, const DGSoln & y, Time )
{
	// J = Int_0^1 u => G[u;J] = J - Int_[0,1] u

  return y.Scalar( 0 ) - boost::math::quadrature::gauss_kronrod<double, 31>::integrate( [ & ]( double x ){ return y.u( 0 )( x ); }, 0, 1 );
}

void ScalarTestLD2::ScalarGPrime( Index i, State &s, const DGSoln &y, std::function<double( double )> P, Interval I, Time )
{
    s.Variable[ 0 ] = -boost::math::quadrature::gauss_kronrod<double, 31>::integrate( P, I.x_l, I.x_u );
	s.Derivative[ 0 ] = 0.0;
	s.Flux[ 0 ] = 0.0;
	s.Scalars[ 0 ] = 1.0;
}

void ScalarTestLD2::dSources_dScalars( Index, Values &v, const State &, Position x, Time )
{
	v[ 0 ] = ScaledSource( x );
}

Value ScalarTestLD2::InitialScalarValue( Index s ) const
{
	// Our job to make sure this is consistent!
	return u0; // Int_0^1 u(t=0) = u0
}

