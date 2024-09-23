
#include "ScalarTestLD2a.hpp"
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

<<<<<<< HEAD
=======
	and

	J = 0.198675

    Also implemented as two identical uncoupled copies to check nScalars > 1 is OK

>>>>>>> relax-sources
	*/

// Needed to register the class
REGISTER_PHYSICS_IMPL(ScalarTestLD2a);

const double a1 = 0.5;

ScalarTestLD2a::ScalarTestLD2a(toml::value const &config, Grid const&)
{
	// Always set nVars in a derived constructor
	nVars = 1;
	nScalars = 2;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if (config.count("DiffusionProblem") != 1)
		throw std::invalid_argument("There should be a [DiffusionProblem] section if you are using the ScalarTestLD2a physics model.");

	auto const &DiffConfig = config.at("DiffusionProblem");

	kappa = toml::find_or(DiffConfig, "Kappa", 1.0);
	alpha = toml::find_or(DiffConfig, "alpha", 0.2);
	beta = toml::find_or(DiffConfig, "beta", 10.0);
	u0 = toml::find_or(DiffConfig, "u0", 0.1);

}

// Dirichlet Boundary Conditon
Value ScalarTestLD2a::LowerBoundary(Index, Time) const
{
	return 0.0;
}

Value ScalarTestLD2a::UpperBoundary(Index, Time) const
{
	return u0;
}

bool ScalarTestLD2a::isLowerBoundaryDirichlet(Index) const { return false; };
bool ScalarTestLD2a::isUpperBoundaryDirichlet(Index) const { return true; };

Value ScalarTestLD2a::SigmaFn(Index, const State &s, Position x, Time)
{
	return kappa * s.Derivative[0];
}

Value ScalarTestLD2a::ScaledSource( Position x ) const
{
	if ( x > alpha )
		return 0;
	else 
		return beta*( 1.0 - x/alpha );
}

Value ScalarTestLD2a::Sources(Index i, const State &s, Position x, Time)
{
<<<<<<< HEAD
	double J = s.Scalars[ 0 ];

	return J * ScaledSource( x );
=======
	double J1 = s.Scalars[ 0 ];
    double J2 = s.Scalars[ 1 ];

	return ( J1 + J2 ) * ScaledSource( x );
>>>>>>> relax-sources
}

void ScalarTestLD2a::dSigmaFn_dq(Index, Values &v, const State &, Position, Time)
{
	v[0] = kappa;
};

void ScalarTestLD2a::dSigmaFn_du(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void ScalarTestLD2a::dSources_du(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void ScalarTestLD2a::dSources_dq(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void ScalarTestLD2a::dSources_dsigma(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value ScalarTestLD2a::InitialValue(Index, Position x) const
{
	return u0;
}

Value ScalarTestLD2a::InitialDerivative(Index, Position x) const
{
	return 0.0;
}

Value ScalarTestLD2a::ScalarG( Index i, const DGSoln & y, Time )
{
<<<<<<< HEAD
    // J^2 + E^2 - 7 = 0
    // E - sqrt(6) = 0

    double J = y.Scalar(0);
    double E = y.Scalar(1);
    switch( i ) {
        case 0:
            return J*J + E*E - 7.0;
            break;
        case 1:
            return E - std::sqrt(6);
=======
	// J = Int_0^1 u => G[u;J] = J - Int_[0,1] u

    switch( i ) {
        case 0:
            return y.Scalar( 0 ) - a1 * boost::math::quadrature::gauss_kronrod<double, 31>::integrate( [ & ]( double x ){ return y.u( 0 )( x ); }, 0, 1 );
            break;
        case 1:
            return y.Scalar( 1 ) - (1.0 - a1) * boost::math::quadrature::gauss_kronrod<double, 31>::integrate( [ & ]( double x ){ return y.u( 0 )( x ); }, 0, 1 );
>>>>>>> relax-sources
            break;
    }
    return 0;
}

void ScalarTestLD2a::ScalarGPrime( Index i, State &s, const DGSoln &y, std::function<double( double )> P, Interval I, Time )
{
    s.zero();
<<<<<<< HEAD
    double J = y.Scalar(0);
    double E = y.Scalar(1);
    if ( i == 0 ) {
        s.Scalars[ 0 ] = 2.0 * J;
        s.Scalars[ 1 ] = 2.0 * E;
    } else if ( i == 1 ) {
=======
    if ( i == 0 ) {
        s.Variable[ 0 ] = -a1 * boost::math::quadrature::gauss_kronrod<double, 31>::integrate( P, I.x_l, I.x_u );
        s.Scalars[ 0 ] = 1.0;
    } else if ( i == 1 ) {
        s.Variable[ 0 ] = ( a1 - 1.0 ) * boost::math::quadrature::gauss_kronrod<double, 31>::integrate( P, I.x_l, I.x_u );
>>>>>>> relax-sources
        s.Scalars[ 1 ] = 1.0;
    }
}

void ScalarTestLD2a::dSources_dScalars( Index, Values &v, const State &, Position x, Time )
{
<<<<<<< HEAD
    v.setZero();
	v[ 0 ] = ScaledSource( x );
=======
	v[ 0 ] = a1*ScaledSource( x );
	v[ 1 ] = (1 - a1)*ScaledSource( x );
>>>>>>> relax-sources
}

Value ScalarTestLD2a::InitialScalarValue( Index s ) const
{
    switch(s) {
        case 0:
<<<<<<< HEAD
            return 1.0;
            break;
        case 1:
            return std::sqrt(6);
            break;
    }
    throw std::logic_error("ARGH!");
=======
            return a1 * u0;
            break;
        case 1:
            return (1.0 - a1) * u0;
            break;
    }
>>>>>>> relax-sources
    return 0.0;
}

