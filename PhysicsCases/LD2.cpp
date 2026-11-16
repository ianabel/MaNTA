
#include "LD2.hpp"
#include <iostream>

/*
	Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL( LD2 );

// RegridPolicy::InPlace: the constructor reads nothing from its grid and the
// case keeps nothing from an evaluation plan, so a new plan changes nothing it
// built.
LD2::LD2( toml::value const& config, Grid const& )
	: TransportSystem({.variables = {{"u", "the diffused quantity", ""}},
					   .regrid = RegridPolicy::InPlace})
{
	// Construct your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if ( config.count( "DiffusionProblem" ) != 1 )
		throw std::invalid_argument( "There should be a [DiffusionProblem] section if you are using the LD2 physics model." );

	auto const& DiffConfig = config.at( "DiffusionProblem" );

	kappa =         toml::find_or( DiffConfig, "Kappa", 1.0 );
	InitialWidth  = toml::find_or( DiffConfig, "InitialWidth", 0.2 );
	InitialHeight = toml::find_or( DiffConfig, "InitialHeight", 1.0 );
	Centre =        toml::find_or( DiffConfig, "Centre", 0.0 );

	t0 = InitialWidth * InitialWidth / ( 4.0 * kappa );
    nSigmaCalls = 0;
    nSourceCalls = 0;
    nDerivCalls = 0;
}

// Dirichlet Boundary Condition
Value LD2::LowerBoundary( Index, Time t ) const
{
	return ExactSolution( 0.0, t );
}

Value LD2::UpperBoundary( Index, Time t ) const
{
	return ExactSolution( 1.0, t );
}

// The data move with the exact solution, and a consistent y' at t0 is solved
// from their explicit time derivatives. d/dt of H sqrt(t0/(t+t0)) exp(-eta^2),
// eta^2 = (x-c)^2 / (4 kappa (t+t0)), is that times (eta^2 - 1/2) / (t+t0).
static double ld2ExactRate(double value, double x, double centre, double kappa, double t, double t0)
{
	const double etaSquared = (x - centre) * (x - centre) / (4.0 * kappa * (t + t0));
	return value * (etaSquared - 0.5) / (t + t0);
}

Value LD2::dLowerBoundary_dt( Index, Time t ) const
{
	return ld2ExactRate( ExactSolution( 0.0, t ), 0.0, Centre, kappa, t, t0 );
}

Value LD2::dUpperBoundary_dt( Index, Time t ) const
{
	return ld2ExactRate( ExactSolution( 1.0, t ), 1.0, Centre, kappa, t, t0 );
}


Value LD2::SigmaFn( Index, const State &s, Position, Time )
{
    ++nSigmaCalls;
	return kappa * s.q(0);
}

Value LD2::Sources( Index, const State &, Position, Time )
{
    ++nSourceCalls;
	return 0.0;
}

void LD2::dSigmaFn_dq( Index, VectorRef v, const State &, Position, Time )
{
    ++nDerivCalls;
	v[ 0 ] = kappa;
};

void LD2::dSigmaFn_du( Index, VectorRef v, const State &, Position, Time )
{
    ++nDerivCalls;
	v[ 0 ] = 0.0;
};

void LD2::dSources_du( Index, VectorRef v , const State &, Position, Time )
{
    ++nDerivCalls;
	v[ 0 ] = 0.0;
};

void LD2::dSources_dq( Index, VectorRef v , const State &, Position, Time )
{
    ++nDerivCalls;
	v[ 0 ] = 0.0;
};

void LD2::dSources_dsigma( Index, VectorRef v , const State &, Position, Time )
{
    ++nDerivCalls;
	v[ 0 ] = 0.0;
};



// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value LD2::InitialValue( Index, Position x ) const
{
	double y = ( x - Centre )/InitialWidth;
	return InitialHeight * ::exp( -y*y );
}

Value LD2::InitialDerivative( Index, Position x ) const
{
	double y = ( x - Centre )/InitialWidth;
	return InitialHeight * ( -2.0 * y ) * ::exp( -y*y ) * ( 1.0/InitialWidth );
}


Value LD2::ExactSolution( Position x, Time t ) const
{
	double EtaSquared = ( x - Centre )*( x - Centre )/( 4.0 * kappa * ( t + t0 ) );
	return InitialHeight * ::sqrt( t0/( t + t0 ) ) * ::exp( -EtaSquared );
}
