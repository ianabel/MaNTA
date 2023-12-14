
#include "FishersEquation.hpp"

/*
	Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL( FishersEquation );

FishersEquation::FishersEquation( toml::value const& config, Grid const& )
{
	// Always set nVars in a derived constructor
	nVars = 1;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if ( config.count( "FisherTest" ) != 1 )
		throw std::invalid_argument( "There should be a [FisherTest] section if you are using the FishersEquation physics model." );

	auto const& InternalConfig = config.at( "FisherTest" );

	C = toml::find_or( InternalConfig, "C", 1.0 );

	x_l = -10.0;
	x_u =  10.0;

	c = -5.0/::sqrt( 6.0 );
}

// Dirichlet Boundary Conditon
Value FishersEquation::LowerBoundary( Index, Time t ) const
{
	return AblowitzWaveSolution( x_l - c*t );
}

Value FishersEquation::UpperBoundary( Index, Time t ) const
{
	return AblowitzWaveSolution( x_u - c*t );
}

bool FishersEquation::isLowerBoundaryDirichlet( Index ) const { return true; };
bool FishersEquation::isUpperBoundaryDirichlet( Index ) const { return true; };


Value FishersEquation::SigmaFn( Index, const State &s, Position, Time )
{
	return s.Derivative[ 0 ];
}

Value FishersEquation::Sources( Index, const State & s, Position, Time )
{
	double uVal = s.Variable[ 0 ];
	return uVal * ( 1.0 - uVal );
}

void FishersEquation::dSigmaFn_dq( Index, Values& v, const State &, Position, Time )
{
	v[ 0 ] = 1.0;
};

void FishersEquation::dSigmaFn_du( Index, Values& v, const State&, Position, Time )
{
	v[ 0 ] = 0.0;
};

void FishersEquation::dSources_du( Index, Values&v , const State &s, Position, Time )
{
	double uVal = s.Variable[ 0 ];
	v[ 0 ] = 1.0 - 2.0*uVal;
};

void FishersEquation::dSources_dq( Index, Values&v , const State &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void FishersEquation::dSources_dsigma( Index, Values&v , const State &, Position, Time )
{
	v[ 0 ] = 0.0;
};

// This physics model uses an exact solution for testing purposes

Value FishersEquation::InitialValue( Index, Position x ) const
{
	return AblowitzWaveSolution( x );
}

Value FishersEquation::InitialDerivative( Index, Position x ) const
{
	double S = 1.0 + C * ::exp( x / ::sqrt( 6.0 ) );
	double dSdx = ( C/::sqrt( 6.0 ) ) * ::exp( x / ::sqrt( 6.0 ) );
	return -2.0*( 1.0/( S * S * S ) )* dSdx;
}

double FishersEquation::AblowitzWaveSolution( double z ) const
{
	double S = 1.0 + C * ::exp( z / ::sqrt( 6.0 ) );
	return 1.0 / ( S * S );
}
