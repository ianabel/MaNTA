
#include "VonNeumannTest.hpp"
#include <iostream>

/*
	Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL( VonNeumannTest );

VonNeumannTest::VonNeumannTest( toml::value const& config )
{
	// Always set nVars in a derived constructor
	nVars = 1;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if ( config.count( "DiffusionProblem" ) != 1 )
		throw std::invalid_argument( "There should be a [DiffusionProblem] section if you are using the VonNeumannTest physics model." );

	auto const& DiffConfig = config.at( "DiffusionProblem" );

	kappa =         toml::find_or( DiffConfig, "Kappa", 1.0 );
	InitialWidth  = toml::find_or( DiffConfig, "InitialWidth", 0.2 );
	InitialHeight = toml::find_or( DiffConfig, "InitialHeight", 1.0 );
	Centre =        toml::find_or( DiffConfig, "Centre", 0.0 );

	t0 = InitialWidth * InitialWidth / ( 4.0 * kappa );
	std::cerr << "t0 = " << t0 << std::endl;

}

// Dirichlet Boundary Conditon
Value VonNeumannTest::LowerBoundary( Index, Time t ) const
{
	return ExactSolution( 0.0, t );
}

Value VonNeumannTest::UpperBoundary( Index, Time t ) const
{
	return ExactSolution( 1.0, t );
}

bool VonNeumannTest::isLowerBoundaryDirichlet( Index ) const { return true; };
bool VonNeumannTest::isUpperBoundaryDirichlet( Index ) const { return true; };


Value VonNeumannTest::SigmaFn( Index, const Values &, const Values & q, Position, Time )
{
	return kappa * q[ 0 ];
}

Value VonNeumannTest::Sources( Index, const Values &, const Values &, const Values &, Position, Time )
{
	return 0.0;
}

void VonNeumannTest::dSigmaFn_dq( Index, Values& v, const Values&, const Values&, Position, Time )
{
	v[ 0 ] = kappa;
};

void VonNeumannTest::dSigmaFn_du( Index, Values& v, const Values&, const Values&, Position, Time )
{
	v[ 0 ] = 0.0;
};

void VonNeumannTest::dSources_du( Index, Values&v , const Values &, const Values &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void VonNeumannTest::dSources_dq( Index, Values&v , const Values &, const Values &, Position, Time )
{
	v[ 0 ] = 0.0;
};

void VonNeumannTest::dSources_dsigma( Index, Values&v , const Values &, const Values &, Position, Time )
{
	v[ 0 ] = 0.0;
};



// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value VonNeumannTest::InitialValue( Index, Position x ) const
{
	double y = ( x - Centre )/InitialWidth;
	return InitialHeight * ::exp( -y*y );
}

Value VonNeumannTest::InitialDerivative( Index, Position x ) const
{
	double y = ( x - Centre )/InitialWidth;
	return InitialHeight * ( -2.0 * y ) * ::exp( -y*y ) * ( 1.0/InitialWidth );
}


Value VonNeumannTest::ExactSolution( Position x, Time t ) const
{
	double EtaSquared = ( x - Centre )*( x - Centre )/( 4.0 * kappa * ( t + t0 ) );
	return InitialHeight * ::sqrt( t0/( t + t0 ) ) * ::exp( -EtaSquared );
}
