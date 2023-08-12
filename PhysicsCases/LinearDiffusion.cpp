
#include "LinearDiffusion.hpp"

/*
	Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL( LinearDiffusion );


using TransportSystem::Value;
using TransportSystem::Position;
using TransportSystem::Time;

LinearDiffusion::LinearDiffusion( toml::value const& config )
{
	// Always set nVars in a derived constructor
	nVars = 1;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if ( config.count( "DiffusionProblem" ) != 1 )
		throw std::invalid_argument( "There should be a [DiffusionProblem] section if you are using the LienarDiffusion physics model." );

	auto const& DiffConfig = config.at( "DiffusionProblem" );

	kappa =         toml::find_or( DiffConfig, "Kappa", 1.0 );
	InitialWidth  = toml::find_or( DiffConfig, "InitialWidth", 0.2 );
	InitialHeight = toml::find_or( DiffConfig, "InitialHeight", 1.0 );

}

// Dirichlet Boundary Conditon
Value LinearDiffusion::Dirichlet_g( Index, Position, Time )
{
	return 0;
}

Value LinearDiffusion::vonNeumann_g( Index, Position, Time )
{
	return 0;
}

Value LinearDiffusion::SigmaFn( Index, const ValueVector&, const ValueVector& q, Position, Time )
{
	return kappa * q[ 0 ];
}

Value LinearDiffusion::Sources( Index, const ValueVector&, const ValueVector&, Position, Time )
{
	return 0.0;
}

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value LinearDiffusion::InitialValue( Index, Position x )
{
	Value y = x / InitialWidth;
	return InitialHeight * ::exp( - ( y * y ) );
}

Value LinearDiffusion::InitialDerivative( Index, Position x )
{
	Value y = x / InitialWidth;
	return -2.0 * ( y / InitialWidth ) * InitialValue( 0, x );
}

