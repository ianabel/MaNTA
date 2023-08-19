#ifndef TESTDIFFUSION_HPP
#define TESTDIFFUSION_HPP

#include "TransportSystem.hpp"

/*
	Header-only TransportSystem for the Unit tests to
	enable testing of construction of a SystemSolver
 */

class LinearDiffusion : public TransportSystem {
	public:
		// Must provide a constructor that constructs from a toml configuration snippet
		// you can ignore it, or read problem-dependent parameters from the configuration file
		explicit LinearDiffusion( toml::value const& config ){
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
			Centre =        toml::find_or( DiffConfig, "Centre", 0.5 );

		};

		// You must provide implementations of both, these are your boundary condition functions
		Value LowerBoundary( Index, Position, Time ) override { return 0.0;};
		Value UpperBoundary( Index, Position, Time ) override { return 0.0;};

		bool isLowerBoundaryDirichlet( Index ) override { return true;};
		bool isUpperBoundaryDirichlet( Index ) override { return true;};

		// The guts of the physics problem (these are non-const as they
		// are allowed to alter internal state such as to store computations
		// for future calls)
		Value SigmaFn( Index, const ValueVector &, const ValueVector &q, Position, Time ) override {
			return kappa * q[ 0 ];
		};
		Value Sources( Index, const ValueVector &, const ValueVector &, Position, Time ) override {
			return 0.0;
		};

		// Finally one has to provide initial conditions for u & q
		Value      InitialValue( Index, Position x ) override {
			double y = ( x - Centre )/InitialWidth;
			return InitialHeight * ::exp( -y*y );
		};
		Value InitialDerivative( Index, Position x ) override {
			double y = ( x - Centre )/InitialWidth;
			return InitialHeight * ( -2.0 * y ) * ::exp( -y*y ) * ( 1.0/InitialWidth );
		};

private:
	// Put class-specific data here
	double kappa, InitialWidth, InitialHeight, Centre;

	// Doesn't include class registration, this is header-only for internal testing
};

#endif // TESTDIFFUSION_HPP
