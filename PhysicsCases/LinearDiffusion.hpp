
#include "TransportSystem.hpp"

/*
	Linear Diffusion Test Case, showcasing how to write a physics case that is compiled
	at the same time as the 
 */

// Always inherit from TransportSystem
class LinearDiffusion : public TransportSystem {
	public:
		// Must provide a constructor that constructs from a toml configuration snippet
		// you can ignore it, or read problem-dependent parameters from the configuration file
		explicit LinearDiffusion( toml::value const& config );

		// You must provide implementations of both, these are your boundary condition functions
		Value LowerBoundary( Index, Position, Time ) override;
		Value UpperBoundary( Index, Position, Time ) override;

		bool isLowerBoundaryDirichlet( Index ) override;
		bool isUpperBoundaryDirichlet( Index ) override;

		// The guts of the physics problem (these are non-const as they
		// are allowed to alter internal state such as to store computations
		// for future calls)
		Value SigmaFn( Index, const ValueVector &, const ValueVector &, Position, Time ) override;
		Value Sources( Index, const ValueVector &, const ValueVector &, Position, Time ) override;

		// Finally one has to provide initial conditions for u & q
		Value      InitialValue( Index, Position x ) override;
		Value InitialDerivative( Index, Position x ) override;

private:
	// Put class-specific data here
	double kappa, InitialWidth, InitialHeight, Centre;

	// Without this (and the implementation line in LinearDiffusion.cpp)
	// ManTA won't know how to relate the string 'LinearDiffusion' to the class.
	REGISTER_PHYSICS_HEADER( LinearDiffusion );
};

