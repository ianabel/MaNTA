#ifndef LINEARDIFFUSION_HPP
#define LINEARDIFFUSION_HPP

#include "PhysicsCases.hpp"

/*
 * Exact solutions for the nonlinear equation
 *
 * du    d         du
 * -- - -- D( u )  -- = 0
 * dt   dx         dx
 *
 * for D = ( n/2 ) u^n * ( 1 - u^n/( n + 1 ) ) ;
 * u = ( 1 - x/sqrt( t ) )^( 1/n ) on [ 0,1 ]
 * meant to be initialised at t=1
 */

// Always inherit from TransportSystem
class NonlinearDiffusion : public TransportSystem
{
public:
	// Must provide a constructor that constructs from a toml configuration snippet
	// you can ignore it, or read problem-dependent parameters from the configuration file
	explicit NonlinearDiffusion(toml::value const &config);

	// You must provide implementations of both, these are your boundary condition functions
	Value LowerBoundary(Index, Time) const override;
	Value UpperBoundary(Index, Time) const override;

	bool isLowerBoundaryDirichlet(Index) const override;
	bool isUpperBoundaryDirichlet(Index) const override;

	// The guts of the physics problem (these are non-const as they
	// are allowed to alter internal state such as to store computations
	// for future calls)
	Value SigmaFn(Index, const Values &, const Values &, Position, Time) override;
	Value Sources(Index, const Values &, const Values &, const Values &, Position, Time) override;

	void dSigmaFn_du(Index, Values &, const Values &, const Values &, Position, Time) override;
	void dSigmaFn_dq(Index, Values &, const Values &, const Values &, Position, Time) override;

	void dSources_du(Index, Values &v, const Values &, const Values &, Position, Time) override;
	void dSources_dq(Index, Values &v, const Values &, const Values &, Position, Time) override;
	void dSources_dsigma(Index, Values &v, const Values &, const Values &, Position, Time) override;

	// Finally one has to provide initial conditions for u & q
	Value InitialValue(Index, Position) const override;
	Value InitialDerivative(Index, Position) const override;

private:
	// Put class-specific data here
	double n;

	Value ExactSolution(Position, Time) const;

	// Without this (and the implementation line in NonlinearDiffusion.cpp)
	// ManTA won't know how to relate the string 'NonlinearDiffusion' to the class.
	REGISTER_PHYSICS_HEADER(NonlinearDiffusion)
};

#endif // LINEARDIFFUSION_HPP
