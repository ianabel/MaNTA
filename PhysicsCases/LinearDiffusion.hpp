#ifndef LINEARDIFFUSION_HPP
#define LINEARDIFFUSION_HPP

#include "PhysicsCases.hpp"

/*
	Linear Diffusion Test Case, showcasing how to write a physics case that is compiled
	at the same time as the
 */

// Always inherit from TransportSystem
class LinearDiffusion : public TransportSystem
{
public:
	// Must provide a constructor that constructs from a toml configuration snippet
	// you can ignore it, or read problem-dependent parameters from the configuration file
	explicit LinearDiffusion(toml::value const &config, Grid const &);

	// You must provide implementations of both, these are your boundary condition functions
	Value LowerBoundary(Index, Time) const override;
	Value UpperBoundary(Index, Time) const override;

	bool isLowerBoundaryDirichlet(Index) const override;
	bool isUpperBoundaryDirichlet(Index) const override;

	// The guts of the physics problem (these are non-const as they
	// are allowed to alter internal state such as to store computations
	// for future calls)
	Value SigmaFn(Index, const State &, Position, Time) override;
	Value Sources(Index, const State &, Position, Time) override;

	void dSigmaFn_du(Index, VectorRef , const State &, Position, Time) override;
	void dSigmaFn_dq(Index, VectorRef , const State &, Position, Time) override;

	void dSources_du(Index, VectorRef v, const State &, Position, Time) override;
	void dSources_dq(Index, VectorRef v, const State &, Position, Time) override;
	void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override;

	// Finally one has to provide initial conditions for u & q
	Value InitialValue(Index, Position) const override;
	Value InitialDerivative(Index, Position) const override;

private:
	// Put class-specific data here
	double kappa, InitialWidth, InitialHeight, Centre;
	bool lowerNeumann;
	double SourceStrength;

	// The UseMMS option, its growth/growth_rate parameters, MMS_Solution,
	// MMS_Source and the "MMS" diagnostics group are all gone. The manufactured
	// solution was (1 + growth tanh(rate t)) times the initial Gaussian while
	// LowerBoundary/UpperBoundary return 0, so it did not satisfy the boundary
	// conditions -- with the defaults it is about 0.29 at the domain edges -- and
	// an order-of-accuracy study against it could not show k+1. Nothing set the
	// option: every config that mentioned it set it false, and no regression case
	// uses this physics case at all (Tests/RegressionTests/LinearDiffusion.conf
	// asks for LDTest).
	//
	// Order of accuracy is measured by Tests/UnitTests/MMSConvergenceTests.cpp,
	// which builds its own manufactured problems and never used any of this.

	// Without this (and the implementation line in LinearDiffusion.cpp)
	// ManTA won't know how to relate the string 'LinearDiffusion' to the class.
	REGISTER_PHYSICS_HEADER(LinearDiffusion)
};

#endif // LINEARDIFFUSION_HPP
