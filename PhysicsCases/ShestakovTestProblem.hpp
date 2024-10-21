#ifndef ADTESTPROBLEM
#define ADTESTPROBLEM

#include "AutodiffTransportSystem.hpp"

class ShestakovTestProblem : public AutodiffTransportSystem
{
public:
	ShestakovTestProblem(toml::value const &config, Grid const &grid);

private:
	Real Flux(Index, RealVector, RealVector, Real, Time) override;
	Real Source(Index, RealVector, RealVector, RealVector, RealVector, Real, Time) override;

	Value InitialValue(Index, Position) const override;
	Value InitialDerivative(Index, Position) const override;

	REGISTER_PHYSICS_HEADER(ShestakovTestProblem)
};

#endif
