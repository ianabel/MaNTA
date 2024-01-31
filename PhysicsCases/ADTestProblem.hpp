#ifndef ADTESTPROBLEM
#define ADTESTPROBLEM

#include "AutodiffTransportSystem.hpp"

class ADTestProblem : public AutodiffTransportSystem
{
public:
	ADTestProblem(toml::value const &config, Grid const &grid);

private:
	Real Flux(Index, RealVector, RealVector, Position, Time, std::vector<Position> * = nullptr) override;
	Real Source(Index, RealVector, RealVector, RealVector, Position, Time, std::vector<Position> * = nullptr) override;

	Value InitialValue(Index, Position) const override;
	Value InitialDerivative(Index, Position) const override;

	double T_s, a, SourceWidth, SourceCentre;

	REGISTER_PHYSICS_HEADER(ADTestProblem)
};

#endif
