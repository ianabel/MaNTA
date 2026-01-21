#ifndef ADTESTPROBLEM
#define ADTESTPROBLEM

#include "AutodiffTransportSystem.hpp"

class ADTestProblem : public AutodiffTransportSystem
{
public:
	ADTestProblem(toml::value const &config, Grid const &grid);
	virtual Value aFn(Index i, Position x) override;

private:
	Real Flux(Index, RealVector, RealVector, Real, Time) override;
	Real Source(Index, RealVector, RealVector, RealVector, RealVector, Real, Time) override;

	double T_s, a, SourceWidth, SourceCentre;

	double afn_test;

	REGISTER_PHYSICS_HEADER(ADTestProblem)
};

#endif
