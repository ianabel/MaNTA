#ifndef ADTWOCHANNEL
#define ADTWOCHANNEL

#include "AutodiffTransportSystem.hpp"

class ADTwoChannel : public AutodiffTransportSystem
{
public:
	ADTwoChannel(toml::value const &config, Grid const &grid);

private:
	Real Flux(Index, RealVector, RealVector, Real, Time) override;
	Real Source(Index, RealVector, RealVector, RealVector, RealVector, Real, Time) override;

	Value InitialValue(Index, Position) const override;
	Value InitialDerivative(Index, Position) const override;

	double A, c, kappa, S_w;
	std::vector<Value> H;

	REGISTER_PHYSICS_HEADER(ADTwoChannel)
};

#endif
