#ifndef ADTWOCHANNEL
#define ADTWOCHANNEL

#include "AutodiffTransportSystem.hpp"

class ADTwoChannel : public AutodiffTransportSystem
{
public:
	ADTwoChannel(toml::value const &config, Grid const &grid);

private:
	Real Flux(Index, RealVector, RealVector, Position, Time, std::vector<Position> * = nullptr) override;
	Real Source(Index, RealVector, RealVector, RealVector, Position, Time, std::vector<Position> * = nullptr) override;

	Real Postprocessor(const FluxWrapper &f, Position x, Time t) override { return f(x, t, nullptr); };
	Values Postprocessor(const GradWrapper &f, Position x, Time t) override { return f(x, t, nullptr); };

	Value InitialValue(Index, Position) const override;
	Value InitialDerivative(Index, Position) const override;

	double A, c, kappa, S_w;
	std::vector<Value> H;

	REGISTER_PHYSICS_HEADER(ADTwoChannel)
};

#endif
