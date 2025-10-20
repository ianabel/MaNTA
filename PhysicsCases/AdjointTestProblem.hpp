#ifndef ADTESTPROBLEM
#define ADTESTPROBLEM

#include "AutodiffTransportSystem.hpp"

class AdjointTestProblem : public AutodiffTransportSystem
{
public:
    AdjointTestProblem(toml::value const &config, Grid const &grid);

private:
    Real Flux(Index, RealVector, RealVector, Real, Time) override;
    Real Source(Index, RealVector, RealVector, RealVector, RealVector, Real, Time) override;

    double T_s, D, SourceWidth, SourceCentre;

    REGISTER_PHYSICS_HEADER(AdjointTestProblem)
};

#endif
