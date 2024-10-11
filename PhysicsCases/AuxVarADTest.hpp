#ifndef AUXVARADTEST
#define AUXVARADTEST

#include "AutodiffTransportSystem.hpp"

class AuxVarADTest : public AutodiffTransportSystem
{
public:
    AuxVarADTest(toml::value const &config, Grid const &grid);

private:
    // You must provide implementations of both, these are your boundary condition functions
    Value LowerBoundary(Index, Time) const override;
    Value UpperBoundary(Index, Time) const override;

    bool isLowerBoundaryDirichlet(Index) const override;
    bool isUpperBoundaryDirichlet(Index) const override;

    Real Flux(Index, RealVector, RealVector, Real, Time) override;
    Real Source(Index, RealVector, RealVector, RealVector, RealVector, Real, Time) override;
    Real GFunc(Index, RealVector, RealVector, RealVector, RealVector, Position, Time) override;

    autodiff::dual2nd InitialFunction(Index i, autodiff::dual2nd x, autodiff::dual2nd t) const override;
    Value InitialAuxValue(Index, Position) const override;

    double kappa, InitialWidth, InitialHeight, Centre;

    REGISTER_PHYSICS_HEADER(AuxVarADTest)
};

#endif
