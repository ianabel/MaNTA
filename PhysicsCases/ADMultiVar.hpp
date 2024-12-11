#ifndef ADTESTPROBLEM
#define ADTESTPROBLEM

#include "AutodiffTransportSystem.hpp"

class ADMultiVar : public AutodiffTransportSystem
{
  public:
    ADMultiVar(toml::value const &config, Grid const &grid);

  private:
    Real Flux(Index, RealVector, RealVector, Real, Time) override;
    Real Source(Index, RealVector, RealVector, RealVector, RealVector, Real, Time) override;

    Real S0( Real );
    Real S1( Real );

    virtual Value InitialValue(Index i, Position x) const override;
    virtual Value InitialDerivative(Index i, Position x) const override;

    double a, b, A0, A1, x0, x1, w0, w1;

    REGISTER_PHYSICS_HEADER(ADMultiVar)
};

#endif
