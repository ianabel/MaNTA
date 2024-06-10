#ifndef NONLINAUTODIFF_HPP
#define NONLINAUTODIFF_HPP

#include "AutodiffTransportSystem.hpp"

class NonlinAutodiff : public AutodiffTransportSystem
{
public:
    NonlinAutodiff(toml::value const &config, Grid const &grid);
    Real2nd InitialFunction(Index i, Real2nd x, Real2nd t) const override;

private:
    Real Flux(Index, RealVector, RealVector, Real, Time) override;
    Real Source(Index, RealVector, RealVector, RealVector, Real, Time) override;

    double SourceStrength, SourceWidth, SourceCenter, InitialWidth, InitialHeight;
    double Kappa, Beta;

    void initialiseDiagnostics(NetCDFIO &nc) override;
    void writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex) override;

    REGISTER_PHYSICS_HEADER(NonlinAutodiff)
};

#endif