#ifndef LINEARDIFFSOURCETEST_HPP
#define LINEARDIFFSOURCETEST_HPP

#include "AutodiffTransportSystem.hpp"

class LinearDiffSourceTest : public AutodiffTransportSystem
{
public:
    LinearDiffSourceTest(toml::value const &config, Grid const &grid);
    virtual Real2nd InitialFunction(Index i, Real2nd x, Real2nd t) const override;

private:
    static SystemSpec buildSpec(toml::value const &config);

    enum Sources : int
    {
        PeakedEdge = 0,
        Gaussian = 1,
        Uniform = 2,
        Step = 3
    };


    std::map<std::string, Sources> SourceMap = {{"PeakedEdge", Sources::PeakedEdge}, {"Gaussian", Sources::Gaussian}, {"Uniform", Sources::Uniform}, {"Step", Sources::Step}};

    Real Flux(Index, RealVector, RealVector, Real, Time) override;
    Real Source(Index, RealVector, RealVector, RealVector, RealVector, Real, Time) override;

    // Real Phi(Index, RealVector, RealVector, RealVector, RealVector, Position, Time) override;

    // Value InitialAuxValue(Index, Position) const override;

    Value LowerBoundary(Index i, Time t) const override;
    Value UpperBoundary(Index i, Time t) const override;

    std::vector<double> uL, uR, SourceStrength, SourceWidth, SourceCenter, InitialWidth, InitialHeight;
    std::vector<Sources> SourceTypes;
    Index nSources;
    Matrix Kappa;

    void initialiseDiagnostics(NetCDFIO &nc) override;
    void writeDiagnostics(DGSoln const &y, DGSoln const& dydt, Time t, NetCDFIO &nc, size_t tIndex) override;

    REGISTER_PHYSICS_HEADER(LinearDiffSourceTest)
};

#endif
