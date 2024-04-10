#ifndef LINEARDIFFSOURCETEST_HPP
#define LINEARDIFFSOURCETEST_HPP

#include "AutodiffTransportSystem.hpp"

class LinearDiffSourceTest : public AutodiffTransportSystem
{
public:
    LinearDiffSourceTest(toml::value const &config, Grid const &grid);
    virtual Real2nd InitialFunction(Index i, Real2nd x, Real2nd t) const override;

private:
    enum Sources : int
    {
        PeakedEdge = 0,
        Gaussian = 1,
        Uniform = 2
    };

    std::map<std::string, Sources> SourceMap = {{"PeakedEdge", Sources::PeakedEdge}, {"Gaussian", Sources::Gaussian}, {"Uniform", Sources::Uniform}};

    Real Flux(Index, RealVector, RealVector, Position, Time) override;
    Real Source(Index, RealVector, RealVector, RealVector, Position, Time) override;

    Value LowerBoundary(Index i, Time t) const override;
    Value UpperBoundary(Index i, Time t) const override;

    virtual bool isLowerBoundaryDirichlet(Index i) const override;
    virtual bool isUpperBoundaryDirichlet(Index i) const override;

    std::vector<double> uL, uR, SourceStrength, SourceWidth, SourceCenter, InitialWidth, InitialHeight;
    std::vector<Sources> SourceTypes;
    std::vector<bool> upperBoundaryConditions, lowerBoundaryConditions;
    Matrix Kappa;

    REGISTER_PHYSICS_HEADER(LinearDiffSourceTest);
};

#endif