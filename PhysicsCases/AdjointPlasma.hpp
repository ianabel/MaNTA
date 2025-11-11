#ifndef ADJOINTPLASMA_HPP
#define ADJOINTPLASMA_HPP

#include "MagneticFields.hpp"
#include "AutodiffTransportSystem.hpp"
#include "AutodiffAdjointProblem.hpp"

class AdjointPlasma : public AutodiffTransportSystem
{
public:
    AdjointPlasma(toml::value const &config, Grid const &grid);
    ~AdjointPlasma() = default;

    virtual Real2nd InitialFunction(Index i, Real2nd x, Real2nd t) const override;

    virtual AdjointProblem *createAdjointProblem() override;

    Real g(Position, RealVector &, RealVector &, RealVector &, RealVector &);

    void initialiseDiagnostics(NetCDFIO &) override;
    void writeDiagnostics(DGSoln const &y, DGSoln const &dydt, Time t, NetCDFIO &nc, size_t tIndex) override;

private:
    Real Flux(Index, RealVector, RealVector, Real, Time) override;
    Real Source(Index, RealVector, RealVector, RealVector, RealVector, RealVector, Real, Time) override;

    Value LowerBoundary(Index i, Time t) const override;
    Value UpperBoundary(Index i, Time t) const override;

    virtual bool isLowerBoundaryDirichlet(Index i) const override;
    virtual bool isUpperBoundaryDirichlet(Index i) const override;

    Real Gamma(RealVector u, RealVector q, Real x, Time t) const;
    Real qe(RealVector u, RealVector q, Real x, Time t) const;
    Real qi(RealVector u, RealVector q, Real x, Time t) const;

    Real Sn(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real x, Time t) const;
    Real Spe(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real x, Time t) const;
    Real Spi(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real x, Time t) const;

    Real DensityFn(Real x) const;
    Real DensityPrime(Real x) const;

    Real SafetyFactor(Real r) const;
    Real Shear(Real r) const;

    enum Channel : Index
    {
        IonEnergy = 0,
        ElectronEnergy = 1,
        Density = 2
    };

    enum ObjectiveFunctions : Index
    {
        StoredEnergy = 0,
        FusionYield = 1
    };

    std::map<std::string, Index>
        objectiveFunctions = {{"StoredEnergy", 0}, {"FusionYield", 1}};

private:
    // std::unique_ptr<PlasmaConstants> Plasma = nullptr;
    // std::shared_ptr<MagneticField> B = nullptr;

    //
    bool evolveDensity = false;
    Real grad_n;

    Real R0 = 3.0;
    Real a = 0.4;
    Real AspectRatio = R0 / a;

    Real alpha = 2.0;
    Real Chi_min = 0.1;
    Real EquilibrationFactor = 1.0;

    Real Xe_Xi = 1.0;
    Real C = 1.0;

    Real nu = 1.0;

    Real HeatFraction = 0.5;

    Real Btor = 1.0,
         Bpol = 0.1;

    Real SourceCenter = 0.5;
    Real SourceStrength = 10.0;
    Real SourceWidth = 0.03;

    Value nEdge, TeEdge, TiEdge;
    Value InitialPeakDensity, InitialPeakTe, InitialPeakTi;

    std::map<Index, std::string> adjointNames;

    Index objectiveFunction = 0;

    REGISTER_PHYSICS_HEADER(AdjointPlasma)
};

#endif