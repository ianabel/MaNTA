#ifndef SlabPlasma
#define SlabPlamsa

#include "AutodiffTransportSystem.hpp"

/*

Physics case to test plasma fluxes for ion and electron energy

 */

#include <cmath>

class SlabPlasma : public AutodiffTransportSystem
{
public:
    SlabPlasma(toml::value const &config, Grid const &grid);

    Real2nd InitialFunction(Index i, Real2nd x, Real2nd t) const override;

private:
    enum Channel : Index
    {
        IonEnergy = 0,
        ElectronEnergy = 1
    };
    Real Flux(Index, RealVector, RealVector, Real, Time) override;
    Real Source(Index, RealVector, RealVector, RealVector, Real, Time) override;

    Real2nd MMS_Solution(Index i, Real2nd x, Real2nd t) override
    {
        return InitialFunction(i, x, t);
    };

    enum DensityType : Index
    {
        Uniform = 0,
        Gaussian = 1,
    };

    std::map<std::string, Index> DensityMap = {{"Uniform", DensityType::Uniform}, {"Gaussian", DensityType::Gaussian}};
    std::string DensityProfile;
    Real Density(Real2nd, Real2nd) const;
    Real DensityPrime(Real, Real) const;

    Real qi(RealVector, RealVector, Real, Time);
    Real qe(RealVector, RealVector, Real, Time);

    Real Si(RealVector, RealVector, RealVector, Real, Time);
    Real Se(RealVector, RealVector, RealVector, Real, Time);

    double InitialWidth, InitialHeight;
    double nEdge, TiEdge, TeEdge;

    // Reference Values
    constexpr static double ElectronMass = 9.1094e-31;         // Electron Mass, kg
    constexpr static double IonMass = 1.6726e-27;              // 2.5* Ion Mass ( = proton mass) kg (DT fusion)
    constexpr static double ElementaryCharge = 1.60217663e-19; // Coulombs
    constexpr static double VacuumPermittivity = 8.8541878128e-12;

    // All reference values should be in SI (non SI variants are explicitly noted)
    constexpr static double n0 = 1.0e20, n0cgs = n0 / 1.0e6;
    constexpr static double T0 = 1000.0 * ElementaryCharge, T0eV = T0 / ElementaryCharge;
    constexpr static double B0 = 1.0; // Reference field in T
    constexpr static double a = 1.0;  // Reference length in m

    // Underlying functions
    double RhoStarRef() const;
    Real LogLambda_ii(Real, Real) const;
    Real LogLambda_ei(Real, Real) const;
    Real ElectronCollisionTime(Real, Real) const;
    Real IonCollisionTime(Real, Real) const;
    double ReferenceElectronCollisionTime() const;
    double ReferenceIonCollisionTime() const;

    double EnergyExchangeFactor; // tune energy exchange
    Real IonElectronEnergyExchange(Real n, Real pe, Real pi, Real x, double t) const;

    void initialiseDiagnostics(NetCDFIO &nc) override;
    void writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex) override;

    REGISTER_PHYSICS_HEADER(SlabPlasma)
};

#endif
