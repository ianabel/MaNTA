#ifndef CURVEDMIRRORPLASMA_HPP
#define CURVEDMIRRORPLASMA_HPP

#include "MagneticFields.hpp"
#include "AutodiffTransportSystem.hpp"
// #include "Constants.hpp"
#include "MirrorPlasma/PlasmaConstants.hpp"

/*
    Ground-up reimplementation of a collisional cylindrical plasma, with a single ion species
    we use V( psi ), the volume enclosed by a flux surface as the radial coordinate.

    The equations to be solved are then

    d_dt n_e + d_dV ( V' Gamma_e ) = Sn
    d_dt p_e + d_dV ( V' q_e ) = Spe + Cei
    d_dt p_i + d_dV ( V' q_i ) = Spi - pi_i_cl * domega_dpsi + Gamma m_i omega^2/B_z + Cie
    d_t L_phi + d_dV ( V' pi ) = Somega - j_R R B_z

    pi = Sum_s { pi_s_cl + m_s omega R^2 Gamma_s }

 */

#include <cmath>

class CurvedMirrorPlasma : public AutodiffTransportSystem
{
public:
    CurvedMirrorPlasma(toml::value const &config, Grid const &grid);
    ~CurvedMirrorPlasma() = default;

    virtual Real2nd InitialFunction(Index i, Real2nd x, Real2nd t) const override;

private:
    double RelaxFactor;
    double MinDensity;
    double MinTemp;
    Real floor(Real x, double val) const
    {
        if (x >= val)
            return x;
        else
        {
            x.val = val;
            return x;
        }
    }

    enum Channel : Index
    {
        Density = 0,
        IonEnergy = 1,
        ElectronEnergy = 2,
        AngularMomentum = 3
    };
    std::map<std::string, Index> ChannelMap = {{"Density", Channel::Density}, {"IonEnergy", Channel::IonEnergy}, {"ElectronEnergy", Channel::ElectronEnergy}, {"AngularMomentum", Channel::AngularMomentum}};
    std::map<Index, bool> ConstantChannelMap = {{Channel::Density, false}, {Channel::IonEnergy, false}, {Channel::ElectronEnergy, false}, {Channel::AngularMomentum, false}};

    enum ParticleSourceType
    {
        None = 0,
        Gaussian = 1,
        Distributed = 2,
        Ionization = 3,
    };

    std::vector<bool> upperBoundaryConditions;
    std::vector<bool> lowerBoundaryConditions;

    double nEdge, TeEdge, TiEdge, MUpper, MLower, MEdge;
    double InitialPeakDensity, InitialPeakTe, InitialPeakTi, InitialPeakMachNumber;
    double MachWidth;
    bool useAmbipolarPhi;

    Real Flux(Index i, RealVector u, RealVector q, Real V, Time t) override
    {
        return B->FluxSurfaceAverage([&](Real V, Real s)
                                     { return Flux(i, u, q, V, s, t); }, V);
    };
    Real Flux(Index, RealVector, RealVector, Real, Real, Time);

    Real Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, Time t) override
    {
        return B->FluxSurfaceAverage([&](Real V, Real s)
                                     { return Source(i, u, q, sigma, phi, V, s, t); }, V);
    };
    Real Source(Index, RealVector, RealVector, RealVector, RealVector, Real, Real, Time);

    Real GFunc(Index i, RealVector u, RealVector q, RealVector sigma, RealVector phi, Position V, Time t) override
    {
        return B->FluxSurfaceAverage([&](Real V, Real s)
                                     { return GFunc(i, u, q, sigma, phi, V, s, t); }, V);
    };
    Real GFunc(Index, RealVector, RealVector, RealVector, RealVector, Real, Real, Time);
    Value InitialAuxValue(Index, Position, Time) const override;

    Value LowerBoundary(Index i, Time t) const override;
    Value UpperBoundary(Index i, Time t) const override;

    virtual bool isLowerBoundaryDirichlet(Index i) const override;
    virtual bool isUpperBoundaryDirichlet(Index i) const override;

    Real2nd MMS_Solution(Index i, Real2nd V, Real2nd t) override;

    std::vector<double> growth_factors;

    double ParticleSourceStrength, ParticleSourceCenter,
        ParticleSourceWidth, UniformHeatSource;

    double IRadial;

    std::unique_ptr<PlasmaConstants> Plasma;
    std::shared_ptr<MagneticField> B;

    // // Reference Values

    // // All reference values should be in SI (non SI variants are explicitly noted)
    constexpr static double n0 = 1.e20, n0cgs = n0 / 1.e6;
    constexpr static double T0 = 1000.0 * ElementaryCharge, T0eV = T0 / ElementaryCharge;
    constexpr static double B0 = 1.0; // Reference field in T
    constexpr static double a = 1.0;  // Reference length in m
    constexpr static double Z_eff = 3.0;

    Real Gamma(RealVector u, RealVector q, Real V, Real s, Time t) const;
    Real qe(RealVector u, RealVector q, Real V, Real s, Time t) const;
    Real qi(RealVector u, RealVector q, Real V, Real s, Time t) const;
    Real Pi(RealVector u, RealVector q, Real V, Real s, Time t) const;
    Real Sn(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, Real s, Time t) const;
    Real Spe(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, Real s, Time t) const;
    Real Spi(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, Real s, Time t) const;
    Real Somega(RealVector u, RealVector q, RealVector sigma, RealVector phi, Real V, Real s, Time t) const;

    // Underlying functions

    Real IonClassicalAngularMomentumFlux(Real V, Real s, Real n, Real Ti, Real dOmegadV, Time t) const;

    // Add a cap to sources to prevent ridiculous values
    double SourceCap;

    Real ParticleSource(double R, double t) const;

    Real ElectronPastukhovLossRate(Real V, Real s, Real Xi_e, Real n, Real Te) const;
    Real IonPastukhovLossRate(Real V, Real s, Real Xi_i, Real n, Real Ti) const;

    Real CentrifugalPotential(Real V, Real s, Real omega, Real Ti, Real Te) const;

    Real phi0(RealVector u, Real V, Real s) const;

    Real Xi_i(Real V, Real s, Real phi, Real Ti, Real Te, Real omega) const;

    Real Xi_e(Real V, Real s, Real phi, Real Ti, Real Te, Real omega) const;

    // First order correction to phi, only used if not calculating auxiliary phi1
    Real AmbipolarPhi(Real V, Real n, Real Ti, Real Te) const;

    Real ParallelCurrent(Real V, Real s, Real omega, Real n, Real Ti, Real Te, Real phi) const;
    double R_Lower, R_Upper;

    Real RelaxSource(Real A, Real B) const
    {
        return RelaxFactor * (A - B);
    };

    // template <typename T1, typename T2>
    // double Voltage(T1 &L_phi, T2 &n);
    // void initialiseDiagnostics(NetCDFIO &nc) override;
    // void writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex) override;

    REGISTER_PHYSICS_HEADER(CurvedMirrorPlasma)
};

#endif
