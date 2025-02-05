#ifndef PLASMACONSTANTS_HPP
#define PLASMACONSTANTS_HPP

#include <map>
#include <autodiff/forward/dual.hpp>
#include "../MagneticFields.hpp"

#include <boost/math/quadrature/gauss_kronrod.hpp>

using Real = autodiff::dual;

// Reference Values
constexpr static double ElectronMass = 9.1094e-31;         // Electron Mass, kg
constexpr static double ProtonMass = 1.6726e-27;           // Proton Mass,kg
constexpr static double ElementaryCharge = 1.60217663e-19; // Coulombs
constexpr static double VacuumPermittivity = 8.8541878128e-12;

// Base ion species class for species-specific values, for now just has the fusion rate and mass
class IonSpecies
{
public:
    // IonSpecies() = default;
    IonSpecies(double A) : IonMass(A * ProtonMass) {};
    virtual ~IonSpecies() = default;

    const double IonMass;
    virtual Real FusionRate(Real n, Real pi) const = 0;

    // Cross sections
    // only need to overload if we're not using hydrogen ions

    // Energy in electron volts, returns cross section in cm^2
    virtual Real electronImpactIonizationCrossSection(Real Energy) const
    {
        // Minimum energy of cross section in eV
        constexpr double ionizationEnergy = 13.6;
        constexpr double minimumEnergySigma = ionizationEnergy;

        // Contribution from ground state
        // Janev 1993, ATOMIC AND PLASMA-MATERIAL INTERACTION DATA FOR FUSION, Volume 4
        // Equation 1.2.1
        // e + H(1s) --> e + H+ + e
        // Accuracy is 10% or better
        constexpr double fittingParamA = 0.18450;
        constexpr std::array<double, 5> fittingParamB{-0.032226, -0.034539, 1.4003, -2.8115, 2.2986};

        Real sigma;
        if (Energy < minimumEnergySigma)
        {
            return 0.0;
        }
        else
        {
            Real sum = 0.0;
            Real x = 1.0 - ionizationEnergy / Energy;
            if (x <= 0)
                return 0.0;
            sum = x * (fittingParamB[0] + x * (fittingParamB[1] + x * (fittingParamB[2] + x * (fittingParamB[3] + x * fittingParamB[4]))));
            sigma = (1.0e-13 / (ionizationEnergy * Energy)) * (fittingParamA * log(Energy / ionizationEnergy) + sum);
            return sigma;
        }
    }

    // Energy in eV, returns cross section in cm^2
    virtual Real protonImpactIonizationCrossSection(Real Energy) const
    {
        // Minimum energy of cross section in keV
        const double minimumEnergySigma = 0.5;
        // Convert to keV
        Real EnergyKEV = Energy / 1000;

        // Contribution from ground state
        // Janev 1993, ATOMIC AND PLASMA-MATERIAL INTERACTION DATA FOR FUSION, Volume 4
        // Equation 2.2.1
        // H+ + H(1s) --> H+ + H+ + e
        // Accuracy is 30% or better
        constexpr double A1 = 12.899;
        constexpr double A2 = 61.897;
        constexpr double A3 = 9.2731e3;
        constexpr double A4 = 4.9749e-4;
        constexpr double A5 = 3.9890e-2;
        constexpr double A6 = -1.5900;
        constexpr double A7 = 3.1834;
        constexpr double A8 = -3.7154;

        Real sigma;
        if (EnergyKEV < minimumEnergySigma)
        {
            sigma = 0;
        }
        else
        {
            // Energy is in units of keV
            sigma = 1e-16 * A1 * (exp(-A2 / EnergyKEV) * log(1 + A3 * EnergyKEV) / EnergyKEV + A4 * exp(-A5 * EnergyKEV) / (pow(EnergyKEV, A6) + A7 * pow(EnergyKEV, A8)));
        }
        return sigma;
    }

    // Energy in eV, returns cross section in cm^2
    virtual double HydrogenChargeExchangeCrossSection(double Energy)
    {
        // Minimum energy of cross section in eV
        constexpr double minimumEnergySigma_n1 = 0.12;

        // Contribution from ground -> ground state
        // Janev 1993 2.3.1
        // p + H(n=1) --> H + p
        double sigma_n1;
        if (Energy < minimumEnergySigma_n1)
        {
            sigma_n1 = 0;
        }
        else
        {
            double EnergyKEV = Energy / 1000;
            sigma_n1 = 1e-16 * 3.2345 * log(235.88 / EnergyKEV + 2.3713) / (1 + 0.038371 * EnergyKEV + 3.8068e-6 * pow(EnergyKEV, 3.5) + 1.1832e-10 * pow(EnergyKEV, 5.4));
        }
        return sigma_n1;
    }
};

class Hydrogen : public IonSpecies
{
public:
    Hydrogen() : IonSpecies(A) {};

    Real FusionRate(Real n, Real pi) const override
    {
        return 0.0;
    }

private:
    constexpr static double A = 1.0;
};

class Deuterium : public IonSpecies
{
public:
    Deuterium() : IonSpecies(A) {};

    Real FusionRate(Real n, Real pi) const override
    {
        return 0.0;
    }

private:
    constexpr static double A = 2.0;
};

class DeuteriumTritium : public IonSpecies
{
public:
    DeuteriumTritium() : IonSpecies(A) {};

    // Density in m^-3, temperature in keV
    Real FusionRate(Real n, Real Ti) const override
    {
        Real Factor = 1e-6 * 3.68e-12;
        Real sigmav;
        if (Ti < 25)
            sigmav = Factor * pow(Ti, -2. / 3.) * exp(-19.94 * pow(Ti, -1. / 3.));
        else
            sigmav = 1e-6 * 2.7e-16; // Formula is only valid for Ti < 25 keV, just set to a constant after

        // nD*nT = n/2*n/2
        Real R = 0.25 * n * n * sigmav;

        return R;
    }

private:
    constexpr static double A = 2.5;
};

// Hold the different ion species types
template <typename T>
IonSpecies *createIonSpecies() { return new T(); };

// Map to the constructor for each of the different ion species
static std::map<std::string, IonSpecies *(*)()> PlasmaMap = {{"Hydrogen", &createIonSpecies<Hydrogen>}, {"Deuterium", &createIonSpecies<Deuterium>}, {"DeuteriumTritium", &createIonSpecies<DeuteriumTritium>}};

// Class for computing common plasma constants
class PlasmaConstants
{
public:
    // Constructor takes ion type, magnetic field, other normalizing parameters if desired
    PlasmaConstants(std::string IonType, std::shared_ptr<StraightMagneticField> B, double PlasmaWidth) : B(B), PlasmaWidth(PlasmaWidth)
    {
        auto it = PlasmaMap.find(IonType);
        if (it == PlasmaMap.end())
            throw std::logic_error("Requested ion species does not exist.");
        Plasma = it->second();
    }
    PlasmaConstants(std::string IonType, std::shared_ptr<StraightMagneticField> B, double n0, double T0, double Z_eff, double PlasmaWidth) : B(B), n0(n0), T0(T0), Z_eff(Z_eff), PlasmaWidth(PlasmaWidth)
    {
        auto it = PlasmaMap.find(IonType);
        if (it == PlasmaMap.end())
            throw std::logic_error("Requested ion species does not exist.");
        Plasma = it->second();
    };
    virtual ~PlasmaConstants() { delete Plasma; }

    template <typename T>
    T LogLambda_ii(T ni, T Ti) const
    {
        double LogLambdaRef = 24.0 - log(n0cgs) / 2.0 + log(T0eV);
        T LogLambda = 24.0 - log(n0cgs * ni) / 2.0 + log(T0eV * Ti);
        return LogLambda / LogLambdaRef; // really needs to know Ti as well
    }

    template <typename T>
    T LogLambda_ei(T ne, T Te) const
    {
        double LogLambdaRef = 23.0 - log(2.0) - log(n0cgs) / 2.0 + log(T0eV) * 1.5;
        T LogLambda = 23.0 - log(2.0) - log(ne * n0cgs) / 2.0 + log(Te * T0eV) * 1.5;
        return LogLambda / LogLambdaRef; // really needs to know Ti as well
    }

    // Return tau_ei (Helander & Sigmar notation ) normalised to tau_ei( n0, T0 )
    // This is equal to tau_e as used in Braginskii
    template <typename T>
    T ElectronCollisionTime(T ne, T Te) const
    {
        return pow(Te, 1.5) / (ne * LogLambda_ei(ne, Te));
    }

    // Return sqrt(2) * tau_ii (Helander & Sigmar notation ) normalised to tau_ii( n0, T0 )
    // This is equal to tau_i as used in Braginskii
    template <typename T>
    T IonCollisionTime(T ni, T Ti) const
    {
        return pow(Ti, 1.5) / (ni * LogLambda_ii(ni, Ti));
    }

    template <typename T>
    T Om_i(T B) const
    {
        return ElementaryCharge * B / IonMass();
    }

    template <typename T>
    T Om_e(T B) const
    {
        return ElementaryCharge * B / ElectronMass;
    }

    template <typename T>
    T c_s(T Te) const
    {
        return static_cast<T>(sqrt(T0 * Te / IonMass()));
    }

    Real FusionRate(Real n, Real pi) const;
    Real TotalAlphaPower(Real n, Real pi) const;
    Real BremsstrahlungLosses(Real n, Real pe) const;
    Real CyclotronLosses(Real V, Real n, Real Te) const;
    Real IonizationRate(Real n, Real NeutralDensity, Real v, Real Te, Real Ti) const;
    Real ChargeExchangeLossRate(Real n, Real NeutralDensity, Real v, Real Ti) const;

    Real IonElectronEnergyExchange(Real n, Real pe, Real pi, Real V, double t) const;

    double IonMass() const { return Plasma->IonMass; }
    double ReferenceElectronCollisionTime() const;
    double ReferenceIonCollisionTime() const;
    double ReferenceElectronThermalVelocity() const;
    double ReferenceIonThermalVelocity() const;
    double RhoStarRef() const;
    double mu() const;
    double NormalizingTime() const;

private:
    using integrator = boost::math::quadrature::gauss_kronrod<double, 15>;
    constexpr static unsigned max_depth = 0;
    constexpr static double tol = 1e-6;

    // Calculates the velocity space integral for collisional processes with neutrals
    // CrossSection is a lambda with units of cm^2 that returns an autodiff compatible type (Real or double)
    template <class XS>
    Real NeutralProcess(XS const &CrossSection, Real vtheta, Real T, double Mass, double minEnergy) const
    {
        Real vth2 = 2 * T * T0 / Mass;
        double cs0 = sqrt(T0 / IonMass());
        Real Mth = vtheta * cs0 / sqrt(vth2);

        auto Integrand = [this](double Energy, Real Mass, Real M, Real T, Real CrossSection)
        {
            Real MmE = M - sqrt(Energy / T);
            Real MpE = M + sqrt(Energy / T);
            Real v = sqrt(2 * ElementaryCharge * Energy / Mass);
            Real I = v * ElementaryCharge / Mass * (CrossSection * 1e-4) * (exp(-MmE * MmE) - exp(-MpE * MpE));
            return I;
        };

        Real Integral = 0;
        Integral.val = integrator::integrate([&](double Energy)
                                             { return Integrand(Energy, Mass, Mth, T * T0eV, CrossSection(Energy)).val; }, minEnergy, std::numeric_limits<double>::infinity(), max_depth, tol);

        // boost isn't compatible with autodiff so we calculate .grad integral separately if needed
        if (T.grad != 0 || vtheta.grad != 0)
            Integral.grad = integrator::integrate([&](double Energy)
                                                  { return Integrand(Energy, Mass, Mth, T * T0eV, CrossSection(Energy)).grad; }, minEnergy, std::numeric_limits<double>::infinity(), max_depth, tol);

        return Integral / (sqrt(M_PI) * Mth * vth2);
    }

private:
    IonSpecies *Plasma = nullptr;
    std::shared_ptr<StraightMagneticField> B;
    const double n0 = 1e20;
    const double n0cgs = n0 * 1e-6;
    const double T0 = 1000.0 * ElementaryCharge, T0eV = T0 / ElementaryCharge;
    const double a = 1.0; // m
    const double Z_eff = 3.0;
    double B0 = 1.0; // T

    double PlasmaWidth;
};

#endif