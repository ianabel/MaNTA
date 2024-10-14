#ifndef PLASMACONSTANTS_HPP
#define PLASMACONSTANTS_HPP

#include <map>
#include <autodiff/forward/dual.hpp>
#include "../MagneticFields.hpp"

using Real = autodiff::dual;

// Reference Values
constexpr static double ElectronMass = 9.1094e-31;         // Electron Mass, kg
constexpr static double ProtonMass = 1.6726e-27;           // Proton Mass,kg
constexpr static double ElementaryCharge = 1.60217663e-19; // Coulombs
constexpr static double VacuumPermittivity = 8.8541878128e-12;

enum PlasmaTypes : int
{
    H = 0,
    DD = 1,
    DT = 2,
};

// Base ion species class for species-specific values, for now just has the fusion rate and mass
class IonSpecies
{
public:
    IonSpecies(double A) : IonMass(A * ProtonMass) {};
    virtual ~IonSpecies() = default;

    const double IonMass;
    virtual Real FusionRate(Real n, Real pi) const = 0;
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

static std::map<int, IonSpecies *(*)()> PlasmaMap = {{PlasmaTypes::H, &createIonSpecies<Hydrogen>}, {PlasmaTypes::DD, &createIonSpecies<Deuterium>}, {PlasmaTypes::DT, &createIonSpecies<DeuteriumTritium>}};

// Class for computing common plasma constants
class PlasmaConstants
{
public:
    // Constructor takes ion type, pointer to magnetic field, other normalizing parameters if desired
    PlasmaConstants(PlasmaTypes p, std::shared_ptr<StraightMagneticField> B, double PlasmaWidth) : B(B), PlasmaWidth(PlasmaWidth) { Plasma = PlasmaMap[p](); };
    PlasmaConstants(PlasmaTypes p, std::shared_ptr<StraightMagneticField> B, double n0, double T0, double Z_eff, double PlasmaWidth) : B(B), n0(n0), T0(T0), Z_eff(Z_eff), PlasmaWidth(PlasmaWidth) { Plasma = PlasmaMap[p](); };
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

    Real FusionRate(Real n, Real pi) const;
    Real TotalAlphaPower(Real n, Real pi) const;
    Real BremsstrahlungLosses(Real n, Real pe) const;
    Real CyclotronLosses(Real V, Real n, Real Te) const;

    Real IonElectronEnergyExchange(Real n, Real pe, Real pi, Real V, double t) const;

    double IonMass() const { return Plasma->IonMass; }
    double ReferenceElectronCollisionTime() const;
    double ReferenceIonCollisionTime() const;
    double RhoStarRef() const;
    double NormalizingTime() const;

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