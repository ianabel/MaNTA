#include "PlasmaConstants.hpp"

// Just normalize the fusion rate from the ion class
Real PlasmaConstants::FusionRate(Real n, Real pi) const
{
    Real Ti_keV = pi / n * T0 / (1000 * ElementaryCharge);
    return NormalizingTime() / n0 * Plasma->FusionRate(n0 * n, Ti_keV);
}

Real PlasmaConstants::TotalAlphaPower(Real n, Real pi) const
{
    double Factor = 5.6e-13 / (T0);
    Real AlphaPower = Factor * FusionRate(n, pi);
    return AlphaPower;
};
// Implements Bremsstrahlung radiative losses from NRL plasma formulary
Real PlasmaConstants::BremsstrahlungLosses(Real n, Real pe) const
{
    double p0 = n0 * T0;
    Real Normalization = p0 / NormalizingTime();
    Real Factor = (1 + Z_eff) * 1.69e-32 * 1e-6 * n0 * n0 / Normalization;

    Real Te_eV = (pe / n) * T0 / ElementaryCharge;
    Real Pbrem = Factor * n * n * sqrt(Te_eV);

    return Pbrem;
};
Real PlasmaConstants::CyclotronLosses(Real V, Real n, Real Te) const
{
    // NRL formulary with reference values factored out
    // Return units are W/m^3
    Real Te_eV = T0 / ElementaryCharge * Te;
    Real n_e20 = n * n0 / 1e20;
    Real B_z = B->B(B->Psi(V), 0.0) * B0; // in Tesla
    Real P_vacuum = 6.21 * n_e20 * Te_eV * B_z * B_z;

    // Characteristic absorption length
    // lambda_0 = (Electron Inertial Lenght) / ( Plasma Frequency / Cyclotron Frequency )  ; Eq (4) of Tamor
    //				= (5.31 * 10^-4 / (n_e20)^1/2) / ( 3.21 * (n_e20)^1/2 / B ) ; From NRL Formulary, converted to our units (Tesla for B & 10^20 /m^3 for n_e)
    Real LambdaZero = (5.31e-4 / 3.21) * (B_z / n_e20);
    double WallReflectivity = 0.95;
    Real OpticalThickness = (PlasmaWidth / (1.0 - WallReflectivity)) / LambdaZero;
    // This is the Phi introduced by Trubnikov and later approximated by Tamor
    Real TransparencyFactor = pow(Te_eV, 1.5) / (200.0 * sqrt(OpticalThickness));
    // Moderate the vacuum emission by the transparency factor
    Real Normalization = n0 * T0 / NormalizingTime();

    Real P_cy = P_vacuum * TransparencyFactor / Normalization;
    return P_cy;
};

/*
   In SI this is

   Q_i = 3 n_e m_e ( T_e - T_i ) / ( m_i tau_e )

   Normalised for the Energy equation, whose overall normalising factor is
   n0 T0^2 / ( m_e Omega_e_ref^2 tau_e_ref )


   Q_i = 3 * (p_e - p_i) / (tau_e) * (m_e/m_i) / (rho_s/R_ref)^2

 */
Real PlasmaConstants::IonElectronEnergyExchange(Real n, Real pe, Real pi, Real V, double t) const
{
    Real Te = pe / n;
    double RhoStar = RhoStarRef();
    Real pDiff = pe - pi;
    Real IonHeating = (pDiff / (ElectronCollisionTime(n, Te))) * ((3.0 / (RhoStar * RhoStar))); //* (ElectronMass / IonMass));

    if (std::isfinite(IonHeating.val))
        return IonHeating;
    else
        throw std::logic_error("Non-finite value computed for the ion heating at x = " + std::to_string(V.val) + " and t = " + std::to_string(t));
};

// Return the actual value in SI units
double PlasmaConstants::ReferenceElectronCollisionTime() const
{
    double LogLambdaRef = 24.0 - log(n0cgs) / 2.0 + log(T0eV); // 24 - ln( n^1/2 T^-1 ) from NRL pg 34
    return 12.0 * pow(M_PI, 1.5) * sqrt(ElectronMass) * pow(T0, 1.5) * VacuumPermittivity * VacuumPermittivity / (sqrt(2) * n0 * pow(ElementaryCharge, 4) * LogLambdaRef);
};

// Return the actual value in SI units
double PlasmaConstants::ReferenceIonCollisionTime() const
{
    double LogLambdaRef = 23.0 - log(2.0) - log(n0cgs) / 2.0 + log(T0eV) * 1.5; // 23 - ln( (2n)^1/2 T^-3/2 ) from NRL pg 34
    return 12.0 * pow(M_PI, 1.5) * sqrt(IonMass()) * pow(T0, 1.5) * VacuumPermittivity * VacuumPermittivity / (n0 * pow(ElementaryCharge, 4) * LogLambdaRef);
};

/*
Normalisation:
   All lengths to a, densities to n0, temperatures to T0
   We normalise time to   [ n0 T0 R_ref B_ref^2 / ( m_e Omega_e(B_ref)^2 tau_e(n0,T0) ) ]^-1
    in effect we are normalising to the particle diffusion time across a distance 1

 */

// This is c_s / ( Omega_i * a )
// = sqrt( T0 / mi ) / ( e B0 / mi ) =  [ sqrt( T0 mi ) / ( e B0 ) ] / a
double PlasmaConstants::RhoStarRef() const
{
    return sqrt(T0 * IonMass()) / (ElementaryCharge * B0 * a);
}

// The normalizing time can also be written this way
double PlasmaConstants::NormalizingTime() const
{
    double RhoStar = RhoStarRef();

    return IonMass() / ElectronMass * 1 / (RhoStar * RhoStar) * ReferenceElectronCollisionTime();
};
