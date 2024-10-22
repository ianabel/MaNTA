#include "MagneticFields.hpp"

Real CurvedMagneticField::Psi_V(Real V) const
{
    Real PsiV = B0 / (2 * M_PI * L_z) * V / (1 - A / gamma * sin(gamma * L_z));
    return PsiV;
}

Real CurvedMagneticField::B(Real Psi, Real z) const
{
    Real Bmag = sqrt(pow(B_r(Psi, z), 2) + pow(B_z(z), 2));
    return Bmag;
}

Real CurvedMagneticField::R(Real Psi, Real z) const
{
    Real R_psi = sqrt(2 * Psi / B_z(z));
    return R_psi;
}

Real CurvedMagneticField::dRdV(Real V, Real z) const
{
    Real drdv = 1.0 / VPrime(V) * 1.0 / sqrt(2 * (Psi_V(V) * B_z(z)));
    return drdv;
}

Real CurvedMagneticField::VPrime(Real V) const
{
    Real Vp = 2 * M_PI / B0 * L_z * (1 - A / gamma * sin(gamma * L_z));
    return Vp;
}

Real CurvedMagneticField::MirrorRatio(Real V, Real z) const
{
    return B(Psi_V(V), 0) / B(Psi_V(V), z);
}

Real CurvedMagneticField::LeftEndpoint(Real Psi) const
{
    return 0.0;
}

Real CurvedMagneticField::RightEndpoint(Real Psi) const
{
    return L_z;
}

Real CurvedMagneticField::B_z(Real z) const
{
    Real Bz = B0 / (1 - A * cos(gamma * z));
    return Bz;
}

Real CurvedMagneticField::B_r(Real Psi, Real z) const
{
    Real Br = 0.5 * B0 * R(Psi, z) * A * gamma * sin(gamma * z) / pow(1 - A * cos(gamma * z), 2);
    return Br;
}
