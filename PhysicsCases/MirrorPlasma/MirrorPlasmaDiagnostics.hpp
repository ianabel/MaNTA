#pragma once

#include "PhysicsCases.hpp"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
typedef std::function<double(double)> Fn;

using Real = autodiff::dual;
using RealVector = autodiff::VectorXdual;

enum Channel : int
{
    Density = 0,
    IonEnergy = 1,
    ElectronEnergy = 2,
    AngularMomentum = 3
};

// Functions for mirror diagnostics, templated so we can have multiple mirror plasma files
// Need to be very careful that each mirror plasma has all the members used here
// Magnetic field calls only use base class methods
template <class Plasma>
void initializeMirrorDiagnostics(NetCDFIO &nc, Plasma const &p)
{

    // Add diagnostics here
    //
    double TauNorm = p.Plasma->NormalizingTime(); //(IonMass / ElectronMass) * (1.0 / (RhoStar * RhoStar)) * (ReferenceElectronCollisionTime());
    nc.AddScalarVariable("Tau", "Normalising time", "s", TauNorm);

    auto nVars = p.nVars;
    auto nAux = p.nAux;

    // lambda wrappers for DGSoln object

    auto L = [&](double V)
    {
        return p.InitialValue(Channel::AngularMomentum, V);
    };
    auto LPrime = [&](double V)
    {
        return p.InitialDerivative(Channel::AngularMomentum, V);
    };
    auto n = [&](double V)
    {
        return p.InitialValue(Channel::Density, V);
    };
    auto nPrime = [&](double V)
    {
        return p.InitialDerivative(Channel::Density, V);
    };
    auto p_i = [&](double V)
    {
        return (2. / 3.) * p.InitialValue(Channel::IonEnergy, V);
    };
    auto p_e = [&](double V)
    {
        return (2. / 3.) * p.InitialValue(Channel::ElectronEnergy, V);
    };

    auto u = [&](double V)
    {
        RealVector U(nVars);
        for (Index i = 0; i < nVars; ++i)
            U(i) = p.InitialValue(i, V);

        return U;
    };

    auto q = [&](double V)
    {
        RealVector Q(nVars);
        for (Index i = 0; i < nVars; ++i)
            Q(i) = p.InitialDerivative(i, V);

        return Q;
    };

    auto aux = [&](double V)
    {
        RealVector Aux(nAux);
        for (Index i = 0; i < nAux; ++i)
            Aux(i) = p.InitialAuxValue(i, V, 0.0);

        return Aux;
    };

    Fn Phi0 = [&](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = p.B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real p0 = p.CentrifugalPotential((Real)V, omega, Ti, Te);

        return p0.val;
    };

    Fn phi = [&](double V)
    {
        if (p.useAmbipolarPhi)
        {
            return Phi0(V) + p.InitialAuxValue(0, V, 0);
        }
        else
        {
            return Phi0(V);
        }
    };

    auto ShearingRate = [&](double V)
    {
        auto qV = q(V);
        auto uV = u(V);
        Real Ti = 2. / 3. * uV(Channel::IonEnergy) / uV(Channel::Density);
        Real R = p.B->R_V(V, 0.0);
        Real vtheta = 1 / R * uV(Channel::AngularMomentum) / uV(Channel::Density);

        Real dVdR = 1 / p.B->dRdV(V, 0.0);
        Real SR = 1.0 / sqrt(Ti) * (dVdR / uV(Channel::Density) * (qV(Channel::AngularMomentum) - R * vtheta * qV(Channel::Density)) - vtheta);
        return SR.val;
    };

    auto ElectrostaticPotential = [&](double V)
    {
        if (p.useAmbipolarPhi)
        {
            return p.phi0(u(V), (Real)V).val + p_i(V) / n(V) * aux(V)[0].val;
        }
        else
        {
            return p.phi0(u(V), (Real)V).val;
        }
    };

    double initialVoltage = p.Voltage(L, n);

    const std::function<double(const double &)> initialZero = [](const double &V)
    { return 0.0; };

    // Wrap DGApprox with lambdas for heating functions
    Fn ViscousHeating = [&](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real R = p.B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i
        Real dRdV = p.B->dRdV(V, 0.0);
        Real JPrime = R * R * nPrime(V) + 2.0 * dRdV * R * n(V);
        Real dOmegadV = LPrime(V) / J - JPrime * L(V) / (J * J);

        try
        {
            double Heating = p.IonClassicalAngularMomentumFlux(V, n(V), Ti, dOmegadV, 0).val * dOmegadV.val;
            return Heating;
        }
        catch (...)
        {
            return 0.0;
        };
    };

    Fn AlphaHeating = [&](double V)
    {
        double MirrorRatio = p.B->MirrorRatio(V, 0.0).val;

        double Heating = sqrt(1 - 1 / MirrorRatio) * p.Plasma->TotalAlphaPower(n(V), p_i(V)).val;
        return Heating;
    };

    Fn RadiationLosses = [&](double V)
    {
        double Losses = p.Plasma->BremsstrahlungLosses(n(V), p_e(V)).val;

        return Losses;
    };

    Fn ElectronParallelLosses = [&](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = p.B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real Xe = p.Xi_e((Real)V, aux(V)[0], Ti, Te, omega);

        double ParallelLosses = Te.val * (1 + Xe.val) * p.ElectronPastukhovLossRate(V, Xe, n(V), Te).val;

        return ParallelLosses;
    };

    Fn IonParallelLosses = [&](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = p.B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real Xi = p.Xi_i((Real)V, aux(V)[0], Ti, Te, omega);

        double ParallelLosses = Ti.val * (1 + Xi.val) * p.IonPastukhovLossRate(V, Xi, n(V), Te).val;

        return ParallelLosses;
    };

    Fn AngularMomentumLosses = [&](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = p.B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real AngularMomentumPerParticle = L(V) / n(V);

        Real Xi = p.Xi_i((Real)V, aux(V)[0], Ti, Te, omega);

        double ParallelLosses = AngularMomentumPerParticle.val * p.IonPastukhovLossRate(V, Xi, n(V), Te).val;

        return ParallelLosses;
    };

    Fn EnergyExchange = [&](double V)
    {
        return p.Plasma->IonElectronEnergyExchange(n(V), p_e(V), p_i(V), V, 0.0).val;
    };

    Fn IonPotentialHeating = [&](double V)
    {
        return p.IonPotentialHeating(u(V), q(V), aux(V), V).val;
    };

    Fn ElectronPotentialHeating = [&](double V)
    {
        return p.ElectronPotentialHeating(u(V), q(V), aux(V), V).val;
    };

    Fn ParticleSourceHeating = [&](double V)
    {
        Real R = p.B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;
        Real ParticleSourceHeating = 0.5 * omega * omega * R * R * p.ParticleSource(R.val, 0.0);
        return ParticleSourceHeating.val;
    };

    Real tnorm = p.n0 * TauNorm; // n0 * T0 * a * B0 * B0 / (electronMass * Om_e(B0) * Om_e(B0) * tau_e(n0, n0 * T0));
    nc.AddScalarVariable("tnorm", "time normalization", "s", 1 / tnorm.val);
    nc.AddScalarVariable("Lnorm", "Length normalization", "m", 1 / p.a);
    nc.AddScalarVariable("n0", "Density normalization", "m^-3", p.n0);
    nc.AddScalarVariable("T0", "Temperature normalization", "J", p.T0);
    nc.AddScalarVariable("B0", "Reference magnetic field", "T", p.B0);
    nc.AddScalarVariable("IRadial", "Radial current", "A", p.IRadial * (p.Plasma->IonMass() * sqrt(p.T0 / p.Plasma->IonMass()) * p.a) * tnorm.val / p.B0);
    nc.AddTimeSeries("Voltage", "Total voltage drop across the plasma", "Volts", initialVoltage);
    nc.AddGroup("MMS", "Manufactured solutions");
    for (int j = 0; j < nVars; ++j)
        nc.AddVariable("MMS", "Var" + std::to_string(j), "Manufactured solution", "-", [&](double V)
                       { return p.InitialFunction(j, V, 0.0).val.val; });
    nc.AddVariable("ShearingRate", "Plasma shearing rate", "-", ShearingRate);
    nc.AddVariable("ElectrostaticPotential", "electrostatic potential (phi0+phi1)", "-", ElectrostaticPotential);

    nc.AddGroup("MomentumFlux", "Separating momentum fluxes");
    nc.AddGroup("ParallelLosses", "Separated parallel losses");
    nc.AddVariable("ParallelLosses", "ElectronParLoss", "Parallel particle losses", "-", ElectronParallelLosses);
    nc.AddVariable("ParallelLosses", "IonParLoss", "Parallel particle losses", "-", IonParallelLosses);
    nc.AddVariable("ParallelLosses", "CentrifugalPotential", "Centrifugal potential", "-", phi);
    nc.AddVariable("ParallelLosses", "AngularMomentumLosses", "Angular momentum loss rate", "-", AngularMomentumLosses);

    // Heat Sources
    nc.AddGroup("Heating", "Separated heating sources");
    nc.AddVariable("Heating", "AlphaHeating", "Alpha heat source", "-", AlphaHeating);
    nc.AddVariable("Heating", "ViscousHeating", "Viscous heat source", "-", ViscousHeating);
    nc.AddVariable("Heating", "RadiationLosses", "Bremsstrahlung heat losses", "-", RadiationLosses);
    nc.AddVariable("Heating", "EnergyExchange", "Collisional ion-electron energy exhange", "-", EnergyExchange);
    nc.AddVariable("Heating", "IonPotentialHeating", "Ion potential heating", "-", IonPotentialHeating);
    nc.AddVariable("Heating", "ElectronPotentialHeating", "Ion potential heating", "-", ElectronPotentialHeating);
    nc.AddVariable("Heating", "ParticleSourceHeating", "Heating due to particle source", "-", ParticleSourceHeating);
};

template <class Plasma>
void writeMirrorDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex, Plasma const &p)
{
    auto nVars = p.nVars;
    auto nAux = p.nAux;
    // lambda wrappers for DGSoln object
    auto L = [&y](double V)
    {
        return y.u(Channel::AngularMomentum)(V);
    };
    auto LPrime = [&y](double V)
    {
        return y.q(Channel::AngularMomentum)(V);
    };
    auto n = [&y](double V)
    {
        return y.u(Channel::Density)(V);
    };
    auto nPrime = [&y](double V)
    {
        return y.q(Channel::Density)(V);
    };
    auto p_i = [&y](double V)
    {
        return (2. / 3.) * y.u(Channel::IonEnergy)(V);
    };
    auto p_e = [&y](double V)
    {
        return (2. / 3.) * y.u(Channel::ElectronEnergy)(V);
    };

    auto u = [&](double V)
    {
        RealVector U(nVars);
        for (Index i = 0; i < nVars; ++i)
            U(i) = y.u(i)(V);

        return U;
    };

    auto q = [&](double V)
    {
        RealVector Q(nVars);
        for (Index i = 0; i < nVars; ++i)
            Q(i) = y.q(i)(V);

        return Q;
    };

    auto aux = [&](double V)
    {
        RealVector Aux(nAux);
        for (Index i = 0; i < nAux; ++i)
            Aux(i) = y.Aux(i)(V);

        return Aux;
    };

    Fn Phi0 = [&](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = p.B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real p0 = p.CentrifugalPotential((Real)V, omega, Ti, Te);

        return p0.val;
    };

    Fn phi = [&](double V)
    {
        if (p.useAmbipolarPhi)
        {
            return Phi0(V) + aux(V)[0].val;
        }
        else
        {
            return Phi0(V);
        }
    };

    auto ShearingRate = [&](double V)
    {
        auto qV = q(V);
        auto uV = u(V);
        Real Ti = 2. / 3. * uV(Channel::IonEnergy) / uV(Channel::Density);
        Real R = p.B->R_V(V, 0.0);
        Real vtheta = 1 / R * uV(Channel::AngularMomentum) / uV(Channel::Density);

        Real dVdR = 1 / p.B->dRdV(V, 0.0);
        Real SR = 1.0 / sqrt(Ti) * (dVdR / uV(Channel::Density) * (qV(Channel::AngularMomentum) - R * vtheta * qV(Channel::Density)) - vtheta);
        return SR.val;
    };

    auto ElectrostaticPotential = [&](double V)
    {
        if (p.useAmbipolarPhi)
        {
            return p.phi0(u(V), (Real)V).val + p_i(V) / n(V) * aux(V)[0].val;
        }
        else
        {
            return p.phi0(u(V), (Real)V).val;
        }
    };

    double initialVoltage = p.Voltage(L, n);

    const std::function<double(const double &)> initialZero = [](const double &V)
    { return 0.0; };

    // Wrap DGApprox with lambdas for heating functions
    Fn ViscousHeating = [&](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real R = p.B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i
        Real dRdV = p.B->dRdV(V, 0.0);
        Real JPrime = R * R * nPrime(V) + 2.0 * dRdV * R * n(V);
        Real dOmegadV = LPrime(V) / J - JPrime * L(V) / (J * J);

        try
        {
            double Heating = p.IonClassicalAngularMomentumFlux(V, n(V), Ti, dOmegadV, 0).val * dOmegadV.val;
            return Heating;
        }
        catch (...)
        {
            return 0.0;
        };
    };

    Fn AlphaHeating = [&](double V)
    {
        double MirrorRatio = p.B->MirrorRatio(V, 0.0).val;

        double Heating = sqrt(1 - 1 / MirrorRatio) * p.Plasma->TotalAlphaPower(n(V), p_i(V)).val;
        return Heating;
    };

    Fn RadiationLosses = [&](double V)
    {
        double Losses = p.Plasma->BremsstrahlungLosses(n(V), p_e(V)).val;

        return Losses;
    };

    Fn ElectronParallelLosses = [&](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = p.B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real Xe = p.Xi_e((Real)V, aux(V)[0], Ti, Te, omega);

        double ParallelLosses = Te.val * (1 + Xe.val) * p.ElectronPastukhovLossRate(V, Xe, n(V), Te).val;

        return ParallelLosses;
    };

    Fn IonParallelLosses = [&](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = p.B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real Xi = p.Xi_i((Real)V, aux(V)[0], Ti, Te, omega);

        double ParallelLosses = Ti.val * (1 + Xi.val) * p.IonPastukhovLossRate(V, Xi, n(V), Te).val;

        return ParallelLosses;
    };

    Fn AngularMomentumLosses = [&](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = p.B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real AngularMomentumPerParticle = L(V) / n(V);

        Real Xi = p.Xi_i((Real)V, aux(V)[0], Ti, Te, omega);

        double ParallelLosses = AngularMomentumPerParticle.val * p.IonPastukhovLossRate(V, Xi, n(V), Te).val;

        return ParallelLosses;
    };

    Fn EnergyExchange = [&](double V)
    {
        return p.Plasma->IonElectronEnergyExchange(n(V), p_e(V), p_i(V), V, 0.0).val;
    };

    Fn IonPotentialHeating = [&](double V)
    {
        return p.IonPotentialHeating(u(V), q(V), aux(V), V).val;
    };

    Fn ElectronPotentialHeating = [&](double V)
    {
        return p.ElectronPotentialHeating(u(V), q(V), aux(V), V).val;
    };

    Fn ParticleSourceHeating = [&](double V)
    {
        Real R = p.B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;
        Real ParticleSourceHeating = 0.5 * omega * omega * R * R * p.ParticleSource(R.val, 0.0);
        return ParticleSourceHeating.val;
    };

    Fn DensitySol = [&](double V)
    { return p.InitialFunction(Channel::Density, V, t).val.val; };
    Fn IonEnergySol = [&](double V)
    { return p.InitialFunction(Channel::IonEnergy, V, t).val.val; };
    Fn ElectronEnergySol = [&](double V)
    { return p.InitialFunction(Channel::ElectronEnergy, V, t).val.val; };
    Fn AngularMomentumSol = [&](double V)
    { return p.InitialFunction(Channel::AngularMomentum, V, t).val.val; };

    // Add the appends for the heating stuff
    nc.AppendToGroup<Fn>("Heating", tIndex, {{"AlphaHeating", AlphaHeating}, {"ViscousHeating", ViscousHeating}, {"RadiationLosses", RadiationLosses}, {"EnergyExchange", EnergyExchange}, {"IonPotentialHeating", IonPotentialHeating}, {"ElectronPotentialHeating", ElectronPotentialHeating}, {"ParticleSourceHeating", ParticleSourceHeating}});

    nc.AppendToGroup<Fn>("ParallelLosses", tIndex, {{"ElectronParLoss", ElectronParallelLosses}, {"IonParLoss", IonParallelLosses}, {"CentrifugalPotential", phi}, {"AngularMomentumLosses", AngularMomentumLosses}});

    nc.AppendToGroup<Fn>("MMS", tIndex, {{"Var0", DensitySol}, {"Var1", IonEnergySol}, {"Var2", ElectronEnergySol}, {"Var3", AngularMomentumSol}});

    nc.AppendToVariable("ElectrostaticPotential", ElectrostaticPotential, tIndex);
    nc.AppendToVariable("ShearingRate", ShearingRate, tIndex);
};
