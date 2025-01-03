#include "../MirrorPlasma.hpp"

// omega & n are callables
template <typename T1, typename T2>
double MirrorPlasma::Voltage(T1 &L_phi, T2 &n)
{
    auto integrator = boost::math::quadrature::gauss<double, 15>();
    auto integrand = [this, &L_phi, &n](double V)
    {
        double R = B->R_V(V);
        return L_phi(V) / (n(V) * R * R * B->VPrime(V));
    };
    double cs0 = std::sqrt(T0 / Plasma->IonMass());
    return cs0 * integrator.integrate(integrand, xL, xR);
}

void MirrorPlasma::initialiseDiagnostics(NetCDFIO &nc)
{

    AutodiffTransportSystem::initialiseDiagnostics(nc);
    // Add diagnostics here
    //
    double TauNorm = Plasma->NormalizingTime(); //(IonMass / ElectronMass) * (1.0 / (RhoStar * RhoStar)) * (ReferenceElectronCollisionTime());
    nc.AddScalarVariable("Tau", "Normalising time", "s", TauNorm);

    // lambda wrappers for DGSoln object

    auto L = [this](double V)
    {
        return InitialValue(Channel::AngularMomentum, V);
    };
    auto LPrime = [this](double V)
    {
        return InitialDerivative(Channel::AngularMomentum, V);
    };
    auto n = [this](double V)
    {
        return uToDensity(InitialValue(Channel::Density, V)).val;
    };
    auto nPrime = [this](double V)
    {
        return qToDensityGradient(InitialDerivative(Channel::Density, V), InitialValue(Channel::Density, V)).val;
    };
    auto p_i = [this](double V)
    {
        return (2. / 3.) * InitialValue(Channel::IonEnergy, V);
    };
    auto p_e = [this](double V)
    {
        return (2. / 3.) * InitialValue(Channel::ElectronEnergy, V);
    };

    auto Te = [&](double V)
    {
        return p_e(V) / n(V);
    };
    auto Ti = [&](double V)
    {
        return p_i(V) / n(V);
    };

    auto omega = [&](double V)
    {
        Position R = B->R_V(V);
        Value J = n(V) * R * R;
        return L(V) / J;
    };

    auto u = [this](double V)
    {
        RealVector U(nVars);
        for (Index i = 0; i < nVars; ++i)
            U(i) = InitialValue(i, V);

        return U;
    };

    auto q = [this](double V)
    {
        RealVector Q(nVars);
        for (Index i = 0; i < nVars; ++i)
            Q(i) = InitialDerivative(i, V);

        return Q;
    };

    auto aux = [this](double V)
    {
        Values Aux(nAux);
        for (Index i = 0; i < nAux; ++i)
            Aux(i) = AutodiffTransportSystem::InitialAuxValue(i, V);

        return Aux;
    };

    Fn Phi0 = [&](double V)
    {
        return CentrifugalPotential(V, omega(V), Ti(V), Te(V));
    };

    Fn phi = [this, &Phi0](double V)
    {
        if (useAmbipolarPhi)
        {
            return Phi0(V) + AutodiffTransportSystem::InitialAuxValue(0, V);
        }
        else
        {
            return Phi0(V);
        }
    };

    auto ShearingRate = [this, &u, &q](double V)
    {
        auto qV = q(V);
        auto uV = u(V);
        Real Ti = 2. / 3. * uV(Channel::IonEnergy) / uV(Channel::Density);
        double R = B->R_V(V);
        Real vtheta = 1 / R * uV(Channel::AngularMomentum) / uV(Channel::Density);

        double dVdR = 1 / B->dRdV(V);
        Real SR = 1.0 / sqrt(Ti) * (dVdR / uV(Channel::Density) * (qV(Channel::AngularMomentum) - R * vtheta * qV(Channel::Density)) - vtheta);
        return SR.val;
    };

    auto ElectrostaticPotential = [this, &u, &aux, &p_i, &n](double V)
    {
        if (useAmbipolarPhi)
        {
            return phi0(u(V), (Real)V).val + p_i(V) / n(V) * aux(V)[0];
        }
        else
        {
            return phi0(u(V), (Real)V).val;
        }
    };

    double initialVoltage = Voltage(L, n);

    const std::function<double(const double &)> initialZero = [](const double &V)
    { return 0.0; };

    // Wrap DGApprox with lambdas for heating functions
    Fn ViscousHeating = [this, &n, &nPrime, &L, &LPrime, &p_i](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real R = this->B->R_V(V);
        Real J = n(V) * R * R; // Normalisation includes the m_i
        Real dRdV = B->dRdV(V);
        Real JPrime = R * R * nPrime(V) + 2.0 * dRdV * R * n(V);
        Real dOmegadV = LPrime(V) / J - JPrime * L(V) / (J * J);

        try
        {
            double Heating = this->IonClassicalAngularMomentumFlux(V, n(V), Ti, dOmegadV, 0).val * dOmegadV.val;
            return Heating;
        }
        catch (...)
        {
            return 0.0;
        };
    };

    Fn AlphaHeating = [this, &n, &p_i](double V)
    {
        double MirrorRatio = this->B->MirrorRatio(V);

        double Heating = sqrt(1 - 1 / MirrorRatio) * Plasma->TotalAlphaPower(n(V), p_i(V)).val;
        return Heating;
    };

    Fn RadiationLosses = [this, &n, &p_e](double V)
    {
        double Losses = Plasma->BremsstrahlungLosses(n(V), p_e(V)).val;

        return Losses;
    };

    Fn ElectronParallelLosses = [&](double V)
    {
        double Xe = Xi_e(V, aux(V)[0], Ti(V), Te(V), omega(V));

        double ParallelLosses = Te(V) * (1 + Xe) * ElectronPastukhovLossRate(V, Xe, n(V), Te(V)).val;

        return ParallelLosses;
    };

    Fn IonParallelLosses = [&](double V)
    {
        double Xi = Xi_i(V, aux(V)[0], Ti(V), Te(V), omega(V));

        double ParallelLosses = Ti(V) * (1 + Xi) * IonPastukhovLossRate(V, Xi, n(V), Te(V)).val;

        return ParallelLosses;
    };

    Fn AngularMomentumLosses = [&](double V)
    {
        Real AngularMomentumPerParticle = L(V) / n(V);

        double Xi = Xi_i(V, aux(V)[0], Ti(V), Te(V), omega(V));

        double ParallelLosses = AngularMomentumPerParticle.val * IonPastukhovLossRate(V, Xi, n(V), Te(V)).val;

        return ParallelLosses;
    };

    Fn EnergyExchange = [this, &n, &p_i, &p_e](double V)
    {
        return Plasma->IonElectronEnergyExchange(n(V), p_e(V), p_i(V), V, 0.0).val;
    };

    Fn IonPotentialHeating = [&](double V)
    {
        Real S = -Gamma(u(V), q(V), V, 0.0) * (omega(V) * omega(V) / (2 * pi * a) - dphidV(u(V), q(V), aux(V), V));

        return S.val;
    };

    Fn ElectronPotentialHeating = [&](double V)
    {
        Real S = -Gamma(u(V), q(V), V, 0.0) * (dphidV(u(V), q(V), aux(V), V));

        return S.val;
    };

    // Real tnorm = n0 * TauNorm; // n0 * T0 * a * B0 * B0 / (electronMass * Om_e(B0) * Om_e(B0) * tau_e(n0, n0 * T0));
    double omega0 = 1 / a * sqrt(T0 / Plasma->IonMass());

    // reference values
    nc.AddScalarVariable("tnorm", "time normalization", "s", TauNorm);
    nc.AddScalarVariable("Lnorm", "Length normalization", "m", 1 / a);
    nc.AddScalarVariable("n0", "Density normalization", "m^-3", n0);
    nc.AddScalarVariable("T0", "Temperature normalization", "J", T0);
    nc.AddScalarVariable("B0", "Reference magnetic field", "T", B0);
    nc.AddScalarVariable("IRadial", "Radial current", "A", IRadial);
    nc.AddTimeSeries("Voltage", "Total voltage drop across the plasma", "Volts", initialVoltage);
    nc.AddGroup("MMS", "Manufactured solutions");
    for (int j = 0; j < nVars; ++j)
        nc.AddVariable("MMS", "Var" + std::to_string(j), "Manufactured solution", "-", [this, j](double V)
                       { return this->InitialFunction(j, V, 0.0).val.val; });
    nc.AddVariable("ShearingRate", "Plasma shearing rate", "-", ShearingRate);
    nc.AddVariable("ElectrostaticPotential", "electrostatic potential (phi0+phi1)", "-", ElectrostaticPotential);

    // Parallel losses
    nc.AddGroup("ParallelLosses", "Separated parallel losses");
    nc.AddVariable("ParallelLosses", "ElectronParLoss", "Parallel particle losses", "-", ElectronParallelLosses);
    nc.AddVariable("ParallelLosses", "IonParLoss", "Parallel particle losses", "-", IonParallelLosses);
    nc.AddVariable("ParallelLosses", "CentrifugalPotential", "Centrifugal potential", "-", phi);
    nc.AddVariable("ParallelLosses", "AngularMomentumLosses", "Angular momentum loss rate", "-", AngularMomentumLosses);

    // Collisions with neutrals
    nc.AddGroup("Neutrals", "Collisional neutral terms");
    nc.AddVariable("Neutrals", "Ionization", "Ionization rate", "m^-3 s^-1", [&](double V)
                   { return Plasma->IonizationRate(n(V), NeutralDensity(B->R(V), 0.0), L(V) / (n(V) * B->R_V(V)), Te(V), Ti(V)).val; });
    nc.AddVariable("Neutrals", "ChargeExchange", "Charge exchange rate", "m^-3 s^-1", [&](double V)
                   { return Plasma->ChargeExchangeLossRate(n(V), NeutralDensity(B->R(V), 0.0), L(V) / (n(V) * B->R_V(V)), Ti(V)).val; });

    // Heat Sources
    nc.AddGroup("Heating", "Separated heating sources");
    nc.AddVariable("Heating", "AlphaHeating", "Alpha heat source", "-", AlphaHeating);
    nc.AddVariable("Heating", "ViscousHeating", "Viscous heat source", "-", ViscousHeating);
    nc.AddVariable("Heating", "RadiationLosses", "Bremsstrahlung heat losses", "-", RadiationLosses);
    nc.AddVariable("Heating", "EnergyExchange", "Collisional ion-electron energy exhange", "-", EnergyExchange);
    nc.AddVariable("Heating", "IonPotentialHeating", "Ion potential heating", "-", IonPotentialHeating);
    nc.AddVariable("Heating", "ElectronPotentialHeating", "Ion potential heating", "-", ElectronPotentialHeating);
    nc.AddVariable("dPhi0dV", "Phi0 derivative", "-", [this, &u, &q](double V)
                   { return dphi0dV(u(V), q(V), V).val; });
    nc.AddVariable("dPhi1dV", "Phi1 derivative", "-", [this, &u, &q, &aux](double V)
                   {
		if (nAux > 0)
			return dphi1dV(u(V), q(V), aux(V)[0], V).val;
		else
			return 0.0; });
}

void MirrorPlasma::writeDiagnostics(DGSoln const &y, DGSoln const &dydt, Time t, NetCDFIO &nc, size_t tIndex)
{

    AutodiffTransportSystem::writeDiagnostics(y, dydt, t, nc, tIndex);

    // lambda wrappers for DGSoln object
    auto L = [&y](double V)
    {
        return y.u(Channel::AngularMomentum)(V);
    };
    auto LPrime = [&y](double V)
    {
        return y.q(Channel::AngularMomentum)(V);
    };
    auto n = [&](double V)
    {
        return uToDensity(y.u(Channel::Density)(V)).val;
    };
    auto nPrime = [&](double V)
    {
        return qToDensityGradient(y.q(Channel::Density)(V), y.u(Channel::Density)(V)).val;
    };
    auto p_i = [&y](double V)
    {
        return (2. / 3.) * y.u(Channel::IonEnergy)(V);
    };
    auto p_e = [&y](double V)
    {
        return (2. / 3.) * y.u(Channel::ElectronEnergy)(V);
    };

    auto Te = [&](double V)
    {
        return p_e(V) / n(V);
    };
    auto Ti = [&](double V)
    {
        return p_i(V) / n(V);
    };

    auto u = [this, &y](double V)
    {
        RealVector U(nVars);
        for (Index i = 0; i < nVars; ++i)
            U(i) = y.u(i)(V);

        return U;
    };

    auto q = [this, &y](double V)
    {
        RealVector Q(nVars);
        for (Index i = 0; i < nVars; ++i)
            Q(i) = y.q(i)(V);

        return Q;
    };

    auto aux = [this, &y](double V)
    {
        RealVector Aux(nAux);
        for (Index i = 0; i < nAux; ++i)
            Aux(i) = y.Aux(i)(V);

        return Aux;
    };

    Fn Phi0 = [this, &n, &p_i, &p_e, &L](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = this->B->R_V(V);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real p0 = CentrifugalPotential((Real)V, omega, Ti, Te) + AmbipolarPhi(V, n(V), Ti, Te) / 2.0;

        return p0.val;
    };

    Fn phi = [this, &Phi0, &y](double V)
    {
        if (useAmbipolarPhi)
        {
            return Phi0(V) + y.Aux(0)(V);
        }
        else
        {
            return Phi0(V);
        }
    };

    auto ShearingRate = [this, &u, &q](double V)
    {
        auto qV = q(V);
        auto uV = u(V);
        Real Ti = 2. / 3. * uV(Channel::IonEnergy) / uV(Channel::Density);
        double R = B->R_V(V);
        Real vtheta = 1 / R * uV(Channel::AngularMomentum) / uToDensity(uV(Channel::Density));

        double dVdR = 1 / B->dRdV(V);
        Real SR = 1.0 / sqrt(Ti) * (dVdR / uV(Channel::Density) * (qV(Channel::AngularMomentum) - R * vtheta * qToDensityGradient(qV(Channel::Density), uV(Channel::Density)) - vtheta));
        return SR.val;
    };

    auto ElectrostaticPotential = [this, &u, &aux, &p_i, &n](double V)
    {
        if (useAmbipolarPhi)
        {
            return phi0(u(V), (Real)V).val + p_i(V) / n(V) * aux(V)[0].val;
        }
        else
        {
            return phi0(u(V), (Real)V).val;
        }
    };

    double voltage = Voltage(L, n);
    nc.AppendToTimeSeries("Voltage", voltage, tIndex);

    // Wrap DGApprox with lambdas for heating functions
    Fn ViscousHeating = [this, &n, &nPrime, &L, &LPrime, &p_i, &t](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real R = this->B->R_V(V);
        Real J = n(V) * R * R; // Normalisation includes the m_i
        Real dRdV = B->dRdV(V);
        Real JPrime = R * R * nPrime(V) + 2.0 * dRdV * R * n(V);
        Real dOmegadV = LPrime(V) / J - JPrime * L(V) / (J * J);

        try
        {
            double Heating = this->IonClassicalAngularMomentumFlux(V, n(V), Ti, dOmegadV, t).val * dOmegadV.val;
            return Heating;
        }
        catch (...)
        {
            return 0.0;
        };
    };

    Fn AlphaHeating = [this, &n, &p_i](double V)
    {
        double MirrorRatio = this->B->MirrorRatio(V);

        double Heating = sqrt(1 - 1 / MirrorRatio) * Plasma->TotalAlphaPower(n(V), p_i(V)).val;
        return Heating;
    };

    Fn RadiationLosses = [this, &n, &p_e](double V)
    {
        double Losses = Plasma->BremsstrahlungLosses(n(V), p_e(V)).val;

        return Losses;
    };

    Fn ElectronParallelLosses = [this, &n, &p_i, &p_e, &L, &aux](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = this->B->R_V(V);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real Xe = Xi_e((Real)V, aux(V)[0], Ti, Te, omega);

        double ParallelLosses = Te.val * (1 + Xe.val) * ElectronPastukhovLossRate(V, Xe, n(V), Te).val;

        return ParallelLosses;
    };

    Fn IonParallelLosses = [this, &n, &p_i, &p_e, &L, &aux](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = this->B->R_V(V);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real Xi = Xi_i((Real)V, aux(V)[0], Ti, Te, omega);

        double ParallelLosses = Ti.val * (1 + Xi.val) * IonPastukhovLossRate(V, Xi, n(V), Te).val;

        return ParallelLosses;
    };

    Fn AngularMomentumLosses = [this, &n, &p_i, &p_e, &L, &aux](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = this->B->R_V(V);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real AngularMomentumPerParticle = L(V) / n(V);

        Real Xi = Xi_i((Real)V, aux(V)[0], Ti, Te, omega);

        double ParallelLosses = AngularMomentumPerParticle.val * IonPastukhovLossRate(V, Xi, n(V), Te).val;

        return ParallelLosses;
    };

    Fn EnergyExchange = [this, &n, &p_i, &p_e](double V)
    {
        return Plasma->IonElectronEnergyExchange(n(V), p_e(V), p_i(V), V, 0.0).val;
    };

    Fn dPhi0dV = [this, &u, &q](double V)
    {
        return dphi0dV(u(V), q(V), V).val;
    };

    Fn IonPotentialHeating = [this, &u, &q, &n, &L, &aux](double V)
    {
        Real R = this->B->R_V(V);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;
        Real S = -Gamma(u(V), q(V), V, 0.0) * (omega * omega / (2 * pi * a) - dphidV(u(V), q(V), aux(V), V));

        return S.val;
    };

    Fn ElectronPotentialHeating = [this, &u, &q, &aux](double V)
    {
        Real S = -Gamma(u(V), q(V), V, 0.0) * (dphidV(u(V), q(V), aux(V), V));

        return S.val;
    };

    Fn DensitySol = [this, t](double V)
    { return this->InitialFunction(Channel::Density, V, t).val.val; };
    Fn IonEnergySol = [this, t](double V)
    { return this->InitialFunction(Channel::IonEnergy, V, t).val.val; };
    Fn ElectronEnergySol = [this, t](double V)
    { return this->InitialFunction(Channel::ElectronEnergy, V, t).val.val; };
    Fn AngularMomentumSol = [this, t](double V)
    { return this->InitialFunction(Channel::AngularMomentum, V, t).val.val; };

    // Add the appends for the heating stuff
    nc.AppendToGroup<Fn>("Heating", tIndex, {{"AlphaHeating", AlphaHeating}, {"ViscousHeating", ViscousHeating}, {"RadiationLosses", RadiationLosses}, {"EnergyExchange", EnergyExchange}, {"IonPotentialHeating", IonPotentialHeating}, {"ElectronPotentialHeating", ElectronPotentialHeating}});

    nc.AppendToGroup<Fn>("ParallelLosses", tIndex, {{"ElectronParLoss", ElectronParallelLosses}, {"IonParLoss", IonParallelLosses}, {"CentrifugalPotential", phi}, {"AngularMomentumLosses", AngularMomentumLosses}});

    nc.AppendToGroup<Fn>("MMS", tIndex, {{"Var0", DensitySol}, {"Var1", IonEnergySol}, {"Var2", ElectronEnergySol}, {"Var3", AngularMomentumSol}});

    nc.AppendToGroup("Neutrals", tIndex, "Ionization", [&](double V)
                     { return Plasma->IonizationRate(n(V), NeutralDensity(B->R_V(V), t), L(V) / (n(V) * B->R_V(V)), Te(V), Ti(V)).val; });
    nc.AppendToGroup("Neutrals", tIndex, "ChargeExchange", [&](double V)
                     { return Plasma->ChargeExchangeLossRate(n(V), NeutralDensity(B->R_V(V), t), L(V) / (n(V) * B->R_V(V)), Ti(V)).val; });

    nc.AppendToVariable("dPhi0dV", [this, &u, &q](double V)
                        { return dphi0dV(u(V), q(V), V).val; }, tIndex);

    nc.AppendToVariable("dPhi1dV", [this, &u, &q, &aux](double V)
                        {
		if (nAux > 0)
			return dphi1dV(u(V), q(V), aux(V)[0], V).val;
		else
			return 0.0; }, tIndex);
    nc.AppendToVariable("ElectrostaticPotential", ElectrostaticPotential, tIndex);
    nc.AppendToVariable("ShearingRate", ShearingRate, tIndex);
}
