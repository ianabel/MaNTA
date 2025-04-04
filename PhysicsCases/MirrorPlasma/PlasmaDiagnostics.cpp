#include "../MirrorPlasma.hpp"

// omega & n are callables
template <typename T1, typename T2>
double MirrorPlasma::Voltage(T1 &L_phi, T2 &n)
{
    auto integrator = boost::math::quadrature::gauss<double, 15>();
    auto integrand = [this, &L_phi, &n](double V)
    {
        double R = B->R_V(V, 0.0);
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
    { return InitialValue(Channel::AngularMomentum, V); };
    auto LPrime = [this](double V)
    { return InitialDerivative(Channel::AngularMomentum, V); };
    auto n = [this](double V)
    { return uToDensity(InitialValue(Channel::Density, V)).val; };
    auto nPrime = [this](double V)
    { return qToDensityGradient(InitialDerivative(Channel::Density, V), InitialValue(Channel::Density, V)).val; };
    auto p_i = [this](double V)
    { return (2. / 3.) * InitialValue(Channel::IonEnergy, V); };
    auto p_i_prime = [this](double V)
    { return (2. / 3.) * InitialDerivative(Channel::IonEnergy, V); };
    auto p_e = [this](double V)
    { return (2. / 3.) * InitialValue(Channel::ElectronEnergy, V); };
    auto p_e_prime = [this](double V)
    { return (2. / 3.) * InitialDerivative(Channel::ElectronEnergy, V); };

    auto Te = [&](double V)
    { return p_e(V) / n(V); };
    auto Ti = [&](double V)
    { return p_i(V) / n(V); };

    auto Ti_prime = [&](double V)
    {
        return (p_i_prime(V) - nPrime(V) * Ti(V)) / n(V);
    };
    auto Te_prime = [&](double V)
    {
        return (p_e_prime(V) - nPrime(V) * Te(V)) / n(V);
    };

    auto omega = [&](double V)
    {
        Position R = B->R_V(V, 0.0);
        Value J = n(V) * R * R;
        return L(V) / J;
    };
    auto dOmegadV = [&](double V)
    {
        double R = B->R_V(V, 0.0);
        double J = n(V) * R * R; // Normalisation includes the m_i

        double dRdV = B->dRdV(V, 0.0);
        double JPrime = R * R * nPrime(V) + 2.0 * dRdV * R * n(V);
        return LPrime(V) / J - JPrime * L(V) / (J * J);
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

    Fn ShearingRate = [&](double V)
    {
        double dRdV = B->dRdV(V, 0.0);
        double R = B->R_V(V, 0.0);
        // double dOmegadR = dOmegadV(V) / dRdV;
        // return sqrt(Te(V))*B->R_V(V)/omega*dOmegadR;
        // double gradV = B->R_V(V, 0.0) * dOmegadV(V) / dRdV + omega(V);

        return R * R / (sqrt(Te(V)))*dOmegadV(V) / dRdV; // sqrt(Te(V)) * dOmegadV(V) / (dRdV * omega(V));
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
    Fn ViscousHeating = [&](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real R = this->B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i
        Real dRdV = B->dRdV(V, 0.0);
        Real JPrime = R * R * nPrime(V) + 2.0 * dRdV * R * n(V);
        Real dOmegadV = LPrime(V) / J - JPrime * L(V) / (J * J);

        try
        {
            double Heating = this->IonClassicalAngularMomentumFlux(V, n(V), Ti, omega(V), dOmegadV, 0).val * dOmegadV.val;
            return Heating;
        }
        catch (...)
        {
            return 0.0;
        };
    };

    Fn AlphaHeating = [this, &n, &p_i](double V)
    {
        double MirrorRatio = this->B->MirrorRatio(V, 0.0);

        double Heating = sqrt(1 - 1 / MirrorRatio) * Plasma->TotalAlphaPower(n(V), p_i(V)).val;
        return Heating;
    };

    Fn RadiationLosses = [this, &n, &p_e](double V)
    {
        double Losses = Plasma->BremsstrahlungLosses(n(V), p_e(V)).val;

        return Losses;
    };

    Fn ParallelLosses = [&](double V)
    {
        double Xe = Xi_e(V, aux(V)[0], Ti(V), Te(V), omega(V));

        double ParallelLosses = ElectronPastukhovLossRate(V, Xe, n(V), Te(V)).val;

        return ParallelLosses;
    };

    Fn ElectronParallelHeatLosses = [&](double V)
    {
        double Xe = Xi_e(V, aux(V)[0], Ti(V), Te(V), omega(V));

        double ParallelLosses = Te(V) * (1 + Xe) * ElectronPastukhovLossRate(V, Xe, n(V), Te(V)).val;

        return ParallelLosses;
    };

    Fn IonParallelHeatLosses = [&](double V)
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

    Fn DensityArtificialDiffusion = [&](double V)
    {
        double GeometricFactor = (B->VPrime(V) * B->R_V(V, 0.0));

        // double Chi_n = 1.0; // sqrt(Plasma->mu()) * pow(Ti(V), 3. / 2.) * lambda_n;

        // double x = sqrt(Ti(V)) * lambda_n / (lowNThreshold / Plasma->RhoStarRef()) - 1.0;

        double rhon = abs(sqrt(Ti(V)) * Plasma->RhoStarRef() * nPrime(V) / B->dRdV(V, 0.0) / n(V));
        // Real lambda_n = (B->L_V(V) * (R_Upper - R_Lower)) * abs(nPrime);

        double x = (rhon - lowNThreshold) / lowNThreshold;

        if (x > 0)
            return x * lowNDiffusivity * nPrime(V) * GeometricFactor * B->R_V(V, 0.0);
        else
            return 0.0;
    };

    Fn IonPressureArtificialDiffusion = [&](double V)
    {
        double GeometricFactor = (B->VPrime(V) * B->R_V(V, 0.0));
        // double lambda_T = 1 / B->dRdV(V, 0.0) * abs(Ti_prime(V) / sqrt(Ti(V)));

        // double x = lambda_T / (lowPThreshold / Plasma->RhoStarRef()) - 1.0;

        double rhoTi = abs(sqrt(Ti(V)) * Plasma->RhoStarRef() * Ti_prime(V) / B->dRdV(V, 0.0) / Ti(V));

        double Chi_i = 1.0; // sqrt(Plasma->mu()) * pow(Ti(V), 3. / 2.) * lambda_T;

        double x = (rhoTi - lowPThreshold) / lowPThreshold;

        if (x > 0)
            return x * lowPDiffusivity * Chi_i * GeometricFactor * B->R_V(V, 0.0) * Ti_prime(V);
        else
            return 0.0;
    };
    Fn ElectronPressureArtificialDiffusion = [&](double V)
    {
        // double lambda_T = V * abs(Te_prime(V) / Te(V));
        // double lambda_N = V * abs(nPrime(V) / n(V));
        // double Chi_e = pow(Te(V), 3. / 2.) * abs(1 / B->dRdV(V, 0.0) * Te_prime(V) / Te(V));
        double GeometricFactor = (B->VPrime(V) * B->R_V(V, 0.0));
        return TeDiffusivity * GeometricFactor * Te_prime(V);
        // double x = lambda_T - lowPThreshold * std::max(4.5, 0.8 * lambda_N); //- lowPThreshold * sqrt(Plasma->mu()) / Plasma->RhoStarRef(); //-1 / sqrt(Plasma->mu()) * dvt;
        // if (x > 0)
        //     return SmoothTransition(x, transitionLength, TeDiffusivity) * Te_prime(V) / Te(V);
        // else
        //     return 0.0;
        // return TeDiffusivity * Te_prime(V);
    };
    Fn AngularMomentumArtificialDiffusion = [&](double V)
    {
        double GeometricFactor = (B->VPrime(V) * B->R_V(V, 0.0) * B->R_V(V, 0.0));
        double R = B->R_V(V, 0.0);
        // double dRdV = B->dRdV(V, 0.0);
        // double lambda_omega = 1 / dRdV * abs((2 * R * dRdV * omega(V) + R * R * dOmegadV(V)) / (R * R * omega(V)));
        double rho_omega = abs(sqrt(Ti(V)) * Plasma->RhoStarRef() * dOmegadV(V) / B->dRdV(V, 0.0) / omega(V));
        double x = (rho_omega - lowLThreshold) / lowLThreshold;
        double Chi_L = 1.0; // sqrt(Plasma->mu()) * pow(Ti(V), 3. / 2.) * lambda_omega;

        if (x > 0)
            return x * lowLDiffusivity * GeometricFactor * Chi_L * (R * R * dOmegadV(V));
        else
            return 0.0;
    };

    auto rho_i = [&](double V)
    { return sqrt(Ti(V)) * Plasma->RhoStarRef() / B->B(V, 0.0); };

    auto rho_e = [&](double V)
    { return sqrt(Te(V)) * Plasma->RhoStarRef() / sqrt(Plasma->mu()) / B->B(V, 0.0); };

    auto collisionality = [&](double V)
    { return 1.0 / (Plasma->IonCollisionTime(n(V), Ti(V)) * Plasma->ReferenceIonCollisionTime()) * B->L_V(V) / Plasma->c_s(Te(V)); };

    double Iout = 0;
    if (useConstantVoltage)
        Iout = -InitialScalarValue(Scalar::Current) * I0;
    else
        Iout = IRadial * I0;

    // reference values
    nc.AddScalarVariable("Lnorm", "Length normalization", "m", a);
    nc.AddScalarVariable("n0", "Density normalization", "m^-3", n0);
    nc.AddScalarVariable("T0", "Temperature normalization", "J", T0);
    nc.AddScalarVariable("B0", "Reference magnetic field", "T", B0);
    nc.AddScalarVariable("L_z", "Plasma axial length", "m", a * B->L_V(xL));
    nc.AddTimeSeries("Voltage", "Total voltage drop across the plasma", "Volts", initialVoltage);
    nc.AddTimeSeries("Current", "Radial current through plasma", "A", Iout);
    nc.AddTimeSeries("ComputedCurrent", "Radial current computed from momentum equation", "A", -Iout);
    nc.AddGroup("MMS", "Manufactured solutions");
    for (int j = 0; j < nVars; ++j)
        nc.AddVariable("MMS", "Var" + std::to_string(j), "Manufactured solution", "-", [this, j](double V)
                       { return this->InitialFunction(j, V, 0.0).val.val; });

    nc.AddVariable("B", "Magnetic field", "T", [&](double V)
                   { return B0 * B->B(V, 0.0); });
    nc.AddVariable("Rm", "Mirror Ratio", "-", [&](double V)
                   { return B->MirrorRatio(V, 0.0); });
    nc.AddVariable("R", "R(V)", "m", [&](double V)
                   { return a * B->R_V(V, 0.0); });
    nc.AddVariable("L", "Fieldline length", "m", [&](double V)
                   { return a * B->L_V(V); });

    nc.AddGroup("GradientScaleLengths", "Gradient scale lengths");
    nc.AddVariable("GradientScaleLengths", "Ln", "", "m", [&](double V)
                   { return a * B->dRdV(V, 0.0) * n(V) / nPrime(V); });
    nc.AddVariable("GradientScaleLengths", "Lpi", "", "m", [&](double V)
                   { return a * B->dRdV(V, 0.0) * p_i(V) / p_i_prime(V); });
    nc.AddVariable("GradientScaleLengths", "Lpe", "", "m", [&](double V)
                   { return a * B->dRdV(V, 0.0) * p_e(V) / p_e_prime(V); });
    nc.AddVariable("GradientScaleLengths", "LTi", "", "m", [&](double V)
                   { return a * B->dRdV(V, 0.0) * Ti(V) / Ti_prime(V); });
    nc.AddVariable("GradientScaleLengths", "LTe", "", "m", [&](double V)
                   { return a * B->dRdV(V, 0.0) * Te(V) / Te_prime(V); });
    nc.AddVariable("GradientScaleLengths", "LL", "", "m", [&](double V)
                   { return a * B->dRdV(V, 0.0) * L(V) / LPrime(V); });

    nc.AddGroup("DimensionlessNumbers", "Useful dimensionless values");
    nc.AddVariable("DimensionlessNumbers", "eta_e", "L_N/L_T", "-", [&](double V)
                   { return Te_prime(V) / Te(V) * n(V) / nPrime(V); });
    nc.AddVariable("DimensionlessNumbers", "ShearingRate", "Plasma shearing rate", "m^-1", ShearingRate);
    nc.AddVariable("DimensionlessNumbers", "RhoN", "Gyroradius over the gradient scale length", "", [&](double V)
                   { return rho_i(V) * nPrime(V) / B->dRdV(V, 0.0) / n(V); });
    nc.AddVariable("DimensionlessNumbers", "RhoTi", "Gyroradius over the gradient scale length", "", [&](double V)
                   { return rho_i(V) * Ti_prime(V) / B->dRdV(V, 0.0) / Ti(V); });
    nc.AddVariable("DimensionlessNumbers", "RhoTe", "Gyroradius over the gradient scale length", "", [&](double V)
                   { return rho_e(V) * Te_prime(V) / B->dRdV(V, 0.0) / Te(V); });

    nc.AddVariable("DimensionlessNumbers", "RhoL", "Gyroradius over the gradient scale length", "",
                   [&](double V)
                   {
                       double R = B->R_V(V, 0.0);
                       double dRdV = B->dRdV(V, 0.0);
                       double lambda_omega = (2 * R * dRdV * omega(V) + R * R * dOmegadV(V)) / (R * R * omega(V));

                       return rho_i(V) * lambda_omega;
                   });

    nc.AddVariable("DimensionlessNumbers", "Collisionality", "Plasma collisionality", "-", collisionality);

    nc.AddVariable("ElectrostaticPotential", "electrostatic potential (phi0+phi1)", "-", ElectrostaticPotential);

    // Parallel losses
    nc.AddGroup("ParallelLosses", "Separated parallel losses");
    nc.AddVariable("ParallelLosses", "ParLoss", "Parallel particle losses", "-", ParallelLosses);
    nc.AddVariable("ParallelLosses", "ElectronParLoss", "Parallel heat losses", "-", ElectronParallelHeatLosses);
    nc.AddVariable("ParallelLosses", "IonParLoss", "Parallel heat losses", "-", IonParallelHeatLosses);
    nc.AddVariable("ParallelLosses", "CentrifugalPotential", "Centrifugal potential", "-", phi);
    nc.AddVariable("ParallelLosses", "AngularMomentumLosses", "Angular momentum loss rate", "-", AngularMomentumLosses);

    // Collisions with neutrals
    nc.AddGroup("Neutrals", "Collisional neutral terms");
    nc.AddVariable("Neutrals", "Ionization", "Ionization rate", "m^-3 s^-1", [&](double V)
                   { return Plasma->IonizationRate(n(V), NeutralDensity(B->R_V(V, 0.0), 0.0), L(V) / (n(V) * B->R_V(V, 0.0)), Te(V), Ti(V)).val; });
    nc.AddVariable("Neutrals", "ChargeExchange", "Charge exchange rate", "m^-3 s^-1", [&](double V)
                   { return Plasma->ChargeExchangeLossRate(n(V), NeutralDensity(B->R_V(V, 0.0), 0.0), L(V) / (n(V) * B->R_V(V, 0.0)), Ti(V)).val; });

    // Heat Sources
    nc.AddGroup("Heating", "Separated heating sources");
    nc.AddVariable("Heating", "AlphaHeating", "Alpha heat source", "-", AlphaHeating);
    nc.AddVariable("Heating", "ViscousHeating", "Viscous heat source", "-", ViscousHeating);
    nc.AddVariable("Heating", "RadiationLosses", "Bremsstrahlung heat losses", "-", RadiationLosses);
    nc.AddVariable("Heating", "EnergyExchange", "Collisional ion-electron energy exhange", "-", EnergyExchange);
    nc.AddVariable("Heating", "IonPotentialHeating", "Ion potential heating", "-", IonPotentialHeating);
    nc.AddVariable("Heating", "ElectronPotentialHeating", "Ion potential heating", "-", ElectronPotentialHeating);
    nc.AddVariable("Heating", "CyclotronLosses", "Cyclotron heat losses", "-", [&](double V)
                   { return Plasma->CyclotronLosses(V, n(V), Te(V)).val; });

    nc.AddVariable("dPhi0dV", "Phi0 derivative", "-", [this, &u, &q](double V)
                   { return dphi0dV(u(V), q(V), V).val; });
    nc.AddVariable("dPhi1dV", "Phi1 derivative", "-",
                   [this, &u, &q, &aux](double V)
                   {
                       if (nAux > 0)
                           return dphi1dV(u(V), q(V), aux(V)[0], V).val;
                       else
                           return 0.0;
                   });

    // Artificial Diffusion

    nc.AddGroup("ArtificialDiffusion", "Extra diffusion added at high gradients");

    nc.AddVariable("ArtificialDiffusion", "DensityArtificialDiffusion", "AD for density", "-", DensityArtificialDiffusion);
    nc.AddVariable("ArtificialDiffusion", "IonPressureArtificialDiffusion", "AD for IonPressure", "-", IonPressureArtificialDiffusion);
    nc.AddVariable("ArtificialDiffusion", "ElectronPressureArtificialDiffusion", "AD for ElectionPressure", "-", ElectronPressureArtificialDiffusion);
    nc.AddVariable("ArtificialDiffusion", "AngularMomentumArtificialDiffusion", "AD for AngularMomentum", "-", AngularMomentumArtificialDiffusion);
}

void MirrorPlasma::writeDiagnostics(DGSoln const &y, DGSoln const &dydt, Time t, NetCDFIO &nc, size_t tIndex)
{

    AutodiffTransportSystem::writeDiagnostics(y, dydt, t, nc, tIndex);

    // lambda wrappers for DGSoln object
    auto L = [&y](double V)
    { return y.u(Channel::AngularMomentum)(V); };
    auto LPrime = [&y](double V)
    { return y.q(Channel::AngularMomentum)(V); };
    auto n = [&](double V)
    { return uToDensity(y.u(Channel::Density)(V)).val; };
    auto nPrime = [&](double V)
    { return qToDensityGradient(y.q(Channel::Density)(V), y.u(Channel::Density)(V)).val; };
    auto p_i = [&y](double V)
    { return (2. / 3.) * y.u(Channel::IonEnergy)(V); };
    auto p_e = [&y](double V)
    { return (2. / 3.) * y.u(Channel::ElectronEnergy)(V); };

    auto p_i_prime = [&y](double V)
    { return (2. / 3.) * y.q(Channel::IonEnergy)(V); };
    auto p_e_prime = [&y](double V)
    { return (2. / 3.) * y.q(Channel::ElectronEnergy)(V); };

    auto Te = [&](double V)
    { return p_e(V) / n(V); };
    auto Ti = [&](double V)
    { return p_i(V) / n(V); };

    auto Ti_prime = [&](double V)
    { return (p_i_prime(V) - nPrime(V) * Ti(V)) / n(V); };
    auto Te_prime = [&](double V)
    { return (p_e_prime(V) - nPrime(V) * Te(V)) / n(V); };

    auto omega = [&](double V)
    {
        Position R = B->R_V(V, 0.0);
        Value J = n(V) * R * R;
        return L(V) / J;
    };

    auto dOmegadV = [&](double V)
    {
        double R = B->R_V(V, 0.0);
        double J = n(V) * R * R; // Normalisation includes the m_i

        double dRdV = B->dRdV(V, 0.0);
        double JPrime = R * R * nPrime(V) + 2.0 * dRdV * R * n(V);
        return LPrime(V) / J - JPrime * L(V) / (J * J);
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

        Real R = this->B->R_V(V, 0.0);
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

    Fn ShearingRate = [&](double V)
    {
        double dRdV = B->dRdV(V, 0.0);
        double R = B->R_V(V, 0.0);
        // double dOmegadR = dOmegadV(V) / dRdV;
        // return sqrt(Te(V))*B->R_V(V)/omega*dOmegadR;
        // double gradV = B->R_V(V, 0.0) * dOmegadV(V) / dRdV + omega(V);

        return R * R / (sqrt(Te(V)))*dOmegadV(V) / dRdV;
        // return 1 / a * gradV / sqrt(Ti(V));
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

    double Iout = 0;
    if (useConstantVoltage)
        Iout = -y.Scalar(Scalar::Current) * I0;
    else
        Iout = IRadial * I0;
    nc.AppendToTimeSeries("Current", Iout, tIndex);
    nc.AppendToTimeSeries("ComputedCurrent", TotalCurrent(y, t) * I0, tIndex);

    // Wrap DGApprox with lambdas for heating functions
    Fn ViscousHeating = [&](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real R = this->B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i
        Real dRdV = B->dRdV(V, 0.0);
        Real JPrime = R * R * nPrime(V) + 2.0 * dRdV * R * n(V);
        Real dOmegadV = LPrime(V) / J - JPrime * L(V) / (J * J);

        try
        {
            double Heating = this->IonClassicalAngularMomentumFlux(V, n(V), Ti, omega(V), dOmegadV, t).val * dOmegadV.val;
            return Heating;
        }
        catch (...)
        {
            return 0.0;
        };
    };

    Fn AlphaHeating = [this, &n, &p_i](double V)
    {
        double MirrorRatio = this->B->MirrorRatio(V, 0.0);

        double Heating = sqrt(1 - 1 / MirrorRatio) * Plasma->TotalAlphaPower(n(V), p_i(V)).val;
        return Heating;
    };

    Fn RadiationLosses = [this, &n, &p_e](double V)
    {
        double Losses = Plasma->BremsstrahlungLosses(n(V), p_e(V)).val;

        return Losses;
    };

    Fn ParallelLosses = [&](double V)
    {
        double Xe = Xi_e(V, aux(V)[0].val, Ti(V), Te(V), omega(V));

        double ParallelLosses = ElectronPastukhovLossRate(V, Xe, n(V), Te(V)).val;

        return ParallelLosses;
    };

    Fn ElectronParallelHeatLosses = [this, &n, &p_i, &p_e, &L, &aux](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = this->B->R_V(V, 0.0);
        Real J = n(V) * R * R; // Normalisation includes the m_i

        Real omega = L(V) / J;

        Real Xe = Xi_e((Real)V, aux(V)[0], Ti, Te, omega);

        double ParallelLosses = Te.val * (1 + Xe.val) * ElectronPastukhovLossRate(V, Xe, n(V), Te).val;

        return ParallelLosses;
    };

    Fn IonParallelHeatLosses = [this, &n, &p_i, &p_e, &L, &aux](double V)
    {
        Real Ti = p_i(V) / n(V);
        Real Te = p_e(V) / n(V);

        Real R = this->B->R_V(V, 0.0);
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

        Real R = this->B->R_V(V, 0.0);
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
        Real R = this->B->R_V(V, 0.0);
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

    Fn CyclotronLosses = [&](double V)
    { return Plasma->CyclotronLosses(V, n(V), Te(V)).val; };

    Fn DensityArtificialDiffusion = [&](double V)
    {
        double GeometricFactor = (B->VPrime(V) * B->R_V(V, 0.0));

        // double Chi_n = 1.0; // sqrt(Plasma->mu()) * pow(Ti(V), 3. / 2.) * lambda_n;

        // double x = sqrt(Ti(V)) * lambda_n / (lowNThreshold / Plasma->RhoStarRef()) - 1.0;

        double rhon = abs(sqrt(Ti(V)) * Plasma->RhoStarRef() * nPrime(V) / B->dRdV(V, 0.0) / n(V));
        // Real lambda_n = (B->L_V(V) * (R_Upper - R_Lower)) * abs(nPrime);

        double x = (rhon - lowNThreshold) / lowNThreshold;

        if (x > 0)
            return x * lowNDiffusivity * nPrime(V) * GeometricFactor * B->R_V(V, 0.0);
        else
            return 0.0;
    };

    Fn IonPressureArtificialDiffusion = [&](double V)
    {
        double GeometricFactor = (B->VPrime(V) * B->R_V(V, 0.0));
        // double lambda_T = 1 / B->dRdV(V, 0.0) * abs(Ti_prime(V) / sqrt(Ti(V)));

        // double x = lambda_T / (lowPThreshold / Plasma->RhoStarRef()) - 1.0;

        double rhoTi = abs(sqrt(Ti(V)) * Plasma->RhoStarRef() * Ti_prime(V) / B->dRdV(V, 0.0) / Ti(V));

        double Chi_i = 1.0; // sqrt(Plasma->mu()) * pow(Ti(V), 3. / 2.) * lambda_T;

        double x = (rhoTi - lowPThreshold) / lowPThreshold;

        if (x > 0)
            return x * lowPDiffusivity * Chi_i * GeometricFactor * B->R_V(V, 0.0) * Ti_prime(V);
        else
            return 0.0;
    };
    Fn ElectronPressureArtificialDiffusion = [&](double V)
    {
        // double lambda_T = V * abs(Te_prime(V) / Te(V));
        // double lambda_N = V * abs(nPrime(V) / n(V));
        // double Chi_e = pow(Te(V), 3. / 2.) * abs(1 / B->dRdV(V, 0.0) * Te_prime(V) / Te(V));
        double GeometricFactor = (B->VPrime(V) * B->R_V(V, 0.0));
        return TeDiffusivity * GeometricFactor * Te_prime(V);
        // double x = lambda_T - lowPThreshold * std::max(4.5, 0.8 * lambda_N); //- lowPThreshold * sqrt(Plasma->mu()) / Plasma->RhoStarRef(); //-1 / sqrt(Plasma->mu()) * dvt;
        // if (x > 0)
        //     return SmoothTransition(x, transitionLength, TeDiffusivity) * Te_prime(V) / Te(V);
        // else
        //     return 0.0;
        // return TeDiffusivity * Te_prime(V);
    };
    Fn AngularMomentumArtificialDiffusion = [&](double V)
    {
        double GeometricFactor = (B->VPrime(V) * B->R_V(V, 0.0) * B->R_V(V, 0.0));
        double R = B->R_V(V, 0.0);
        // double dRdV = B->dRdV(V, 0.0);
        // double lambda_omega = 1 / dRdV * abs((2 * R * dRdV * omega(V) + R * R * dOmegadV(V)) / (R * R * omega(V)));
        double rho_omega = abs(sqrt(Ti(V)) * Plasma->RhoStarRef() * dOmegadV(V) / B->dRdV(V, 0.0) / omega(V));
        double x = (rho_omega - lowLThreshold) / lowLThreshold;
        double Chi_L = 1.0; // sqrt(Plasma->mu()) * pow(Ti(V), 3. / 2.) * lambda_omega;

        if (x > 0)
            return x * lowLDiffusivity * GeometricFactor * Chi_L * (R * R * dOmegadV(V));
        else
            return 0.0;
    };
    auto rho_i = [&](double V)
    { return sqrt(Ti(V)) * Plasma->RhoStarRef() / B->B(V, 0.0); };

    auto rho_e = [&](double V)
    { return sqrt(Te(V)) * Plasma->RhoStarRef() / sqrt(Plasma->mu()) / B->B(V, 0.0); };

    auto collisionality = [&](double V)
    { return 1.0 / (Plasma->IonCollisionTime(n(V), Ti(V)) * Plasma->ReferenceIonCollisionTime()) * B->L_V(V) / Plasma->c_s(Te(V)); };

    Fn DensitySol = [this, t](double V)
    { return this->InitialFunction(Channel::Density, V, t).val.val; };
    Fn IonEnergySol = [this, t](double V)
    { return this->InitialFunction(Channel::IonEnergy, V, t).val.val; };
    Fn ElectronEnergySol = [this, t](double V)
    { return this->InitialFunction(Channel::ElectronEnergy, V, t).val.val; };
    Fn AngularMomentumSol = [this, t](double V)
    { return this->InitialFunction(Channel::AngularMomentum, V, t).val.val; };

    // Add the appends for the heating stuff
    nc.AppendToGroup<Fn>("Heating", tIndex,
                         {{"AlphaHeating", AlphaHeating},
                          {"ViscousHeating", ViscousHeating},
                          {"RadiationLosses", RadiationLosses},
                          {"EnergyExchange", EnergyExchange},
                          {"IonPotentialHeating", IonPotentialHeating},
                          {"ElectronPotentialHeating", ElectronPotentialHeating},
                          {"CyclotronLosses", CyclotronLosses}});

    nc.AppendToGroup<Fn>("ParallelLosses", tIndex,
                         {{"ParLoss", ParallelLosses},
                          {"ElectronParLoss", ElectronParallelHeatLosses},
                          {"IonParLoss", IonParallelHeatLosses},
                          {"CentrifugalPotential", phi},
                          {"AngularMomentumLosses", AngularMomentumLosses}});

    nc.AppendToGroup<Fn>("MMS", tIndex, {{"Var0", DensitySol}, {"Var1", IonEnergySol}, {"Var2", ElectronEnergySol}, {"Var3", AngularMomentumSol}});

    nc.AppendToGroup("Neutrals", tIndex, "Ionization", [&](double V)
                     { return Plasma->IonizationRate(n(V), NeutralDensity(B->R_V(V, 0.0), t), L(V) / (n(V) * B->R_V(V, 0.0)), Te(V), Ti(V)).val; });
    nc.AppendToGroup("Neutrals", tIndex, "ChargeExchange", [&](double V)
                     { return Plasma->ChargeExchangeLossRate(n(V), NeutralDensity(B->R_V(V, 0.0), t), L(V) / (n(V) * B->R_V(V, 0.0)), Ti(V)).val; });

    nc.AppendToGroup("GradientScaleLengths", tIndex, "Ln", [&](double V)
                     { return a * B->dRdV(V, 0.0) * n(V) / nPrime(V); });
    nc.AppendToGroup("GradientScaleLengths", tIndex, "Lpi", [&](double V)
                     { return a * B->dRdV(V, 0.0) * p_i(V) / p_i_prime(V); });
    nc.AppendToGroup("GradientScaleLengths", tIndex, "Lpe", [&](double V)
                     { return a * B->dRdV(V, 0.0) * p_e(V) / p_e_prime(V); });
    nc.AppendToGroup("GradientScaleLengths", tIndex, "LTi", [&](double V)
                     { return a * B->dRdV(V, 0.0) * Ti(V) / Ti_prime(V); });
    nc.AppendToGroup("GradientScaleLengths", tIndex, "LTe", [&](double V)
                     { return a * B->dRdV(V, 0.0) * Te(V) / Te_prime(V); });
    nc.AppendToGroup("GradientScaleLengths", tIndex, "LL", [&](double V)
                     { return a * B->dRdV(V, 0.0) * L(V) / LPrime(V); });

    // Dimensionless numbers
    nc.AppendToGroup("DimensionlessNumbers", tIndex, "eta_e", [&](double V)
                     { return Te_prime(V) / Te(V) * n(V) / nPrime(V); });
    nc.AppendToGroup("DimensionlessNumbers", tIndex, "ShearingRate", ShearingRate);

    nc.AppendToGroup("DimensionlessNumbers", tIndex, "RhoN", [&](double V)
                     { return rho_i(V) * nPrime(V) / B->dRdV(V, 0.0) / n(V); });
    nc.AppendToGroup("DimensionlessNumbers", tIndex, "RhoTi", [&](double V)
                     { return rho_i(V) * Ti_prime(V) / B->dRdV(V, 0.0) / Ti(V); });
    nc.AppendToGroup("DimensionlessNumbers", tIndex, "RhoTe", [&](double V)
                     { return rho_e(V) * Te_prime(V) / B->dRdV(V, 0.0) / Te(V); });

    nc.AppendToGroup("DimensionlessNumbers", tIndex, "RhoL",
                     [&](double V)
                     {
                         double R = B->R_V(V, 0.0);
                         double dRdV = B->dRdV(V, 0.0);
                         double lambda_omega = (2 * R * dRdV * omega(V) + R * R * dOmegadV(V)) / (R * R * omega(V));

                         return rho_i(V) * lambda_omega;
                     });
    nc.AppendToGroup("DimensionlessNumbers", tIndex, "Collisionality", collisionality);

    nc.AppendToVariable("dPhi0dV", [this, &u, &q](double V)
                        { return dphi0dV(u(V), q(V), V).val; }, tIndex);

    nc.AppendToVariable("dPhi1dV", [this, &u, &q, &aux](double V)
                        {
		if (nAux > 0)
			return dphi1dV(u(V), q(V), aux(V)[0], V).val;
		else
			return 0.0; }, tIndex);
    nc.AppendToVariable("ElectrostaticPotential", ElectrostaticPotential, tIndex);

    nc.AppendToGroup<Fn>("ArtificialDiffusion", tIndex,
                         {{"DensityArtificialDiffusion", DensityArtificialDiffusion},
                          {"IonPressureArtificialDiffusion", IonPressureArtificialDiffusion},
                          {"ElectronPressureArtificialDiffusion", ElectronPressureArtificialDiffusion},
                          {"AngularMomentumArtificialDiffusion", AngularMomentumArtificialDiffusion}});
}
