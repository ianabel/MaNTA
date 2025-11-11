#include "AdjointPlasma.hpp"

REGISTER_PHYSICS_IMPL(AdjointPlasma);

Real H(Real x)
{
    return static_cast<Real>(x > 0.0 ? 1.0 : 0.0);
}

Real G(Real x)
{
    if (x < 0)
        return 0.0;
    Real F = min(x, sqrt(x));
    return static_cast<Real>(F * H(x));
}

const double n_mid = 1.0;
const double n_edge = 0.5;
const double T_mid = 4.0, T_edge = 1.0;

// Fusion yield coeffs
constexpr static double BG = 31.3970;
constexpr static double mc2 = 937814;

constexpr static double C1 = 5.65718e-12;
constexpr static double C2 = 3.41267e-3;
constexpr static double C3 = 1.99167e-3;
constexpr static double C4 = 0.0;
constexpr static double C5 = 1.05060e-5;
constexpr static double C6 = 0.0;
constexpr static double C7 = 0.0;

AdjointPlasma::AdjointPlasma(toml::value const &config, Grid const &grid)
{
    nVars = 2;
    nScalars = 0;
    nAux = 0;

    xL = grid.lowerBoundary();
    xR = grid.upperBoundary();

    if (config.count("AdjointPlasma") == 1)
    {
        auto const &InternalConfig = config.at("AdjointPlasma");

        uL.resize(nVars);
        uR.resize(nVars);

        std::string of = toml::find_or(InternalConfig, "ObjectiveFunction", "StoredEnergy");
        if (objectiveFunctions.contains(of))
            objectiveFunction = objectiveFunctions[of];
        else
            throw std::invalid_argument("Invalid objective function, please use either FusionYield or StoredEnergy");

        EquilibrationFactor = toml::find_or(InternalConfig, "EquilibrationFactor", 1.0);

        SourceCenter = toml::find_or(InternalConfig, "SourceCenter", 0.3);
        SourceStrength = toml::find_or(InternalConfig, "SourceStrength", 10.0);
        SourceWidth = toml::find_or(InternalConfig, "SourceWidth", 0.03);

        HeatFraction = toml::find_or(InternalConfig, "HeatFraction", 0.5);

        //  std::string IonType = toml::find_or(InternalConfig, "IonSpecies", "Deuterium");

        // Plasma = std::make_unique<PlasmaConstants>(IonType, B, n0, T0, Z_eff, a * (R_Upper - R_Lower));
        nEdge = toml::find_or(InternalConfig, "EdgeDensity", n_edge);
        grad_n = toml::find_or(InternalConfig, "DensityGradient", 0.0);
        Value TEdge = toml::find_or(InternalConfig, "EdgeTemperature", T_edge);
        TiEdge = TEdge;
        TeEdge = TEdge;

        Xe_Xi = toml::find_or(InternalConfig, "Xe_Xi", 0.5);
        C = toml::find_or(InternalConfig, "C", 1.0);
        alpha = toml::find_or(InternalConfig, "alpha", 2.0);
        Btor = toml::find_or(InternalConfig, "Btor", 1.0);
        Bpol = toml::find_or(InternalConfig, "Bpol", 0.1);

        R0 = toml::find_or(InternalConfig, "R0", 5.0);
        AspectRatio = toml::find_or(InternalConfig, "AspectRatio", 10.0);
        a = R0 / AspectRatio;
        Chi_min = toml::find_or(InternalConfig, "Chi_min", 0.1);
        nu = toml::find_or(InternalConfig, "nu", 1.0);

        InitialPeakDensity = toml::find_or(InternalConfig, "InitialDensity", n_mid);
        InitialPeakTe = toml::find_or(InternalConfig, "InitialElectronTemperature", T_mid);
        InitialPeakTi = toml::find_or(InternalConfig, "InitialIonTemperature", T_mid);

        uL[Channel::IonEnergy] = (3. / 2.) * InitialPeakDensity * InitialPeakTi;
        uR[Channel::IonEnergy] = (3. / 2.) * nEdge * TiEdge;
        uL[Channel::ElectronEnergy] = (3. / 2.) * InitialPeakDensity * InitialPeakTe;
        uR[Channel::ElectronEnergy] = (3. / 2.) * nEdge * TeEdge;

        if (evolveDensity)
        {
            nVars++;
            uL.push_back(nEdge);
            uR.push_back(nEdge);
        }
    }
    else if (config.count("AdjointPlasma") == 0)
    {
        throw std::invalid_argument("To use the Mirror Plasma physics model, a [MirrorPlasma] configuration section is required.");
    }
    else
    {
        throw std::invalid_argument("Unable to find unique [MirrorPlasma] configuration section in configuration file.");
    }
}

Real2nd AdjointPlasma::InitialFunction(Index i, Real2nd x, Real2nd t) const
{
    // Real2nd m = (uR[i] - uL[i]) / (xR - xL);
    // return m * (x - xL) + uL[i];
    return 3. / 2. * TiEdge * DensityFn(x.val);
}

AdjointProblem *AdjointPlasma::createAdjointProblem()
{
    AutodiffAdjointProblem *p = new AutodiffAdjointProblem(this);
    p->setG([this](Position x, RealVector &u, RealVector &q, RealVector &sigma, RealVector &phi)
            { return g(x, u, q, sigma, phi); });

    addP(grad_n);
    addP(AspectRatio);
    addP(Xe_Xi);
    addP(alpha);

    p->addUpperBoundarySensitivity(Channel::IonEnergy, pvals.size());
    p->addUpperBoundarySensitivity(Channel::ElectronEnergy, pvals.size() + 1);

    p->setNp(pvals.size() + 2);
    return p;
}

Real AdjointPlasma::g(Position x, RealVector &u, RealVector &q, RealVector &sigma, RealVector &)
{
    Real gout = 0;
    switch (static_cast<ObjectiveFunctions>(objectiveFunction))
    {
    case ObjectiveFunctions::StoredEnergy:
    {
        Real pe = 2. / 3. * u(Channel::ElectronEnergy),
             pi = 2. / 3. * u(Channel::IonEnergy);
        // Real p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy);

        gout = 3. / 2. * (pe + pi);
        break;
    }
    case ObjectiveFunctions::FusionYield:
    {
        Real n = DensityFn(x);
        Real pi = 2. / 3. * u(Channel::IonEnergy);
        Real Ti = pi / n;

        Real theta = Ti / (1 - (Ti * (C2 + Ti * (C4 + Ti * C6)) / (1 + Ti * (C3 + Ti * (C5 + Ti * C7)))));
        Real xi = pow(BG * BG / (4 * theta), 1. / 3.);
        Real sigmav = C1 * theta * sqrt(xi / (mc2 * pow(Ti, 3))) * exp(-3 * xi);

        gout = 17.6 * 0.25 * n * n * sigmav * pow(100.0, -3) * 1e40;
        break;
    }
    default:
        break;
    }
    // Real r = sqrt(x); // treat x as r/a for now
    return gout;
}

void AdjointPlasma::initialiseDiagnostics(NetCDFIO &nc)
{
    auto p_i = [this](double x)
    { return (2. / 3.) * InitialValue(Channel::IonEnergy, x); };
    // auto p_i_prime = [this](double x)
    // { return (2. / 3.) * InitialDerivative(Channel::IonEnergy, x); };
    auto p_e = [this](double x)
    { return (2. / 3.) * InitialValue(Channel::ElectronEnergy, x); };
    // auto p_e_prime = [this](double x)
    // { return (2. / 3.) * InitialDerivative(Channel::ElectronEnergy, x); };

    auto T_i = [&](double x)
    { return p_i(x) / DensityFn(x).val; };
    auto T_e = [&](double x)
    { return p_e(x) / DensityFn(x).val; };

    nc.AddGroup("Temperatures", "");
    nc.AddVariable("Temperatures", "Ti", "ion temperature", "-", T_i);
    nc.AddVariable("Temperatures", "Te", "electron temperature", "-", T_e);
    nc.AddVariable("Density", "density function", "-", [this](double x)
                   { return DensityFn(x).val; });
}

void AdjointPlasma::writeDiagnostics(DGSoln const &y, DGSoln const &dydt, Time t, NetCDFIO &nc, size_t tIndex)
{
    auto p_i = [&](double x)
    { return (2. / 3.) * y.u(Channel::IonEnergy)(x); };
    // auto p_i_prime = [this](double x)
    // { return (2. / 3.) * InitialDerivative(Channel::IonEnergy, x); };
    auto p_e = [&](double x)
    { return (2. / 3.) * y.u(Channel::ElectronEnergy)(x); };
    // auto p_e_prime = [this](double x)
    // { return (2. / 3.) * InitialDerivative(Channel::ElectronEnergy, x); };

    Fn T_i = [&](double x)
    { return p_i(x) / DensityFn(x).val; };
    Fn T_e = [&](double x)
    { return p_e(x) / DensityFn(x).val; };

    nc.AppendToGroup<Fn>("Temperatures", tIndex,
                         {{"Ti", T_i}, {"Te", T_e}});
    nc.AppendToVariable("Density", [this](double x)
                        { return DensityFn(x).val; }, tIndex);
}

Real AdjointPlasma::Flux(Index i, RealVector u, RealVector q, Real x, Time t)
{

    switch (static_cast<Channel>(i))
    {
    case Channel::IonEnergy:
    {
        return qi(u, q, x, t);
    }
    case Channel::ElectronEnergy:
    {
        return qe(u, q, x, t);
    }
    case Channel::Density:
    default:
        return 0.0;
    }
}

Real AdjointPlasma::Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector phi, RealVector scalars, Real x, Time t)
{
    switch (static_cast<Channel>(i))
    {
    case Channel::IonEnergy:
    {
        return Spi(u, q, sigma, phi, x, t);
    }
    case Channel::ElectronEnergy:
    {
        return Spe(u, q, sigma, phi, x, t);
    }
    case Channel::Density:
        return Sn(u, q, sigma, phi, x, t);
    default:
        return 0.0;
    }
}

Value AdjointPlasma::LowerBoundary(Index i, Time t) const
{
    return 0.0;
}

Value AdjointPlasma::UpperBoundary(Index i, Time t) const
{
    return uR[i];
}

bool AdjointPlasma::isLowerBoundaryDirichlet(Index i) const
{
    return false;
}

bool AdjointPlasma::isUpperBoundaryDirichlet(Index i) const
{
    return true;
}

Real AdjointPlasma::Gamma(RealVector u, RealVector q, Real x, Time t) const
{
    return 0.0;
}

Real AdjointPlasma::qe(RealVector u, RealVector q, Real x, Time t) const
{
    Real r = sqrt(x);
    Real pe = 2. / 3. * u(Channel::ElectronEnergy),
         pi = 2. / 3. * u(Channel::IonEnergy);
    // Real p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy);
    Real p_i_prime = (2. / 3.) * q(Channel::IonEnergy), p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy);

    Real n = DensityFn(x);
    Real n_prime = DensityPrime(x);
    Real Te = pe / n, Ti = pi / n; // Temps in keV

    Real Te_prime = (p_e_prime - Te * n_prime) / n;

    Real tau = Ti / Te;

    Real R_LTi = -(2 * r * AspectRatio * (p_i_prime / pi - n_prime / n));

    Real GeometricFactor = r * r; // pi * a * Btor * r;

    Real R_LTi_crit = 4 / 3 * (1 + tau) * (1 + 2 * abs(Shear(r)) / SafetyFactor(r));

    Real Chi_i = Chi_min;

    if (R_LTi >= R_LTi_crit)
        Chi_i += pow(Ti, 3. / 2.) * C * pow(R_LTi - R_LTi_crit, alpha);

    Real Chi_e = Xe_Xi * Chi_i;

    Real q_out = GeometricFactor * Chi_e * n * Te_prime;

    // for neumann condition
    if (x <= 0.05)
        q_out += 2.0 * (1 - 1 / 0.05 * x) * Te_prime;

    return q_out;
}

Real AdjointPlasma::qi(RealVector u, RealVector q, Real x, Time t) const
{
    Real r = sqrt(x);
    Real pe = 2. / 3. * u(Channel::ElectronEnergy),
         pi = 2. / 3. * u(Channel::IonEnergy);
    // Real p_e_prime = (2. / 3.) * q(Channel::ElectronEnergy);

    Real n = DensityFn(x);
    Real n_prime = DensityPrime(x);

    Real p_i_prime = (2. / 3.) * q(Channel::IonEnergy);
    Real Te = pe / n, Ti = pi / n; // Temps in keV

    Real Ti_prime = (p_i_prime - n_prime * Ti) / n;

    Real tau = Ti / Te;
    Real R_LTi = -(2 * r * AspectRatio * (p_i_prime / pi - n_prime / n));

    Real GeometricFactor = r * r; // pi * a * Btor * r;

    Real R_LTi_crit = 4 / 3 * (1 + tau) * (1 + 2 * abs(Shear(r)) / SafetyFactor(r));

    Real Chi_i = Chi_min;

    if (R_LTi >= R_LTi_crit)
    {
        // std::cout << "R_LT exceeds critical value at " << r.val << std::endl;
        Chi_i += C * pow(Ti, 3. / 2.) * pow(R_LTi - R_LTi_crit, alpha);
    }
    Real q_out = GeometricFactor * Chi_i * n * Ti_prime;

    // for neumann condition
    if (x <= 0.05)
        q_out += 2.0 * (1 - 1 / 0.05 * x) * Ti_prime;

    return q_out;
}

Real AdjointPlasma::Sn(RealVector u, RealVector q, RealVector sigma, RealVector, Real x, Time t) const
{
    return 0.0;
}

Real AdjointPlasma::Spe(RealVector u, RealVector q, RealVector sigma, RealVector, Real x, Time t) const
{
    Real pe = 2. / 3. * u(Channel::ElectronEnergy), pi = 2. / 3. * u(Channel::IonEnergy);
    Real n = DensityFn(x);
    Real Te = pe / n, Ti = pi / n; // Temps in keV
    Real EnergyExchange = EquilibrationFactor * n * (Ti - Te);
    Real S = SourceStrength * exp(-1. / SourceWidth * pow(x - SourceCenter, 2));
    return EnergyExchange + HeatFraction * S;
}

Real AdjointPlasma::Spi(RealVector u, RealVector q, RealVector sigma, RealVector, Real x, Time t) const
{
    Real pe = 2. / 3. * u(Channel::ElectronEnergy), pi = 2. / 3. * u(Channel::IonEnergy);
    Real n = DensityFn(x);
    Real Te = pe / n, Ti = pi / n; // Temps in keV
    Real EnergyExchange = EquilibrationFactor * n * (Te - Ti);

    Real S = SourceStrength * exp(-1. / SourceWidth * pow(x - SourceCenter, 2));
    return EnergyExchange + (1 - HeatFraction) * S;
}

Real AdjointPlasma::DensityFn(Real x) const
{
    // Real r = sqrt(x);

    // Real b = 0.02;
    // Real d = 20.0;

    // Real exponent = -0.5;
    // Real c = 0.5;

    // Real y = (r - c) / sqrt(b);
    // Real G = (b * d / (4 * a)) * (exp(-pow(1 - c, 2) / b) - exp(-y * y)) + (d * sqrt(b * M_PI) / (4 * a)) * ((c - 1) * erf((c - 1) / sqrt(b)) + (1 - x) * erf(c / sqrt(b)) - (x - c) * erf(y));
    // Real u2 = pow(nEdge, (1 + exponent)) + 2 * (1 + exponent) * G;
    // return pow(u2, (1.0 / (1 + exponent)));
    return nEdge + grad_n * (1 - x);
}

Real AdjointPlasma::DensityPrime(Real x) const
{
    return -grad_n;
}

Real AdjointPlasma::SafetyFactor(Real r) const
{
    return (nu + 1) * Btor / Bpol * 1 / AspectRatio * pow(r, 2) / (1 - pow(1 - r * r, nu + 1));
}
Real AdjointPlasma::Shear(Real r) const
{
    return r / SafetyFactor(r) * autodiff::derivative([this](Real r)
                                                      { return SafetyFactor(r); }, wrt(r), at(r));
}