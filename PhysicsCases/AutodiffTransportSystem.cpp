#include "AutodiffTransportSystem.hpp"
#include "Constants.hpp"
#include <autodiff/forward/dual.hpp>
#include <iostream>
using namespace autodiff;

REGISTER_PHYSICS_IMPL(AutodiffTransportSystem);

enum
{
    Gaussian = 0,
    Dirichlet = 1,
    Cosine = 2,
    Uniform = 3,
};

AutodiffTransportSystem::AutodiffTransportSystem(toml::value const &config)
{

    if (config.count("AutodiffTransportSystem") != 1)
        throw std::invalid_argument("There should be a [AutodiffTransportSystem] section if you are using the AutodiffTransportSystem physics model.");

    auto const &InternalConfig = config.at("AutodiffTransportSystem");
    nVars = toml::find_or(InternalConfig, "nVars", 2);
    // if (config.count("FluxType") != 1)
    //     throw std::invalid_argument("FluxType needs to specified exactly once in Autodiff configuration section");

    std::string FluxType = toml::find_or(InternalConfig, "FluxType", "MatrixFlux");

    fluxObject = AutodiffFlux::InstantiateProblem(FluxType, config, nVars);

    if (fluxObject == nullptr)
    {
        std::cerr << " Could not instantiate a physics model for TransportSystem = " << FluxType << std::endl;
        std::cerr << " Available physics models include: " << std::endl;
        for (auto pair : *AutodiffFlux::map)
        {
            std::cerr << '\t' << pair.first << std::endl;
        }
        std::cerr << std::endl;
    }

    xL = toml::find_or(InternalConfig, "x_L", 0.1);
    xR = toml::find_or(InternalConfig, "x_R", 1.0);

    isUpperDirichlet = toml::find_or(InternalConfig, "isUpperDirichlet", true);
    isLowerDirichlet = toml::find_or(InternalConfig, "isLowerDirichlet", true);

    isTestProblem = toml::find_or(InternalConfig, "isTestProblem", false);

    std::vector<double> InitialHeights_v = toml::find<std::vector<double>>(InternalConfig, "InitialHeights");
    InitialHeights = VectorWrapper(InitialHeights_v.data(), nVars);
    if (!isTestProblem)
    {
        std::string profile = toml::find_or(InternalConfig, "InitialProfile", "Dirichlet");
        InitialProfile = InitialProfiles[profile];
    }

    std::vector<double> uL_v = toml::find<std::vector<double>>(InternalConfig, "uL");
    std::vector<double> uR_v = toml::find<std::vector<double>>(InternalConfig, "uR");
    uR = VectorWrapper(uR_v.data(), nVars);
    uL = VectorWrapper(uL_v.data(), nVars);
}

Vector AutodiffTransportSystem::InitialHeights;

int AutodiffTransportSystem::InitialProfile;

Value AutodiffTransportSystem::LowerBoundary(Index i, Time t) const { return InitialFunction(i, xL, 0.0, uR(i), uL(i), xL, xR).val.val; }

Value AutodiffTransportSystem::UpperBoundary(Index i, Time t) const { return InitialFunction(i, xR, 0.0, uR(i), uL(i), xL, xR).val.val; }

bool AutodiffTransportSystem::isLowerBoundaryDirichlet(Index i) const { return isLowerDirichlet; }

bool AutodiffTransportSystem::isUpperBoundaryDirichlet(Index i) const { return isUpperDirichlet; }

// The same for the flux and source functions -- the vectors have length nVars

Value AutodiffTransportSystem::SigmaFn(Index i, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);

    Value sigma = fluxObject->sigma[i](uw, qw, x, t).val;
    return sigma;
}
Value AutodiffTransportSystem::Sources(Index i, const Values &u, const Values &q, const Values &sigma, Position x, Time t)
{

    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(sigma);
    Value S = fluxObject->source[i](uw, qw, sw, x, t).val;

    if (isTestProblem)
        S += TestSource(i, x, t);
    return S;
}

// We need derivatives of the flux functions
void AutodiffTransportSystem::dSigmaFn_du(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);

    grad = gradient(fluxObject->sigma[i], wrt(uw), at(uw, qw, x, t));
}
void AutodiffTransportSystem::dSigmaFn_dq(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{

    VectorXdual uw(u);
    VectorXdual qw(q);

    grad = gradient(fluxObject->sigma[i], wrt(qw), at(uw, qw, x, t));
}

// and for the sources
void AutodiffTransportSystem::dSources_du(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(nVars);
    for (int j = 0; j < nVars; j++)
        sw(j) = fluxObject->sigma[i](uw, qw, x, t);

    grad = gradient(fluxObject->source[i], wrt(uw), at(uw, qw, sw, x, t));
}
void AutodiffTransportSystem::dSources_dq(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(nVars);
    for (int j = 0; j < nVars; j++)
        sw(j) = fluxObject->sigma[i](uw, qw, x, t);

    grad = gradient(fluxObject->source[i], wrt(qw), at(uw, qw, sw, x, t));
}
void AutodiffTransportSystem::dSources_dsigma(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(nVars);
    for (int j = 0; j < nVars; j++)
        sw(j) = fluxObject->sigma[i](uw, qw, x, t);

    grad = gradient(fluxObject->source[i], wrt(sw), at(uw, qw, sw, x, t));
}

// and initial conditions for u & q
Value AutodiffTransportSystem::InitialValue(Index i, Position x) const
{
    if (isTestProblem)
    {
        double sol = TestDirichlet(i, x, 0.0, UpperBoundary(i, 0.0), LowerBoundary(i, 0.0), xL, xR).val.val;
        return sol; // TestSols[i](x, 0)(0).val;
    }

    else
    {
        return InitialFunction(i, x, 0.0, UpperBoundary(i, 0.0), LowerBoundary(i, 0.0), xL, xR).val.val;
    }
}
Value AutodiffTransportSystem::InitialDerivative(Index i, Position x) const
{

    if (isTestProblem)
    {
        dual2nd pos = x;
        dual2nd t = 0.0;
        double deriv = derivative(TestDirichlet, wrt(pos), at(i, pos, t, UpperBoundary(i, 0.0), LowerBoundary(i, 0.0), xL, xR));
        return deriv;
    }
    else
    {
        dual2nd pos = x;
        dual2nd t = 0.0;
        double deriv = derivative(InitialFunction, wrt(pos), at(i, pos, t, UpperBoundary(i, 0.0), LowerBoundary(i, 0.0), xL, xR));
        return deriv;
    }
}
double AutodiffTransportSystem::TestSource(Index i, Position x, Time t)
{
    dual2nd T = t;
    dual2nd pos = x;
    double u_R = UpperBoundary(i, t);
    double u_L = LowerBoundary(i, t);
    double ut = derivative(TestDirichlet, wrt(T), at(i, pos, T, u_R, u_L, xL, xR));
    VectorXdual q(nVars);
    VectorXdual u(nVars);

    VectorXdual dq(nVars);
    VectorXdual sigma(nVars);

    for (Index j = 0; j < nVars; j++)
    {
        double u_R = UpperBoundary(j, t);
        double u_L = LowerBoundary(j, t);

        auto [q0, q1, q2] = derivatives(TestDirichlet, wrt(pos, pos), at(i, pos, T, u_R, u_L, xL, xR));

        u(j) = q0;
        q(j) = q1;
        dq(j) = q2;
    }

    for (Index j = 0; j < nVars; j++)
    {
        sigma(j) = fluxObject->sigma[i](u, q, x, t);
    }

    Values ugrad = gradient(fluxObject->sigma[i], wrt(u), at(u, q, x, t));
    Values qgrad = gradient(fluxObject->sigma[i], wrt(q), at(u, q, x, t));
    dual xdual = x;
    Values xgrad = gradient(fluxObject->sigma[i], wrt(xdual), at(u, q, xdual, t));
    double uxd = xgrad(0);

    for (Index j = 0; j < nVars; j++)
    {
        uxd += ugrad(j) * q(j).val + qgrad(j) * dq(j).val;
    }

    double S = fluxObject->source[i](u, q, sigma, x, t).val;
    double St = ut + uxd - S;

    return St;
}

dual2nd AutodiffTransportSystem::TestDirichlet(Index i, dual2nd x, dual2nd t, double u_R, double u_L, double x_L, double x_R)
{
    double k = 0.5;
    double shape = 5 * (x_R - x_L);
    dual2nd Center = 0.5 * (x_R + x_L);

    dual2nd a = (asinh(u_L) - asinh(u_R)) / (x_L - x_R);
    dual2nd b = (asinh(u_L) - x_L / x_R * asinh(u_R)) / (a * (x_L / x_R - 1));
    dual2nd c = (M_PI / 2 - 3 * M_PI / 2) / (x_L - x_R);
    dual2nd d = (M_PI / 2 - x_L / x_R * (3 * M_PI / 2)) / (c * (x_L / x_R - 1));

    dual2nd u = sinh(a * (x - b)) - cos(c * (x - d)) * InitialHeights[i] * exp(-k * t) * exp(-1 / shape * (x - Center) * (x - Center));

    //  dual2nd u = -cos(c * (x - d)) * exp(-k * t) * exp(-0.5 * x * x);
    return u;
}

dual2nd AutodiffTransportSystem::InitialFunction(Index i, dual2nd x, dual2nd t, double u_R, double u_L, double x_L, double x_R)
{
    dual2nd c, d;
    dual2nd u = 0;
    dual2nd C = pow(0.5 * (x_R + x_L), 2) / 2;
    double shape = 100; // 10 / (x_R - x_L) * ::log(10);
    switch (InitialProfile)
    {
    case Gaussian:
        u = u_L + InitialHeights[i] * (exp(-(x - C) * (x - C) * shape) - exp(-(x_L - C) * (x_L - C) * shape));
        break;
    case Dirichlet:
        u = TestDirichlet(i, x, t, u_R, u_L, x_L, x_R);
        break;
    case Cosine:

        c = (M_PI / 2 - 3 * M_PI / 2) / (x_L - x_R);
        d = (M_PI / 2 - x_L / x_R * (3 * M_PI / 2)) / (c * (x_L / x_R - 1));

        u = u_L - cos(c * (x - d)) * InitialHeights[i] * exp(-shape * (x - C) * (x - C));

        break;
    case Uniform:
        u = u_L;
        break;
    default:
        break;
    };
    return u;
}
