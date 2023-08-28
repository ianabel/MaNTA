#include "Autodiff3VarCyl.hpp"
#include "Constants.hpp"
#include <autodiff/forward/dual.hpp>
#include <iostream>
using namespace autodiff;

REGISTER_PHYSICS_IMPL(Autodiff3VarCyl);

Autodiff3VarCyl::Autodiff3VarCyl(toml::value const &config)
{
    nVars = 3;

    uR(nVars);
    uL(nVars);

    if (config.count("Autodiff3VarCyl") != 1)
        throw std::invalid_argument("There should be a [Autodiff3VarCyl] section if you are using the Autodiff3VarCyl physics model.");

    auto const &InternalConfig = config.at("Autodiff3VarCyl");

    xL = toml::find_or(InternalConfig, "x_L", 0.1);
    xR = toml::find_or(InternalConfig, "x_R", 1.0);

    isUpperDirichlet = toml::find_or(InternalConfig, "isUpperDirichlet", true);
    isLowerDirichlet = toml::find_or(InternalConfig, "isLowerDirichlet", true);

    double nL = toml::find_or(InternalConfig, "nL", 3e18);
    double nR = toml::find_or(InternalConfig, "nR", 4e18);

    double peL = toml::find_or(InternalConfig, "peL", 2); // keV
    double peR = toml::find_or(InternalConfig, "peR", 2); // keV

    double piL = toml::find_or(InternalConfig, "piL", 2); // keV
    double piR = toml::find_or(InternalConfig, "piR", 2); // keV

    uR << nR, peR, piR;
    uL << nL, peL, piL;
}

Value Autodiff3VarCyl::LowerBoundary(Index i, Time t) const { return uL(i); }

Value Autodiff3VarCyl::UpperBoundary(Index i, Time t) const { return uR(i); }

bool Autodiff3VarCyl::isLowerBoundaryDirichlet(Index i) const { return isLowerDirichlet; }

bool Autodiff3VarCyl::isUpperBoundaryDirichlet(Index i) const { return isUpperDirichlet; }

// The same for the flux and source functions -- the vectors have length nVars

Value Autodiff3VarCyl::SigmaFn(Index i, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);

    Value sigma = sigmaVec[i](uw, qw, x, t).val;
    return sigma;
}
Value Autodiff3VarCyl::Sources(Index i, const Values &u, const Values &q, const Values &sigma, Position x, Time t)
{

    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(sigma);
    Value S = SourceVec[i](uw, qw, sw, x, t).val;

    if (isTestProblem)
        S += TestSource(i, x, t);
    return S;
}

// We need derivatives of the flux functions
void Autodiff3VarCyl::dSigmaFn_du(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);

    grad = gradient(sigmaVec[i], wrt(uw), at(uw, qw, x, t));
}
void Autodiff3VarCyl::dSigmaFn_dq(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{

    VectorXdual uw(u);
    VectorXdual qw(q);

    grad = gradient(sigmaVec[i], wrt(qw), at(uw, qw, x, t));
}

// and for the sources
void Autodiff3VarCyl::dSources_du(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(nVars);
    for (int j = 0; j < nVars; j++)
        sw(j) = sigmaVec[j](uw, qw, x, t);

    grad = gradient(SourceVec[i], wrt(uw), at(uw, qw, sw, x, t));
}
void Autodiff3VarCyl::dSources_dq(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(nVars);
    for (int j = 0; j < nVars; j++)
        sw(j) = sigmaVec[j](uw, qw, x, t);

    grad = gradient(SourceVec[i], wrt(qw), at(uw, qw, sw, x, t));
}
void Autodiff3VarCyl::dSources_dsigma(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(nVars);
    for (int j = 0; j < nVars; j++)
        sw(j) = sigmaVec[j](uw, qw, x, t);

    grad = gradient(SourceVec[i], wrt(sw), at(uw, qw, sw, x, t));
}

// and initial conditions for u & q
Value Autodiff3VarCyl::InitialValue(Index i, Position x) const
{
    if (isTestProblem)
    {
        double sol = TestDirichlet(x, 0.0, UpperBoundary(i, 0.0), LowerBoundary(i, 0.0), xL, xR).val.val;
        return sol; // TestSols[i](x, 0)(0).val;
    }

    else
        return 0;
}
Value Autodiff3VarCyl::InitialDerivative(Index i, Position x) const
{

    if (isTestProblem)
    {
        dual2nd pos = x;
        dual2nd t = 0.0;
        double deriv = derivative(TestDirichlet, wrt(pos), at(x, t, UpperBoundary(i, 0.0), LowerBoundary(i, 0.0), xL, xR));
        return deriv;
    }
    else
    {
        return 0;
    }
}
double Autodiff3VarCyl::TestSource(Index i, Position x, Time t)
{
    dual2nd T = t;
    dual2nd pos = x;

    double ut = derivative(TestDirichlet, wrt(T), at(x, T, UpperBoundary(i, t), LowerBoundary(i, t), xL, xR));

    VectorXdual q(nVars);
    VectorXdual u(nVars);
    VectorXdual dq(nVars);
    VectorXdual sigma(nVars);

    for (Index j = 0; j < nVars; j++)
    {
        auto [q0, q1, q2] = derivatives(TestDirichlet, wrt(pos, pos), at(x, T, UpperBoundary(i, t), LowerBoundary(i, t), xL, xR));

        u(j) = q0;
        q(j) = q1;
        dq(j) = q2;
    }
    for (Index j = 0; j < nVars; j++)
    {
        sigma(j) = sigmaVec[j](u, q, x, t);
    }
    Values ugrad = gradient(sigmaVec[i], wrt(u), at(u, q, x, t));
    Values qgrad = gradient(sigmaVec[i], wrt(q), at(u, q, x, t));
    double uxd = 0.0;

    for (Index j = 0; j < nVars; j++)
    {
        uxd += ugrad(j) * q(j).val + qgrad(j) * dq(j).val;
    }
    double St = ut + uxd - SourceVec[i](u, q, sigma, x, t).val;

    return St;
}

dual2nd Autodiff3VarCyl::TestDirichlet(dual2nd x, dual2nd t, double u_R, double u_L, double x_L, double x_R)
{
    double k = 0.5;

    dual2nd a = (asinh(u_L) - asinh(u_R)) / (x_L - x_R);
    dual2nd b = (asinh(u_L) - x_L / x_R * asinh(u_R)) / (a * (x_L / x_R - 1));
    dual2nd c = (M_PI / 2 - 3 * M_PI / 2) / (x_L - x_R);
    dual2nd d = (M_PI / 2 - x_L / x_R * (3 * M_PI / 2)) / (c * (x_L / x_R - 1));

    dual2nd u = sinh(a * (x - b)) - cos(c * (x - d)) * u_L * exp(-k * t) * exp(-0.5 * x * x);

    return u;
}

const double B_mid = 0.3; // Tesla
const double Om_i = e_charge * B_mid / ionMass;
const double Om_e = e_charge * B_mid / electronMass;
const double lambda = 15.0;

double tau_i(Value n, Value Pi)
{
    if (Pi > 0)
        return 3.44e11 * (1.0 / ::pow(n, 5.0 / 2.0)) * (::pow(J_eV(Pi), 3.0 / 2.0)) * (1.0 / lambda) * (::sqrt(ionMass / electronMass));
    else
        return 3.44e11 * (1.0 / ::pow(n, 5.0 / 2.0)) * (::pow(n, 3.0 / 2.0)) * (1.0 / lambda) * (::sqrt(ionMass / electronMass)); // if we have a negative temp just treat it as 1eV
}

double tau_e(Value n, Value Pe)
{
    if (Pe > 0)
        return (3.0 / 2.0) * 3.44e11 * (1.0 / ::pow(n, 5.0 / 2.0)) * (::pow(J_eV(Pe), 1.0 / 2.0)) * (1.0 / lambda);
    else
        return (3.0 / 2.0) * 3.44e11 * (1.0 / ::pow(n, 5.0 / 2.0)) * (::pow(n, 1.0 / 2.0)) * (1.0 / lambda);
}

double nu(double n, double Pe)
{
    // double alpha = ::pow(e_charge*e_charge/(2*M_PI*eps_0*me),2)*n(R)*lambda(R);
    // return alpha/n(R)*::pow(mi*Pe+me*Pi, -3.0/2.0);
    // std::cerr << 3.44e-11*::pow(n(R),5.0/2.0)*lambda(R)/::pow(J_eV(Pe),3/2) << std::endl << std::endl;
    return 3.44e-11 * ::pow(n, 5.0 / 2.0) * lambda / ::pow(J_eV(Pe), 3 / 2);
}

double Ce(double n, double Pi, double Pe)
{
    return nu(n, Pe) * (Pi - Pe) / n;
}

double Ci(double n, double Pi, double Pe)
{
    return -Ce(n, Pi, Pe);
}

sigmaFn Gamma = [](VectorXdual u, VectorXdual q, dual x, double t)
{
    // maybe add a factor of sqrt x if x = r^2/2
    dual R = sqrt(2.0 * x);
    dual G = R * R * 2. / 3. * u(1) / (electronMass * Om_e * Om_e * tau_e(u(0).val, u(1).val)) * ((-q(1) / 2 + q(2) / u(1)) - 3 / 2 * q(0) / u(0));
    return G;
};

sigmaFn qi = [](VectorXdual u, VectorXdual q, dual x, double t)
{
    dual sigma = Gamma(u, q, x, t);
    dual R = sqrt(2. * x);
    dual kappa = 2. * u(2) / (ionMass * Om_i * Om_i * tau_i(u(0).val, u(2).val));
    dual Q = R * (2. / 3.) * ((5. / 2.) * u(2) / u(0) * sigma + kappa * R * q(2));
    return Q;
};
sigmaFn qe = [](VectorXdual u, VectorXdual q, dual x, double t)
{
    dual sigma = Gamma(u, q, x, t);
    dual R = sqrt(2. * x);
    dual kappa = 4.66 * u(1) / (electronMass * Om_e * Om_e * tau_e(u(0).val, u(1).val));
    dual Q = R * (2. / 3.) * (5. / 2. * u(1) / u(0) * sigma + kappa * R * q(1));
    return Q;
};

sigmaFnArray Autodiff3VarCyl::sigmaVec = {Gamma, qi, qe};

SourceFn Sn = [](VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    return 0;
};

// look at ion and electron sources again -- they should be opposite
SourceFn Spi = [](VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual R = sqrt(2 * x);
    dual S = (2 / 3) * (5 / 2 * q(2) * R * q(2) / (u(0) * ionMass * Om_i)) - R * Ci(u(2).val, u(0).val, R.val);

    return S;
};
SourceFn Spe = [](VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    dual R = sqrt(2 * x);
    dual S = (2 / 3) * (5 / 2 * q(1) * R * q(1) / (u(0) * ionMass * Om_i)) - R * Ce(u(2).val, u(0).val, R.val);
    return S;
};

SourceFnArray Autodiff3VarCyl::SourceVec = {Sn, Spi, Spe};
