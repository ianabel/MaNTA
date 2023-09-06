#include "AutodiffMatrix.hpp"
#include "Constants.hpp"
#include <autodiff/forward/dual.hpp>
#include <iostream>
using namespace autodiff;

REGISTER_PHYSICS_IMPL(AutodiffMatrix);

AutodiffMatrix::AutodiffMatrix(toml::value const &config)
{
    nVars = 3;

    if (config.count("AutodiffMatrix") != 1)
        throw std::invalid_argument("There should be a [AutodiffMatrix] section if you are using the AutodiffMatrix physics model.");

    auto const &InternalConfig = config.at("AutodiffMatrix");

    xL = toml::find_or(InternalConfig, "x_L", 0.1);
    xR = toml::find_or(InternalConfig, "x_R", 1.0);

    isUpperDirichlet = toml::find_or(InternalConfig, "isUpperDirichlet", true);
    isLowerDirichlet = toml::find_or(InternalConfig, "isLowerDirichlet", true);

    isTestProblem = toml::find_or(InternalConfig, "isTestProblem", false);
    Kappa = Matrix::Identity(nVars, nVars);
    double nL = toml::find_or(InternalConfig, "nL", 1.5);
    double nR = toml::find_or(InternalConfig, "nR", 1.0);

    double peL = toml::find_or(InternalConfig, "peL", 1.5); // J
    double peR = toml::find_or(InternalConfig, "peR", 1.0); // J

    double piL = toml::find_or(InternalConfig, "piL", 1.5); // J
    double piR = toml::find_or(InternalConfig, "piR", 1.0); // J
    Values upperValues(nVars);
    Values lowerValues(nVars);
    upperValues << nR, peR, piR;
    lowerValues << nL, peL, piL;
    uR = upperValues;
    uL = lowerValues;
}

Value AutodiffMatrix::LowerBoundary(Index i, Time t) const { return uL(i); }

Value AutodiffMatrix::UpperBoundary(Index i, Time t) const { return uR(i); }

bool AutodiffMatrix::isLowerBoundaryDirichlet(Index i) const { return isLowerDirichlet; }

bool AutodiffMatrix::isUpperBoundaryDirichlet(Index i) const { return isUpperDirichlet; }

// The same for the flux and source functions -- the vectors have length nVars

Value AutodiffMatrix::SigmaFn(Index i, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);

    Value sigma = sigmaVec[i](uw, qw, x, t).val;
    return sigma;
}
Value AutodiffMatrix::Sources(Index i, const Values &u, const Values &q, const Values &sigma, Position x, Time t)
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
void AutodiffMatrix::dSigmaFn_du(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);

    grad = gradient(sigmaVec[i], wrt(uw), at(uw, qw, x, t));
}
void AutodiffMatrix::dSigmaFn_dq(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{

    VectorXdual uw(u);
    VectorXdual qw(q);

    grad = gradient(sigmaVec[i], wrt(qw), at(uw, qw, x, t));
}

// and for the sources
void AutodiffMatrix::dSources_du(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(nVars);
    for (int j = 0; j < nVars; j++)
        sw(j) = sigmaVec[j](uw, qw, x, t);

    grad = gradient(SourceVec[i], wrt(uw), at(uw, qw, sw, x, t));
}
void AutodiffMatrix::dSources_dq(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(nVars);
    for (int j = 0; j < nVars; j++)
        sw(j) = sigmaVec[j](uw, qw, x, t);

    grad = gradient(SourceVec[i], wrt(qw), at(uw, qw, sw, x, t));
}
void AutodiffMatrix::dSources_dsigma(Index i, Values &grad, const Values &u, const Values &q, Position x, Time t)
{
    VectorXdual uw(u);
    VectorXdual qw(q);
    VectorXdual sw(nVars);
    for (int j = 0; j < nVars; j++)
        sw(j) = sigmaVec[j](uw, qw, x, t);

    grad = gradient(SourceVec[i], wrt(sw), at(uw, qw, sw, x, t));
}

// and initial conditions for u & q
Value AutodiffMatrix::InitialValue(Index i, Position x) const
{
    if (isTestProblem)
    {
        double sol = TestDirichlet(x, 0.0, UpperBoundary(i, 0.0), LowerBoundary(i, 0.0), xL, xR).val.val;
        return sol; // TestSols[i](x, 0)(0).val;
    }

    else
        return 0;
}
Value AutodiffMatrix::InitialDerivative(Index i, Position x) const
{

    if (isTestProblem)
    {
        dual2nd pos = x;
        dual2nd t = 0.0;
        double deriv = derivative(TestDirichlet, wrt(pos), at(pos, t, UpperBoundary(i, 0.0), LowerBoundary(i, 0.0), xL, xR));
        return deriv;
    }
    else
    {
        return 0;
    }
}
double AutodiffMatrix::TestSource(Index i, Position x, Time t)
{
    dual2nd T = t;
    dual2nd pos = x;
    double u_R = UpperBoundary(i, t);
    double u_L = LowerBoundary(i, t);
    double ut = derivative(TestDirichlet, wrt(T), at(pos, T, u_R, u_L, xL, xR));
    VectorXdual q(nVars);
    VectorXdual u(nVars);

    VectorXdual dq(nVars);
    VectorXdual sigma(nVars);

    for (Index j = 0; j < nVars; j++)
    {
        double u_R = UpperBoundary(j, t);
        double u_L = LowerBoundary(j, t);

        auto [q0, q1, q2] = derivatives(TestDirichlet, wrt(pos, pos), at(pos, T, u_R, u_L, xL, xR));

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
    dual xdual = x;
    Values xgrad = gradient(sigmaVec[i], wrt(xdual), at(u, q, xdual, t));
    double uxd = xgrad(0);

    for (Index j = 0; j < nVars; j++)
    {
        uxd += ugrad(j) * q(j).val + qgrad(j) * dq(j).val;
    }
    double S = SourceVec[i](u, q, sigma, x, t).val;
    double St = ut + uxd - S;

    return St;
}

dual2nd AutodiffMatrix::TestDirichlet(dual2nd x, dual2nd t, double u_R, double u_L, double x_L, double x_R)
{
    double k = 5 / 0.025;

    dual2nd a = (asinh(u_L) - asinh(u_R)) / (x_L - x_R);
    dual2nd b = (asinh(u_L) - x_L / x_R * asinh(u_R)) / (a * (x_L / x_R - 1));
    dual2nd c = (M_PI / 2 - 3 * M_PI / 2) / (x_L - x_R);
    dual2nd d = (M_PI / 2 - x_L / x_R * (3 * M_PI / 2)) / (c * (x_L / x_R - 1));

    dual2nd u = sinh(a * (x - b)) - cos(c * (x - d)) * u_L * exp(-k * t) * exp(-0.5 * x * x);

    //  dual2nd u = -cos(c * (x - d)) * exp(-k * t) * exp(-0.5 * x * x);
    return u;
}
// Problem specific stuff
Matrix Kappa = Matrix::Identity(3, 3);

sigmaFn F1 = [](VectorXdual u, VectorXdual q, dual x, double t)
{
    // maybe add a factor of sqrt x if x = r^2/2

    auto sigma = Kappa * q;

    return sigma(0);
};

sigmaFn F2 = [](VectorXdual u, VectorXdual q, dual x, double t)
{
    auto sigma = Kappa * q;

    return sigma(1);
};
sigmaFn F3 = [](VectorXdual u, VectorXdual q, dual x, double t)
{
    auto sigma = Kappa * q;

    return sigma(2);
};
sigmaFnArray AutodiffMatrix::sigmaVec = {F1, F2, F3};

SourceFn S1 = [](VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    return 0.0;
};

// look at ion and electron sources again -- they should be opposite
SourceFn S2 = [](VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    return 0.0;
};
SourceFn S3 = [](VectorXdual u, VectorXdual q, VectorXdual sigma, dual x, double t)
{
    return 0.0;
};

SourceFnArray AutodiffMatrix::SourceVec = {S1, S2, S3};