#include "AutodiffTransportSystem.hpp"
#include "Constants.hpp"
#include <autodiff/forward/dual.hpp>
#include <iostream>
using namespace autodiff;

enum
{
    Gaussian = 0,
    Dirichlet = 1,
    Cosine = 2,
    Uniform = 3,
    Linear = 4,
};

AutodiffTransportSystem::AutodiffTransportSystem(toml::value const &config, Grid const& grid )
{
    if (config.count("AutodiffTransportSystem") != 1)
        throw std::invalid_argument("There should be a [AutodiffTransportSystem] section if you are using the AutodiffTransportSystem physics model.");

    auto const &InternalConfig = config.at("AutodiffTransportSystem");

    xL = grid.lowerBoundary();
    xR = grid.upperBoundary();

    isUpperDirichlet = toml::find_or(InternalConfig, "isUpperDirichlet", true);
    isLowerDirichlet = toml::find_or(InternalConfig, "isLowerDirichlet", true);

    std::vector<double> InitialHeights_v = toml::find<std::vector<double>>(InternalConfig, "InitialHeights");
    InitialHeights = VectorWrapper(InitialHeights_v.data(), nVars);

	 std::vector<std::string> profile = toml::find<std::vector<std::string>>(InternalConfig, "InitialProfile");

	 for (auto &p : profile)
	 {
		 InitialProfile.push_back(InitialProfiles[p]);
	 }

    std::vector<double> uL_v = toml::find<std::vector<double>>(InternalConfig, "uL");
    std::vector<double> uR_v = toml::find<std::vector<double>>(InternalConfig, "uR");

    uR = VectorWrapper(uR_v.data(), nVars);
    uL = VectorWrapper(uL_v.data(), nVars);
}

Vector AutodiffTransportSystem::InitialHeights;

std::vector<int> AutodiffTransportSystem::InitialProfile;

Value AutodiffTransportSystem::SigmaFn(Index i, const State &s, Position x, Time t)
{
    VectorXdual uw(s.Variable);
    VectorXdual qw(s.Derivative);

	return Flux( i, uw, qw, x, t).val;
}

Value AutodiffTransportSystem::Sources(Index i, const State &s, Position x, Time t)
{
    VectorXdual uw(s.Variable);
    VectorXdual qw(s.Derivative);
    VectorXdual sw(s.Flux);

    return Source(i, uw, qw, sw, x, t).val;
}

// We need derivatives of the flux functions
void AutodiffTransportSystem::dSigmaFn_du(Index i, Values &grad, const State &s, Position x, Time t)
{
	autodiff::VectorXdual uw(s.Variable);
	autodiff::VectorXdual qw(s.Derivative);

	grad = autodiff::gradient( [ this,i ]( VectorXdual uD, VectorXdual qD, Position X, Time T ) { return this->Flux( i, uD, qD, X, T ); }, wrt(uw), at(uw, qw, x, t));
}

void AutodiffTransportSystem::dSigmaFn_dq(Index i, Values &grad, const State &s, Position x, Time t)
{
	autodiff::VectorXdual uw(s.Variable);
	autodiff::VectorXdual qw(s.Derivative);

	grad = autodiff::gradient( [ this,i ]( VectorXdual uD, VectorXdual qD, Position X, Time T ) { return this->Flux( i, uD, qD, X, T ); }, wrt(qw), at(uw, qw, x, t));
}

// and for the sources
void AutodiffTransportSystem::dSources_du(Index i, Values &grad, const State &s, Position x, Time t)
{
    VectorXdual uw(s.Variable);
    VectorXdual qw(s.Derivative);
    VectorXdual sw(s.Flux);

    grad = gradient( [ this,i ]( VectorXdual uD, VectorXdual qD, VectorXdual sD, Position X, Time T ) { return this->Source( i, uD, qD, sD, X, T ); },
                     wrt(uw), at(uw, qw, sw, x, t));
}

void AutodiffTransportSystem::dSources_dq(Index i, Values &grad, const State &s, Position x, Time t)
{
    VectorXdual uw(s.Variable);
    VectorXdual qw(s.Derivative);
    VectorXdual sw(s.Flux);

    grad = gradient( [ this,i ]( VectorXdual uD, VectorXdual qD, VectorXdual sD, Position X, Time T ) { return this->Source( i, uD, qD, sD, X, T ); },
                     wrt(qw), at(uw, qw, sw, x, t));
}

void AutodiffTransportSystem::dSources_dsigma(Index i, Values &grad, const State &s, Position x, Time t)
{
    VectorXdual uw(s.Variable);
    VectorXdual qw(s.Derivative);
    VectorXdual sw(s.Flux);

    grad = gradient( [ this,i ]( VectorXdual uD, VectorXdual qD, VectorXdual sD, Position X, Time T ) { return this->Source( i, uD, qD, sD, X, T ); },
                     wrt(sw), at(uw, qw, sw, x, t));
}

// and initial conditions for u & q
Value AutodiffTransportSystem::InitialValue(Index i, Position x) const
{
	return InitialFunction(i, x, 0.0, UpperBoundary(i, 0.0), LowerBoundary(i, 0.0), xL, xR).val.val;
}
Value AutodiffTransportSystem::InitialDerivative(Index i, Position x) const
{
	dual2nd pos = x;
	dual2nd t = 0.0;
	double deriv = derivative(InitialFunction, wrt(pos), at(i, pos, t, UpperBoundary(i, 0.0), LowerBoundary(i, 0.0), xL, xR));
	return deriv;
}

dual2nd AutodiffTransportSystem::InitialFunction(Index i, dual2nd x, dual2nd t, double u_R, double u_L, double x_L, double x_R)
{
    dual2nd a, b, c, d;
    dual2nd u = 0;
    dual2nd C = 0.5 * (x_R + x_L);
    double m = (u_L - u_R) / (x_L - x_R);
    double shape = 5; // 10 / (x_R - x_L) * ::log(10);
    switch (InitialProfile[i])
    {
    case Gaussian:
        u = u_L + InitialHeights[i] * (exp(-(x - C) * (x - C) * shape) - exp(-(x_L - C) * (x_L - C) * shape));
        break;
    case Dirichlet:
		  u = u_L;
        break;
    case Cosine:
        a = (asinh(u_L) - asinh(u_R)) / (x_L - x_R);
        b = (asinh(u_L) - x_L / x_R * asinh(u_R)) / (a * (x_L / x_R - 1));
        c = (M_PI / 2 - 3 * M_PI / 2) / (x_L - x_R);
        d = (M_PI / 2 - x_L / x_R * (3 * M_PI / 2)) / (c * (x_L / x_R - 1));
        if (u_L == u_R)
        {
            u = u_L - cos(c * (x - d)) * InitialHeights[i] * exp(-shape * (x - C) * (x - C));
        }
        else
        {
            u = sinh(a * (x - b)) - cos(c * (x - d)) * InitialHeights[i] * exp(-shape * (x - C) * (x - C));
        }

        break;
    case Uniform:
        u = u_L;
        break;
    case Linear:
        u = u_L + m * (x - x_L);
        break;
    default:
        break;
    };
    return u;
}

