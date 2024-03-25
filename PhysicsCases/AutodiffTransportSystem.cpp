#include "AutodiffTransportSystem.hpp"
#include "Constants.hpp"
#include <autodiff/forward/dual.hpp>
#include <iostream>
using namespace autodiff;

AutodiffTransportSystem::AutodiffTransportSystem(toml::value const &config, Grid const &grid, Index nV, Index nS)
{
	nVars = nV;
	nScalars = nS;

	if (config.count("AutodiffTransportSystem") == 1)
	{

		auto const &InternalConfig = config.at("AutodiffTransportSystem");

		isUpperDirichlet = toml::find_or(InternalConfig, "isUpperDirichlet", true);
		isLowerDirichlet = toml::find_or(InternalConfig, "isLowerDirichlet", true);

		xL = grid.lowerBoundary();
		xR = grid.upperBoundary();

		std::vector<double> InitialHeights_v = toml::find<std::vector<double>>(InternalConfig, "InitialHeights");
		InitialHeights = VectorWrapper(InitialHeights_v.data(), nVars);

		std::vector<std::string> profile = toml::find<std::vector<std::string>>(InternalConfig, "InitialProfile");

		for (auto &p : profile)
		{
			InitialProfile.push_back(InitialProfiles[p]);
		}

		uL = toml::find<std::vector<double>>(InternalConfig, "uL");
		uR = toml::find<std::vector<double>>(InternalConfig, "uR");
	}
}

Index AutodiffTransportSystem::getConstantI(Index i) const
{
	for (auto &constI : ConstantProfiles)
	{
		if (i >= constI)
			i++;
	}
	return i;
}
void AutodiffTransportSystem::InsertConstantValues(Index i, RealVector &u, RealVector &q, Position x)
{
	int nTotal = nVars + nConstantProfiles;
	u.conservativeResize(nTotal);
	q.conservativeResize(nTotal);
	int count = nConstantProfiles;
	for (auto &constI : ConstantProfiles)
	{
		u(Eigen::seq(constI + 1, nTotal - count)) = u(Eigen::seq(constI, nTotal - count - 1));
		u(constI) = InitialValue(i, x);
		q(Eigen::seq(constI + 1, nTotal - count)) = q(Eigen::seq(constI, nTotal - count - 1));
		q(constI) = InitialDerivative(i, x);
		count--;
	}
}

void AutodiffTransportSystem::RemoveConstantValues(Values &v)
{
	int nTotal = nVars + nConstantProfiles;
	int count = nConstantProfiles;
	for (auto &i : ConstantProfiles)
	{
		v(Eigen::seq(i, nTotal - count - 1)) = v(Eigen::seq(i + 1, nTotal - count));
		count--;
	}
	v.conservativeResize(nVars);
}

Value AutodiffTransportSystem::SigmaFn(Index i, const State &s, Position x, Time t)
{
	VectorXdual uw(s.Variable);
	VectorXdual qw(s.Derivative);

	if (nConstantProfiles > 0)
	{
		InsertConstantValues(i, uw, qw, x);
	}

	return Flux(i, uw, qw, x, t).val;
}

Value AutodiffTransportSystem::Sources(Index i, const State &s, Position x, Time t)
{
	VectorXdual uw(s.Variable);
	VectorXdual qw(s.Derivative);
	VectorXdual sw(s.Flux);

	if (nConstantProfiles > 0)
	{
		InsertConstantValues(i, uw, qw, x);
		sw.resize(nVars + nConstantProfiles);
		for (Index i = 0; i < nVars + nConstantProfiles - 1; i++)
		{
			sw(i) = Flux(i, uw, qw, x, t);
		}
	}

	return Source(i, uw, qw, sw, x, t).val;
}

// We need derivatives of the flux functions
void AutodiffTransportSystem::dSigmaFn_du(Index i, Values &grad, const State &s, Position x, Time t)
{
	autodiff::VectorXdual uw(s.Variable);
	autodiff::VectorXdual qw(s.Derivative);

	if (nConstantProfiles > 0)
	{
		InsertConstantValues(i, uw, qw, x);
		grad = autodiff::gradient([this, i](VectorXdual uD, VectorXdual qD, Position X, Time T)
								  { return this->Flux(i, uD, qD, X, T); },
								  wrt(uw), at(uw, qw, x, t));
		RemoveConstantValues(grad);
	}
	else
	{
		grad = autodiff::gradient([this, i](VectorXdual uD, VectorXdual qD, Position X, Time T)
								  { return this->Flux(i, uD, qD, X, T); },
								  wrt(uw), at(uw, qw, x, t));
	}
}

void AutodiffTransportSystem::dSigmaFn_dq(Index i, Values &grad, const State &s, Position x, Time t)
{
	autodiff::VectorXdual uw(s.Variable);
	autodiff::VectorXdual qw(s.Derivative);

	if (nConstantProfiles > 0)
	{
		InsertConstantValues(i, uw, qw, x);
		grad = autodiff::gradient([this, i](VectorXdual uD, VectorXdual qD, Position X, Time T)
								  { return this->Flux(i, uD, qD, X, T); },
								  wrt(qw), at(uw, qw, x, t));
		RemoveConstantValues(grad);
	}
	else
	{

		grad = autodiff::gradient([this, i](VectorXdual uD, VectorXdual qD, Position X, Time T)
								  { return this->Flux(i, uD, qD, X, T); },
								  wrt(qw), at(uw, qw, x, t));
	}
}

// and for the sources
void AutodiffTransportSystem::dSources_du(Index i, Values &grad, const State &s, Position x, Time t)
{
	VectorXdual uw(s.Variable);
	VectorXdual qw(s.Derivative);
	VectorXdual sw(s.Flux);

	if (nConstantProfiles > 0)
	{
		InsertConstantValues(i, uw, qw, x);
		sw.resize(nVars + nConstantProfiles);
		for (Index i = 0; i < nVars + nConstantProfiles - 1; i++)
		{
			sw(i) = Flux(i, uw, qw, x, t);
		}
		grad = gradient([this, i](VectorXdual uD, VectorXdual qD, VectorXdual sD, Position X, Time T)
						{ return this->Source(i, uD, qD, sD, X, T); },
						wrt(uw), at(uw, qw, sw, x, t));
		RemoveConstantValues(grad);
	}
	else
	{

		grad = gradient([this, i](VectorXdual uD, VectorXdual qD, VectorXdual sD, Position X, Time T)
						{ return this->Source(i, uD, qD, sD, X, T); },
						wrt(uw), at(uw, qw, sw, x, t));
	}
}

void AutodiffTransportSystem::dSources_dq(Index i, Values &grad, const State &s, Position x, Time t)
{
	VectorXdual uw(s.Variable);
	VectorXdual qw(s.Derivative);
	VectorXdual sw(s.Flux);

	if (nConstantProfiles > 0)
	{
		InsertConstantValues(i, uw, qw, x);
		sw.resize(nVars + nConstantProfiles);
		for (Index i = 0; i < nVars + nConstantProfiles - 1; i++)
		{
			sw(i) = Flux(i, uw, qw, x, t);
		}
		grad = gradient([this, i](VectorXdual uD, VectorXdual qD, VectorXdual sD, Position X, Time T)
						{ return this->Source(i, uD, qD, sD, X, T); },
						wrt(qw), at(uw, qw, sw, x, t));
		RemoveConstantValues(grad);
	}
	else
	{
		grad = gradient([this, i](VectorXdual uD, VectorXdual qD, VectorXdual sD, Position X, Time T)
						{ return this->Source(i, uD, qD, sD, X, T); },
						wrt(qw), at(uw, qw, sw, x, t));
	}
}

void AutodiffTransportSystem::dSources_dsigma(Index i, Values &grad, const State &s, Position x, Time t)
{
	VectorXdual uw(s.Variable);
	VectorXdual qw(s.Derivative);
	VectorXdual sw(s.Flux);

	if (nConstantProfiles > 0)
	{
		InsertConstantValues(i, uw, qw, x);
		sw.resize(nVars + nConstantProfiles);
		for (Index i = 0; i < nVars + nConstantProfiles - 1; i++)
		{
			sw(i) = Flux(i, uw, qw, x, t);
		}
		grad = gradient([this, i](VectorXdual uD, VectorXdual qD, VectorXdual sD, Position X, Time T)
						{ return this->Source(i, uD, qD, sD, X, T); },
						wrt(sw), at(uw, qw, sw, x, t));
		RemoveConstantValues(grad);
	}
	else
	{
		grad = gradient([this, i](VectorXdual uD, VectorXdual qD, VectorXdual sD, Position X, Time T)
						{ return this->Source(i, uD, qD, sD, X, T); },
						wrt(sw), at(uw, qw, sw, x, t));
	}
}

// and initial conditions for u & q
Value AutodiffTransportSystem::InitialValue(Index i, Position x) const
{
	return InitialFunction(getConstantI(i), x, 0.0).val.val;
}

Value AutodiffTransportSystem::InitialDerivative(Index i, Position x) const
{
	dual2nd pos = x;
	dual2nd t = 0.0;
	auto InitialValueFn = [this](Index j, dual2nd X, dual2nd T)
	{
		return InitialFunction(j, X, T);
	};
	double deriv = derivative(InitialValueFn, wrt(pos), at(getConstantI(i), pos, t));
	return deriv;
}

dual2nd AutodiffTransportSystem::InitialFunction(Index i, dual2nd x, dual2nd t) const
{
	dual2nd a, b, c, d;
	dual2nd u = 0;
	dual2nd v = 0;
	dual2nd xMid = 0.5 * (xR + xL);
	double u_L = uL[i];
	double u_R = uR[i];
	double m = (u_L - u_R) / (xL - xR);
	double shape = 5; // 10 / (xR - xL) * ::log(10);

	switch (InitialProfile[i])
	{
	case ProfileType::Gaussian:
		u = u_L + InitialHeights[i] * (exp(-(x - xMid) * (x - xMid) * shape) - exp(-(xL - xMid) * (xL - xMid) * shape));
		break;
	case ProfileType::Cosine:
		u = u_L + m * (x - xL) + InitialHeights[i] * cos(M_PI * (x - xMid) / (xR - xL));
		break;
	case ProfileType::CosineSquared:
		v = cos(M_PI * (x - xMid) / (xR - xL));
		u = u_L + m * (x - xL) + InitialHeights[i] * v * v;
		break;
	case ProfileType::Uniform:
		u = u_L;
		break;
	case ProfileType::Linear:
		u = u_L + m * (x - xL);
		break;
	default:
		break;
	};
	return u;
}
