#include "AutodiffTransportSystem.hpp"
#include "Constants.hpp"
#include <autodiff/forward/dual.hpp>
#include <iostream>
#include <filesystem>
#include <string>
using namespace autodiff;

AutodiffTransportSystem::AutodiffTransportSystem(toml::value const &config, Grid const &grid, Index nV, Index nS, Index nA)
{
	nVars = nV;
	nScalars = nS;
	nAux = nA;

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

Value AutodiffTransportSystem::SigmaFn(Index i, const State &s, Position x, Time t)
{
	VectorXdual uw(s.Variable);
	VectorXdual qw(s.Derivative);

	return Flux(i, uw, qw, x, t).val;
}

Value AutodiffTransportSystem::Sources(Index i, const State &s, Position x, Time t)
{
	VectorXdual uw(s.Variable);
	VectorXdual qw(s.Derivative);
	VectorXdual sw(s.Flux);
	VectorXdual phiw(s.Aux);

	return Source(i, uw, qw, sw, phiw, x, t).val;
}

// We need derivatives of the flux functions
void AutodiffTransportSystem::dSigmaFn_du(Index i, Values &grad, const State &s, Position x, Time t)
{
	autodiff::VectorXdual uw(s.Variable);
	autodiff::VectorXdual qw(s.Derivative);

	grad = autodiff::gradient([this, i](VectorXdual uD, VectorXdual qD, Position X, Time T)
							  { return this->Flux(i, uD, qD, X, T); }, wrt(uw), at(uw, qw, x, t));
}

void AutodiffTransportSystem::dSigmaFn_dq(Index i, Values &grad, const State &s, Position x, Time t)
{
	autodiff::VectorXdual uw(s.Variable);
	autodiff::VectorXdual qw(s.Derivative);

	grad = autodiff::gradient([this, i](VectorXdual uD, VectorXdual qD, Position X, Time T)
							  { return this->Flux(i, uD, qD, X, T); }, wrt(qw), at(uw, qw, x, t));
}

// and for the sources
void AutodiffTransportSystem::dSources_du(Index i, Values &grad, const State &s, Position x, Time t)
{
	VectorXdual uw(s.Variable);
	VectorXdual qw(s.Derivative);
	VectorXdual sw(s.Flux);
	VectorXdual phiw(s.Aux);

	grad = gradient([this, i](VectorXdual uD, VectorXdual qD, VectorXdual sD, VectorXdual phiD, Position X, Time T)
					{ return this->Source(i, uD, qD, sD, phiD, X, T); },
					wrt(uw), at(uw, qw, sw, phiw, x, t));
}

void AutodiffTransportSystem::dSources_dq(Index i, Values &grad, const State &s, Position x, Time t)
{
	VectorXdual uw(s.Variable);
	VectorXdual qw(s.Derivative);
	VectorXdual sw(s.Flux);
	VectorXdual phiw(s.Aux);

	grad = gradient([this, i](VectorXdual uD, VectorXdual qD, VectorXdual sD, VectorXdual phiD, Position X, Time T)
					{ return this->Source(i, uD, qD, sD, phiD, X, T); },
					wrt(qw), at(uw, qw, sw, phiw, x, t));
}

void AutodiffTransportSystem::dSources_dsigma(Index i, Values &grad, const State &s, Position x, Time t)
{
	VectorXdual uw(s.Variable);
	VectorXdual qw(s.Derivative);
	VectorXdual sw(s.Flux);
	VectorXdual phiw(s.Aux);

	grad = gradient([this, i](VectorXdual uD, VectorXdual qD, VectorXdual sD, VectorXdual phiD, Position X, Time T)
					{ return this->Source(i, uD, qD, sD, phiD, X, T); },
					wrt(sw), at(uw, qw, sw, phiw, x, t));
}

Value AutodiffTransportSystem::AuxG(Index i, const State &s, Position x, Time t)
{
	VectorXdual uw(s.Variable);
	VectorXdual qw(s.Derivative);
	VectorXdual sw(s.Flux);
	VectorXdual phiw(s.Aux);

	return Phi(i, uw, qw, sw, phiw, x, t).val;
}

void AutodiffTransportSystem::AuxGPrime(Index i, State &out, const State &s, Position x, Time t)
{
	VectorXdual uw(s.Variable);
	VectorXdual qw(s.Derivative);
	VectorXdual sw(s.Flux);
	VectorXdual phiw(s.Aux);

	out.Variable = autodiff::gradient([this, i](VectorXdual uD, VectorXdual qD, VectorXdual sD, VectorXdual phiD, Position X, Time T)
									  { return this->Phi(i, uD, qD, sD, phiD, X, T); }, wrt(uw), at(uw, qw, sw, phiw, x, t));

	out.Derivative = autodiff::gradient([this, i](VectorXdual uD, VectorXdual qD, VectorXdual sD, VectorXdual phiD, Position X, Time T)
										{ return this->Phi(i, uD, qD, sD, phiD, X, T); }, wrt(qw), at(uw, qw, sw, phiw, x, t));

	out.Flux = autodiff::gradient([this, i](VectorXdual uD, VectorXdual qD, VectorXdual sD, VectorXdual phiD, Position X, Time T)
								  { return this->Phi(i, uD, qD, sD, phiD, X, T); }, wrt(sw), at(uw, qw, sw, phiw, x, t));

	out.Aux = autodiff::gradient([this, i](VectorXdual uD, VectorXdual qD, VectorXdual sD, VectorXdual phiD, Position X, Time T)
								 { return this->Phi(i, uD, qD, sD, phiD, X, T); }, wrt(phiw), at(uw, qw, sw, phiw, x, t));
}

void AutodiffTransportSystem::dSources_dPhi(Index i, Values &grad, const State &s, Position x, Time t)
{
	VectorXdual uw(s.Variable);
	VectorXdual qw(s.Derivative);
	VectorXdual sw(s.Flux);
	VectorXdual phiw(s.Aux);

	grad = gradient([this, i](VectorXdual uD, VectorXdual qD, VectorXdual sD, VectorXdual phiD, Position X, Time T)
					{ return this->Source(i, uD, qD, sD, phiD, X, T); },
					wrt(phiw), at(uw, qw, sw, phiw, x, t));
}

// and initial conditions for u & q
Value AutodiffTransportSystem::InitialValue(Index i, Position x) const
{

	return InitialFunction(i, x, 0.0).val.val;
}

Value AutodiffTransportSystem::InitialDerivative(Index i, Position x) const
{
	dual2nd pos = x;
	dual2nd t = 0.0;
	auto InitialValueFn = [this](Index j, dual2nd X, dual2nd T)
	{
		return InitialFunction(j, X, T);
	};
	double deriv = derivative(InitialValueFn, wrt(pos), at(i, pos, t));
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

void AutodiffTransportSystem::LoadDataToSpline(const std::string &file)
{
	try
	{
		data_file.open(file, netCDF::NcFile::FileMode::read);
	}
	catch (...)
	{
		std::string msg = "Failed to open netCDF file at: " + std::string(std::filesystem::absolute(std::filesystem::path(file)));
		throw std::runtime_error(msg);
	}

	auto x_dim = data_file.getDim("x");
	auto t_dim = data_file.getDim("t");
	auto nPoints = x_dim.getSize();
	auto nTime = t_dim.getSize();

	std::vector<double> x(nPoints);
	data_file.getVar("x").getVar(x.data());
	double h = x[1] - x[0];

	std::vector<double> temp(nPoints);
	std::vector<double> temp_deriv(nPoints);
	netCDF::NcGroup tempGroup;

	std::vector<size_t> start = {nTime - 1, 0};
	std::vector<size_t> count = {1, nPoints};

	double xmid = 0.5 * (x.back() + x.front());

	for (Index i = 0; i < nVars; ++i)
	{
		tempGroup = data_file.getGroup("Var" + std::to_string(i));

		tempGroup.getVar("u").getVar(start, count, temp.data());
		tempGroup.getVar("q").getVar(start, count, temp_deriv.data());
		NcFileInitialValues.push_back(std::make_unique<spline>(temp.begin(), temp.end(), x[0], h, temp_deriv.front(), temp_deriv.back()));

		NcFileInitialDerivatives.push_back(std::make_unique<spline>(temp_deriv.begin(), temp_deriv.end(), x[0], h));
	}
	data_file.close();
}

autodiff::dual2nd AutodiffTransportSystem::MMS_Solution(Index i, Real2nd x, Real2nd t)
{
	Real2nd tfac = growth * tanh(growth_rate * t);
	Real2nd S = (1 + tfac) * InitialFunction(i, x, 0.0);
	return S;
}

Value AutodiffTransportSystem::MMS_Source(Index i, Position x, Time t)
{
	Real2nd xval = x;
	Real2nd tval = t;

	State s(nVars, nScalars);
	Values d2udx2(nVars);
	VectorXdual sigma(nVars);

	for (Index j = 0; j < nVars; ++j)
	{
		auto [uval, qval, d2udx2val] = derivatives([this, j](Real2nd x, Real2nd t)
												   { return this->MMS_Solution(j, x, t); }, wrt(xval, xval), at(xval, tval));

		s.Variable(j) = uval;
		s.Derivative(j) = qval;
		d2udx2(j) = d2udx2val;
	}

	Values gradu(nVars);
	Values gradq(nVars);

	dSigmaFn_du(i, gradu, s, x, t);
	dSigmaFn_dq(i, gradq, s, x, t);

	VectorXdual uw(s.Variable);
	VectorXdual qw(s.Derivative);

	Real xreal = x;

	double dSdx = derivative([this, i](VectorXdual uD, VectorXdual qD, Real X, Time T)
							 { return this->Flux(i, uD, qD, X, T); },
							 wrt(xreal), at(uw, qw, xreal, t));

	double dSigma_dx = dSdx;

	for (Index j = 0; j < nVars; ++j)
	{
		sigma(j) = Flux(i, uw, qw, xreal, t);
		dSigma_dx += s.Derivative[j] * gradu[j] + d2udx2[j] * gradq[j];
	}

	VectorXdual phi(nAux);

	for (Index j = 0; j < nAux; ++j)
	{
		phi(j) = 0.0;
	}
	double dudt = derivative([this, i](Real2nd x, Real2nd t)
							 { return this->MMS_Solution(i, x, t); }, wrt(tval), at(xval, tval));

	double S = Source(i, uw, qw, sigma, phi, xreal, t).val;

	double MMS = dudt - dSigma_dx - S;

	return MMS;
}

void AutodiffTransportSystem::initialiseDiagnostics(NetCDFIO &nc)
{
	nc.AddGroup("MMSSource", "MMS sources");
	for (Index j = 0; j < nVars; ++j)
		nc.AddVariable("MMSSource", "Var" + std::to_string(j), "MMS source", "-", [this, j](double x)
					   { return this->MMS_Source(j, x, 0.0); });
}

void AutodiffTransportSystem::writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex)
{
	for (Index j = 0; j < nVars; ++j)
		nc.AppendToGroup("MMSSource", tIndex, "Var" + std::to_string(j), [this, j, t](double x)
						 { return this->MMS_Source(j, x, t); });
}
