#include "AutodiffTransportSystem.hpp"
#include "PyIntegrator.hpp"
#include <autodiff/forward/dual.hpp>
#include <iostream>
#include <filesystem>
#include <string>

SystemSpec AutodiffTransportSystem::withBoundaryConfig(SystemSpec spec, toml::value const &config)
{
	if (config.count("AutodiffTransportSystem") != 1)
		return spec;

	auto const &InternalConfig = config.at("AutodiffTransportSystem");

	auto kind = [](bool dirichlet)
	{ return dirichlet ? BoundaryKind::Dirichlet : BoundaryKind::Neumann; };

	if (InternalConfig.contains("isUpperDirichlet"))
	{
		const auto upper = kind(toml::find<bool>(InternalConfig, "isUpperDirichlet"));
		for (auto &v : spec.variables)
			v.upper = upper;
	}

	if (InternalConfig.contains("isLowerDirichlet"))
	{
		const auto lower = kind(toml::find<bool>(InternalConfig, "isLowerDirichlet"));
		for (auto &v : spec.variables)
			v.lower = lower;
	}

	return spec;
}

AutodiffTransportSystem::AutodiffTransportSystem(toml::value const &config, Grid const &grid, SystemSpec spec)
	: TransportSystem(withBoundaryConfig(std::move(spec), config))
{
	if (config.count("AutodiffTransportSystem") == 1)
	{
		auto const &InternalConfig = config.at("AutodiffTransportSystem");

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
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector geom(s.geom());

	return Flux(i, u, q, geom, x, t).val;
}

Value AutodiffTransportSystem::Sources(Index i, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector sigma(s.sigma());
	RealVector phi(s.phi());
	RealVector Scalar(s.scalars());
	RealVector geom(s.geom());

	return Source(i, u, q, sigma, phi, Scalar, geom, x, t).val;
}

// We need derivatives of the flux functions
void AutodiffTransportSystem::dSigmaFn_du(Index i, VectorRef grad, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector geom(s.geom());

	Real uout;

	autodiff::gradient([this, i](RealVector uD, RealVector qD, RealVector geomD, Position X, Time T)
					   { return this->Flux(i, uD, qD, geomD, X, T); }, wrt(u), at(u, q, geom, x, t), uout, grad);
}

void AutodiffTransportSystem::dSigmaFn_dq(Index i, VectorRef grad, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector geom(s.geom());

	Real uout;

	autodiff::gradient([this, i](RealVector uD, RealVector qD, RealVector geomD, Position X, Time T)
					   { return this->Flux(i, uD, qD, geomD, X, T); }, wrt(q), at(u, q, geom, x, t), uout, grad);
}

void AutodiffTransportSystem::dSigmaFn_dGeometry(Index i, VectorRef grad, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector geom(s.geom());

	Real uout;

	autodiff::gradient([this, i](RealVector uD, RealVector qD, RealVector geomD, Position X, Time T)
					   { return this->Flux(i, uD, qD, geomD, X, T); }, wrt(geom), at(u, q, geom, x, t), uout, grad);
}

// and for the sources
void AutodiffTransportSystem::dSources_du(Index i, VectorRef grad, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector sigma(s.sigma());
	RealVector phi(s.phi());
	RealVector Scalar(s.scalars());
	RealVector geom(s.geom());

	Real uout;

	gradient([this, i](RealVector uD, RealVector qD, RealVector sD, RealVector phiD, RealVector ScalarD, RealVector geomD, Position X, Time T)
			 { return this->Source(i, uD, qD, sD, phiD, ScalarD, geomD, X, T); },
			 wrt(u), at(u, q, sigma, phi, Scalar, geom, x, t), uout, grad);
}

void AutodiffTransportSystem::dSources_dq(Index i, VectorRef grad, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector sigma(s.sigma());
	RealVector phi(s.phi());
	RealVector Scalar(s.scalars());
	RealVector geom(s.geom());

	Real uout;

	gradient([this, i](RealVector uD, RealVector qD, RealVector sD, RealVector phiD, RealVector ScalarD, RealVector geomD, Position X, Time T)
			 { return this->Source(i, uD, qD, sD, phiD, ScalarD, geomD, X, T); },
			 wrt(q), at(u, q, sigma, phi, Scalar, geom, x, t), uout, grad);
}

void AutodiffTransportSystem::dSources_dsigma(Index i, VectorRef grad, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector sigma(s.sigma());
	RealVector phi(s.phi());
	RealVector Scalar(s.scalars());
	RealVector geom(s.geom());

	Real uout;

	gradient([this, i](RealVector uD, RealVector qD, RealVector sD, RealVector phiD, RealVector ScalarD, RealVector geomD, Position X, Time T)
			 { return this->Source(i, uD, qD, sD, phiD, ScalarD, geomD, X, T); },
			 wrt(sigma), at(u, q, sigma, phi, Scalar, geom, x, t), uout, grad);
}

void AutodiffTransportSystem::dSources_dPhi(Index i, VectorRef grad, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector sigma(s.sigma());
	RealVector phi(s.phi());
	RealVector Scalar(s.scalars());
	RealVector geom(s.geom());
	// phi.resize(nVars);
	Real uout;

	gradient([this, i](RealVector uD, RealVector qD, RealVector sD, RealVector phiD, RealVector ScalarD, RealVector geomD, Position X, Time T)
			 { return this->Source(i, uD, qD, sD, phiD, ScalarD, geomD, X, T); },
			 wrt(phi), at(u, q, sigma, phi, Scalar, geom, x, t), uout, grad);
}

void AutodiffTransportSystem::dSources_dScalars(Index i, VectorRef grad, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector sigma(s.sigma());
	RealVector phi(s.phi());
	RealVector Scalar(s.scalars());
	RealVector geom(s.geom());

	Real uout;

	gradient([this, i](RealVector uD, RealVector qD, RealVector sD, RealVector phiD, RealVector ScalarD, RealVector geomD, Position X, Time T)
			 { return this->Source(i, uD, qD, sD, phiD, ScalarD, geomD, X, T); },
			 wrt(Scalar), at(u, q, sigma, phi, Scalar, geom, x, t), uout, grad);
}

void AutodiffTransportSystem::dSources_dGeometry(Index i, VectorRef grad, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector sigma(s.sigma());
	RealVector phi(s.phi());
	RealVector Scalar(s.scalars());
	RealVector geom(s.geom());

	Real uout;

	gradient([this, i](RealVector uD, RealVector qD, RealVector sD, RealVector phiD, RealVector ScalarD, RealVector geomD, Position X, Time T)
			 { return this->Source(i, uD, qD, sD, phiD, ScalarD, geomD, X, T); },
			 wrt(geom), at(u, q, sigma, phi, Scalar, geom, x, t), uout, grad);
}

// Through the *geometry-aware* Flux overload, like every other hook here.
//
// This differentiated the 4-arg one until the adjoint learned about the field
// coupling, and the difference was not academic. The 4-arg overload is the
// mandatory pure virtual, so a geometry-dependent case is forced to write some
// body for it to compile at all; whatever that stand-in computes is what this
// function would have differentiated, rather than the real flux. dSigmaFn_dp
// feeds F_p in computeAdjointGradients, so the symptom would have been a wrong
// gradient beside a perfectly good G -- the one failure shape the adjoint has
// nothing to report it with.
void AutodiffTransportSystem::dSigmaFn_dp(Index i, Index pIndex, Value &grad, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector geom(s.geom());
	// make sure all gradients are zero
	Real p = getPval(pIndex);

	grad = autodiff::derivative(
		[this, i, pIndex](Real p, RealVector uD, RealVector qD, RealVector geomD, Position X, Time T)
		{
			setPval(pIndex, p);
			return Flux(i, uD, qD, geomD, X, T);
		},
		wrt(p), at(p, u, q, geom, x, t));

	clearGradients();
}

// The same fix as dSigmaFn_dp above, for Source: the 9-arg geometry-aware
// overload, not the 8-arg one every geometry-dependent case would only have
// written to satisfy the compiler.
void AutodiffTransportSystem::dSources_dp(Index i, Index pIndex, Value &grad, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector sigma(s.sigma());
	RealVector phi(s.phi());
	RealVector Scalar(s.scalars());
	RealVector geom(s.geom());

	Real p = getPval(pIndex);

	grad = autodiff::derivative(
		[this, i, pIndex](Real p, RealVector uD, RealVector qD, RealVector sD, RealVector phiD, RealVector ScalarD, RealVector geomD, Position X, Time T)
		{
			setPval(pIndex, p);
			Real S = Source(i, uD, qD, sD, phiD, ScalarD, geomD, X, T);
			return S;
		},
		wrt(p), at(p, u, q, sigma, phi, Scalar, geom, x, t));

	// make sure all gradients are zero
	clearGradients();
}

Value AutodiffTransportSystem::AuxG(Index i, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector sigma(s.sigma());
	RealVector phi(s.phi());
	RealVector geom(s.geom());

	return GFunc(i, u, q, sigma, phi, geom, x, t).val;
}

void AutodiffTransportSystem::AuxGPrime(Index i, State &out, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector sigma(s.sigma());
	RealVector phi(s.phi());
	RealVector geom(s.geom());

	Real uout;

	autodiff::gradient([this, i](RealVector uD, RealVector qD, RealVector sD, RealVector phiD, RealVector geomD, Position X, Time T)
					   { return this->GFunc(i, uD, qD, sD, phiD, geomD, X, T); }, wrt(u), at(u, q, sigma, phi, geom, x, t), uout, out.u());

	autodiff::gradient([this, i](RealVector uD, RealVector qD, RealVector sD, RealVector phiD, RealVector geomD, Position X, Time T)
					   { return this->GFunc(i, uD, qD, sD, phiD, geomD, X, T); }, wrt(q), at(u, q, sigma, phi, geom, x, t), uout, out.q());

	autodiff::gradient([this, i](RealVector uD, RealVector qD, RealVector sD, RealVector phiD, RealVector geomD, Position X, Time T)
					   { return this->GFunc(i, uD, qD, sD, phiD, geomD, X, T); }, wrt(sigma), at(u, q, sigma, phi, geom, x, t), uout, out.sigma());

	autodiff::gradient([this, i](RealVector uD, RealVector qD, RealVector sD, RealVector phiD, RealVector geomD, Position X, Time T)
					   { return this->GFunc(i, uD, qD, sD, phiD, geomD, X, T); }, wrt(phi), at(u, q, sigma, phi, geom, x, t), uout, out.phi());
}

void AutodiffTransportSystem::dAuxG_dGeometry(Index i, VectorRef grad, const State &s, Position x, Time t)
{
	RealVector u(s.u());
	RealVector q(s.q());
	RealVector sigma(s.sigma());
	RealVector phi(s.phi());
	RealVector geom(s.geom());

	Real uout;

	autodiff::gradient([this, i](RealVector uD, RealVector qD, RealVector sD, RealVector phiD, RealVector geomD, Position X, Time T)
					   { return this->GFunc(i, uD, qD, sD, phiD, geomD, X, T); }, wrt(geom), at(u, q, sigma, phi, geom, x, t), uout, grad);
}

// and initial conditions for u & q
Value AutodiffTransportSystem::InitialValue(Index i, Position x) const
{
	if (loadInitialConditionsFromFile)
	{
		return (*NcFileInitialValues[i])(x);
	}
	else
	{
		return InitialFunction(i, x, 0.0).val.val;
	}
}

Value AutodiffTransportSystem::InitialDerivative(Index i, Position x) const
{
	if (loadInitialConditionsFromFile)
	{
		return (*NcFileInitialValues[i]).prime(x);
	}
	else
	{
		Real2nd pos = x;
		Real2nd t = 0.0;
		auto InitialValueFn = [this](Index j, Real2nd X, Real2nd T)
		{
			return InitialFunction(j, X, T);
		};
		double deriv = derivative(InitialValueFn, wrt(pos), at(i, pos, t));
		return deriv;
	}
}

Real2nd AutodiffTransportSystem::InitialFunction(Index i, Real2nd x, Real2nd t) const
{
	Real2nd a, b, c, d;
	Real2nd u = 0;
	Real2nd v = 0;
	Real2nd xMid = 0.5 * (xR + xL);
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
		u = InitialHeights[i];
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

	for (Index i = 0; i < nVars; ++i)
	{
		// getVariableName(i), not "Var" + i: this reads back a file MaNTA wrote,
		// and NetCDFIO names those groups after the case's own variables. The
		// literal only worked while every case used the placeholder names.
		tempGroup = data_file.getGroup(getVariableName(i));
		if (tempGroup.isNull())
			throw std::runtime_error(
				"Initial-condition file " + file + " has no group '" + getVariableName(i) +
				"'. A file written before this case's variables were named will use the "
				"old names; regenerate it, or rename the groups.");

		tempGroup.getVar("u").getVar(start, count, temp.data());
		tempGroup.getVar("q").getVar(start, count, temp_deriv.data());
		NcFileInitialValues.push_back(std::make_unique<spline>(temp.begin(), temp.end(), x[0], h, temp_deriv.front(), temp_deriv.back()));

		NcFileInitialDerivatives.push_back(std::make_unique<spline>(temp_deriv.begin(), temp_deriv.end(), x[0], h));
	}
	for (Index i = 0; i < nAux; ++i)
	{
		data_file.getVar("AuxVariable" + std::to_string(i)).getVar(start, count, temp.data());
		NcFileInitialAuxValue.push_back(std::make_unique<spline>(temp.begin(), temp.end(), x[0], h));
	}
	data_file.close();
}

Real2nd AutodiffTransportSystem::MMS_Solution(Index i, Real2nd x, Real2nd t)
{
	Real2nd tfac = growth * tanh(growth_rate * t);
	Real2nd S = (1 + tfac) * InitialFunction(i, x, 0.0);
	return S;
}

// TODO(manufactured coupled sources): the direct Flux(...)/Source(...) calls
// below (building `sigma`/`dSigma_dx`/`S`) go through the geometry-blind
// overloads, not the geometry-aware ones dSigmaFn_du/dq above already use via
// `s.geom()` -- so this function is internally inconsistent about which overload
// it evaluates. Harmless today because `s` here is a bare State(nVars,
// nScalars) with no geometry rows (ngeom = 0 by default) and no case yet gives
// the two Flux/Source overloads different content, so both paths agree. It will
// stop agreeing the moment a geometry-dependent case tries useMMS: the
// manufactured source would then be built against a flux that ignores geometry
// while the solver's actual residual uses one that does not, which is exactly
// the "manufactured source differentiated with the wrong function" trap this
// file's own header comment warns about for the sign convention -- same shape,
// different cause.
//
// The two markers this sat beside -- on dSigmaFn_dp and dSources_dp -- are gone,
// fixed with the adjoint transposes, because those two feed F_p and so were an
// adjoint defect. This one is not, and it is not a two-line fix either: MMS_Source
// is handed (i, x, t) and nothing else, so there is no psi to evaluate geometry
// at. AutodiffTransportSystem holds no field model and has no way to reach the
// solver's. Threading one in is a change to the manufactured-solution facility,
// which is what the coupled order study owns.
Value AutodiffTransportSystem::MMS_Source(Index i, Position x, Time t)
{
	Real2nd xval = x;
	Real2nd tval = t;

	State s(nVars, nScalars);
	Values d2udx2(nVars);
	RealVector sigma(nVars);

	for (Index j = 0; j < nVars; ++j)
	{
		auto [uval, qval, d2udx2val] = derivatives([this, j](Real2nd x, Real2nd t)
												   { return this->MMS_Solution(j, x, t); }, wrt(xval, xval), at(xval, tval));

		s.u(j) = uval;
		s.q(j) = qval;
		d2udx2(j) = d2udx2val;
	}

	Values gradu(nVars);
	Values gradq(nVars);

	dSigmaFn_du(i, gradu, s, x, t);
	dSigmaFn_dq(i, gradq, s, x, t);

	RealVector u(s.u());
	RealVector q(s.q());

	Real xreal = x;

	double dSdx = derivative([this, i](RealVector uD, RealVector qD, Real X, Time T)
							 { return this->Flux(i, uD, qD, X, T); },
							 wrt(xreal), at(u, q, xreal, t));

	double dSigma_dx = dSdx;

	for (Index j = 0; j < nVars; ++j)
	{
		sigma(j) = Flux(i, u, q, xreal, t);
		dSigma_dx += s.q(j) * gradu[j] + d2udx2[j] * gradq[j];
	}

	RealVector phi(nAux);

	for (Index j = 0; j < nAux; ++j)
	{
		phi(j) = InitialAuxValue(j, x, t);
	}
	double dudt = derivative([this, i](Real2nd x, Real2nd t)
							 { return this->MMS_Solution(i, x, t); }, wrt(tval), at(xval, tval));

	double S = Source(i, u, q, sigma, phi, xreal, t).val;

	double MMS = dudt - dSigma_dx - S;

	return MMS;
}

void AutodiffTransportSystem::initialiseDiagnostics(NetCDFIO &nc)
{
	if (nAux > 0)
	{
		nc.AddGroup("AuxG", "Auxiliary functions");
		for (Index i = 0; i < nAux; ++i)
			nc.AddVariable("AuxG", "Aux" + std::to_string(i), "Auxiliary function", "-", [this, i](double x)
						   { return this->InitialAuxValue(i, x); });
	}

	if (nScalars > 0)
	{
		nc.AddGroup("ScalarG", "Scalar functions");
		for (Index i = 0; i < nScalars; ++i)
			nc.AddTimeSeries("ScalarG", "ScalarG" + std::to_string(i), "Scalar function", "-", InitialScalarValue(i));
	}

	if (useMMS)
	{
		nc.AddGroup("MMSSource", "MMS sources");
		for (Index i = 0; i < nVars; ++i)
			nc.AddVariable("MMSSource", "Var" + std::to_string(i), "MMS source", "-", [this, i](double x)
						   { return this->MMS_Source(i, x, 0.0); });
	}
}

void AutodiffTransportSystem::writeDiagnostics(DGSoln const &y, DGSoln const &dydt, Time t, NetCDFIO &nc, size_t tIndex)
{
	if (nAux > 0)
	{
		for (Index i = 0; i < nAux; ++i)
			nc.AppendToGroup("AuxG", tIndex, "Aux" + std::to_string(i), [this, i, &y, &t](double x)
							 {  State s = y.eval(x);
								return this->AuxG(i, s, x, t); });
	}

	if (nScalars > 0)
	{
		// Sampled once for all the scalars, as the residual does.
		const GlobalState states = y.evalOnNodes();
		const GlobalState states_dt = dydt.evalOnNodes();
		const Grid &grid = y.getGrid();
		const Vector &weights = m_integrator.integrationWeights(y.getBasis(), grid);
		const Matrix &phiBoundary = m_integrator.phiBoundary(y.getBasis(), grid);

		for (Index i = 0; i < nScalars; ++i)
			nc.AppendToTimeSeries("ScalarG", "ScalarG" + std::to_string(i),
								  ScalarG(i, states, states_dt, y.getPoints(), weights, phiBoundary, t),
								  tIndex);
	}

	if (useMMS)
	{
		for (Index i = 0; i < nVars; ++i)
			nc.AppendToGroup("MMSSource", tIndex, "Var" + std::to_string(i), [this, i, t](double x)
							 { return this->MMS_Source(i, x, t); });
	}
}
