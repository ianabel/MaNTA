
#include "LinearDiffusion.hpp"

/*
	Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL(LinearDiffusion);

LinearDiffusion::LinearDiffusion(toml::value const &config, Grid const &)
{
	// Always set nVars in a derived constructor
	nVars = 1;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if (config.count("DiffusionProblem") != 1)
		throw std::invalid_argument("There should be a [DiffusionProblem] section if you are using the LinearDiffusion physics model.");

	auto const &DiffConfig = config.at("DiffusionProblem");

	kappa = toml::find_or(DiffConfig, "Kappa", 1.0);
	InitialWidth = toml::find_or(DiffConfig, "InitialWidth", 0.2);
	InitialHeight = toml::find_or(DiffConfig, "InitialHeight", 1.0);
	Centre = toml::find_or(DiffConfig, "Centre", 0.5);
	SourceStrength = toml::find_or(DiffConfig, "SourceStrength", 0.0);

	// MMS
	useMMS = toml::find_or(DiffConfig, "UseMMS", false);
	growth = toml::find_or(DiffConfig, "growth", 1.0);
	growth_rate = toml::find_or(DiffConfig, "growth_rate", 0.5);

	lowerNeumann = toml::find_or(DiffConfig, "LowerNeumann", false);
}

// Dirichlet Boundary Conditon
Value LinearDiffusion::LowerBoundary(Index, Time) const
{
	return 0.0;
}

Value LinearDiffusion::UpperBoundary(Index, Time) const
{
	return 0.0;
}

bool LinearDiffusion::isLowerBoundaryDirichlet(Index) const { return !lowerNeumann; };
bool LinearDiffusion::isUpperBoundaryDirichlet(Index) const { return true; };

Value LinearDiffusion::SigmaFn(Index, const State &s, Position x, Time)
{
	return kappa * s.Derivative[0];
}

Value LinearDiffusion::Sources(Index, const State &s, Position x, Time t)
{
	double u = s.Variable[0];
	double S = SourceStrength * u;
	if (useMMS)
		S += MMS_Source(0, x, t);
	return S;
}

void LinearDiffusion::dSigmaFn_dq(Index, Values &v, const State &, Position, Time)
{
	v[0] = kappa;
};

void LinearDiffusion::dSigmaFn_du(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void LinearDiffusion::dSources_du(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void LinearDiffusion::dSources_dq(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void LinearDiffusion::dSources_dsigma(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value LinearDiffusion::InitialValue(Index, Position x) const
{
	double alpha = 1 / InitialWidth;
	double y = (x - Centre);
	return InitialHeight * ::exp(-alpha * y * y);
}

Value LinearDiffusion::InitialDerivative(Index, Position x) const
{
	double y = (x - Centre);
	double alpha = 1 / InitialWidth;
	return -InitialHeight * (2.0 * y) * ::exp(-alpha * y * y) * alpha;
}

double LinearDiffusion::MMS_Solution(Index, Position x, Time t)
{
	return (1 + growth * tanh(growth_rate * t)) * InitialValue(0, x);
}

double LinearDiffusion::MMS_Source(Index, Position x, Time t)
{
	double u = MMS_Solution(0, x, t);

	double dudt = growth * growth_rate * 1 / (cosh(growth_rate * t) * cosh(growth_rate * t)) * InitialValue(0, x);

	double alpha = 1 / InitialWidth;
	double y = x - Centre;
	double d2udx2 = 2 * alpha * (2 * alpha * y * y - 1) * u;

	// double q = -2 * alpha * x * u;
	double S = SourceStrength * u;

	return dudt -
		   kappa * d2udx2 - S;
}

void LinearDiffusion::initialiseDiagnostics(NetCDFIO &nc)
{
	nc.AddGroup("MMS", "Manufactured solutions");
	nc.AddVariable("MMS", "Var0", "Manufactured solution", "-", [this](double x)
				   { return this->InitialValue(0, x); });
}

void LinearDiffusion::writeDiagnostics(DGSoln const &y, Time t, NetCDFIO &nc, size_t tIndex)
{
	Fn s1 = [this, t](double x)
	{ return this->MMS_Solution(0, x, t); };

	nc.AppendToGroup<Fn>("MMS", tIndex, {{"Var0", s1}});
}
