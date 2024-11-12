
#include "ScalarTestLD.hpp"

/*
	Implementation of the Linear Diffusion case
    with an uncoupled scalar that satisfies ds/dt = s
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL(ScalarTestLD);

ScalarTestLD::ScalarTestLD(toml::value const &config, Grid const&)
{
	// Always set nVars in a derived constructor
	nVars = 1;
	nScalars = 1;

	// Construst your problem from user-specified config
	// throw an exception if you can't. NEVER leave a part-constructed object around
	// here we need the actual value of the diffusion coefficient, and the shape of the initial gaussian

	if (config.count("DiffusionProblem") != 1)
		throw std::invalid_argument("There should be a [DiffusionProblem] section if you are using the ScalarTestLD physics model.");

	auto const &DiffConfig = config.at("DiffusionProblem");

	kappa = toml::find_or(DiffConfig, "Kappa", 1.0);
	InitialWidth = toml::find_or(DiffConfig, "InitialWidth", 0.2);
	InitialHeight = toml::find_or(DiffConfig, "InitialHeight", 1.0);
	Centre = toml::find_or(DiffConfig, "Centre", 0.0);

}

// Dirichlet Boundary Conditon
Value ScalarTestLD::LowerBoundary(Index, Time) const
{
	return 0.0;
}

Value ScalarTestLD::UpperBoundary(Index, Time) const
{
	return 0.0;
}

bool ScalarTestLD::isLowerBoundaryDirichlet(Index) const { return false; };
bool ScalarTestLD::isUpperBoundaryDirichlet(Index) const { return true; };

Value ScalarTestLD::SigmaFn(Index, const State &s, Position x, Time)
{
	return kappa * s.Derivative[0];
}

Value ScalarTestLD::Sources(Index, const State &, Position x, Time)
{
	return 1.0 - x;
}

void ScalarTestLD::dSigmaFn_dq(Index, Values &v, const State &, Position, Time)
{
	v[0] = kappa;
};

void ScalarTestLD::dSigmaFn_du(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void ScalarTestLD::dSources_du(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void ScalarTestLD::dSources_dq(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void ScalarTestLD::dSources_dsigma(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

// Initialise with a Gaussian at x = 0
Value ScalarTestLD::InitialValue(Index, Position x) const
{
	double y = (x - Centre) / InitialWidth;
	return InitialHeight * ::exp(-y * y);
}

Value ScalarTestLD::InitialDerivative(Index, Position x) const
{
	double y = (x - Centre) / InitialWidth;
	return InitialHeight * (-2.0 * y) * ::exp(-y * y) * (1.0 / InitialWidth);
}

Value ScalarTestLD::ScalarGExtended( Index, const DGSoln & y, const DGSoln &dydt, Time )
{
	return (dydt.Scalar( 0 ) - y.Scalar( 0 ));
}

void ScalarTestLD::ScalarGPrimeExtended( Index, State &out, State &out_dt, const DGSoln &y, const DGSoln &dydt, std::function<double( double )>, Interval, Time )
{
    out.zero(); out_dt.zero();
    out.Scalars( 0 ) = -1.0;
    out_dt.Scalars( 0 ) = 1.0;
}

void ScalarTestLD::dSources_dScalars( Index, Values &v, const State &, Position, Time )
{
	v[ 0 ] = 0.0;
}

Value ScalarTestLD::InitialScalarValue( Index ) const
{
    return 1.0;
}

