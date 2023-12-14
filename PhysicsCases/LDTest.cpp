
#include "LDTest.hpp"

/*
	Implementation of the Linear Diffusion case
 */

// Needed to register the class
REGISTER_PHYSICS_IMPL(LDTest);

LDTest::LDTest(toml::value const &config, Grid const& )
{
	// Always set nVars in a derived constructor
	nVars = 1;

	auto const& DiffConfig = config.at( "DiffusionProblem" );
	kappa = toml::find_or(DiffConfig, "Kappa", 1.0);
	EdgeValue = 0.3;
}

// Dirichlet Boundary Conditon
Value LDTest::LowerBoundary(Index, Time) const
{
	return EdgeValue;
}

Value LDTest::UpperBoundary(Index, Time) const
{
	return EdgeValue;
}

bool LDTest::isLowerBoundaryDirichlet(Index) const { return true; };
bool LDTest::isUpperBoundaryDirichlet(Index) const { return true; };

Value LDTest::SigmaFn(Index, const State &s, Position x, Time)
{
	return kappa * s.Derivative[0];
}

Value LDTest::Sources(Index, const State &, Position x, Time)
{
	return ::exp( -25.0*x*x );
}

void LDTest::dSigmaFn_dq(Index, Values &v, const State &, Position, Time)
{
	v[0] = kappa;
};

void LDTest::dSigmaFn_du(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void LDTest::dSources_du(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void LDTest::dSources_dq(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

void LDTest::dSources_dsigma(Index, Values &v, const State &, Position, Time)
{
	v[0] = 0.0;
};

// We don't need the index variables as nVars is 1, so the index argument should
// always be 0

Value LDTest::InitialValue(Index, Position) const
{
	return EdgeValue;
}

Value LDTest::InitialDerivative(Index, Position) const
{
	return 0;
}
