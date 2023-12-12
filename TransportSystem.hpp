#ifndef TRANSPORTSYSTEM_HPP
#define TRANSPORTSYSTEM_HPP

#include "Types.hpp"
#include "DGSoln.hpp"
#include "NetCDFIO.hpp"

/*
	Pure interface class
	defines a problem in the form
		a_i d_t u_i + d_x ( sigma_i ) = S_i( u(x), q(x), x, t ) ; S_i can depend on the entire u & q vector, but only locally.
		sigma_i = sigma_hat_i( u( x ), q( x ), x, t ) ; so can sigma_hat_i

 */


class TransportSystem
{
public:
	virtual ~TransportSystem() = default;

	Index getNumVars() const { return nVars; };
	Index getNumScalars() const { return nScalars; };

	// Function for passing boundary conditions to the solver
	virtual Value LowerBoundary(Index i, Time t) const = 0;
	virtual Value UpperBoundary(Index i, Time t) const = 0;

	virtual bool isLowerBoundaryDirichlet(Index i) const = 0;
	virtual bool isUpperBoundaryDirichlet(Index i) const = 0;

	// The same for the flux and source functions -- the vectors have length nVars
	virtual Value SigmaFn(Index i, const State &s, Position x, Time t) = 0;
	virtual Value Sources(Index i, const State &s, Position x, Time t) = 0;

	// This determines the a_i functions. Only one with a default option, but can be overriden
	virtual Value aFn(Index i, Position x) { return 1.0; };

	// We need derivatives of the flux functions
	virtual void dSigmaFn_du(Index i, Values &, const State &s, Position x, Time t) = 0;
	virtual void dSigmaFn_dq(Index i, Values &, const State &s, Position x, Time t) = 0;

	// and for the sources
	virtual void dSources_du(Index i, Values &, const State &, Position x, Time t) = 0;
	virtual void dSources_dq(Index i, Values &, const State &, Position x, Time t) = 0;
	virtual void dSources_dsigma(Index i, Values &, const State &, Position x, Time t) = 0;

	// and initial conditions for u & q
	virtual Value InitialValue(Index i, Position x) const = 0;
	virtual Value InitialDerivative(Index i, Position x) const = 0;

	virtual Value InitialScalarValue( Index s ) const {
		if ( nScalars != 0 )
			throw std::logic_error( "nScalars > 0 but no initial value provided" );
		return 0.0;
	}

	// Scalar functions
	virtual Value ScalarG( Index, const DGSoln&, Time ) {
		if ( nScalars != 0 )
			throw std::logic_error( "nScalars > 0 but no scalar G provided" );
		return 0.0;
	}

	virtual void ScalarGPrime( Index, State &, const DGSoln &, std::function<double( double )>, Interval, Time ) {
		if ( nScalars != 0 )
			throw std::logic_error( "nScalars > 0 but no scalar G derivative provided" );
	}
	
	virtual void dSources_dScalars( Index, Values &, const State &, Position, Time ) {
		if ( nScalars != 0 )
			throw std::logic_error( "nScalars > 0 but no coupling function provided" );
	}

	virtual std::string getVariableName(Index i)
	{
		return std::string("Var") + std::to_string(i);
	}

	virtual std::string getVariableDescription(Index i)
	{
		return std::string("Variable ") + std::to_string(i);
	}

	virtual std::string getVariableUnits(Index i)
	{
		return std::string("");
	}

	// Hooks for adding extra NetCDF outputs
	virtual void initialiseDiagnostics( NetCDFIO & ) {
		return;
	}

	// Parameters are ( solution, time, netcdf output object, time index )
	virtual void writeDiagnostics( DGSoln const&, double, NetCDFIO &, size_t ) {
		return;
	}

	virtual void finaliseDiagnostics( NetCDFIO & ) {
		return;
	}

	std::map<int, std::string> subVars = {{0, "u"}, {1, "q"}, {2, "sigma"}, {3, "S"}};

protected:
	Index nVars;
	Index nScalars = 0;
};

#endif // TRANSPORTSYSTEM_HPP
