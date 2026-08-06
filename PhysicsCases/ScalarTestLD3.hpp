#ifndef SCALARTESTLD3_HPP
#define SCALARTESTLD3_HPP

#include "PhysicsCases.hpp"

/*
	Linear Diffusion Test Case with a trivial scalar
 */

// Always inherit from TransportSystem
class ScalarTestLD3 : public TransportSystem
{
public:
	// Must provide a constructor that constructs from a toml configuration snippet
	// you can ignore it, or read problem-dependent parameters from the configuration file
	explicit ScalarTestLD3(toml::value const &config, Grid const &);

	// You must provide implementations of both, these are your boundary condition functions
	Value LowerBoundary(Index, Time) const override;
	Value UpperBoundary(Index, Time) const override;

	bool isLowerBoundaryDirichlet(Index) const override;
	bool isUpperBoundaryDirichlet(Index) const override;

	// The guts of the physics problem (these are non-const as they
	// are allowed to alter internal state such as to store computations
	// for future calls)
	Value SigmaFn(Index, const State &, Position, Time) override;
	Value Sources(Index, const State &, Position, Time) override;

	void dSigmaFn_du(Index, VectorRef , const State &, Position, Time) override;
	void dSigmaFn_dq(Index, VectorRef , const State &, Position, Time) override;

	void dSources_du(Index, VectorRef v, const State &, Position, Time) override;
	void dSources_dq(Index, VectorRef v, const State &, Position, Time) override;
	void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override;

	Value ScalarGExtended(Index, const DGSoln &, const DGSoln &, Time) override;
	void ScalarGPrimeExtended(Index, State &, State &, const DGSoln &, const DGSoln &, std::function<double(double)>, Interval, Time) override;
	void dSources_dScalars(Index, VectorRef, const State &, Position, Time) override;

	// Finally one has to provide initial conditions for u & q
	Value InitialValue(Index, Position) const override;
	Value InitialDerivative(Index, Position) const override;

	Value InitialScalarValue(Index) const override;
	Value InitialScalarDerivative(Index s, const DGSoln &y, const DGSoln &dydt) const override;

	void initialiseDiagnostics(NetCDFIO &nc) override;
	void writeDiagnostics(DGSoln const &, double, NetCDFIO &, size_t) override;
        
    bool isScalarDifferential( Index ) override;

private:
	// Put class-specific data here
	double kappa, alpha, beta, gamma, u0, M0, gamma_d, gamma_I;

	Value ScaledSource(Position) const;

	/// Int over the whole domain of variable `var`, integrated cell by cell.
	///
	/// A DG solution is only piecewise polynomial, so a single adaptive rule
	/// over the whole domain resolves the cell-face kinks to its own tolerance
	/// rather than exactly -- and, worse, its subdivision choices shift
	/// discontinuously as the coefficients change, so the integral is not a
	/// smooth function of them. Per cell the rule is exact, which is what the
	/// hand-written dG_0/du claims.
	static double Mass(DGSoln const &y, Index var);

	// Without this (and the implementation line in ScalarTestLD3.cpp)
	// ManTA won't know how to relate the string 'ScalarTestLD3' to the class.
	REGISTER_PHYSICS_HEADER(ScalarTestLD3)
};

#endif // SCALARTESTLD_HPP
