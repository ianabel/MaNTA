#ifndef SCALARTESTAD_HPP
#define SCALARTESTAD_HPP

#include "AutodiffTransportSystem.hpp"

/*
    Linear Diffusion Test Case with a trivial scalar
 */

// Always inherit from TransportSystem
class ScalarTestAD : public AutodiffTransportSystem
{
public:
    // Must provide a constructor that constructs from a toml configuration snippet
    // you can ignore it, or read problem-dependent parameters from the configuration file
    explicit ScalarTestAD(toml::value const &config, Grid const &);

    // You must provide implementations of both, these are your boundary condition functions
    Value LowerBoundary(Index, Time) const override;
    Value UpperBoundary(Index, Time) const override;

    bool isLowerBoundaryDirichlet(Index) const override;
    bool isUpperBoundaryDirichlet(Index) const override;

    // The guts of the physics problem (these are non-const as they
    // are allowed to alter internal state such as to store computations
    // for future calls)

    Value ScalarGExtended(Index, const DGSoln &, const DGSoln &, Time) override;
    void ScalarGPrimeExtended(Index, State &, State &, const DGSoln &, const DGSoln &, std::function<double(double)>, Interval, Time) override;

    // Finally one has to provide initial conditions for u & q
    Value InitialValue(Index, Position) const override;
    Value InitialDerivative(Index, Position) const override;

    Value InitialScalarValue(Index) const override;
    Value InitialScalarDerivative(Index s, const DGSoln &y, const DGSoln &dydt) const override;

    void initialiseDiagnostics(NetCDFIO &nc) override;
    void writeDiagnostics(DGSoln const &, DGSoln const &, double, NetCDFIO &, size_t) override;

    bool isScalarDifferential(Index) override;

private:
    // Put class-specific data here
    double kappa, alpha, beta, gamma, u0, M0, gamma_d;

    virtual Real Flux(Index i, RealVector u, RealVector q, Real x, Time t) override;
    virtual Real Source(Index i, RealVector u, RealVector q, RealVector sigma, RealVector phi, RealVector Scalars, Real x, Time t) override;

    Real ScaledSource(Real) const;

    // Without this (and the implementation line in ScalarTestAD.cpp)
    // ManTA won't know how to relate the string 'ScalarTestAD' to the class.
    REGISTER_PHYSICS_HEADER(ScalarTestAD)
};

#endif // SCALARTESTLD_HPP
