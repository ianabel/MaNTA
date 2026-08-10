#ifndef NEUMANNTEST_HPP
#define NEUMANNTEST_HPP

#include "PhysicsCases.hpp"

/*
    Linear Diffusion Test Case, showcasing how to write a physics case that is compiled
    at the same time as the
 */

// Always inherit from TransportSystem
class NeumannTest : public TransportSystem
{
public:
    // Must provide a constructor that constructs from a toml configuration snippet
    // you can ignore it, or read problem-dependent parameters from the configuration file
    explicit NeumannTest(toml::value const &config, Grid const &);

    // You must provide implementations of both, these are your boundary condition functions
    Value LowerBoundary(Index, Time) const override;
    Value UpperBoundary(Index, Time) const override;

    // The guts of the physics problem (these are non-const as they
    // are allowed to alter internal state such as to store computations
    // for future calls)
    Value SigmaFn(Index, const State &, Position, Time) override;
    Value Sources(Index, const State &, Position, Time) override;

    void dSigmaFn_du(Index, VectorRef, const State &, Position, Time) override;
    void dSigmaFn_dq(Index, VectorRef, const State &, Position, Time) override;

    void dSources_du(Index, VectorRef v, const State &, Position, Time) override;
    void dSources_dq(Index, VectorRef v, const State &, Position, Time) override;
    void dSources_dsigma(Index, VectorRef v, const State &, Position, Time) override;

    // Finally one has to provide initial conditions for u & q
    Value InitialValue(Index, Position) const override;
    Value InitialDerivative(Index, Position) const override;

private:
    static SystemSpec buildSpec(toml::value const &config);

    // Put class-specific data here
    double kappa, InitialWidth, InitialHeight, Centre;
    double xL, xR;
    double growth, growth_rate, SourceStrength;

    // Without this (and the implementation line in NeumannTest.cpp)
    // ManTA won't know how to relate the string 'NeumannTest' to the class.
    REGISTER_PHYSICS_HEADER(NeumannTest)
};

#endif // LINEARDIFFUSION_HPP
