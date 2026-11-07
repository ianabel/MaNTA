// A physics case built outside the tree: compiled against the installed headers
// with the flags `pkg-config --cflags manta` reports, not linked against
// libmanta, and loaded by naming it in PhysicsPlugins. docs/out_of_tree.rst
// includes this file, and CI's out-of-tree job builds and runs it, so the
// example in the documentation is one that is known to work.
//
//   d_t u = d_x( kappa d_x u ) + 1  on [0, 1],  u = 0 at both ends.

#include <manta/PhysicsCases.hpp>

class MyCase : public TransportSystem
{
public:
    MyCase(toml::value const &config, Grid const &)
        : TransportSystem({.variables = {{"u", "the diffused quantity", "kg/m^3"}}})
    {
        // The whole configuration file arrives; the case reads its own table.
        if (config.contains("MyCase"))
            kappa = toml::find_or(config.at("MyCase"), "Kappa", kappa);
    }

    Value SigmaFn(Index, const State &s, Position, Time) override { return kappa * s.q(0); }
    Value Sources(Index, const State &, Position, Time) override { return 1.0; }

    // Derivative out-parameters arrive zeroed, so only nonzero entries are
    // written. These five are pure virtual in C++, unlike in Python.
    void dSigmaFn_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSigmaFn_dq(Index, VectorRef v, const State &, Position, Time) override { v[0] = kappa; }
    void dSources_du(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dq(Index, VectorRef, const State &, Position, Time) override {}
    void dSources_dsigma(Index, VectorRef, const State &, Position, Time) override {}

    Value LowerBoundary(Index, Time) const override { return 0.0; }
    Value UpperBoundary(Index, Time) const override { return 0.0; }
    Value InitialValue(Index, Position) const override { return 0.0; }
    Value InitialDerivative(Index, Position) const override { return 0.0; }

private:
    double kappa = 1.0;

    REGISTER_PHYSICS_HEADER(MyCase)
};

REGISTER_PHYSICS_IMPL(MyCase);
