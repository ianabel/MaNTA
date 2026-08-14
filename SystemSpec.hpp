#ifndef SYSTEMSPEC_HPP
#define SYSTEMSPEC_HPP

#include "Types.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

/*
    What a physics case *is*, as data.

    Everything here used to be a virtual function on TransportSystem: nine of
    them returned names, descriptions and units; two more reported the boundary
    condition kind, backed by a pair of uninitialised bools; one more reported
    whether a scalar was differential. A case declared itself by assigning
    protected members in its constructor body, so there was no point at which
    the description could be checked, and no way to ask a half-built case what
    it was.

    A SystemSpec is built once, validated once, and handed to the
    TransportSystem constructor. After that it is immutable, and nVars /
    nScalars / nAux are derived from it rather than assigned.
 */

enum class BoundaryKind
{
    Dirichlet, // fixes u
    Neumann,   // fixes q, the gradient -- *not* the flux; see docs/physics_interface.rst
    Mixed,     // a u + b q + d sigma = c, with the coefficients below
};

/*
    A boundary condition: the kind, plus the three coefficients a Mixed one needs.

    The row a Mixed end imposes is

        a u + b q + d sigma = c

    where c is what LowerBoundary/UpperBoundary returns -- so the coefficients are
    spec data and only the datum varies with t, which is the split
    docs/physics_interface.rst already describes. `sigma` is the *stored* flux,
    which is -sigma_hat (docs/formulation.rst); `d` multiplies that, because that
    is what the assembly multiplies.

    Neumann is the b = 1 case and Dirichlet is *not* the a = 1 case: a Dirichlet
    end is imposed by a different mechanism -- an identically zero trace row and
    column, with the datum substituted into the cell rows -- where a = 1 gives a
    weakly imposed (penalised) Dirichlet. Same solution, same order, different
    discretisation. Do not treat them as interchangeable.

    The converting constructor is load bearing rather than a convenience: it is
    what keeps every existing `{"n", "density", "m^-3", BoundaryKind::Neumann,
    BoundaryKind::Dirichlet}` spec, and the four `v.lower = someBoundaryKind`
    assignments in PhysicsCases/, compiling unchanged. Its cost is that
    BoundaryCondition is not an aggregate, so designated initialisers do not work
    on it -- hence `mixed()`.
 */
struct BoundaryCondition
{
    BoundaryKind kind = BoundaryKind::Dirichlet;
    double a = 0.0, b = 0.0, d = 0.0; // read only when kind == Mixed

    constexpr BoundaryCondition() = default;
    constexpr BoundaryCondition(BoundaryKind k) : kind(k) {} // implicit, deliberately

    static constexpr BoundaryCondition mixed(double a, double b, double d)
    {
        BoundaryCondition bc;
        bc.kind = BoundaryKind::Mixed;
        bc.a = a;
        bc.b = b;
        bc.d = d;
        return bc;
    }
};

/// Compare an end against a kind, ignoring the coefficients.
///
/// C++20 synthesises the reversed form, so `BoundaryKind::Mixed == bc` works too.
/// There is deliberately no BoundaryCondition == BoundaryCondition: a defaulted
/// one would compare coefficients, and "is this end Neumann" is the only question
/// anything in the tree asks.
constexpr bool operator==(BoundaryCondition const &bc, BoundaryKind k) { return bc.kind == k; }

struct FieldSpec
{
    std::string name;
    std::string description;
    std::string units;
    BoundaryCondition lower = BoundaryKind::Dirichlet;
    BoundaryCondition upper = BoundaryKind::Dirichlet;
};

struct ScalarSpec
{
    std::string name;
    std::string description;
    std::string units;
    // Algebraic by default: G_s constrains y alone. Set this when G_s depends
    // explicitly on dy/dt, which is also what makes InitialScalarDerivative be
    // consulted for this scalar.
    bool differential = false;
};

struct AuxSpec
{
    std::string name;
    std::string description;
    std::string units;
};

struct SystemSpec
{
    std::vector<FieldSpec> variables;
    std::vector<ScalarSpec> scalars;
    std::vector<AuxSpec> aux;

    Index numVars() const { return static_cast<Index>(variables.size()); }
    Index numScalars() const { return static_cast<Index>(scalars.size()); }
    Index numAux() const { return static_cast<Index>(aux.size()); }

    /// Throw unless this describes a system the solver can actually build.
    ///
    /// Called from the TransportSystem constructor, so a case that fails this
    /// never becomes an object -- which is what every case constructor's
    /// "NEVER leave a part-constructed object around" comment has been asking
    /// for by hand.
    void validate() const
    {
        if (variables.empty())
            throw std::invalid_argument("A transport system must declare at least one variable");

        // Names are how the netCDF groups, the .dat columns and the by-name
        // State accessors are keyed, so they have to be unique across all three
        // groups, not just within one.
        std::vector<std::string> seen;
        seen.reserve(variables.size() + scalars.size() + aux.size());

        auto check = [&seen](std::string const &name, char const *what)
        {
            if (name.empty())
                throw std::invalid_argument(std::string("A ") + what + " was declared with an empty name");
            if (std::find(seen.begin(), seen.end(), name) != seen.end())
                throw std::invalid_argument("Duplicate name '" + name + "' declared for a " + what +
                                            "; variable, scalar and auxiliary names share one namespace");
            seen.push_back(name);
        };

        for (auto const &v : variables)
            check(v.name, "variable");
        for (auto const &s : scalars)
            check(s.name, "scalar");
        for (auto const &a : aux)
            check(a.name, "auxiliary variable");

        // A Mixed end has to constrain a derivative quantity. Rejecting b = d = 0
        // keeps there from being two inequivalent spellings of a Dirichlet
        // boundary reached by accident: `mixed(1, 0, 0)` is a *weakly* imposed
        // Dirichlet, which converges to the same answer by a different
        // discretisation, and a case that wanted the real thing and wrote this
        // would get no complaint and a different set of numbers.
        auto checkMixed = [](BoundaryCondition const &bc, std::string const &name, char const *end)
        {
            if (bc.kind != BoundaryKind::Mixed)
                return;
            if (bc.a == 0.0 && bc.b == 0.0 && bc.d == 0.0)
                throw std::invalid_argument("Variable '" + name + "' declares a Mixed " + end +
                                            " boundary with a = b = d = 0, which constrains nothing");
            if (bc.b == 0.0 && bc.d == 0.0)
                throw std::invalid_argument("Variable '" + name + "' declares a Mixed " + end +
                                            " boundary with b = d = 0, so it constrains only u. That is a"
                                            " weakly imposed Dirichlet condition rather than the Dirichlet"
                                            " kind; use BoundaryKind::Dirichlet if that is what you meant,"
                                            " or give b or d a nonzero coefficient");
        };

        for (auto const &v : variables)
        {
            checkMixed(v.lower, v.name, "lower");
            checkMixed(v.upper, v.name, "upper");
        }
    }

    /// Index of a variable / scalar / auxiliary variable by name, or -1.
    Index variableIndex(std::string_view name) const { return indexOf(variables, name); }
    Index scalarIndex(std::string_view name) const { return indexOf(scalars, name); }
    Index auxIndex(std::string_view name) const { return indexOf(aux, name); }

private:
    template <typename Specs>
    static Index indexOf(Specs const &specs, std::string_view name)
    {
        for (size_t i = 0; i < specs.size(); ++i)
            if (specs[i].name == name)
                return static_cast<Index>(i);
        return -1;
    }
};

/*
    Placeholder names.

    Before the spec existed, a case that did not override getVariableName got
    "Var0", "Var1", ... and these are the names baked into every checked-in
    .ref.nc, so a case ported to a SystemSpec has to keep them to stay
    comparable against its reference output.

    They are here rather than spelled out in each case so that giving a case
    real names later is a matter of finding the calls below, and so that no one
    reads `{"Var0", "Variable 0", ""}` in a physics case and takes it for the
    house style. A case written from scratch should name its variables.
 */
// BoundaryCondition rather than BoundaryKind, so a Mixed end is expressible here
// too; BoundaryKind converts, so every existing call is unchanged.
inline std::vector<FieldSpec> numberedFields(Index n,
                                             BoundaryCondition lower = BoundaryKind::Dirichlet,
                                             BoundaryCondition upper = BoundaryKind::Dirichlet)
{
    std::vector<FieldSpec> out;
    out.reserve(static_cast<size_t>(n));
    for (Index i = 0; i < n; ++i)
        out.push_back({"Var" + std::to_string(i), "Variable " + std::to_string(i), "", lower, upper});
    return out;
}

inline std::vector<ScalarSpec> numberedScalars(Index n, bool differential = false)
{
    std::vector<ScalarSpec> out;
    out.reserve(static_cast<size_t>(n));
    for (Index i = 0; i < n; ++i)
        out.push_back({"Scalar" + std::to_string(i), "Scalar " + std::to_string(i), "", differential});
    return out;
}

inline std::vector<AuxSpec> numberedAux(Index n)
{
    std::vector<AuxSpec> out;
    out.reserve(static_cast<size_t>(n));
    for (Index i = 0; i < n; ++i)
        out.push_back({"AuxVariable" + std::to_string(i), "Auxiliary Variable " + std::to_string(i), ""});
    return out;
}

#endif // SYSTEMSPEC_HPP
