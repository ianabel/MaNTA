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
    Neumann,   // fixes the flux
};

struct FieldSpec
{
    std::string name;
    std::string description;
    std::string units;
    BoundaryKind lower = BoundaryKind::Dirichlet;
    BoundaryKind upper = BoundaryKind::Dirichlet;
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
inline std::vector<FieldSpec> numberedFields(Index n,
                                             BoundaryKind lower = BoundaryKind::Dirichlet,
                                             BoundaryKind upper = BoundaryKind::Dirichlet)
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
