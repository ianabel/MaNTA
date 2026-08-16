#ifndef FIELDMODELSPEC_HPP
#define FIELDMODELSPEC_HPP

#include "Types.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

/*
    What a magnetic-field model *is*, as data.

    Built once, validated once, handed to the FieldModel constructor, immutable
    after -- the contract SystemSpec.hpp already established, so that a
    part-built model cannot exist and nFieldDOF/nGeometry are derived rather
    than assigned.

    The name is deliberately not the shorter `FieldSpec`: `SystemSpec.hpp`
    already defines a global `struct FieldSpec`, the per-transport-variable
    descriptor bound to Python as `manta.Field`, and the two are unrelated
    types. Reusing the name compiles every translation unit cleanly and fails
    only at link time, as an `-Werror=odr` violation naming neither type by
    the name that clashed -- and "field" in MaNTA already means a transport
    variable, so `FieldModelSpec` is the accurate name here, not merely the
    available one. Do not shorten it back.

    Two vectors, and they mean different things:

      * `dofs` are unknowns. They join the IDA vector after the global scalars,
        and each declares whether it is differential or algebraic -- which goes
        into the `id` vector IDASetId receives.

      * `geometry` are *derived* slots. They are not unknowns: they are a
        function of (psi, x) evaluated at the physics nodes and cached per
        residual, in the same standing as sigmaHat. A physics case reads them
        through State::geom(g).

    `label` names the spatial coordinate the model's geometry is expressed
    against. MaNTA does not interpret it -- the provider declares its own label
    and supplies the metric on it -- but it is recorded in the output so a run
    says what its x meant.

    `name` is the netCDF group a coupled run writes psi and the geometry slots
    into. It defaults rather than being required, because the registered name a
    config selects the model by is a property of the *configuration* and the
    solver never sees it: SystemSolver holds a FieldModel, not the string
    FieldModels::InstantiateFieldModel was given. A model that wants its output
    labelled says so here.
 */

struct FieldDOF
{
    std::string name;
    std::string description;
    std::string units;

    // True if this unknown's residual row carries a d/dt. Declaring it true
    // when the row has no time derivative is the IDA_LINESEARCH_FAIL (-13)
    // misdeclaration: IDA_YA_YDP_INIT holds every differential *value* fixed,
    // so a row reaching no unknown it may move is irreducible and the
    // backtracking loop runs to exhaustion. Task 6 refuses it at run time,
    // where the residual can actually be interrogated.
    bool differential = false;
};

struct FieldSlot
{
    std::string name;
    std::string description;
    std::string units;
};

class FieldModelSpec
{
public:
    std::vector<FieldDOF> dofs;
    std::vector<FieldSlot> geometry;
    std::string label;
    std::string name = "Field";

    Index nFieldDOF() const { return static_cast<Index>(dofs.size()); }
    Index nGeometry() const { return static_cast<Index>(geometry.size()); }

    void validate() const
    {
        if (dofs.empty())
            throw std::invalid_argument("FieldModelSpec: a field model must declare at least one degree of freedom");
        if (geometry.empty())
            throw std::invalid_argument(
                "FieldModelSpec: a field model must declare at least one geometry slot; "
                "geometry is the only channel from the field DOFs into the transport physics");
        if (label.empty())
            throw std::invalid_argument("FieldModelSpec: the spatial label must be named");
        // Only reachable by clearing the default, which is the one way to ask
        // for a netCDF group with no name -- netCDF accepts that and produces a
        // file whose field group cannot be looked up.
        if (name.empty())
            throw std::invalid_argument("FieldModelSpec: the output group name cannot be empty");

        checkNames(dofs, "degree of freedom");
        checkNames(geometry, "geometry slot");
    }

private:
    template <typename T>
    static void checkNames(std::vector<T> const &v, char const *what)
    {
        for (auto const &e : v)
            if (e.name.empty())
                throw std::invalid_argument(std::string("FieldModelSpec: an unnamed ") + what);

        for (size_t i = 0; i < v.size(); ++i)
            for (size_t j = i + 1; j < v.size(); ++j)
                if (v[i].name == v[j].name)
                    throw std::invalid_argument(std::string("FieldModelSpec: duplicate ") + what + " name '" + v[i].name + "'");
    }
};

#endif // FIELDMODELSPEC_HPP
