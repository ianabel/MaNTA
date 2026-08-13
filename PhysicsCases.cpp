// Just a translation unit to hold the static global map

#include "PhysicsCases.hpp"

#include <stdexcept>

PhysicsCases::map_type *PhysicsCases::map;

namespace
{
    /// "AuxVarTest, LD2, LDTest, ..." for an error message.
    std::string registeredNames(PhysicsCases::map_type const &m)
    {
        if (m.empty())
            return "(none -- no physics case object files are linked in)";

        std::string out;
        for (auto const &entry : m)
        {
            if (!out.empty())
                out += ", ";
            out += entry.first;
        }
        return out;
    }
}

void PhysicsCases::RegisterPhysicsCase(std::string const &s, function_type creator)
{
    auto [it, inserted] = getMap()->insert(std::make_pair(s, creator));
    if (!inserted)
        throw std::invalid_argument(
            "Two physics cases are registered under the name '" + s +
            "'. Registration used to be a bare map::insert, so the first one silently won and "
            "the second was never instantiable -- with no diagnostic at build or run time. "
            "Rename one of them.");
}

std::unique_ptr<TransportSystem> PhysicsCases::InstantiateProblem(std::string const& s, toml::value const& config, Grid const& grid ) {
    map_type::iterator it = getMap()->find(s);
    if (it == getMap()->end())
        // Throws rather than returning nullptr. Every caller had to remember to
        // check, and the one place that forgot dereferenced it -- an unknown
        // TransportSystem name segfaulted instead of saying so. The list of
        // what *is* available is the useful half of the message, because the
        // usual cause is a physics case whose object file is not linked in:
        // nothing references it directly, so a missing entry in PHYSICS_SOURCES
        // produces no compile error.
        throw std::invalid_argument(
            "There is no physics case named '" + s + "'. Available cases: " +
            registeredNames(*getMap()));
    return it->second( config, grid );
}

PhysicsCases::map_type* PhysicsCases::getMap() {
    // never delete'ed. (exist until program termination)
    // because we can't guarantee correct destruction order
    if(!map) { map = new map_type; }
    return map;
}
