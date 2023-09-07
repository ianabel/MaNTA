#ifndef AUTODIFFFLUX_HPP
#define AUTODIFFFLUX_HPP

#include <map>
#include <string>
#include <utility>

#include "Types.hpp"
#include <toml.hpp>
#include "FluxObject.hpp"

template <typename T>
std::shared_ptr<FluxObject> createFluxObject(toml::value const &config, Index nVars) { return std::make_shared<FluxObject>(T(config, nVars)); }

struct AutodiffFlux
{
public:
    typedef std::map<std::string, std::shared_ptr<FluxObject> (*)(toml::value const &config, Index nVars)> map_type;

    static std::shared_ptr<FluxObject> InstantiateProblem(std::string const &s, toml::value const &config, Index nVars)
    {
        map_type::iterator it = getMap()->find(s);
        if (it == getMap()->end())
            return nullptr;
        return it->second(config, nVars);
    }

    // To register explicitly
    static void RegisterFluxCase(std::string const &s, std::shared_ptr<FluxObject> (*creator)(toml::value const &config, Index nVars))
    {
        getMap()->insert(std::make_pair(s, creator));
    }

protected:
    static map_type *getMap()
    {
        // never delete'ed. (exist until program termination)
        // because we can't guarantee correct destruction order
        if (!map)
        {
            map = new map_type;
        }
        return map;
    }

public:
    static map_type *map;
};

// For auto-registering
template <typename T>
struct AutodiffFluxRegister : AutodiffFlux
{
    AutodiffFluxRegister(std::string const &s) { getMap()->insert(std::make_pair(s, &createFluxObject<T>)); }
};

#define REGISTER_FLUX_HEADER(TypeName) static AutodiffFluxRegister<TypeName> _reg;
#define REGISTER_FLUX_IMPL(TypeName) AutodiffFluxRegister<TypeName> TypeName::_reg(#TypeName);

#endif // AUTODIFFFLUX_HPP
