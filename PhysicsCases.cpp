// Just a translation unit to hold the static global map 

#include "PhysicsCases.hpp"

PhysicsCases::map_type *PhysicsCases::map;

std::unique_ptr<TransportSystem> PhysicsCases::InstantiateProblem(std::string const& s, toml::value const& config, Grid const& grid ) {
    map_type::iterator it = getMap()->find(s);
    if(it == getMap()->end())
        return nullptr;
    return it->second( config, grid );
}

PhysicsCases::map_type* PhysicsCases::getMap() {
    // never delete'ed. (exist until program termination)
    // because we can't guarantee correct destruction order 
    if(!map) { map = new map_type; } 
    return map; 
}

