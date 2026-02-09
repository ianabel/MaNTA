// Just a translation unit to hold the static global map 

#include "PhysicsCases.hpp"

PhysicsCases::map_type *PhysicsCases::map;

#include <iostream>
#include <dlfcn.h>

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

void LoadFromFile( std::string const& filename )
{
	void *handle = dlopen( filename.c_str(), RTLD_LAZY | RTLD_LOCAL );
	if ( handle == nullptr ) {
		std::cerr << "Error loading dynamic shared object at " << filename << std::endl;
		std::cerr << "\t" << dlerror() << std::endl;
		return;
	}
	using allocatorType = std::unique_ptr<TransportSystem>(toml::value const&, Grid const&);
	allocatorType *creator = reinterpret_cast<allocatorType*>( dlsym( handle, "createTransportSystem" ) );
	using classnameFn = std::string(void);
	classnameFn* pGCN = reinterpret_cast<classnameFn*>( dlsym( handle, "getClassName" ) );
	std::string className = pGCN();
	PhysicsCases::RegisterPhysicsCase( className, creator );
}

