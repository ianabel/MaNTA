// Just a translation unit to hold the static global map 

#include "PhysicsCases.hpp"

PhysicsCases::map_type *PhysicsCases::map;
#include <iostream>
#include <dlfcn.h>

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

