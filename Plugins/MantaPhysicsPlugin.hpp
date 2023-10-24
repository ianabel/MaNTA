#ifndef MANTAPHYSICSPLUGIN_HPP
#define MANTAPHYSICSPLUGIN_HPP

#include "PhysicsCases.hpp"

/* 
 * Define MANTA_PHYSICS_CASE to be the name of the physics model
 * before including this file
 */

#define DECLARE_PHYSICS_PLUGIN( MANTA_PHYSICS_CASE ) extern "C"\
	{\
	TransportSystem* createTransportSystem( toml::value const& config ) { return new MANTA_PHYSICS_CASE( config ); };\
	void deleteTransportSystem( TransportSystem* ptr ) {\
		MANTA_PHYSICS_CASE *pClass = dynamic_cast<MANTA_PHYSICS_CASE *>( ptr );\
		if ( pClass != nullptr )\
			delete pClass;\
		return;\
	};\
		std::string getClassName() { return #MANTA_PHYSICS_CASE; };\
	}

#endif // MANTAPHYSICSPLUGIN_HPP
