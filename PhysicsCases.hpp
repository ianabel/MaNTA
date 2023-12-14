#ifndef PHYSICSCASES_HPP
#define PHYSICSCASES_HPP


#include <map>
#include <string>
#include <utility>

#include "Types.hpp"
#include <toml.hpp>
#include "TransportSystem.hpp"
#include "gridStructures.hpp"

template<typename T> TransportSystem* createTransportSystem( toml::value const& config, Grid const& grid ) { return new T( config, grid ); }

struct PhysicsCases {
	public:
		typedef std::map<std::string, TransportSystem*(*)( toml::value const&, Grid const & )> map_type;

		static TransportSystem* InstantiateProblem(std::string const& s, toml::value const& config, Grid const& grid ) {
			map_type::iterator it = getMap()->find(s);
			if(it == getMap()->end())
				return nullptr;
			return it->second( config, grid );
		}

		// To register explicitly
		static void RegisterPhysicsCase( std::string const& s, TransportSystem*(*creator)( toml::value const& config, Grid const& grid ) ) {
			getMap()->insert( std::make_pair( s, creator ) );
		}

	protected:
		static map_type * getMap() {
			// never delete'ed. (exist until program termination)
			// because we can't guarantee correct destruction order 
			if(!map) { map = new map_type; } 
			return map; 
		}

	public:
		static map_type * map;
};

// For auto-registering
template<typename T> struct PhysicsCaseRegister : PhysicsCases {
	PhysicsCaseRegister(std::string const& s) { getMap()->insert(std::make_pair(s, &createTransportSystem<T>)); }
};

#define REGISTER_PHYSICS_HEADER( TypeName ) static PhysicsCaseRegister<TypeName> _reg;
#define REGISTER_PHYSICS_IMPL( TypeName ) PhysicsCaseRegister<TypeName> TypeName::_reg( #TypeName );

#endif // PHYSICSCASES_HPP
