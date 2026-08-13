#ifndef PHYSICSCASES_HPP
#define PHYSICSCASES_HPP


#include <map>
#include <string>
#include <utility>

#include "Types.hpp"
#include <toml.hpp>
#include "TransportSystem.hpp"
#include "gridStructures.hpp"

template<typename T> std::unique_ptr<TransportSystem> createTransportSystem( toml::value const& config, Grid const& grid ) { return std::make_unique<T>( config, grid ); }

struct PhysicsCases {
	public:
		typedef std::function< std::unique_ptr<TransportSystem>( toml::value const&,  Grid const& ) > function_type;
		typedef std::map<std::string, function_type> map_type;

		static std::unique_ptr<TransportSystem> InstantiateProblem(std::string const& s, toml::value const& , Grid const& );

		// To register explicitly.
		//
		// Throws on a duplicate name rather than quietly keeping the first.
		// This used to be a bare map::insert, which is a no-op when the key
		// exists -- so a case whose name collided with an existing one was
		// simply never instantiated, with nothing said at build or run time.
		static void RegisterPhysicsCase( std::string const& s, function_type creator );

	protected:
		static map_type* getMap();

	public:
		static map_type* map;
};

// For auto-registering. Goes through RegisterPhysicsCase so that the automatic
// and explicit paths agree about what a duplicate name means; this runs during
// static initialisation, so a collision terminates before main() with the
// message above rather than being dropped.
template<typename T> struct PhysicsCaseRegister : PhysicsCases {
	PhysicsCaseRegister(std::string const& s) { RegisterPhysicsCase(s, &createTransportSystem<T>); }
};

#define REGISTER_PHYSICS_HEADER( TypeName ) static PhysicsCaseRegister<TypeName> _reg;
#define REGISTER_PHYSICS_IMPL( TypeName ) PhysicsCaseRegister<TypeName> TypeName::_reg( #TypeName );

#endif // PHYSICSCASES_HPP
