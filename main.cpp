
#include <format>
#include <string>
#include <print>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

#include "ConfigSchema.hpp"

int runManta( std::string const& );

namespace
{

// A schema default, written the way a config file would write it.
std::string defaultText( ConfigSchema::Value const &v )
{
	return std::visit( []( auto const &x ) -> std::string
	{
		using T = std::decay_t<decltype( x )>;
		if constexpr ( std::is_same_v<T, bool> )
			return x ? "true" : "false";
		else if constexpr ( std::is_same_v<T, std::string> )
			return "\"" + x + "\"";
		else if constexpr ( std::is_same_v<T, std::vector<double>> ||
		                    std::is_same_v<T, std::vector<std::string>> )
		{
			std::string out = "[";
			for ( std::size_t i = 0; i < x.size(); ++i )
				out += ( i ? ", " : "" ) + std::format( "{}", x[ i ] );
			return out + "]";
		}
		else
			return std::format( "{}", x );
	}, v );
}

// Two lines per key rather than four aligned columns. ConfigSchema::typeName
// carries its own article -- it returns "an integer", not "integer", because
// its other callers splice it into "Configuration key 'X' must be ..." -- so it
// reads as a sentence and would read as a stutter down a column. The doc lines
// run to eighty characters on their own, which a column layout would either
// wrap or truncate.
void printEntries( ConfigSchema::Category category )
{
	for ( auto const &e : ConfigSchema::schema() )
	{
		if ( e.category != category )
			continue;

		std::string requirement;
		if ( e.requiredToml && e.requiredDict )
			requirement = "required";
		else if ( e.requiredToml )
			requirement = "required in a config file";
		else if ( e.requiredDict )
			requirement = "required by Runner.configure";
		else
			requirement = "default " + defaultText( e._default );

		std::string aliases;
		for ( auto const &a : e.aliases )
			aliases += std::format( " (was {})", a );

		// A semicolon rather than a comma before the requirement: the list types
		// already spell themselves "a number, or an array of numbers".
		std::println( "  {} -- {}; {}{}", e.name, ConfigSchema::typeName( e.type ),
		              requirement, aliases );
		std::println( "      {}", e.doc );
	}
}

void listOptions()
{
	std::println( "Every configuration key MaNTA accepts. These go in the [configuration]" );
	std::println( "table of a config file, and in the dict passed to manta.Runner.configure;" );
	std::println( "both read them from one declaration, in ConfigSchema.cpp." );
	std::println( "" );
	std::println( "A physics case reads its own parameters from its own table, which is the" );
	std::println( "case's business and is not listed here." );
	std::println( "" );

	std::println( "Solver options -- both surfaces:" );
	printEntries( ConfigSchema::Category::Solver );
	std::println( "" );

	std::println( "Problem selection -- config file only. A Runner is handed the physics" );
	std::println( "object, so passing one of these to configure() is an error rather than" );
	std::println( "being ignored:" );
	printEntries( ConfigSchema::Category::ProblemSelection );
	std::println( "" );

	std::println( "Read by the `manta` command, not by the solver. Listed so that a config" );
	std::println( "file carrying them is not rejected:" );
	printEntries( ConfigSchema::Category::Cli );
	std::println( "" );

	std::println( "A deprecated spelling is accepted with a warning naming its replacement." );
	std::println( "Any other unrecognised key is an error, with the nearest key suggested." );
}

} // namespace

int main( int argc, char** argv )
{
	std::string fname("MaNTA.conf");

	// Before anything is opened: this asks about the schema, not about a run.
	if (argc == 2 && std::string(argv[ 1 ]) == "--list-options")
	{
		listOptions();
		return 0;
	}

	if (argc == 2)
		fname = argv[1];
	if (argc > 2)
	{
		std::println(stderr, "Usage: {} [ConfigFile.conf | --list-options] (config default: MaNTA.conf)", argv[ 0 ]);
		return 1;
	}

	// runManta lets a bad configuration propagate, so that `manta.run()` raises
	// in Python rather than returning a code a caller can ignore. On the command
	// line that would reach std::terminate and print nothing useful, so it is
	// caught here and reported as one line.
	try
	{
		return runManta( fname );
	}
	catch ( std::exception const &e )
	{
		std::println(stderr, "ERROR: {}", e.what());
		return 1;
	}
}
