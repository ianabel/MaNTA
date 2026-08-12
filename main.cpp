
#include <string>
#include <print>
#include <stdexcept>

int runManta( std::string const& );

int main( int argc, char** argv )
{
	std::string fname("MaNTA.conf");
	if (argc == 2)
		fname = argv[1];
	if (argc > 2)
	{
		std::println(stderr, "Usage: {} ConfigFile.conf [default: MaNTA.conf]", argv[ 0 ]);
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
