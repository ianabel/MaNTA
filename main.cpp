
#include <string>
#include <print>

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

	return runManta( fname );
}
