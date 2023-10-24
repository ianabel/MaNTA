
#include <string>
#include <iostream>

int runManta( std::string const& );

int main( int argc, char** argv )
{
	std::string fname("MaNTA.conf");
	if (argc == 2)
		fname = argv[1];
	if (argc > 2)
	{
		std::cerr << "Usage: " << argv[ 0 ] << " ConfigFile.conf [default: MaNTA.conf]" << std::endl;
		return 1;
	}

	return runManta( fname );
}
