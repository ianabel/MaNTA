#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <boost/math/tools/roots.hpp>

#include "SystemSolver.hpp"

void runSolver(std::shared_ptr<SystemSolver> system, std::string const& inputFile);

int main( int argc, char** argv )
{
	//std::cerr.precision(17);
	std::string fname( "mts.conf" );
	if ( argc == 2 )
		fname = argv[ 1 ];
	if ( argc > 2 )
	{
		std::cerr << "Usage: MTS++ ConfigFile.conf" << std::endl;
		return 1;
	}

	std::shared_ptr<SystemSolver> system = std::make_shared<SystemSolver>(fname);
	runSolver(system, fname);
}