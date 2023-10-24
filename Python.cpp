#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

int runManta( std::string const& );

PYBIND11_MODULE( MaNTA, m ) 
{
	m.doc() = "Python bindings for MaNTA";
	m.def( "run", runManta, "Runs the MaNTA suite using given configuration file" );
}
