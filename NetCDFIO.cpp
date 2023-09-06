
#include "NetCDFIO.hpp"
#include "SystemSolver.hpp"

// Code for NetCDF interface
//

using namespace netCDF;

NetCDFIO::NetCDFIO() 
{
}

void NetCDFIO::Open( const std::string &file )
{
	filename = file;
	data_file.open( file, netCDF::NcFile::FileMode::replace );
	TimeDim = data_file.addDim( "t" );
	TimeVar = data_file.addVar( "t", netCDF::NcDouble(), TimeDim );
	TimeVar.putAtt( "description", "Time since start of simulation" );
	TimeVar.putAtt( "units", "s" );
	TimeVar.putVar( {0}, 0.0 );
}

void NetCDFIO::Close()
{
	filename = "";
	data_file.close();
}

NetCDFIO::~NetCDFIO()
{
	if ( filename != "" )
		Close();
}

void NetCDFIO::AddScalarVariable( std::string name, std::string description, std::string units, double value )
{
	NcVar newvar = data_file.addVar( name, netCDF::NcDouble() );
	newvar.putAtt( "description", description );
	if ( units != "" )
		newvar.putAtt( "units", units );
	double tmp = value;
	newvar.putVar( &tmp );
}

void NetCDFIO::AddTextVariable( std::string name, std::string description, std::string units, std::string text )
{
	NcVar newvar = data_file.addVar( name, netCDF::NcString() );
	newvar.putAtt( "description", description );
	if ( units != "" )
		newvar.putAtt( "units", units );
	std::string value( text );
	const char *pStr = value.c_str();
	newvar.putVar( &pStr );
}

void NetCDFIO::AddTimeSeries( std::string name, std::string description, std::string units, double InitialValue )
{
	NcVar newvar = data_file.addVar( name, netCDF::NcDouble(), TimeDim );
	newvar.putAtt( "description", description );
	if ( units != "" )
		newvar.putAtt( "units", units );
	newvar.putVar( {0}, InitialValue );
}

size_t NetCDFIO::AddTimeSlice( double T )
{
	size_t next = TimeDim.getSize();
	std::vector<size_t> v = {next};
	TimeVar.putVar( v, T );
	return next;
}

void NetCDFIO::AppendToTimeSeries( std::string const& name, double value, size_t tIndex )
{
	NcVar variable = data_file.getVar( name );
	std::vector<size_t> v = {tIndex};
	variable.putVar( v, value );
}

template<typename T> void NetCDFIO::AppendToVariable( std::string const& name, T const& var, size_t tIndex )
{
	NcVar variable = data_file.getVar( name );
	std::vector<double> gridValues;
	gridValues.resize( gridpoints.size() );
	for ( size_t i = 0; i < gridpoints.size(); ++i )
		gridValues[ i ] = var( gridpoints[ i ] );

	variable.putVar( {tIndex,0}, {1,gridpoints.size()}, gridValues.data() );
}

void NetCDFIO::SetOutputGrid( std::vector<double> const& gridpoints_ )
{
	gridpoints = gridpoints_;
	SpaceDim = data_file.addDim( "x", gridpoints.size() );
	SpaceVar = data_file.addVar( "x", netCDF::NcDouble(), SpaceDim );
	SpaceVar.putVar( {0}, {gridpoints.size()}, gridpoints.data() );
}

template<typename T> void NetCDFIO::AddVariable( std::string name, std::string description, std::string units, T const& initialValue )
{
	NcVar newvar = data_file.addVar( name, netCDF::NcDouble(), {TimeDim, SpaceDim} );
	newvar.putAtt( "description", description );
	if ( units != "" )
		newvar.putAtt( "units", units );
	std::vector<double> gridValues;
	gridValues.resize( gridpoints.size() );
	for ( size_t i = 0; i < gridpoints.size(); ++i )
		gridValues[ i ] = initialValue( gridpoints[ i ] );

	newvar.putVar( {0,0}, {1,gridpoints.size()}, gridValues.data() );
}


// SystemSolver routines that use NetCDFIO

void SystemSolver::initialiseNetCDF( std::string const& NetcdfOutputFile, size_t nOut )
{
	nc_output.Open( NetcdfOutputFile );
	std::vector<double> gridpoints( nOut );
	std::ranges::generate( gridpoints, [ this,nOut ](){
		static int i = 0;
		double delta_x = ( grid.upperBoundary() - grid.lowerBoundary() ) * ( 1.0/( nOut - 1.0 ) );
		return static_cast<double>( i++ )*delta_x + grid.lowerBoundary();
	} );

	nc_output.SetOutputGrid( gridpoints );
	// Add diagnostic hooks
	
	nc_output.AddScalarVariable( "nVariables", "Number of independent variables", "", static_cast<double>( nVars ) );
	
	for ( Index i=0; i<nVars; ++i ) {
		auto initial_v = [this,i]( Position x ) { return problem->InitialValue( i, x ); };
		nc_output.AddVariable( problem->getVariableName( i ), problem->getVariableDescription( i ), problem->getVariableUnits( i ), initial_v );
	}
}

void SystemSolver::WriteTimeslice( double tNew )
{
	size_t tIndex = nc_output.AddTimeSlice( tNew );

	for ( Index i=0; i<nVars; ++i ) {
		nc_output.AppendToVariable( problem->getVariableName( i ), y.u( i ), tIndex );
	}

}




