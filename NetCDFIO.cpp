
#include "NetCDFIO.hpp"
#include "MirrorPlasma.hpp"

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

void NetCDFIO::SetOutputGrid( std::vector<double> const& gridpoints )
{
	SpaceDim = data_file.addDim( "x", gridpoints.size() );
	SpaceVar = data_file.addVar( "x", netCDF::NcDouble(), SpaceDim );

}


// SystemSolver routines that use NetCDFIO

void SystemSolver::InitialiseNetCDF( std::String const& NetcdfOutputFile )
{
	nc_output.Open( NetcdfOutputFile );
	nc_output.AddScalarVariable( "R_min","Innermost plasma radius", "m", PlasmaInnerRadius() );
	nc_output.AddScalarVariable( "R_max","Outermost plasma radius", "m", PlasmaOuterRadius() );
	nc_output.AddScalarVariable( "R_plasma","Plasma radius on centreline", "m", PlasmaCentralRadius() );
	nc_output.AddScalarVariable( "IonDensity","Density of bulk ion species in the central cell", "10^20 m^-3", IonDensity );
	nc_output.AddScalarVariable( "ElectronDensity","Density of electrons in the central cell", "10^20 m^-3", ElectronDensity );
	nc_output.AddScalarVariable( "MirrorRatio","Ratio of Minimum to Maximum B along a field line", "", MirrorRatio );

	// Time Dependent Variables
	nc_output.AddTimeSeries( "Voltage", "Voltage drop across the plasma","V", ImposedVoltage );
	nc_output.AddTimeSeries( "AmbipolarPhi", "Parallel phi drop","V", AmbipolarPhi() );
	nc_output.AddTimeSeries( "MachNumber", "Plasma velocity divided by Sqrt(T_e/m_i)", "", MachNumber );
	nc_output.AddTimeSeries( "IonTemperature", "Temperature of the bulk ion species", "keV", IonTemperature );
	nc_output.AddTimeSeries( "ElectronTemperature", "Temperature of the bulk ion species", "keV", ElectronTemperature );

	nc_output.AddTimeSeries( "Current", "Radial current through the plasma","A", RadialCurrent() );
	nc_output.AddTimeSeries( "ViscousTorque", "Viscous Torque","", ViscousTorque() );
	nc_output.AddTimeSeries( "ParAngMomLoss", "Parallel Angular Momentum Loss","", ParallelAngularMomentumLossRate() );

	nc_output.AddTimeSeries( "ViscousHeating", "Viscous Heating","W/m^3", ViscousHeating() );
	nc_output.AddTimeSeries( "ParIonHeatLoss", "Parallel Ion Heat Loss","W/m^3", ParallelIonHeatLoss() );
	nc_output.AddTimeSeries( "ParElecHeatLoss", "Parallel Electron Heat Loss","W/m^3", ParallelElectronHeatLoss() );
	nc_output.AddTimeSeries( "PerpHeatLoss", "Perp Ion Heat Loss","W/m^3", ClassicalIonHeatLoss() );


}

void MirrorPlasma::WriteTimeslice( double tNew )
{
	if ( !isTimeDependent )
		return;

	if ( NetcdfOutputFile == "" )
		return;

	int tIndex = nc_output.AddTimeSlice( tNew );
	if ( ::fabs( tNew - time ) > 1e-6 )
	{
		std::cerr << "Irregularity in output times" << std::endl;
	}
	nc_output.AppendToTimeSeries( "Voltage", ImposedVoltage, tIndex );
	nc_output.AppendToTimeSeries( "AmbipolarPhi", AmbipolarPhi(), tIndex );
	nc_output.AppendToTimeSeries( "MachNumber", MachNumber, tIndex );
	nc_output.AppendToTimeSeries( "IonTemperature", IonTemperature, tIndex );
	nc_output.AppendToTimeSeries( "ElectronTemperature", ElectronTemperature, tIndex );

	nc_output.AppendToTimeSeries( "Current", RadialCurrent(), tIndex );
	nc_output.AppendToTimeSeries( "ViscousTorque", ViscousTorque(), tIndex );
	nc_output.AppendToTimeSeries( "ParAngMomLoss", ParallelAngularMomentumLossRate(), tIndex );

	nc_output.AppendToTimeSeries( "ViscousHeating", ViscousHeating(), tIndex );
	nc_output.AppendToTimeSeries( "ParIonHeatLoss", ParallelIonHeatLoss(), tIndex );
	nc_output.AppendToTimeSeries( "ParElecHeatLoss", ParallelElectronHeatLoss(), tIndex );
	nc_output.AppendToTimeSeries( "PerpHeatLoss", ClassicalIonHeatLoss(), tIndex );

}




