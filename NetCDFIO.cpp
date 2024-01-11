
#include <ranges>

#include "NetCDFIO.hpp"
#include "SystemSolver.hpp"

// Code for NetCDF interface
//

using namespace netCDF;

NetCDFIO::NetCDFIO()
{
}

void NetCDFIO::Open(const std::string &file)
{
	filename = file;
	data_file.open(file, netCDF::NcFile::FileMode::replace);
	TimeDim = data_file.addDim("t");
	TimeVar = data_file.addVar("t", netCDF::NcDouble(), TimeDim);
	TimeVar.putAtt("description", "Time since start of simulation");
	TimeVar.putAtt("units", "s");
	TimeVar.putVar({0}, 0.0);
}

void NetCDFIO::Close()
{
	filename = "";
	data_file.close();
}

NetCDFIO::~NetCDFIO()
{
	if (filename != "")
		Close();
}

void NetCDFIO::AddScalarVariable(std::string name, std::string description, std::string units, double value)
{
	NcVar newvar = data_file.addVar(name, netCDF::NcDouble());
	newvar.putAtt("description", description);
	if (units != "")
		newvar.putAtt("units", units);
	double tmp = value;
	newvar.putVar(&tmp);
}

void NetCDFIO::AddTextVariable(std::string name, std::string description, std::string units, std::string text)
{
	NcVar newvar = data_file.addVar(name, netCDF::NcString());
	newvar.putAtt("description", description);
	if (units != "")
		newvar.putAtt("units", units);
	std::string value(text);
	const char *pStr = value.c_str();
	newvar.putVar(&pStr);
}

void NetCDFIO::AddTimeSeries(std::string name, std::string description, std::string units, double InitialValue)
{
	NcVar newvar = data_file.addVar(name, netCDF::NcDouble(), TimeDim);
	newvar.putAtt("description", description);
	if (units != "")
		newvar.putAtt("units", units);
	newvar.putVar({0}, InitialValue);
}

size_t NetCDFIO::AddTimeSlice(double T)
{
	size_t next = TimeDim.getSize();
	std::vector<size_t> v = {next};
	TimeVar.putVar(v, T);
	return next;
}

void NetCDFIO::AppendToTimeSeries(std::string const &name, double value, size_t tIndex)
{
	NcVar variable = data_file.getVar(name);
	std::vector<size_t> v = {tIndex};
	variable.putVar(v, value);
}

template <typename T>
void NetCDFIO::AppendToVariable(std::string const &name, T const &var, size_t tIndex)
{
	NcVar variable = data_file.getVar(name);
	std::vector<double> gridValues;
	gridValues.resize(gridpoints.size());
	for (size_t i = 0; i < gridpoints.size(); ++i)
		gridValues[i] = var(gridpoints[i]);

	variable.putVar({tIndex, 0}, {1, gridpoints.size()}, gridValues.data());
}

template <typename T>
void NetCDFIO::AppendToGroup(std::string const &name, size_t tIndex, const std::initializer_list<std::pair<std::string, T>> &vars)
{

	std::vector<double> gridValues;
	gridValues.resize(gridpoints.size());

	NcGroup group = data_file.getGroup(name);
	for (auto &var : vars)
	{
		for (size_t i = 0; i < gridpoints.size(); ++i)
			gridValues[i] = var.second(gridpoints[i]);

		group.getVar(var.first).putVar({tIndex, 0}, {1, gridpoints.size()}, gridValues.data());
	}
}

void NetCDFIO::SetOutputGrid(std::vector<double> const &gridpoints_)
{
	gridpoints = gridpoints_;
	SpaceDim = data_file.addDim("x", gridpoints.size());
	SpaceVar = data_file.addVar("x", netCDF::NcDouble(), SpaceDim);
	SpaceVar.putVar({0}, {gridpoints.size()}, gridpoints.data());
}

template <typename T>
void NetCDFIO::AddVariable(std::string name, std::string description, std::string units, T const &initialValue)
{
	NcVar newvar = data_file.addVar(name, netCDF::NcDouble(), {TimeDim, SpaceDim});
	newvar.putAtt("description", description);
	if (units != "")
		newvar.putAtt("units", units);
	std::vector<double> gridValues;
	gridValues.resize(gridpoints.size());
	for (size_t i = 0; i < gridpoints.size(); ++i)
		gridValues[i] = initialValue(gridpoints[i]);

	newvar.putVar({0, 0}, {1, gridpoints.size()}, gridValues.data());
}

template <typename T>
void NetCDFIO::AddVariable(std::string groupName, std::string name, std::string description, std::string units, T const &initialValue)
{
	NcGroup group = data_file.getGroup(groupName);
	NcVar newvar = group.addVar(name, netCDF::NcDouble(), {TimeDim, SpaceDim});
	newvar.putAtt("description", description);
	if (units != "")
		newvar.putAtt("units", units);
	std::vector<double> gridValues;
	gridValues.resize(gridpoints.size());
	for (size_t i = 0; i < gridpoints.size(); ++i)
		gridValues[i] = initialValue(gridpoints[i]);

	newvar.putVar({0, 0}, {1, gridpoints.size()}, gridValues.data());
}

void NetCDFIO::AddGroup(std::string name, std::string description, std::vector<double> const &gridpoints_)
{
	NcGroup newgroup = data_file.addGroup(name);
	newgroup.putAtt("description", description);
}

// SystemSolver routines that use NetCDFIO

void NetCDFIO::StoreGridInfo( const Grid& grid, unsigned int k )
{
	std::vector<double> CellBoundaries( grid.getNCells() + 1 );
	CellBoundaries[ 0 ] = grid.lowerBoundary();
	for ( Grid::Index i = 0; i < grid.getNCells(); ++i )
		CellBoundaries[ i + 1 ] = grid[ i ].x_u;
	NcGroup gridGroup = data_file.addGroup( "Grid" );
	gridGroup.putAtt( "Description", "Information about the underlying grid used for the simulation" );
	std::vector<int> indexes( CellBoundaries.size() );
	int n = 0;
	std::ranges::generate( indexes, [&n]() mutable { return n++; } );
	NcDim indexDim = gridGroup.addDim( "Index", CellBoundaries.size() );
	NcVar indexVar = gridGroup.addVar( "Index", netCDF::NcInt(), indexDim );
	indexVar.putVar({0}, {indexes.size()}, indexes.data());
	NcVar cellBoundaries = gridGroup.addVar("CellBoundaries", netCDF::NcDouble(), indexDim );
	cellBoundaries.putVar({0}, {CellBoundaries.size()}, CellBoundaries.data());

	NcVar order = gridGroup.addVar( "PolyOrder", netCDF::NcInt() );
	order.putAtt( "Description", "Order of Polynomials used in HDG representation" );
	order.putVar( &k );

}

void SystemSolver::initialiseNetCDF(std::string const &NetcdfOutputFile, size_t nOut)
{
	nc_output.Open(NetcdfOutputFile);
	std::vector<double> gridpoints(nOut);
	std::ranges::generate(gridpoints, [this, nOut]() {
		static int i = 0;
		double delta_x = ( grid.upperBoundary() - grid.lowerBoundary() ) * ( 1.0/( nOut - 1.0 ) );
		return static_cast<double>( i++ )*delta_x + grid.lowerBoundary(); 
		});

	nc_output.SetOutputGrid(gridpoints);

	nc_output.StoreGridInfo( grid, k );



	nc_output.AddScalarVariable("nVariables", "Number of independent variables", "", static_cast<double>(nVars));

	for (Index i = 0; i < nVars; ++i)
	{
		nc_output.AddGroup(problem->getVariableName(i), problem->getVariableDescription(i), gridpoints);
		nc_output.AddVariable(problem->getVariableName(i), "u", "Value", problem->getVariableUnits(i), y.u(i));
		nc_output.AddVariable(problem->getVariableName(i), "q", "Derivative", problem->getVariableUnits(i), y.q(i));
		nc_output.AddVariable(problem->getVariableName(i), "sigma", "Flux", problem->getVariableUnits(i), y.sigma(i));
	}

	for ( Index i=0; i < nScalars; ++i )
	{
		nc_output.AddTimeSeries( problem->getScalarName( i ), problem->getScalarDescription( i ), problem->getScalarUnits( i ),y.Scalar( i ) );
	}

	problem->initialiseDiagnostics( nc_output );
}

void SystemSolver::WriteTimeslice(double tNew)
{
	size_t tIndex = nc_output.AddTimeSlice(tNew);

	for (Index i = 0; i < nVars; ++i)
	{
		nc_output.AppendToGroup<DGApprox>(problem->getVariableName(i), tIndex, {{"u", y.u(i)}, {"q", y.q(i)}, {"sigma", y.sigma(i)}});
	}

	for ( Index i = 0; i < nScalars; ++i )
		nc_output.AppendToTimeSeries( problem->getScalarName( i ), y.Scalar( 0 ), tIndex );

	problem->writeDiagnostics( y, tNew, nc_output, tIndex );
}


