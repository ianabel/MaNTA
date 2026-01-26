
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

void NetCDFIO::AddScalarVariable(std::string groupName, std::string name, std::string description, std::string units, double value)
{
	NcVar newvar = data_file.getGroup(groupName).addVar(name, netCDF::NcDouble());
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

void NetCDFIO::AddTimeSeries(std::string groupName, std::string name, std::string description, std::string units, double InitialValue)
{
	netCDF::NcGroup group = data_file.getGroup(groupName);
	NcVar newvar = group.addVar(name, netCDF::NcDouble(), TimeDim);
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

void NetCDFIO::AppendToTimeSeries(std::string const &groupName, std::string const &name, double value, size_t tIndex)
{
	netCDF::NcVar variable = data_file.getGroup(groupName).getVar(name);
	std::vector<size_t> v = {tIndex};
	variable.putVar(v, value);
}

void NetCDFIO::SetOutputGrid(std::vector<double> const &gridpoints_)
{
	gridpoints = gridpoints_;
	SpaceDim = data_file.addDim("x", gridpoints.size());
	SpaceVar = data_file.addVar("x", netCDF::NcDouble(), SpaceDim);
	SpaceVar.putVar({0}, {gridpoints.size()}, gridpoints.data());
}

void NetCDFIO::AddGroup(std::string name, std::string description)
{
	NcGroup newgroup = data_file.addGroup(name);
	newgroup.putAtt("description", description);
}

netCDF::NcGroup NetCDFIO::CreateGroup(std::string name, std::string description)
{
	NcGroup newgroup = data_file.addGroup(name);
	newgroup.putAtt("description", description);
	return newgroup;
}

// SystemSolver routines that use NetCDFIO

void NetCDFIO::StoreGridInfo(const Grid &grid, unsigned int k)
{
	std::vector<double> CellBoundaries(grid.getNCells() + 1);
	CellBoundaries[0] = grid.lowerBoundary();
	for (Grid::Index i = 0; i < grid.getNCells(); ++i)
		CellBoundaries[i + 1] = grid[i].x_u;
	NcGroup gridGroup = data_file.addGroup("Grid");
	gridGroup.putAtt("Description", "Information about the underlying grid used for the simulation");
	std::vector<int> indexes(CellBoundaries.size());
	int n = 0;
	std::ranges::generate(indexes, [&n]() mutable
						  { return n++; });
	NcDim indexDim = gridGroup.addDim("Index", CellBoundaries.size());
	NcVar indexVar = gridGroup.addVar("Index", netCDF::NcInt(), indexDim);
	indexVar.putVar({0}, {indexes.size()}, indexes.data());
	NcVar cellBoundaries = gridGroup.addVar("CellBoundaries", netCDF::NcDouble(), indexDim);
	cellBoundaries.putVar({0}, {CellBoundaries.size()}, CellBoundaries.data());

	NcVar order = gridGroup.addVar("PolyOrder", netCDF::NcInt());
	order.putAtt("Description", "Order of Polynomials used in HDG representation");
	order.putVar(&k);
}

void SystemSolver::initialiseNetCDF(std::string const &NetcdfOutputFile, size_t nOut)
{
	nc_output.Open(NetcdfOutputFile);
	std::vector<double> gridpoints(nOut);
	for (unsigned int i = 0; i < nOut; ++i)
		gridpoints[i] = grid.lowerBoundary() + i * (grid.upperBoundary() - grid.lowerBoundary()) / (nOut - 1);

	nc_output.SetOutputGrid(gridpoints);

	nc_output.StoreGridInfo(grid, k);

	nc_output.AddScalarVariable("nVariables", "Number of independent variables", "", static_cast<double>(nVars));

	for (Index i = 0; i < nVars; ++i)
	{
		nc_output.AddGroup(problem->getVariableName(i), problem->getVariableDescription(i));
		nc_output.AddVariable(problem->getVariableName(i), "u", "Value", problem->getVariableUnits(i), y.u(i));
		nc_output.AddVariable(problem->getVariableName(i), "q", "Derivative", problem->getVariableUnits(i), y.q(i));
		nc_output.AddVariable(problem->getVariableName(i), "sigma", "Flux", problem->getVariableUnits(i), y.sigma(i));
	}

	for (Index i = 0; i < nScalars; ++i)
	{
		nc_output.AddTimeSeries(problem->getScalarName(i), problem->getScalarDescription(i), problem->getScalarUnits(i), y.Scalar(i));
	}

	for (Index i = 0; i < nAux; ++i)
	{
		nc_output.AddVariable(problem->getAuxVarName(i), problem->getAuxDescription(i), problem->getAuxUnits(i), y.Aux(i));
	}

	problem->initialiseDiagnostics(nc_output);
}

void SystemSolver::WriteTimeslice(double tNew)
{
	size_t tIndex = nc_output.AddTimeSlice(tNew);

	for (Index i = 0; i < nVars; ++i)
	{
		nc_output.AppendToGroup<DGSoln::DGApprox>(problem->getVariableName(i), tIndex, {{"u", y.u(i)}, {"q", y.q(i)}, {"sigma", y.sigma(i)}});
	}

	for (Index i = 0; i < nAux; ++i)
	{
		nc_output.AppendToVariable(problem->getAuxVarName(i), y.Aux(i), tIndex);
	}

	for (Index i = 0; i < nScalars; ++i)
		nc_output.AppendToTimeSeries(problem->getScalarName(i), y.Scalar(i), tIndex);

	problem->writeDiagnostics(y, dydt, tNew, nc_output, tIndex);
}

void SystemSolver::WriteAdjoints()
{
	nc_output.AddScalarVariable("GFn", "", "", adjointProblem->GFn(0, y));
	nc_output.AddScalarVariable("np", "", "", adjointProblem->getNp());
	nc_output.AddScalarVariable("np_boundary", "", "", adjointProblem->getNpBoundary());
	nc_output.AddGroup("G_p", "Gradients of G using adjoint state method");
	nc_output.AddGroup("G_boundary", "Gradients of G on boundary using adjoint state method");
	for (Index i = 0; i < adjointProblem->getNp() - adjointProblem->getNpBoundary(); ++i)
	{
		nc_output.AddScalarVariable("G_p", "p" + std::to_string(i), "", "", G_p(i));
	}
	for (Index i = 0; i < adjointProblem->getNpBoundary(); ++i)
	{
		nc_output.AddScalarVariable("G_boundary", "p" + std::to_string(i), "", "", G_p(i + adjointProblem->getNp() - adjointProblem->getNpBoundary()));
	}
}

void SystemSolver::WriteRestartFile(std::string const &fname, N_Vector const &Y, N_Vector const &dYdt, size_t nOut)
{
	restart_file.Open(fname);

	// Include profiles for debugging
	std::vector<double> gridpoints(nOut);
	for (unsigned int i = 0; i < nOut; ++i)
		gridpoints[i] = grid.lowerBoundary() + i * (grid.upperBoundary() - grid.lowerBoundary()) / (nOut - 1);

	restart_file.SetOutputGrid(gridpoints);

	restart_file.StoreGridInfo(grid, k);

	restart_file.AddScalarVariable("nVariables", "Number of independent variables", "", static_cast<double>(nVars));

	for (Index i = 0; i < nVars; ++i)
	{
		restart_file.AddGroup(problem->getVariableName(i), problem->getVariableDescription(i));
		restart_file.AddVariable(problem->getVariableName(i), "u", "Value", problem->getVariableUnits(i), y.u(i));
		restart_file.AddVariable(problem->getVariableName(i), "q", "Derivative", problem->getVariableUnits(i), y.q(i));
		restart_file.AddVariable(problem->getVariableName(i), "sigma", "Flux", problem->getVariableUnits(i), y.sigma(i));
	}

	for (Index i = 0; i < nScalars; ++i)
	{
		restart_file.AddTimeSeries(problem->getScalarName(i), problem->getScalarDescription(i), problem->getScalarUnits(i), y.Scalar(i));
	}

	for (Index i = 0; i < nAux; ++i)
	{
		restart_file.AddVariable(problem->getAuxVarName(i), problem->getAuxDescription(i), problem->getAuxUnits(i), y.Aux(i));
	}

	// Save N_Vector directly
	NcGroup RestartGroup = restart_file.CreateGroup("RestartData", "Restart group");

	const size_t nDOF = nVars * 3 * nCells * (k + 1) + nVars * (nCells + 1) + nScalars + nAux * nCells * (k + 1);
	NcDim yDim = RestartGroup.addDim("nDOF", nDOF);
	RestartGroup.addVar("nVars", netCDF::NcInt()).putVar(&nVars);
	RestartGroup.addVar("nAux", netCDF::NcInt()).putVar(&nAux);
	RestartGroup.addVar("nScalars", netCDF::NcInt()).putVar(&nScalars);

	RestartGroup.addVar("Y", netCDF::NcDouble(), yDim).putVar({0}, {nDOF}, N_VGetArrayPointer(Y));
	RestartGroup.addVar("dYdt", netCDF::NcDouble(), yDim).putVar({0}, {nDOF}, N_VGetArrayPointer(dYdt));

	restart_file.Close();
}
