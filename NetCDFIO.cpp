
#include <ranges>

#include "NetCDFIO.hpp"
#include "SystemSolver.hpp"
// SystemSolver.hpp only forward-declares FieldModel; the field group below needs
// the spec that names its variables.
#include "FieldModel.hpp"

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

	// u_star is the element-local postprocessed value: superconvergent where the
	// theory applies, and in any case a strictly better representation of u than
	// u itself. Absent when k = 0, where there is no reconstruction to do.
	if (postprocessor)
		postprocessor->computeUStar(y);

	for (Index i = 0; i < nVars; ++i)
	{
		nc_output.AddGroup(problem->getVariableName(i), problem->getVariableDescription(i));
		nc_output.AddVariable(problem->getVariableName(i), "u", "Value", problem->getVariableUnits(i), y.u(i));
		nc_output.AddVariable(problem->getVariableName(i), "q", "Derivative", problem->getVariableUnits(i), y.q(i));
		nc_output.AddVariable(problem->getVariableName(i), "sigma", "Flux", problem->getVariableUnits(i), y.sigma(i));
		if (postprocessor)
			nc_output.AddVariable(problem->getVariableName(i), "u_star",
								  "Postprocessed value", problem->getVariableUnits(i),
								  postprocessor->uStar(i));
	}

	for (Index i = 0; i < nScalars; ++i)
	{
		nc_output.AddTimeSeries(problem->getScalarName(i), problem->getScalarDescription(i), problem->getScalarUnits(i), y.Scalar(i));
	}

	for (Index i = 0; i < nAux; ++i)
	{
		nc_output.AddVariable(problem->getAuxVarName(i), problem->getAuxDescription(i), problem->getAuxUnits(i), y.Aux(i));
	}

	// `t` rather than t0 spelled out: setInitialConditions sets it to t0 and
	// nothing in the time loop moves it, so the two are the same here -- but t is
	// what the rest of this function's state was built at.
	initialiseFieldOutput(nc_output, t);

	problem->initialiseDiagnostics(nc_output);
}

// See the declaration. Creates the group and writes its t0 slice; the geometry
// slots go in as spatial variables because that is what they are, and the field
// DOFs as time series because psi has no x dependence -- the same split the
// transport variables and the global scalars already get.
void SystemSolver::initialiseFieldOutput(NetCDFIO &file, Time tEval)
{
	if (!fieldModel)
		return;

	FieldModelSpec const &fspec = fieldModel->getSpec();

	NcGroup group = file.CreateGroup(fspec.name, "Self-consistent magnetic field model");
	// What the model's x means. MaNTA does not interpret it -- the provider
	// declares its own coordinate and supplies the metric on it -- so recording
	// it is the only way a reader can tell what the geometry is a function of.
	group.putAtt("label", fspec.label);

	for (Index f = 0; f < nField; ++f)
		file.AddTimeSeries(fspec.name, fspec.dofs[f].name, fspec.dofs[f].description,
						   fspec.dofs[f].units, y.Field(f));

	// A copy, not the map: the lambda outlives the statement that builds it, and
	// `y` is a view over memory SUNDIALS owns.
	const Vector psi = y.getField();
	for (Index g = 0; g < nGeom; ++g)
		file.AddVariable(fspec.name, fspec.geometry[g].name, fspec.geometry[g].description,
						 fspec.geometry[g].units,
						 [this, &psi, g, tEval](Position x)
						 {
							 Vector slots = Vector::Zero(nGeom);
							 fieldModel->Geometry(slots, psi, x, tEval);
							 return slots(g);
						 });
}

void SystemSolver::writeFieldTimeslice(NetCDFIO &file, size_t tIndex, Time tEval)
{
	if (!fieldModel)
		return;

	FieldModelSpec const &fspec = fieldModel->getSpec();

	for (Index f = 0; f < nField; ++f)
		file.AppendToTimeSeries(fspec.name, fspec.dofs[f].name, y.Field(f), tIndex);

	const Vector psi = y.getField();
	for (Index g = 0; g < nGeom; ++g)
		file.AppendToGroup(fspec.name, tIndex, fspec.geometry[g].name,
						   [this, &psi, g, tEval](Position x)
						   {
							   Vector slots = Vector::Zero(nGeom);
							   fieldModel->Geometry(slots, psi, x, tEval);
							   return slots(g);
						   });
}

void SystemSolver::WriteTimeslice(double tNew)
{
	size_t tIndex = nc_output.AddTimeSlice(tNew);

	if (postprocessor)
		postprocessor->computeUStar(y);

	for (Index i = 0; i < nVars; ++i)
	{
		nc_output.AppendToGroup<DGSoln::DGApprox>(problem->getVariableName(i), tIndex, {{"u", y.u(i)}, {"q", y.q(i)}, {"sigma", y.sigma(i)}});
		if (postprocessor)
			nc_output.AppendToGroup<DGSoln::DGApprox>(problem->getVariableName(i), tIndex,
													  "u_star", postprocessor->uStar(i));
	}

	for (Index i = 0; i < nAux; ++i)
	{
		nc_output.AppendToVariable(problem->getAuxVarName(i), y.Aux(i), tIndex);
	}

	for (Index i = 0; i < nScalars; ++i)
		nc_output.AppendToTimeSeries(problem->getScalarName(i), y.Scalar(i), tIndex);

	writeFieldTimeslice(nc_output, tIndex, tNew);

	problem->writeDiagnostics(y, dydt, tNew, nc_output, tIndex);
}

void SystemSolver::WriteAdjoints()
{

	nc_output.AddScalarVariable("ng", "", "", adjointProblem->getNg());
	nc_output.AddScalarVariable("np", "", "", adjointProblem->getNp());
	nc_output.AddScalarVariable("np_boundary", "", "", adjointProblem->getNpBoundary());
	for (Index i = 0; i < adjointProblem->getNg(); ++i)
	{
		nc_output.AddScalarVariable("G" + std::to_string(i), "G function", "", adjointProblem->GFn(i, y));
		nc_output.AddGroup("G" + std::to_string(i) + "_p", "Gradients of G using adjoint state method");
		nc_output.AddGroup("G" + std::to_string(i) + "_boundary", "Gradients of G on boundary using adjoint state method");
		for (Index j = 0; j < adjointProblem->getNpInternal(); ++j)
		{
			// if (adjointProblem->areParametersSpatial())
			// {
			// 	nc_output.AddVariable("G" + std::to_string(i) + "_p", "p" + std::to_string(j), "Gradient of G with respect to p" + std::to_string(j), "", G_p.block(i * nCells, j, nCells, 1));
			// }
			// else
			// {
			nc_output.AddScalarVariable("G" + std::to_string(i) + "_p", adjointProblem->getName(j), "", "", G_p(i, j));
			// }
		}
		for (Index j = 0; j < adjointProblem->getNpBoundary(); ++j)
		{
			nc_output.AddScalarVariable("G" + std::to_string(i) + "_boundary", "p" + std::to_string(j), "", "", G_p(i, j + adjointProblem->getNp() - adjointProblem->getNpBoundary()));
		}
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

	if (postprocessor)
		postprocessor->computeUStar(y);

	for (Index i = 0; i < nVars; ++i)
	{
		restart_file.AddGroup(problem->getVariableName(i), problem->getVariableDescription(i));
		restart_file.AddVariable(problem->getVariableName(i), "u", "Value", problem->getVariableUnits(i), y.u(i));
		restart_file.AddVariable(problem->getVariableName(i), "q", "Derivative", problem->getVariableUnits(i), y.q(i));
		restart_file.AddVariable(problem->getVariableName(i), "sigma", "Flux", problem->getVariableUnits(i), y.sigma(i));
		if (postprocessor)
			restart_file.AddVariable(problem->getVariableName(i), "u_star",
									 "Postprocessed value", problem->getVariableUnits(i),
									 postprocessor->uStar(i));
	}

	for (Index i = 0; i < nScalars; ++i)
	{
		restart_file.AddTimeSeries(problem->getScalarName(i), problem->getScalarDescription(i), problem->getScalarUnits(i), y.Scalar(i));
	}

	for (Index i = 0; i < nAux; ++i)
	{
		restart_file.AddVariable(problem->getAuxVarName(i), problem->getAuxDescription(i), problem->getAuxUnits(i), y.Aux(i));
	}

	// `tret`, not the member `t`: t is set to t0 by setInitialConditions and
	// nothing in the time loop moves it, whereas the state being written here is
	// the one the run *ended* at. Geometry is a function of (psi, x, t), so the
	// two choices differ for any model with explicit time dependence. tret is
	// initialised to t0 in initialize(), so the steady-state path -- which never
	// enters the time loop and freezes time-dependent data at t_initial -- gets
	// the time it means.
	initialiseFieldOutput(restart_file, tret);

	// Save N_Vector directly
	NcGroup RestartGroup = restart_file.CreateGroup("RestartData", "Restart group");

	// Asked of the DGSoln rather than open-coded, which is what this line used to
	// be. There were three copies of this formula -- here, the N_VNew_Serial in
	// Solver.cpp and the read-side consistency check in MaNTA.cpp -- and a field
	// model lengthens the vector, so a copy that did not know about nField wrote
	// a *short* file: psi dropped, and an nDOF recorded that matches the
	// uncoupled formula, so the truncated file would read back as consistent.
	// Under-writing rather than overrunning is what makes that silent. The
	// solver's own `y` maps the vector being written, so it is the authority on
	// how long it is; there is now one fewer place for the two to disagree.
	const size_t nDOF = y.getDoF();
	NcDim yDim = RestartGroup.addDim("nDOF", nDOF);
	RestartGroup.addVar("nVars", netCDF::NcInt()).putVar(&nVars);
	RestartGroup.addVar("nAux", netCDF::NcInt()).putVar(&nAux);
	RestartGroup.addVar("nScalars", netCDF::NcInt()).putVar(&nScalars);

	// psi has been *in* Y since the DoF accounting above stopped being open-coded
	// -- what was missing was any record of how much of Y it is. Without this the
	// reader has to infer nField by subtracting the uncoupled formula from nDOF,
	// which is exactly the arithmetic that was wrong in three places; and a file
	// written with a field model would read back into a solver configured without
	// one as a length mismatch blamed on nVars/nAux/nScalars. Written
	// unconditionally, so an uncoupled restart file says nField = 0 rather than
	// leaving a reader to distinguish "no field" from "old file".
	RestartGroup.addVar("nField", netCDF::NcInt()).putVar(&nField);

	RestartGroup.addVar("Y", netCDF::NcDouble(), yDim).putVar({0}, {nDOF}, N_VGetArrayPointer(Y));
	RestartGroup.addVar("dYdt", netCDF::NcDouble(), yDim).putVar({0}, {nDOF}, N_VGetArrayPointer(dYdt));

	restart_file.Close();
}
