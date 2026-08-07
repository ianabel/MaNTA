#include <memory>
#include <boost/math/tools/roots.hpp>
#include <toml.hpp>
#include <filesystem>

#include "SystemSolver.hpp"
#include "PhysicsCases.hpp"
#include "Config.hpp"

// Load restart data into vectors
int LoadFromFile(netCDF::NcFile &restart_file, std::vector<double> &Y, std::vector<double> &dYdt)
{
	netCDF::NcGroup RestartGroup = restart_file.getGroup("RestartData");

	Index nDOF = RestartGroup.getDim("nDOF").getSize();

	Y.resize(nDOF);
	dYdt.resize(nDOF);

	RestartGroup.getVar("Y").getVar(Y.data());
	RestartGroup.getVar("dYdt").getVar(dYdt.data());

	restart_file.close();

	return nDOF;
}

int runManta(std::string const &fname)
{
	std::filesystem::path config_file_path(fname);
	if (!std::filesystem::exists(config_file_path))
	{
    logmsg<LOG_LEVEL::ERROR>("Configuration file {} does not exist.", fname);
		return 1;
	}

	const auto configObject = toml::parse(fname);

	std::shared_ptr<SystemSolver> system;

	// Parse config file for generic configuration options (not physics specific ones)
	const auto configFile = toml::parse(fname);
	const auto config = toml::find<toml::value>(configFile, "configuration");

	bool isRestarting = toml::find_or(config, "restart", false);
	netCDF::NcFile restart_file;

	if (isRestarting)
	{
		std::string fbase = std::filesystem::path(fname).stem();
		std::string fileName = toml::find_or(config, "RestartFile", fbase + ".restart.nc");
		try
		{
			restart_file.open(fileName, netCDF::NcFile::FileMode::read);
		}
		catch (...)
		{
      logmsg<LOG_LEVEL::ERROR>("Failed to open restart netCDF file at: {}", std::string(std::filesystem::absolute(std::filesystem::path(fileName))));
      return 1;
		}
	}

	std::unique_ptr<Grid> grid;

	//Grid *grid;
	unsigned int k = 1;
	if (!isRestarting)
	{

		// Solver parameters
		double lBound, uBound, lowerBoundaryFraction, upperBoundaryFraction;
		bool highGridBoundary;
		int nCells;

		auto polyDegree = toml::find(config, "Polynomial_degree");
		if (config.count("Polynomial_degree") != 1)
			throw std::invalid_argument("Polynomial_degree unspecified or specified more than once");
		else if (!polyDegree.is_integer())
			throw std::invalid_argument("Polynomial_degree must be specified as an integer");
		else
			k = polyDegree.as_integer();

		if (config.count("High_Grid_Boundary") != 1)
		{
			highGridBoundary = false;
			lowerBoundaryFraction = 0.0;
			upperBoundaryFraction = 0.0;
		}
		else
		{
			highGridBoundary = config.at("High_Grid_Boundary").as_boolean();
			lowerBoundaryFraction = toml::find_or(config, "Lower_Boundary_Fraction", 0.2);
			upperBoundaryFraction = toml::find_or(config, "Upper_Boundary_Fraction", 0.2);
		}

		auto numberOfCells = toml::find(config, "Grid_size");
		if (config.count("Grid_size") != 1)
			throw std::invalid_argument("Grid_size unspecified or specified more than once");
		if (!numberOfCells.is_integer())
			throw std::invalid_argument("Grid_size must be specified as an integer");
		else
			nCells = numberOfCells.as_integer();

		if (nCells < 4 && highGridBoundary)
			throw std::invalid_argument("Grid size must exceed 4 cells in order to implemet dense boundaries");

		// The is_integer() branches called as_floating(), which throws
		// toml::type_error on an integer node -- so `Lower_boundary = 0` failed
		// with "as_floating(): bad_cast" despite the branch existing precisely
		// to accept it. Same defect as the one fixed in Config.cpp; these two
		// were missed because they are open-coded here rather than going
		// through getFloat.
		auto lowerBoundary = toml::find(config, "Lower_boundary");
		if (config.count("Lower_boundary") != 1)
			throw std::invalid_argument("Lower_boundary unspecified or specified more than once");
		else if (lowerBoundary.is_integer())
			lBound = static_cast<double>(lowerBoundary.as_integer());
		else if (lowerBoundary.is_floating())
			lBound = static_cast<double>(lowerBoundary.as_floating());
		else
			throw std::invalid_argument("Lower_boundary specified incorrrectly");

		auto upperBoundary = toml::find(config, "Upper_boundary");
		if (config.count("Upper_boundary") != 1)
			throw std::invalid_argument("Upper_boundary unspecified or specified more than once");
		else if (upperBoundary.is_integer())
			uBound = static_cast<double>(upperBoundary.as_integer());
		else if (upperBoundary.is_floating())
			uBound = static_cast<double>(upperBoundary.as_floating());
		else
			throw std::invalid_argument("Upper_boundary specified incorrrectly");

		grid = std::make_unique<Grid>(lBound, uBound, nCells, highGridBoundary, lowerBoundaryFraction, upperBoundaryFraction);
	}
	else
	{
		// Load grid from restart file
		netCDF::NcGroup GridGroup = restart_file.getGroup("Grid");
		auto nPoints = GridGroup.getDim("Index").getSize();
		std::vector<Position> CellBoundaries(nPoints);

		GridGroup.getVar("CellBoundaries").getVar(CellBoundaries.data());

		grid = std::make_unique<Grid>(CellBoundaries);

		GridGroup.getVar("PolyOrder").getVar(&k);
	}

	double tau = getFloatWithDefault("tau", config, 1.0);
	double delta_t = getFloat("delta_t", config);
	double tZero = getFloatWithDefault("t_initial", config, 0.0);
	double tFinal = getFloat("t_final", config);
	double rtol = getFloatWithDefault("Relative_tolerance", config, 1e-3);

	std::vector<double> absTol;

	if (config.count("Absolute_tolerance") == 1)
	{
		auto atol_toml = toml::find(config, "Absolute_tolerance");
		if (atol_toml.is_array())
			absTol = toml::get<std::vector<double>>(atol_toml);
		else
		{
			absTol.resize(1);
			if (atol_toml.is_integer())
				absTol[0] = static_cast<double>(toml::get<int>(atol_toml));
			else
				absTol[0] = toml::get<double>(atol_toml);
		}
	}
	else if (config.count("Absolute_tolerance") == 0)
	{
		absTol.resize(1);
		absTol[0] = 1e-2;
	}
	else
	{
		throw std::invalid_argument("Absolute_tolerance was specified more than once");
	}

	double dt_min = getFloatWithDefault("MinStepSize", config, 1e-7);

	int nOutput = getIntWithDefault("OutputPoints", config, 301);

	bool solveAdjoint = false;
	if (config.count("solveAdjoint") == 1)
		solveAdjoint = config.at("solveAdjoint").as_boolean();

	if (config.count("TransportSystem") != 1)
		throw std::invalid_argument("TransportSystem needs to specified exactly once in the general configuration section");

	std::string ProblemName = config.at("TransportSystem").as_string();

	// Convert string to TransportSystem* instance

	std::unique_ptr<TransportSystem> pProblem = PhysicsCases::InstantiateProblem(ProblemName, configFile, *grid);

	// This check has to come before the first use of pProblem. InstantiateProblem
	// returns nullptr for an unrecognised name, and both the adjoint setup and
	// the restart block below dereference it -- so an unknown TransportSystem
	// used to segfault instead of printing the list of available models.
	if (pProblem == nullptr)
	{
		logmsg<LOG_LEVEL::ERROR>("Could not instantiate a physics model for TransportSystem = {}\n  Available physics models include:  ", ProblemName);
		for (auto pair : *PhysicsCases::map)
		{
			std::println(stderr, "\t{}", pair.first);
		}
		std::println(stderr, "");
		return 1;
	}

	std::unique_ptr<AdjointProblem> adjoint = nullptr;
	if (solveAdjoint)
		adjoint = pProblem->createAdjointProblem();

	if (isRestarting)
	{
		std::vector<double> Y, dYdt;
		Index nDOF_file = LoadFromFile(restart_file, Y, dYdt);

		// Make sure degrees of freedom are consistent with restart file
		const Index nCells = grid->getNCells();
		const Index nDOF = pProblem->getNumVars() * 3 * nCells * (k + 1) + pProblem->getNumVars() * (nCells + 1) + pProblem->getNumScalars() + pProblem->getNumAux() * nCells * (k + 1);

		if (nDOF_file != nDOF)
			throw std::invalid_argument("nVars/nAux/nScalars in restart file inconsistent with physics case");

		pProblem->setRestartValues(Y, dYdt, *grid, k);
	}

	system = std::make_shared<SystemSolver>(*grid, k, pProblem.get());


	system->setOutputCadence(delta_t);
	system->setTolerances(absTol, rtol);
	system->setTau(tau);
	system->setInitialTime(tZero);
	system->setInputFile(fname);
	system->setSolveAdjoint(solveAdjoint);
	if (solveAdjoint)
		system->setAdjointProblem(adjoint.get());

	system->setNOutput(nOutput);
	system->setMinStepSize(dt_min);

	if (config.count("SteadyStateTolerance") == 1)
	{
		double sst = getFloat("SteadyStateTolerance", config);
		logmsg<LOG_LEVEL::INFO>("Running until steady state achieved (variation below {}) or end time reached.", sst);
		// Without this the option was inert: the value was read and logged but
		// never reached the solver, so TerminateOnSteadyState stayed false and
		// the run always went to t_final.
		system->setSteadyStateTolerance(sst);
	}

	system->runSolver(tFinal);

	// For compiled-in TransportSystems we have the type information and
	// this will call the correct inherited destructor

	//delete grid;
	std::println("Done.");
	return 0;
}
