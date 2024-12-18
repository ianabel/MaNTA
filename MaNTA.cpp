#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <boost/math/tools/roots.hpp>
#include <toml.hpp>
#include <filesystem>

#include "SystemSolver.hpp"
#include "PhysicsCases.hpp"

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

double getFloatWithDefault(std::string const &name, toml::value const &config, double defaultValue)
{
	auto confCount = config.count(name);
	if (confCount == 0)
	{
		std::cerr << "INFO: Using default value " << defaultValue << " for configuration option " << name << std::endl;
		return defaultValue;
	}
	else if (confCount > 1)
	{
		throw std::invalid_argument(name + " was multiply specified.");
	}

	auto configElement = toml::find(config, name);

	if (configElement.is_integer())
		return static_cast<double>(configElement.as_floating());
	else if (configElement.is_floating())
		return static_cast<double>(configElement.as_floating());
	else
		throw std::invalid_argument(name + " specified incorrrectly");
	return 0.0;
}

double getFloat(std::string const &name, toml::value const &config)
{
	auto confCount = config.count(name);
	if (confCount == 0)
		throw std::invalid_argument(name + " was not specified.");
	else if (confCount > 1)
		throw std::invalid_argument(name + " was multiply specified.");

	auto configElement = toml::find(config, name);
	if (configElement.is_integer())
		return static_cast<double>(configElement.as_floating());
	else if (configElement.is_floating())
		return static_cast<double>(configElement.as_floating());
	else
		throw std::invalid_argument(name + " specified incorrrectly");
	return 0.0;
}

int getIntWithDefault(std::string const &name, toml::value const &config, int defaultValue)
{
	auto confCount = config.count(name);
	if (confCount == 0)
	{
		std::cerr << "INFO: Using default value " << defaultValue << " for configuration option " << name << std::endl;
		return defaultValue;
	}
	else if (confCount > 1)
	{
		throw std::invalid_argument(name + " was multiply specified.");
	}

	auto configElement = toml::find(config, name);
	if (configElement.is_integer())
		return static_cast<int>(configElement.as_integer());
	else
		throw std::invalid_argument(name + " specified incorrrectly");
	return 0;
}

int runManta(std::string const &fname)
{
	std::filesystem::path config_file_path(fname);
	if (!std::filesystem::exists(config_file_path))
	{
		std::cerr << "Configuration file " << fname << " does not exist" << std::endl;
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
			std::string msg = "Failed to open restart netCDF file at: " + std::string(std::filesystem::absolute(std::filesystem::path(fileName)));
			throw std::runtime_error(msg);
		}
	}

	Grid *grid;
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
			highGridBoundary = false;
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

		auto lowerBoundary = toml::find(config, "Lower_boundary");
		if (config.count("Lower_boundary") != 1)
			throw std::invalid_argument("Lower_boundary unspecified or specified more than once");
		else if (lowerBoundary.is_integer())
			lBound = static_cast<double>(lowerBoundary.as_floating());
		else if (lowerBoundary.is_floating())
			lBound = static_cast<double>(lowerBoundary.as_floating());
		else
			throw std::invalid_argument("Lower_boundary specified incorrrectly");

		auto upperBoundary = toml::find(config, "Upper_boundary");
		if (config.count("Upper_boundary") != 1)
			throw std::invalid_argument("Upper_boundary unspecified or specified more than once");
		else if (upperBoundary.is_integer())
			uBound = static_cast<double>(upperBoundary.as_floating());
		else if (upperBoundary.is_floating())
			uBound = static_cast<double>(upperBoundary.as_floating());
		else
			throw std::invalid_argument("Upper_boundary specified incorrrectly");

		grid = new Grid(lBound, uBound, nCells, highGridBoundary, upperBoundaryFraction, lowerBoundaryFraction);

	} 
	else 
	{
		// Load grid from restart file
		netCDF::NcGroup GridGroup = restart_file.getGroup("Grid");
		auto nPoints = GridGroup.getDim("Index").getSize();
		std::vector<Position> CellBoundaries(nPoints);

		GridGroup.getVar("CellBoundaries").getVar(CellBoundaries.data());

		grid = new Grid(CellBoundaries, static_cast<Index>(nPoints - 1));

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

	

	if (config.count("TransportSystem") != 1)
		throw std::invalid_argument("TransportSystem needs to specified exactly once in the general configuration section");

	std::string ProblemName = config.at("TransportSystem").as_string();

	// Convert string to TransportSystem* instance

	TransportSystem *pProblem = PhysicsCases::InstantiateProblem(ProblemName, configFile, *grid);

	if(isRestarting)
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

	if (pProblem == nullptr)
	{
		std::cerr << " Could not instantiate a physics model for TransportSystem = " << ProblemName << std::endl;
		std::cerr << " Available physics models include: " << std::endl;
		for (auto pair : *PhysicsCases::map)
		{
			std::cerr << '\t' << pair.first << std::endl;
		}
		std::cerr << std::endl;
		return 1;
	}

	system = std::make_shared<SystemSolver>(*grid, k, pProblem);

	system->setOutputCadence(delta_t);
	system->setTolerances(absTol, rtol);
	system->setTau(tau);
	system->setInitialTime(tZero);
	system->setInputFile(fname);

	system->setNOutput(nOutput);
	system->setMinStepSize(dt_min);

	if (config.count("SteadyStateTolerance") == 1)
	{
		double sst = toml::find<double>(config, "SteadyStateTolerance");
		std::cout << "Running until steady state achieved (variation below " << sst << "or end time reached." << std::endl;
	}

	system->runSolver(tFinal);

	// For compiled-in TransportSystems we have the type information and
	// this will call the correct inherited destructor
	delete pProblem;
	delete grid;

	return 0;
}
