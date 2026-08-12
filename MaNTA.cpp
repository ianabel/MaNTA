#include <dlfcn.h> // physics plugins loaded from the config
#include <memory>
#include <boost/math/tools/roots.hpp>
#include <toml.hpp>
#include <filesystem>

#include "SystemSolver.hpp"
#include "PhysicsCases.hpp"
#include "SolverConfig.hpp"

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

	// Every configuration key MaNTA accepts is declared once, in
	// ConfigSchema.cpp, and read the same way here and in PyRunner::configure.
	// This function used to open-code ~120 lines of toml::find_or against a
	// separate list, which is how the two surfaces came to disagree about the
	// name of the initial time and the default absolute tolerance.
	const auto configFile = toml::parse(fname);
	const auto configuration = toml::find<toml::value>(configFile, "configuration");

	SolverConfig config;
	try
	{
		TomlConfigSource source(configuration, fname);
		config = loadSolverConfig(source, ConfigSchema::Reader::Toml);
	}
	catch (std::invalid_argument const &e)
	{
		logmsg<LOG_LEVEL::ERROR>("{}", e.what());
		return 1;
	}

	// Required of a config file, but not of a dict -- a Runner is told the end
	// time by run(tFinal). The schema records the key; the requirement is here.
	if (!config.t_final)
	{
		logmsg<LOG_LEVEL::ERROR>("Missing required configuration key: t_final.");
		return 1;
	}

	netCDF::NcFile restart_file;
	if (config.restart)
	{
		std::string fileName = config.RestartFile.empty()
								   ? config.OutputFilename + ".restart.nc"
								   : config.RestartFile;
		try
		{
			restart_file.open(fileName, netCDF::NcFile::FileMode::read);
		}
		catch (...)
		{
			logmsg<LOG_LEVEL::ERROR>("Failed to open restart netCDF file at: {}",
									 std::string(std::filesystem::absolute(std::filesystem::path(fileName))));
			return 1;
		}
	}

	unsigned int k = 1;
	std::unique_ptr<Grid> grid = makeGrid(config, config.restart ? &restart_file : nullptr, k);

	// Physics cases built outside this tree.
	//
	// A case registers itself from a static initialiser, so loading the shared
	// object is all that is needed -- its PhysicsCaseRegister runs on dlopen and
	// inserts into the same process-global map the built-in cases use. This has
	// to happen before InstantiateProblem, and duplicate names now throw rather
	// than being dropped, so a plugin colliding with a built-in says so.
	for (auto const &path : config.PhysicsPlugins)
	{
		// RTLD_GLOBAL so a plugin can be linked against another plugin's
		// symbols; RTLD_NOW so an unresolved symbol is reported here rather
		// than at the first call into the case.
		if (dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL) == nullptr)
		{
			logmsg<LOG_LEVEL::ERROR>("Could not load physics plugin {}: {}", path, dlerror());
			return 1;
		}
	}

	// InstantiateProblem throws for an unrecognised name, with the list of what
	// is registered in the message. Caught here so the standalone binary still
	// exits 1 with a readable line rather than terminating on an uncaught
	// exception out of main.
	std::unique_ptr<TransportSystem> pProblem;
	try
	{
		pProblem = PhysicsCases::InstantiateProblem(config.TransportSystem, configFile, *grid);
	}
	catch (std::invalid_argument const &e)
	{
		logmsg<LOG_LEVEL::ERROR>("Could not instantiate a physics model for TransportSystem = {}\n  {}",
								 config.TransportSystem, e.what());
		return 1;
	}

	std::unique_ptr<AdjointProblem> adjoint = nullptr;
	if (config.solveAdjoint)
		adjoint = pProblem->createAdjointProblem();

	if (config.restart)
	{
		std::vector<double> Y, dYdt;
		Index nDOF_file = LoadFromFile(restart_file, Y, dYdt);

		// Make sure degrees of freedom are consistent with restart file
		const Index nCells = grid->getNCells();
		const Index nDOF = pProblem->getNumVars() * 3 * nCells * (k + 1) +
						   pProblem->getNumVars() * (nCells + 1) +
						   pProblem->getNumScalars() +
						   pProblem->getNumAux() * nCells * (k + 1);

		if (nDOF_file != nDOF)
			throw std::invalid_argument("nVars/nAux/nScalars in restart file inconsistent with physics case");

		pProblem->setRestartValues(Y, dYdt, *grid, k);
	}

	auto system = std::make_shared<SystemSolver>(*grid, k, pProblem.get());

	applySolverConfig(config, *system);
	if (config.solveAdjoint)
		system->setAdjointProblem(adjoint.get());

	system->runSolver(*config.t_final);

	// For compiled-in TransportSystems we have the type information and
	// this will call the correct inherited destructor

	//delete grid;
	std::println("Done.");
	return 0;
}
