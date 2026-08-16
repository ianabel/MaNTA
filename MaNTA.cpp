#include <dlfcn.h> // physics plugins loaded from the config
#include <memory>
#include <boost/math/tools/roots.hpp>
#include <toml.hpp>
#include <filesystem>

#include "SystemSolver.hpp"
#include "PhysicsCases.hpp"
#include "SolverConfig.hpp"
#include "FieldModel.hpp"

// Load restart data into vectors. `nField` is filled with how many of the
// trailing entries of Y are a field model's psi, so the caller can shape the
// DGSoln that wraps them.
//
// A file written before the field block existed has no nField variable, and
// reads back as zero rather than as an error: every such file was written by a
// run with no field model, so zero is the truth about it rather than a guess.
int LoadFromFile(netCDF::NcFile &restart_file, std::vector<double> &Y, std::vector<double> &dYdt,
				 Index &nField)
{
	netCDF::NcGroup RestartGroup = restart_file.getGroup("RestartData");

	Index nDOF = RestartGroup.getDim("nDOF").getSize();

	Y.resize(nDOF);
	dYdt.resize(nDOF);

	RestartGroup.getVar("Y").getVar(Y.data());
	RestartGroup.getVar("dYdt").getVar(dYdt.data());

	nField = 0;
	netCDF::NcVar nFieldVar = RestartGroup.getVar("nField");
	if (!nFieldVar.isNull())
	{
		int stored = 0;
		nFieldVar.getVar(&stored);
		nField = stored;
	}

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

	// Deliberately not wrapped in a try/catch. A bad configuration has always
	// propagated out of here: pybind translates it for `manta.run()`, so a
	// Python caller gets an exception rather than a return code it can ignore,
	// and main() catches it for the command line.
	TomlConfigSource source(configuration, fname);
	SolverConfig config = loadSolverConfig(source, ConfigSchema::Reader::Toml);

	// t_final is required of a config file and not of a dict -- a Runner is told
	// the end time by run(tFinal) -- which the schema records per reader, so
	// loadSolverConfig has already reported it alongside any other missing key
	// rather than in a message of its own.
	if (!config.t_final)
		throw std::logic_error("t_final is required of the TOML reader but was not set.");

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

	// A field model is selected by name from the same process-global registry
	// pattern the physics cases use, and is handed the parsed config file so it
	// can read its own table. After the plugin dlopens above, since a model may
	// come from one; and *before* the restart block below, because the field
	// block is part of the solution vector and its length is what the restart
	// file has to be checked against and the restart DGSoln shaped by. It is
	// attached to the solver further down -- setFieldModel needs a solver, and a
	// solver cannot exist until the restart values are in the problem.
	std::shared_ptr<FieldModel> fieldModel;
	if (!config.FieldModel.empty())
	{
		try
		{
			fieldModel = FieldModels::InstantiateFieldModel(config.FieldModel, configFile, *grid);
		}
		catch (std::invalid_argument const &e)
		{
			logmsg<LOG_LEVEL::ERROR>("Could not instantiate a field model for FieldModel = {}\n  {}",
									 config.FieldModel, e.what());
			return 1;
		}
	}
	const Index nField = fieldModel ? fieldModel->nFieldDOF() : 0;

	if (config.restart)
	{
		std::vector<double> Y, dYdt;
		Index nField_file = 0;
		Index nDOF_file = LoadFromFile(restart_file, Y, dYdt, nField_file);

		// Make sure degrees of freedom are consistent with restart file
		const Index nCells = grid->getNCells();
		const Index nDOF = pProblem->getNumVars() * 3 * nCells * (k + 1) +
						   pProblem->getNumVars() * (nCells + 1) +
						   pProblem->getNumScalars() +
						   pProblem->getNumAux() * nCells * (k + 1) +
						   nField;

		// Two checks, not one, because nDOF alone cannot separate them: a file
		// with one extra field unknown and one fewer scalar has exactly the
		// right total length, and would be read back with psi in the scalar
		// slot. The field count is the cheaper thing to be sure of, so it is
		// reported first and by name.
		if (nField_file != nField)
			throw std::invalid_argument(
				"Restart file carries " + std::to_string(nField_file) +
				" field unknowns but the configured FieldModel declares " +
				std::to_string(nField) + ".");

		if (nDOF_file != nDOF)
			throw std::invalid_argument("nVars/nAux/nScalars in restart file inconsistent with physics case");

		pProblem->setRestartValues(Y, dYdt, *grid, k, nField);
	}

	auto system = std::make_shared<SystemSolver>(*grid, k, pProblem.get());

	// Before applySolverConfig, and necessarily before runSolver: setFieldModel
	// reshapes the solution vector and refuses once the solver is initialised.
	if (fieldModel)
		system->setFieldModel(fieldModel);

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
