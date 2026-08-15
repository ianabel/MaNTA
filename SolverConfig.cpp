#include "SolverConfig.hpp"

#include "Logging.hpp"

// makeGrid and applySolverConfig need the concrete types the header only
// forward-declares, so the weight of <netcdf>, the grid and the solver lands
// here rather than on every consumer of SolverConfig.hpp.
#include <netcdf>

#include "SystemSolver.hpp"
#include "gridStructures.hpp"

#include <set>
#include <stdexcept>

using namespace ConfigSchema;

// --- TomlConfigSource -------------------------------------------------------

TomlConfigSource::TomlConfigSource(toml::value const &configuration,
                                   std::filesystem::path configPath)
    : config(configuration), path(std::move(configPath))
{
}

bool TomlConfigSource::contains(std::string_view key) const
{
    return config.contains(std::string(key));
}

std::vector<std::string> TomlConfigSource::keys() const
{
    std::vector<std::string> out;
    for (auto const &[k, v] : config.as_table())
        out.push_back(k);
    return out;
}

std::string TomlConfigSource::outputFilenameFallback() const
{
    return path.empty() ? std::string{} : path.stem().string();
}

ConfigSchema::Value TomlConfigSource::get(std::string_view key, Type t) const
{
    auto const &node = config.at(std::string(key));
    auto bad = [&](const char *wanted) {
        return std::invalid_argument("Configuration key '" + std::string(key) +
                                     "' must be " + wanted + ".");
    };

    switch (t)
    {
    case Type::Bool:
        if (!node.is_boolean())
            throw bad(typeName(t));
        return node.as_boolean();

    case Type::Int:
        if (!node.is_integer())
            throw bad(typeName(t));
        return static_cast<int>(node.as_integer());

    case Type::UInt:
        if (!node.is_integer() || node.as_integer() < 0)
            throw bad(typeName(t));
        return static_cast<unsigned>(node.as_integer());

    case Type::Double:
        // TOML distinguishes 1 from 1.0, and `tau = 1` is entirely natural --
        // so read an integer node with as_integer(). Calling as_floating() on
        // one throws toml::type_error, which is the defect Config.cpp's comment
        // records and which MaNTA.cpp repeated twice for the boundaries.
        if (node.is_integer())
            return static_cast<double>(node.as_integer());
        if (node.is_floating())
            return node.as_floating();
        throw bad(typeName(t));

    case Type::String:
        if (!node.is_string())
            throw bad(typeName(t));
        return node.as_string();

    case Type::DoubleList:
    {
        std::vector<double> out;
        if (node.is_array())
        {
            for (auto const &e : node.as_array())
            {
                if (e.is_integer())
                    out.push_back(static_cast<double>(e.as_integer()));
                else if (e.is_floating())
                    out.push_back(e.as_floating());
                else
                    throw bad(typeName(t));
            }
        }
        else if (node.is_integer())
            out.push_back(static_cast<double>(node.as_integer()));
        else if (node.is_floating())
            out.push_back(node.as_floating());
        else
            throw bad(typeName(t));
        return out;
    }

    case Type::StringList:
    {
        if (!node.is_array())
            throw bad(typeName(t));
        std::vector<std::string> out;
        for (auto const &e : node.as_array())
        {
            if (!e.is_string())
                throw bad(typeName(t));
            out.push_back(e.as_string());
        }
        return out;
    }
    }
    throw bad("of a recognised type");
}

// --- loadSolverConfig -------------------------------------------------------

namespace
{

// The spelling actually present for an entry, and a check that only one is.
std::optional<std::string> presentSpelling(ConfigSource const &source, Entry const &e)
{
    std::vector<std::string> found;
    if (source.contains(e.name))
        found.emplace_back(e.name);
    for (auto const &a : e.aliases)
        if (source.contains(a))
            found.emplace_back(a);

    if (found.size() > 1)
    {
        std::string msg = "Configuration key '" + std::string(e.name) +
                          "' was given under more than one spelling: ";
        for (std::size_t i = 0; i < found.size(); ++i)
            msg += (i ? ", " : "") + found[i];
        msg += ". Give it once.";
        throw std::invalid_argument(msg);
    }

    if (found.empty())
        return std::nullopt;

    if (found.front() != e.name)
        logmsg<LOG_LEVEL::WARNING>(
            "Configuration key '{}' is deprecated; use '{}'. Both are accepted for now.",
            found.front(), std::string(e.name));

    return found.front();
}

template <typename T>
T read(ConfigSource const &source, Entry const &e, std::optional<std::string> const &spelling)
{
    if (spelling)
        return std::get<T>(source.get(*spelling, e.type));
    return std::get<T>(e._default);
}

void rejectUnknownKeys(ConfigSource const &source, Reader reader)
{
    for (auto const &k : source.keys())
    {
        auto const *e = findEntry(k);
        if (e == nullptr)
        {
            std::string msg = "Unknown configuration key '" + k + "'.";
            auto suggestion = nearestKey(k);
            if (!suggestion.empty())
                msg += " Did you mean '" + std::string(suggestion) + "'?";
            throw std::invalid_argument(msg);
        }

        // A key that is in the schema but means nothing to this reader.
        // Silently ignoring it is what the old readers did, and is exactly the
        // failure this schema exists to stop: a driver would pass
        // TransportSystem to configure() and never learn it had no effect.
        if (reader == Reader::Dict && e->category == Category::ProblemSelection)
            throw std::invalid_argument(
                "Configuration key '" + k +
                "' selects the physics case and has no meaning for "
                "Runner.configure -- the transport system is passed to the "
                "Runner as an object. Remove it from the dict.");
    }
}

void checkRequired(ConfigSource const &source, Reader reader)
{
    std::string missing;
    for (auto const &e : schema())
    {
        if (!isRequired(e, reader))
            continue;
        bool present = source.contains(e.name);
        for (auto const &a : e.aliases)
            present = present || source.contains(a);
        if (!present)
            missing += (missing.empty() ? "" : ", ") + std::string(e.name);
    }
    if (!missing.empty())
        throw std::invalid_argument("Missing required configuration key(s): " + missing + ".");
}

} // namespace

SolverConfig loadSolverConfig(ConfigSource const &source, Reader reader)
{
    rejectUnknownKeys(source, reader);
    checkRequired(source, reader);

    SolverConfig c{};
    auto spelling = [&](const char *name) {
        return presentSpelling(source, *findEntry(name));
    };
    auto E = [&](const char *name) -> Entry const & { return *findEntry(name); };

#define READ(field, T) c.field = read<T>(source, E(#field), spelling(#field))
    READ(restart, bool);
    READ(RestartFile, std::string);
    READ(High_Grid_Boundary, bool);
    READ(Lower_Boundary_Fraction, double);
    READ(Upper_Boundary_Fraction, double);
    READ(Polynomial_degree, unsigned);
    READ(Grid_size, int);
    READ(Grid_points, std::vector<double>);
    READ(Lower_boundary, double);
    READ(Upper_boundary, double);
    READ(tau, double);
    READ(delta_t, double);
    READ(t_initial, double);
    READ(Relative_tolerance, double);
    READ(Absolute_tolerance, std::vector<double>);
    READ(MinStepSize, double);
    READ(initialTimestep, double);
    READ(OutputPoints, int);
    READ(OutputFilename, std::string);
    READ(solveAdjoint, bool);
    READ(ObjectiveDecreaseTolerance, double);
    READ(WriteOutput, bool);
    READ(WriteDatFile, bool);
    READ(WriteDebugDatFiles, bool);
    READ(Superconvergent, bool);
    READ(zeroFlux, bool);
    READ(AggressiveTimesteps, bool);
    READ(SuppressAlgebraicError, bool);
    READ(SteadyStateSolver, std::string);
    READ(PseudoTransientInitialStep, double);
    READ(PseudoTransientMaxStep, double);
    READ(PseudoTransientSERRate, double);
    READ(PseudoTransientSERFloor, double);
    READ(TransportSystem, std::string);
    READ(PhysicsPlugins, std::vector<std::string>);
#undef READ

    // The two whose presence is the signal.
    if (auto s = spelling("t_final"))
        c.t_final = std::get<double>(source.get(*s, Type::Double));
    if (auto s = spelling("SteadyStateTolerance"))
        c.SteadyStateTolerance = std::get<double>(source.get(*s, Type::Double));

    if (c.OutputFilename.empty())
        c.OutputFilename = source.outputFilenameFallback();

    // Conditional rules a flat required-list cannot express.
    if (!c.restart && c.Grid_points.empty())
    {
        bool haveLower = source.contains("Lower_boundary");
        bool haveUpper = source.contains("Upper_boundary");
        if (!haveLower || !haveUpper)
            throw std::invalid_argument(
                "Missing required configuration key(s): Lower_boundary, Upper_boundary "
                "-- required unless Grid_points is given or the run is a restart.");
    }

    // Only the dict surface has no file to fall back on. A TomlConfigSource
    // built with no path -- an in-memory configuration, which is what the unit
    // tests use -- has no name to offer either, but the key is not required of
    // it: runManta always passes the config file it parsed, so the fallback is
    // always available on the surface where a file exists. Keying this on the
    // reader rather than on the emptiness of the fallback is what keeps the
    // message ("no config file to take a name from") true wherever it fires.
    // The schema marks OutputFilename required for the dict reader, so an
    // *absent* key is already reported alongside every other missing one --
    // reporting it separately here would mean fixing a config one key per run.
    // This catches the remaining case: the key given, but empty.
    if (reader == Reader::Dict && c.OutputFilename.empty())
        throw std::invalid_argument(
            "Missing required configuration key: OutputFilename -- there is no "
            "config file to take a name from.");

    return c;
}

// --- makeGrid ---------------------------------------------------------------

std::unique_ptr<Grid> makeGrid(SolverConfig const &config,
                               netCDF::NcFile *restart, unsigned int &k)
{
    if (config.restart)
    {
        if (restart == nullptr)
            throw std::invalid_argument("restart is set but no restart file was opened.");

        netCDF::NcGroup GridGroup = restart->getGroup("Grid");
        auto nPoints = GridGroup.getDim("Index").getSize();
        std::vector<Position> CellBoundaries(nPoints);
        GridGroup.getVar("CellBoundaries").getVar(CellBoundaries.data());
        GridGroup.getVar("PolyOrder").getVar(&k);
        return std::make_unique<Grid>(CellBoundaries);
    }

    k = config.Polynomial_degree;

    if (!config.Grid_points.empty())
        return std::make_unique<Grid>(config.Grid_points);

    if (config.Grid_size < 4 && config.High_Grid_Boundary)
        throw std::invalid_argument(
            "Grid size must exceed 4 cells in order to implement dense boundaries");

    // Grid ignores both fractions when High_Grid_Boundary is false
    // (gridStructures.hpp:81), so passing them unconditionally is what the two
    // old readers did between them -- MaNTA.cpp zeroed them, PyRunner did not,
    // and the grids came out identical either way. Worth stating because it
    // looks like a divergence somebody should fix.
    return std::make_unique<Grid>(config.Lower_boundary, config.Upper_boundary,
                                  config.Grid_size, config.High_Grid_Boundary,
                                  config.Lower_Boundary_Fraction,
                                  config.Upper_Boundary_Fraction);
}

// --- applySolverConfig ------------------------------------------------------

void applySolverConfig(SolverConfig const &config, SystemSolver &system)
{
    // The only place a configuration reaches the solver. That is the point --
    // it is what stops the TOML path and the dict path configuring differently
    // -- but it also means a block dropped from here un-configures *both*
    // surfaces at once, where the same slip used to affect one.
    system.setOutputCadence(config.delta_t);
    system.setTolerances(config.Absolute_tolerance, config.Relative_tolerance);
    system.setTau(config.tau);
    system.setInitialTime(config.t_initial);
    system.setInitialTimestep(config.initialTimestep);
    system.setInputFile(config.OutputFilename);
    system.setSolveAdjoint(config.solveAdjoint);
    system.setNOutput(config.OutputPoints);
    system.setMinStepSize(config.MinStepSize);
    system.setZeroFlux(config.zeroFlux);
    system.setSuperconvergent(config.Superconvergent);
    system.setWriteOutput(config.WriteOutput);
    system.setWriteDatFile(config.WriteDatFile);
    system.setWriteDebugDatFiles(config.WriteDebugDatFiles);
    system.setAggressiveTimesteps(config.AggressiveTimesteps);
    system.setSuppressAlgebraicError(config.SuppressAlgebraicError);

    // Rejected here rather than defaulted, because a typo in this key would
    // otherwise silently pick a different algorithm.
    if (config.SteadyStateSolver == "PseudoTransient")
        system.setSteadyMode(SystemSolver::SteadyMode::PseudoTransient);
    else if (config.SteadyStateSolver == "TimeMarch")
        system.setSteadyMode(SystemSolver::SteadyMode::TimeMarch);
    else if (config.SteadyStateSolver == "Newton")
        system.setSteadyMode(SystemSolver::SteadyMode::Newton);
    else
        throw std::invalid_argument(
            "SteadyStateSolver must be \"PseudoTransient\", \"TimeMarch\" or "
            "\"Newton\"; got \"" + config.SteadyStateSolver + "\".");

    if (config.PseudoTransientInitialStep > 0.0)
        system.setPseudoTransientInitialStep(config.PseudoTransientInitialStep);
    if (config.PseudoTransientMaxStep > 0.0)
        system.setPseudoTransientMaxStep(config.PseudoTransientMaxStep);

    // Unconditional, unlike the two above: those use 0 as "unset", which works
    // because a zero step is meaningless, but a zero SER *rate* is a legitimate
    // setting -- grow at the floor alone. So the schema's defaults are the real
    // ones and are applied every time. The setters throw std::logic_error on a
    // bad value; loadSolverConfig's contract is std::invalid_argument, so
    // rewrap rather than letting a different exception type escape this path.
    try
    {
        system.setPseudoTransientSERRate(config.PseudoTransientSERRate);
        system.setPseudoTransientSERFloor(config.PseudoTransientSERFloor);
    }
    catch (std::logic_error const &e)
    {
        throw std::invalid_argument(e.what());
    }

    // Presence arms it, which is what the TOML reader has always done;
    // setSteadyStateTolerance also sets TerminateOnSteadyState.
    if (config.SteadyStateTolerance)
    {
        logmsg<LOG_LEVEL::INFO>(
            "Running until steady state achieved (variation below {}) or end time reached.",
            *config.SteadyStateTolerance);
        system.setSteadyStateTolerance(*config.SteadyStateTolerance);
    }

    // Zero is off, and the setter rejects anything negative.
    if (config.ObjectiveDecreaseTolerance != 0.0)
    {
        logmsg<LOG_LEVEL::INFO>(
            "Abandoning the run if dG/dt falls below {} at the initial condition.",
            -config.ObjectiveDecreaseTolerance);
        system.setObjectiveDecreaseTolerance(config.ObjectiveDecreaseTolerance);
    }
}
