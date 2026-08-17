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

// Present under its canonical name or any alias.
bool given(ConfigSource const &source, const char *canonical)
{
    Entry const *e = findEntry(canonical);
    if (source.contains(e->name))
        return true;
    for (auto const &a : e->aliases)
        if (source.contains(a))
            return true;
    return false;
}

void checkRequired(ConfigSource const &source, Reader reader)
{
    std::string missing;
    auto want = [&](std::string_view name) {
        missing += (missing.empty() ? "" : ", ") + std::string(name);
    };

    for (auto const &e : schema())
        if (isRequired(e, reader) && !given(source, e.name.data()))
            want(e.name);

    // The grid keys, whose requiredness a flat list cannot express: GridPoints
    // supersedes GridSize, LowerBoundary and UpperBoundary outright -- makeGrid
    // ignores all three when it is present -- and a restart reads the mesh from its
    // file. GridSize used to be unconditionally required even so, which meant a run
    // driven by explicit boundaries had to carry a number that was then discarded;
    // every graded-mesh spike in MESH-REFINEMENT.md passed a dummy for that reason.
    //
    // Folded into this aggregation rather than checked after the parse, so that a
    // config missing several of these is told about all of them at once. Checked
    // against the *source* because absent and 0 are the same parsed value and must
    // not be the same diagnosis.
    const bool restarting = given(source, "restart") &&
                            std::get<bool>(source.get("restart", Type::Bool));
    if (!restarting && !given(source, "GridPoints"))
    {
        for (const char *key : {"GridSize", "LowerBoundary", "UpperBoundary"})
            if (!given(source, key))
                want(key);
    }

    if (!missing.empty())
        throw std::invalid_argument(
            "Missing required configuration key(s): " + missing +
            ". The grid keys among those are not needed if GridPoints is given, or "
            "on a restart, which reads its mesh from the file.");
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
    READ(LowerBoundaryFraction, double);
    READ(UpperBoundaryFraction, double);
    READ(GradedGridBoundary, bool);
    READ(GradingRatio, double);
    READ(GradingCells, int);
    READ(GradingEnd, std::string);
    READ(PolynomialDegree, unsigned);
    READ(GridSize, int);
    READ(GridPoints, std::vector<double>);
    READ(LowerBoundary, double);
    READ(UpperBoundary, double);
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
    READ(SteadyStateDiagnostics, bool);
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

    // The alias machinery warns that the *name* changed. It cannot know that the
    // *mesh* changed too, and it did: High_Grid_Boundary spaced its boundary-layer
    // cells by a cosine rule, and this grades them geometrically. A file saying
    // nothing but High_Grid_Boundary = true keeps its layer widths and its
    // one-third-per-layer split and gets different cells inside them, so it will
    // produce a different answer than an older MaNTA did. Silence would be wrong.
    if (source.contains("High_Grid_Boundary"))
        logmsg<LOG_LEVEL::WARNING>(
            "High_Grid_Boundary now grades its boundary layers geometrically rather "
            "than by the cosine rule it used to, so this run's mesh differs from the "
            "one an older MaNTA built from the same file. GradingRatio (default {}) "
            "sets the spacing; GridPoints reproduces a specific mesh exactly.",
            c.GradingRatio);

    if (c.GradedGridBoundary)
    {
        if (c.GradingEnd != "Lower" && c.GradingEnd != "Upper" && c.GradingEnd != "Both")
            throw std::invalid_argument(
                "GradingEnd must be \"Both\", \"Lower\" or \"Upper\"; got \"" +
                c.GradingEnd + "\".");

        // Defaulted from GridSize rather than in the schema, because a schema
        // default cannot see another key. A third per layer when grading both ends
        // is what High_Grid_Boundary did, so a config that only ever said
        // High_Grid_Boundary = true gets the same *split* it always had -- the
        // spacing within each layer is what has changed. Half for a single layer.
        //
        // Both are conservative rather than optimal: MESH-REFINEMENT.md §9 measures
        // more graded cells as better on the one problem where this was studied,
        // 9 of 10 beating 5 of 10 by 48x.
        if (c.GradingCells == 0)
            c.GradingCells = (c.GradingEnd == "Both") ? c.GridSize / 3 : c.GridSize / 2;

        // The geometry proper is validated inside gradedMeshPoints, which is where
        // it can be tested without building a configuration. Only what involves
        // *other* keys is checked here.
        const int layers = (c.GradingEnd == "Both") ? 2 : 1;
        const int least = 2 * layers + 1;
        if (c.GridSize < least)
            throw std::invalid_argument(std::format(
                "GradedGridBoundary with GradingEnd = \"{}\" needs at least {} cells "
                "-- two per graded layer and one outside them -- but GridSize is {}.",
                c.GradingEnd, least, c.GridSize));

        if (!c.GridPoints.empty())
            logmsg<LOG_LEVEL::WARNING>(
                "GradedGridBoundary is set but GridPoints was given too; the "
                "explicit boundaries win and the grading is ignored.");
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

    k = config.PolynomialDegree;

    if (!config.GridPoints.empty())
        return std::make_unique<Grid>(config.GridPoints);

    if (config.GradedGridBoundary)
    {
        const GradedEnd end = config.GradingEnd == "Lower"   ? GradedEnd::Lower
                              : config.GradingEnd == "Upper" ? GradedEnd::Upper
                                                             : GradedEnd::Both;

        auto points = gradedMeshPoints(
            config.LowerBoundary, config.UpperBoundary,
            static_cast<Grid::Index>(config.GridSize),
            static_cast<Grid::Index>(config.GradingCells),
            config.LowerBoundaryFraction, config.UpperBoundaryFraction,
            config.GradingRatio, end);

        // The narrowest cell, relative to the domain, and the reason to say so.
        // MESH-REFINEMENT.md §9 measured the solver -- not the method -- as the
        // ceiling on how hard this can be graded: on Shestakov's problem IDA's
        // corrector failed at h = 1e-7, which is MinStepSize's own default, once
        // the narrowest cell was around 1e-6 of the span, and below about 1e-7 no
        // setting of any key got through. The law that makes grading worth doing
        // was still holding at the last mesh that converged, so a run that dies
        // here has not run out of accuracy to gain, and the failure will point at
        // IDA rather than at the mesh.
        //
        // Taken as a min over every cell rather than from whichever end is graded,
        // so it stays right for Both and cannot be wrong for one end.
        const double span = config.UpperBoundary - config.LowerBoundary;
        double narrowest = span;
        for (std::size_t i = 0; i + 1 < points.size(); ++i)
            narrowest = std::min(narrowest, points[i + 1] - points[i]);

        if (span > 0.0 && narrowest / span < 1.0e-6)
            logmsg<LOG_LEVEL::WARNING>(
                "The graded mesh's narrowest cell is {:.2e} of the domain. Past "
                "roughly 1e-6 the time integrator, not the discretisation, is the "
                "limit: expect IDA corrector failures at |h| = MinStepSize, and try "
                "lowering MinStepSize (1e-12 bought one more level where this was "
                "measured) before suspecting the mesh.", narrowest / span);

        return std::make_unique<Grid>(points);
    }

    return std::make_unique<Grid>(config.LowerBoundary, config.UpperBoundary,
                                  config.GridSize);
}

// --- restartRunOrder --------------------------------------------------------

unsigned int restartRunOrder(SolverConfig const &config, unsigned int fileOrder)
{
    if (!config.restart || config.PolynomialDegree == fileOrder)
        return fileOrder;

    // Loud rather than silent, in both directions. Refining puts the stored
    // solution inside the new space and loses nothing; coarsening is a genuine
    // approximation, and a user who reached this by copying a config from
    // elsewhere should be told which number won.
    logmsg<LOG_LEVEL::WARNING>(
        "Restart file was written at PolynomialDegree = {}, but the "
        "configuration asks for {}. The state will be projected onto the new "
        "space rather than copied{}.",
        fileOrder, config.PolynomialDegree,
        config.PolynomialDegree < fileOrder ? ", which discards information at this resolution" : "");

    return config.PolynomialDegree;
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
        system.setSteadyStateDiagnostics(config.SteadyStateDiagnostics);
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
