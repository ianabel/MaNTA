#ifndef SOLVERCONFIG_HPP
#define SOLVERCONFIG_HPP

// Reading a configuration, once, for both surfaces.
//
// ConfigSource is the only thing that differs between a TOML file and a
// Runner.configure dict; everything downstream -- validation, aliases,
// defaults, grid construction, applying to the solver -- is shared. That is
// what stops the two drifting.
//
// Must not include pybind11: this links into MaNTA, libmanta.so and the unit
// tests. DictConfigSource lives in PyConfigSource.hpp, which PyRunner.cpp
// includes and nothing else does.

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <toml.hpp>

#include "ConfigSchema.hpp"

class Grid;
class SystemSolver;

// Declared rather than included: makeGrid only takes a pointer to one, and
// <netcdf> is a heavy header that every consumer of this one would otherwise
// acquire.
namespace netCDF
{
class NcFile;
}

struct SolverConfig
{
    bool                     restart;
    std::string              RestartFile;
    bool                     High_Grid_Boundary;
    double                   Lower_Boundary_Fraction;
    double                   Upper_Boundary_Fraction;
    unsigned                 Polynomial_degree;
    int                      Grid_size;
    std::vector<double>      Grid_points;
    double                   Lower_boundary;
    double                   Upper_boundary;
    double                   tau;
    double                   delta_t;
    double                   t_initial;
    double                   Relative_tolerance;
    std::vector<double>      Absolute_tolerance;
    double                   MinStepSize;
    double                   initialTimestep;
    int                      OutputPoints;
    std::string              OutputFilename;
    bool                     solveAdjoint;
    double                   ObjectiveDecreaseTolerance;
    bool                     WriteOutput;
    bool                     WriteDatFile;
    bool                     WriteDebugDatFiles;
    bool                     Superconvergent;
    bool                     zeroFlux;
    bool                     AggressiveTimesteps;
    bool                     SuppressAlgebraicError;
    std::string              SteadyStateSolver;
    double                   PseudoTransientInitialStep;
    double                   PseudoTransientMaxStep;
    double                   PseudoTransientSERRate;
    double                   PseudoTransientSERFloor;
    bool                     SteadyStateDiagnostics;
    std::string              TransportSystem;
    std::vector<std::string> PhysicsPlugins;

    // Presence, not value, carries the meaning for these two.
    //
    //   t_final              -- runManta errors when unset; Runner.run() uses
    //                           it and run(tFinal) overrides it.
    //   SteadyStateTolerance -- present arms steady-state termination, which is
    //                           what the TOML reader has always done. run_ss()
    //                           arms it regardless.
    std::optional<double> t_final;
    std::optional<double> SteadyStateTolerance;
};

// The one thing that differs between the two surfaces.
class ConfigSource
{
public:
    virtual ~ConfigSource() = default;
    virtual bool contains(std::string_view key) const = 0;
    // Throws std::invalid_argument, naming the key and the wanted type, if the
    // value present cannot be read as `t`.
    virtual ConfigSchema::Value get(std::string_view key, ConfigSchema::Type t) const = 0;
    virtual std::vector<std::string> keys() const = 0;
    // Base name for output when OutputFilename is absent. Empty means "no
    // fallback".
    virtual std::string outputFilenameFallback() const { return {}; }
};

class TomlConfigSource : public ConfigSource
{
public:
    // configPath is used only for the OutputFilename fallback, which is the
    // config file's stem -- the behaviour Solver.cpp has always had.
    explicit TomlConfigSource(toml::value const &configuration,
                              std::filesystem::path configPath = {});

    bool contains(std::string_view key) const override;
    ConfigSchema::Value get(std::string_view key, ConfigSchema::Type t) const override;
    std::vector<std::string> keys() const override;
    std::string outputFilenameFallback() const override;

private:
    toml::value const    &config;
    std::filesystem::path path;
};

// Validate against the schema and produce a SolverConfig. Throws
// std::invalid_argument for an unknown key, a missing required key, a wrong
// type, a key given alongside its own alias, or a violated conditional rule.
SolverConfig loadSolverConfig(ConfigSource const &source, ConfigSchema::Reader reader);

// The grid the configuration asks for. `restart` is the opened restart file
// when config.restart is set, nullptr otherwise; k is written with the
// polynomial degree, which comes from the restart file on that path.
std::unique_ptr<Grid> makeGrid(SolverConfig const &config,
                               netCDF::NcFile *restart, unsigned int &k);

// Every config-derived set* call on the solver, in one place.
void applySolverConfig(SolverConfig const &config, SystemSolver &system);

#endif // SOLVERCONFIG_HPP
