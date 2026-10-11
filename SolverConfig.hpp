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
    double                   LowerBoundaryFraction;
    double                   UpperBoundaryFraction;
    bool                     GradedGridBoundary;
    double                   GradingRatio;
    int                      GradingCells;   // 0 means "derive from GridSize"; see gradingCellsFor
    std::string              GradingEnd;     // "Lower", "Upper" or "Both"
    unsigned                 PolynomialDegree;
    int                      GridSize;
    std::vector<double>      GridPoints;
    double                   LowerBoundary;
    double                   UpperBoundary;
    double                   tau;
    std::string              tauScaling;
    std::string              tauUpdate;
    double                   tauFloor;
    std::string              TauKappa;
    double                   delta_t;
    double                   t_initial;
    double                   Relative_tolerance;
    std::vector<double>      Absolute_tolerance;
    double                   MinStepSize;
    double                   initialTimestep;
    int                      OutputPoints;
    std::string              OutputFilename;
    bool                     solveAdjoint;
    bool                     WriteOutput;
    bool                     WriteDatFile;
    bool                     WriteDebugDatFiles;
    bool                     zeroFlux;
    bool                     AggressiveTimesteps;
    bool                     SuppressAlgebraicError;
    bool                     ForceConsistentIC;
    std::string              SteadyStateSolver;
    double                   PseudoTransientInitialStep;
    double                   PseudoTransientMaxStep;
    double                   PseudoTransientSERRate;
    double                   PseudoTransientSERFloor;
    bool                     EstimateObjectiveOnFinish;
    unsigned int             MaxContinuationSteps;
    unsigned int             NewtonMaxIterations;
    unsigned int             NewtonJacobianReuse;
    double                   NewtonStepTolerance;
    std::string              NewtonScaling;
    bool                     SteadyStateDiagnostics;
    bool                     SteadyStateStepDiagnostics;
    bool                     SteadyStateSolve;
    unsigned int             MaxRejectedSteps;
    // Intermediate rungs to solve at before the configured resolution, each
    // warm-starting the next. Empty means no ladder. Deliberately a route to
    // Polynomial_degree/Grid_size and not a replacement for them, so adding a
    // ladder cannot change the answer -- only what it costs to reach it.
    std::vector<unsigned>    DegreeLadder;
    std::vector<unsigned>    GridLadder;
    std::string              GridLadderRescaling;   // "None" or "Map"
    bool                     DegreeAdaptation;
    double                   DegreeTolerance;
    unsigned int             MaxPolynomialDegree;
    unsigned int             MaxDegreeIncrement;
    double                   DegreeAdaptationBase;
    bool                     MeshAdaptation;
    double                   MeshAdaptationThreshold;
    double                   MeshAdaptationNeighbourMargin;
    unsigned int             MeshAdaptationAttempts;
    // Read by whoever builds the adaptation driver's PhysicsInstance -- runManta
    // and PyRunner -- not by applySolverConfig: it is about the case, not the
    // solver.
    bool                     RebuildPhysicsOnRegrid;
    // Read by the adaptation controllers, and by runManta and PyRunner for the
    // starting point's warning; see ParallelFill.hpp.
    unsigned int             PhysicsParallelism;
    // Read by applySolverConfig, which arms the solver's own refusal, and by the
    // adaptation controllers, which choose no level past it. 0 is no cap.
    unsigned int             MaxPhysicsBatch;
    std::string              TransportSystem;
    std::vector<std::string> PhysicsPlugins;

    // The magnetic-field coupling. FieldModel names a registered model and is
    // applied by runManta rather than by applySolverConfig, which has neither
    // the parsed config a model reads its own table from nor the grid; the
    // other three are plain solver options and go through applySolverConfig
    // like everything else.
    std::string              FieldModel;
    std::string              FieldSolve;
    double                   FieldSolveTolerance;
    int                      FieldSolveMaxSweeps;
    int                      FieldSolveMaxAdjointSweeps;

    // Presence, not value, carries the meaning for these two.
    //
    //   t_final              -- runManta errors when unset; Runner.run() uses
    //                           it and run(tFinal) overrides it.
    //   SteadyStateTolerance -- present arms steady-state termination, which is
    //                           what the TOML reader has always done. run_ss()
    //                           arms it regardless.
    std::optional<double> t_final;
    std::optional<double> SteadyStateTolerance;

    // Presence, not value, is the signal -- as for the two above. DegreeAdaptation
    // implies the superconvergent scheme, so an absent key is defaulted to true on
    // that path and left false otherwise; an *explicit* false alongside it is a
    // contradiction rather than a preference, and loadSolverConfig refuses it. That
    // distinction is impossible to draw from a plain bool carrying the schema's
    // default.
    std::optional<bool> Superconvergent;

    // Whether LowerBoundaryFraction / UpperBoundaryFraction were given. The manual
    // GradedGridBoundary path reads the values with their schema default either
    // way; MeshAdaptation does not, and sizes its layer from the sampling mesh
    // unless told otherwise -- so for it the absent key and the default value have
    // to be told apart.
    bool LowerBoundaryFractionGiven = false;
    bool UpperBoundaryFractionGiven = false;

    // Which parts of the mesh the configuration itself describes, for a restart:
    // there the file already holds a mesh, and only what the configuration says
    // replaces it. With neither GridSize nor GridPoints the run keeps the file's
    // mesh, cell for cell, whatever its spacing; with GridSize but no
    // LowerBoundary or UpperBoundary, the missing end is the file's. Both are
    // impossible to read off the parsed values, whose schema defaults (0 cells,
    // the domain [0, 1]) are indistinguishable from a config that wrote them.
    bool CellsGiven = false;           // GridSize or GridPoints
    bool LowerBoundaryGiven = false;
    bool UpperBoundaryGiven = false;
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

// The mesh the restart file holds, or -- when not restarting -- the one the
// configuration asks for. `restart` is the opened restart file when
// config.restart is set, nullptr otherwise; k is written with the polynomial
// degree the *file* was written at, which is also the degree it must be read
// back at.
//
// On a restart this is the mesh the stored state is laid out on, which is what
// the DOF check and setRestartValues need, and **not** necessarily the mesh the
// run uses -- see restartRunGrid, exactly as restartRunOrder gives the degree
// the run uses.
std::unique_ptr<Grid> makeGrid(SolverConfig const &config,
                               netCDF::NcFile *restart, unsigned int &k);

// The mesh the configuration asks for, whether or not this is a restart.
std::unique_ptr<Grid> configuredGrid(SolverConfig const &config);

// Whether the mesh this configuration runs on is a list of boundaries rather
// than a recipe that can be asked for another cell count: GridPoints, or a
// restart that gives no cell count and so keeps its file's mesh.
bool meshIsExplicit(SolverConfig const &config);

// The cells in each graded layer of a GradedGridBoundary mesh: GradingCells when
// given, otherwise derived from GridSize -- a third of it per layer when grading
// both ends, half when grading one, and never fewer than the 2 a layer needs.
// Resolved from the configuration at hand rather than once at load, so a ladder
// rung built at a smaller GridSize derives its own count instead of inheriting
// one sized for the final mesh.
int gradingCellsFor(SolverConfig const &config);

// The mesh a restarted run should be solved on, given the mesh its restart file
// was written on. The counterpart of restartRunOrder, and it exists for the
// same reason: makeGrid used to return the file's mesh and the run used that,
// so Grid_size was read, validated, required of every config on both readers --
// and then silently discarded. A ladder written as "solve coarse, restart
// finer, solve again" therefore re-solved the coarse problem at every rung and
// reported it converged, which is indistinguishable from success.
//
// A configuration that gives neither GridSize nor GridPoints describes no mesh,
// and the run keeps the file's -- which is the only way a restart can resume on
// a non-uniform mesh no configuration rule produces. One that gives GridSize
// without an end of the domain takes that end from the file.
//
// An equal mesh returns the file's own, so every existing restart takes the
// copy path in setInitialConditions and is bit for bit unchanged. A different
// one warns and wins; setInitialConditions then projects the stored element
// polynomials onto the new cells and rebuilds the trace, since lambda lives on
// faces that have moved.
std::unique_ptr<Grid> restartRunGrid(SolverConfig const &config, Grid const &fileGrid);

// The polynomial degree a restarted run should use, given the degree its restart
// file was written at.
//
// A restart used to take its degree from the file and ignore
// PolynomialDegree outright, even though the schema makes that key required of
// every config on both readers -- so a user was obliged to write a number that
// was then silently discarded. This honours it, and warns when the two differ,
// because the state is projected across the degree change rather than copied.
//
// Equal degrees return the file's, so every existing restart takes the copy
// path in setInitialConditions and is bit for bit unchanged. Shared by both
// surfaces so they cannot drift on it.
unsigned int restartRunOrder(SolverConfig const &config, unsigned int fileOrder);

// Every config-derived set* call on the solver, in one place.
void applySolverConfig(SolverConfig const &config, SystemSolver &system);

#endif // SOLVERCONFIG_HPP
