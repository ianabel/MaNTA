// Reading a [configuration] table against the schema.
//
// The rules under test here are the ones that used to be open-coded twice and
// differently: what is required, what an absent key defaults to, what happens
// to a key nobody recognises, and the two conditional requirements that a flat
// required-list cannot express.

#include <boost/test/unit_test.hpp>

#include "SolverConfig.hpp"
// applySolverConfig is the single point at which a configuration reaches the
// solver, so the tests below build a real one and read the settings back.
#include "SystemSolver.hpp"
#include "CapturedOutput.hpp"
#include "DegreeAdaptation.hpp"
#include "PhysicsInstance.hpp"
#include "TestDiffusion.hpp"

// SolverConfig.hpp only forward-declares Grid, so that it stays cheap to include
// and pybind11-free. The graded-mesh cases below inspect the Grid makeGrid built,
// so they need the definition; taken here rather than by widening that header.
#include "gridStructures.hpp"

#include <map>
#include <stdexcept>
#include <format>
#include <string>

namespace
{
SolverConfig load(std::string const &body)
{
    auto v = toml::parse_str(body);
    TomlConfigSource src(v);
    return loadSolverConfig(src, ConfigSchema::Reader::Toml);
}

// The smallest config that satisfies every unconditional requirement.
const std::string minimal =
    "PolynomialDegree = 2\n"
    "GridSize = 8\n"
    "delta_t = 0.1\n"
    "t_final = 1.0\n"
    "LowerBoundary = 0.0\n"
    "UpperBoundary = 1.0\n"
    "TransportSystem = \"LinearDiffusion\"\n";

// A second ConfigSource over a plain map, standing in for the dict.
//
// DictConfigSource itself needs pybind11 and so cannot be linked into these
// tests. What these exercise is not pybind's casting but the machinery above
// it -- validation, aliases, defaults -- which is shared. The real dict is
// covered by python/Tests/test_run_config.py.
class MapConfigSource : public ConfigSource
{
public:
    std::map<std::string, ConfigSchema::Value> values;
    std::string fallback;

    bool contains(std::string_view key) const override
    {
        return values.count(std::string(key)) > 0;
    }
    ConfigSchema::Value get(std::string_view key, ConfigSchema::Type) const override
    {
        return values.at(std::string(key));
    }
    std::vector<std::string> keys() const override
    {
        std::vector<std::string> out;
        for (auto const &[k, v] : values)
            out.push_back(k);
        return out;
    }
    std::string outputFilenameFallback() const override { return fallback; }
};
} // namespace

BOOST_AUTO_TEST_SUITE(config_source_tests, *boost::unit_test::tolerance(1e-12))

BOOST_AUTO_TEST_CASE(a_minimal_config_loads_with_every_default_applied)
{
    auto c = load(minimal);

    BOOST_TEST(c.PolynomialDegree == 2u);
    BOOST_TEST(c.GridSize == 8);
    BOOST_TEST(c.tau == 1.0);
    BOOST_TEST(c.Relative_tolerance == 1e-3);
    BOOST_REQUIRE(c.Absolute_tolerance.size() == 1u);
    BOOST_TEST(c.Absolute_tolerance[0] == 1e-3);
    BOOST_TEST(c.MinStepSize == 1e-7);
    BOOST_TEST(c.OutputPoints == 301);
    BOOST_TEST(c.WriteOutput);
    BOOST_TEST(!c.WriteDatFile);
    // Superconvergent is a std::optional: absent, rather than present-and-false.
    // Presence is the signal, because DegreeAdaptation defaults it to true and
    // has to be able to tell "not asked for" from "asked against".
    const bool superconvergentAbsent = !c.Superconvergent.has_value();
    BOOST_TEST(superconvergentAbsent);
    BOOST_TEST(!c.AggressiveTimesteps);
    BOOST_TEST(!c.SuppressAlgebraicError);
    // Off by default: a steady solve and a restart skip IDACalcIC, and this is
    // what puts it back. See setForceConsistentIC.
    BOOST_TEST(!c.ForceConsistentIC);
    // PseudoTransient, not TimeMarch: run_ss() and a config carrying
    // SteadyStateTolerance both take the continuation path unless told not to.
    BOOST_TEST(c.SteadyStateSolver == "PseudoTransient");
    BOOST_TEST(c.PseudoTransientInitialStep == 0.0);
    BOOST_TEST(c.PseudoTransientMaxStep == 0.0);

    // Unlike the two above, these two are real values rather than "unset"
    // sentinels: a zero SER rate means "grow at the floor alone", so the
    // schema default is what the solver is configured with every time.
    BOOST_TEST(c.PseudoTransientSERRate == 1.0);
    BOOST_TEST(c.PseudoTransientSERFloor == 2.0);
    BOOST_TEST(c.NewtonMaxIterations == 20u);
    BOOST_TEST(c.NewtonJacobianReuse == 1u);
    BOOST_TEST(c.NewtonStepTolerance == 0.0);
    BOOST_TEST(c.NewtonScaling == "Unit");
    BOOST_TEST(c.SteadyStateDiagnostics == false);
    BOOST_TEST(c.SteadyStateStepDiagnostics == false);
}

BOOST_AUTO_TEST_CASE(absolute_tolerance_defaults_to_1e_3)
{
    // It was 1e-2 in MaNTA.cpp and 1e-3 in PyRunner.cpp. All 68 configs in the
    // tree set it explicitly, so the TOML default was unreachable in practice
    // and 1e-3 -- the Python value -- wins.
    auto c = load(minimal);
    BOOST_REQUIRE(c.Absolute_tolerance.size() == 1u);
    BOOST_TEST(c.Absolute_tolerance[0] == 1e-3);
}

BOOST_AUTO_TEST_CASE(absolute_tolerance_accepts_a_scalar_or_an_array)
{
    BOOST_TEST(load(minimal + "Absolute_tolerance = 1e-5\n").Absolute_tolerance.size() == 1u);
    BOOST_TEST(load(minimal + "Absolute_tolerance = [1e-5, 1e-6]\n").Absolute_tolerance.size() == 2u);
}

BOOST_AUTO_TEST_CASE(an_integer_is_accepted_where_a_number_is_wanted)
{
    // TOML distinguishes 1 from 1.0, and `tau = 1` is entirely natural.
    BOOST_TEST(load(minimal + "tau = 2\n").tau == 2.0);
}

BOOST_AUTO_TEST_CASE(a_missing_required_key_is_an_error_naming_it)
{
    try
    {
        load("GridSize = 8\ndelta_t = 0.1\nTransportSystem = \"X\"\n");
        BOOST_FAIL("expected a throw");
    }
    catch (std::invalid_argument const &e)
    {
        BOOST_TEST(std::string(e.what()).find("PolynomialDegree") != std::string::npos);
    }
}

BOOST_AUTO_TEST_CASE(every_missing_required_key_is_reported_at_once)
{
    // Reporting only the first turns fixing a config into a guessing game one
    // key at a time.
    try
    {
        load("TransportSystem = \"X\"\n");
        BOOST_FAIL("expected a throw");
    }
    catch (std::invalid_argument const &e)
    {
        std::string msg = e.what();
        BOOST_TEST(msg.find("PolynomialDegree") != std::string::npos);
        BOOST_TEST(msg.find("GridSize") != std::string::npos);
        BOOST_TEST(msg.find("delta_t") != std::string::npos);
    }
}

BOOST_AUTO_TEST_CASE(an_unknown_key_is_an_error_suggesting_the_nearest)
{
    try
    {
        load(minimal + "Superconvergnet = true\n");
        BOOST_FAIL("expected a throw");
    }
    catch (std::invalid_argument const &e)
    {
        std::string msg = e.what();
        BOOST_TEST(msg.find("Superconvergnet") != std::string::npos);
        BOOST_TEST(msg.find("Superconvergent") != std::string::npos);
    }
}

BOOST_AUTO_TEST_CASE(a_deprecated_alias_is_accepted)
{
    BOOST_TEST(load(minimal + "tZero = 0.5\n").t_initial == 0.5);
    BOOST_TEST(load(minimal + "aggressiveTimesteps = true\n").AggressiveTimesteps);
}

BOOST_AUTO_TEST_CASE(a_key_and_its_alias_together_is_an_error)
{
    // Silently preferring one would make the config lie about what it does.
    BOOST_CHECK_THROW(load(minimal + "t_initial = 1.0\ntZero = 2.0\n"),
                      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(a_wrong_type_names_the_key_and_both_types)
{
    try
    {
        load(minimal + "Superconvergent = \"yes\"\n");
        BOOST_FAIL("expected a throw");
    }
    catch (std::invalid_argument const &e)
    {
        std::string msg = e.what();
        BOOST_TEST(msg.find("Superconvergent") != std::string::npos);
        BOOST_TEST(msg.find("boolean") != std::string::npos);
    }
}

BOOST_AUTO_TEST_CASE(the_cli_keys_are_accepted_and_ignored)
{
    // Eight .conf files in the tree carry these for manta.cli. Rejecting them
    // would break every one.
    auto c = load(minimal +
                  "PythonModuleName = \"case\"\nPythonModuleFile = \"case.py\"\n");
    BOOST_TEST(c.TransportSystem == "LinearDiffusion");
}

BOOST_AUTO_TEST_CASE(a_problem_selection_key_is_an_error_for_the_dict_reader)
{
    // TransportSystem is in the schema -- a config file needs it -- but it
    // means nothing to Runner.configure, which is handed the physics object.
    // Accepting and ignoring it is what the old reader did, and is the exact
    // failure mode this schema exists to stop.
    MapConfigSource src;
    src.values = {
        {"PolynomialDegree", 2u}, {"GridSize", 8}, {"delta_t", 0.1},
        {"LowerBoundary", 0.0},   {"UpperBoundary", 1.0},
        {"OutputFilename", std::string("out")},
        {"TransportSystem", std::string("LinearDiffusion")},
    };

    try
    {
        loadSolverConfig(src, ConfigSchema::Reader::Dict);
        BOOST_FAIL("expected a throw");
    }
    catch (std::invalid_argument const &e)
    {
        std::string msg = e.what();
        BOOST_TEST(msg.find("TransportSystem") != std::string::npos);
        BOOST_TEST(msg.find("object") != std::string::npos);
    }
}

BOOST_AUTO_TEST_CASE(a_problem_selection_key_is_fine_for_the_toml_reader)
{
    BOOST_TEST(load(minimal).TransportSystem == "LinearDiffusion");
}

BOOST_AUTO_TEST_CASE(the_field_model_key_is_problem_selection_too)
{
    // Same treatment as TransportSystem: a config file names a registered model
    // and runManta instantiates it; a dict has no equivalent and must be told
    // so rather than have the key quietly ignored.
    BOOST_TEST(load(minimal).FieldModel == "");
    BOOST_TEST(load(minimal + "FieldModel = \"SomeModel\"\n").FieldModel == "SomeModel");

    MapConfigSource src;
    src.values = {
        {"Polynomial_degree", 2u}, {"Grid_size", 8}, {"delta_t", 0.1},
        {"Lower_boundary", 0.0},   {"Upper_boundary", 1.0},
        {"OutputFilename", std::string("out")},
        {"FieldModel", std::string("SomeModel")},
    };
    BOOST_CHECK_THROW(loadSolverConfig(src, ConfigSchema::Reader::Dict),
                      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(the_field_solve_defaults_are_the_ones_the_solver_starts_with)
{
    // Two defaults that have to agree: the schema's, and SystemSolver's own
    // member initialisers. They are separate declarations, so nothing but a
    // test connects them.
    auto c = load(minimal);
    BOOST_TEST(c.FieldSolve == "iterative");
    BOOST_TEST(c.FieldSolveTolerance == 1e-8);
    BOOST_TEST(c.FieldSolveMaxSweeps == 20);
    BOOST_TEST(c.FieldSolveMaxAdjointSweeps == 100);

    Grid grid(0.0, 1.0, 4);
    TestDiffusion problem(toml::parse_str("[DiffusionProblem]\nKappa = 1.0\n"));
    SystemSolver sys(grid, 1, &problem);
    BOOST_TEST((sys.getFieldSolveMode() == SystemSolver::FieldSolveMode::Iterative));
    BOOST_TEST(sys.getFieldSolveTolerance() == 1e-8);
    BOOST_TEST(sys.getFieldSolveMaxSweeps() == 20);
    BOOST_TEST(sys.getFieldSolveMaxAdjointSweeps() == 100);
}

BOOST_AUTO_TEST_CASE(apply_solver_config_carries_the_field_solve_settings_through)
{
    // The one thing a SolverConfig comparison cannot check. applySolverConfig is
    // where a configuration reaches the solver, so a set* call dropped from it
    // un-configures *both* surfaces at once and
    // both_sources_produce_the_same_solver_config would go on passing --
    // it compares SolverConfigs, not solvers.
    Grid grid(0.0, 1.0, 4);
    TestDiffusion problem(toml::parse_str("[DiffusionProblem]\nKappa = 1.0\n"));
    SystemSolver sys(grid, 1, &problem);

    applySolverConfig(load(minimal + "FieldSolve = \"exact\"\nFieldSolveTolerance = 1e-11\n"
                                     "FieldSolveMaxSweeps = 3\n"
                                     "FieldSolveMaxAdjointSweeps = 9\n"),
                      sys);

    BOOST_TEST((sys.getFieldSolveMode() == SystemSolver::FieldSolveMode::Exact));
    BOOST_TEST(sys.getFieldSolveTolerance() == 1e-11);
    BOOST_TEST(sys.getFieldSolveMaxSweeps() == 3);
    BOOST_TEST(sys.getFieldSolveMaxAdjointSweeps() == 9);
}

BOOST_AUTO_TEST_CASE(an_unrecognised_field_solve_is_rejected_rather_than_defaulted)
{
    // As for SteadyStateSolver: a typo would otherwise silently pick a
    // different algorithm.
    Grid grid(0.0, 1.0, 4);
    TestDiffusion problem(toml::parse_str("[DiffusionProblem]\nKappa = 1.0\n"));
    SystemSolver sys(grid, 1, &problem);

    try
    {
        applySolverConfig(load(minimal + "FieldSolve = \"schur\"\n"), sys);
        BOOST_FAIL("expected a throw");
    }
    catch (std::invalid_argument const &e)
    {
        std::string msg = e.what();
        BOOST_TEST(msg.find("FieldSolve") != std::string::npos);
        BOOST_TEST(msg.find("schur") != std::string::npos);
    }
}

BOOST_AUTO_TEST_CASE(apply_solver_config_carries_the_tau_scaling_through)
{
    Grid grid(0.0, 1.0, 4);
    TestDiffusion problem(toml::parse_str("[DiffusionProblem]\nKappa = 1.0\n"));
    SystemSolver sys(grid, 1, &problem);

    applySolverConfig(load(minimal), sys);
    BOOST_TEST((sys.getTauScaling() == SystemSolver::TauScaling::Constant));

    applySolverConfig(load(minimal + "tauScaling = \"Diffusive\"\ntauFloor = 0.02\n"), sys);
    BOOST_TEST((sys.getTauScaling() == SystemSolver::TauScaling::Diffusive));
    BOOST_TEST((sys.getTauUpdate() == SystemSolver::TauUpdate::Residual));
    BOOST_TEST(sys.tauFloorFraction == 0.02);

    applySolverConfig(load(minimal + "tauScaling = \"Diffusive\"\ntauUpdate = \"ContinuationStep\"\n"), sys);
    BOOST_TEST((sys.getTauUpdate() == SystemSolver::TauUpdate::ContinuationStep));

    applySolverConfig(load(minimal + "tauScaling = \"Diffusive\"\ntauUpdate = \"JacobianBuild\"\n"), sys);
    BOOST_TEST((sys.getTauUpdate() == SystemSolver::TauUpdate::JacobianBuild));
}

BOOST_AUTO_TEST_CASE(bad_tau_scaling_configurations_are_refused_by_name)
{
    // An unknown scaling for the reason FieldSolve's is refused; a non-positive
    // floor because a face with kappa = 0 would leave its trace undetermined;
    // and Diffusive with the adjoint because the adjoint cannot carry d tau / dy
    // and the gradient would be wrong with nothing to say so.
    Grid grid(0.0, 1.0, 4);
    TestDiffusion problem(toml::parse_str("[DiffusionProblem]\nKappa = 1.0\n"));
    SystemSolver sys(grid, 1, &problem);

    const std::vector<std::pair<std::string, std::string>> cases{
        {"tauScaling = \"Local\"\n", "Local"},
        {"tauScaling = \"Diffusive\"\ntauFloor = 0.0\n", "tauFloor"},
        {"tauScaling = \"Diffusive\"\ntau = -1.0\n", "tau > 0"},
        {"tauScaling = \"Diffusive\"\nsolveAdjoint = true\n", "solveAdjoint"},
        {"tauScaling = \"Diffusive\"\ntauUpdate = \"Newton\"\n", "tauUpdate"},
        {"tauUpdate = \"ContinuationStep\"\n", "Diffusive"},
    };
    for (auto const &[extra, needle] : cases)
    {
        try
        {
            applySolverConfig(load(minimal + extra), sys);
            BOOST_ERROR("expected a throw for: " << extra);
        }
        catch (std::invalid_argument const &e)
        {
            BOOST_TEST(std::string(e.what()).find(needle) != std::string::npos,
                       "message for `" << extra << "` does not name " << needle << ": " << e.what());
        }
    }
}

BOOST_AUTO_TEST_CASE(boundaries_are_required_unless_grid_points_is_given)
{
    const std::string noBounds =
        "PolynomialDegree = 2\nGrid_size = 8\ndelta_t = 0.1\nt_final = 1.0\n"
        "TransportSystem = \"X\"\n";

    BOOST_CHECK_THROW(load(noBounds), std::invalid_argument);

    auto c = load(noBounds + "GridPoints = [0.0, 0.5, 1.0]\n");
    BOOST_TEST(c.GridPoints.size() == 3u);
}

BOOST_AUTO_TEST_CASE(presence_is_recorded_for_the_three_keys_that_need_it)
{
    // These three are armed by being present, not by their value.
    auto without = load(minimal);
    BOOST_TEST(!without.SteadyStateTolerance.has_value());

    auto with = load(minimal + "SteadyStateTolerance = 1e-4\n");
    BOOST_REQUIRE(with.SteadyStateTolerance.has_value());
    BOOST_TEST(*with.SteadyStateTolerance == 1e-4);

    BOOST_REQUIRE(load(minimal).t_final.has_value());
    BOOST_TEST(*load(minimal).t_final == 1.0);
}

BOOST_AUTO_TEST_CASE(output_filename_falls_back_to_the_config_stem)
{
    auto v = toml::parse_str(minimal);
    TomlConfigSource src(v, "/some/where/myrun.conf");
    auto c = loadSolverConfig(src, ConfigSchema::Reader::Toml);
    BOOST_TEST(c.OutputFilename == "myrun");
}

BOOST_AUTO_TEST_CASE(an_explicit_output_filename_wins_over_the_stem)
{
    // On the TOML side this key used to be read by nothing at all.
    auto v = toml::parse_str(minimal + "OutputFilename = \"chosen\"\n");
    TomlConfigSource src(v, "/some/where/myrun.conf");
    auto c = loadSolverConfig(src, ConfigSchema::Reader::Toml);
    BOOST_TEST(c.OutputFilename == "chosen");
}

// The test this whole change exists for.
//
// Two sources carrying the same configuration must produce the same
// SolverConfig, field for field. Every drift that prompted this work -- two
// names for the initial time, two defaults for Absolute_tolerance, four keys on
// one side only -- would have failed here.
//
// MapConfigSource stands in for the dict: DictConfigSource needs pybind11 and
// cannot link into these tests, and what matters is not pybind's casting but
// the shared machinery above it. python/Tests/test_run_config.py covers the
// real dict.
BOOST_AUTO_TEST_CASE(both_sources_produce_the_same_solver_config)
{
    const std::string body = minimal +
        "tau = 2.5\n"
        "Relative_tolerance = 1e-6\n"
        "Absolute_tolerance = [1e-7, 1e-8]\n"
        "t_initial = 0.25\n"
        "OutputPoints = 51\n"
        "Superconvergent = true\n"
        "AggressiveTimesteps = true\n"
        "SuppressAlgebraicError = true\n"
        "ForceConsistentIC = true\n"
        "SteadyStateSolver = \"Newton\"\n"
        "PseudoTransientInitialStep = 0.25\n"
        "PseudoTransientMaxStep = 1e6\n"
        "PseudoTransientSERRate = 0.5\n"
        "PseudoTransientSERFloor = 1.5\n"
        "NewtonMaxIterations = 7\n"
        "NewtonJacobianReuse = 3\n"
        "NewtonStepTolerance = 1e-9\n"
        "NewtonScaling = \"ErrorWeights\"\n"
        "SteadyStateDiagnostics = true\n"
        "SteadyStateStepDiagnostics = true\n"
        "zeroFlux = true\n"
        "WriteOutput = false\n"
        "SteadyStateTolerance = 1e-5\n"
        "FieldSolve = \"exact\"\n"
        "FieldSolveTolerance = 1e-10\n"
        "FieldSolveMaxSweeps = 7\n"
        "FieldSolveMaxAdjointSweeps = 31\n"
        "OutputFilename = \"shared\"\n";

    auto v = toml::parse_str(body);
    TomlConfigSource toml_src(v, "/tmp/ignored.conf");
    auto fromToml = loadSolverConfig(toml_src, ConfigSchema::Reader::Toml);

    // Each entry must hold exactly the alternative the schema declares --
    // MapConfigSource returns the variant as stored rather than converting, so
    // `8` and `8u` are not interchangeable here.
    MapConfigSource map_src;
    map_src.values = {
        {"PolynomialDegree", 2u}, {"GridSize", 8}, {"delta_t", 0.1},
        {"t_final", 1.0}, {"LowerBoundary", 0.0}, {"UpperBoundary", 1.0},
        {"tau", 2.5}, {"Relative_tolerance", 1e-6},
        {"Absolute_tolerance", std::vector<double>{1e-7, 1e-8}},
        {"t_initial", 0.25}, {"OutputPoints", 51},
        {"Superconvergent", true}, {"AggressiveTimesteps", true},
        {"SuppressAlgebraicError", true}, {"ForceConsistentIC", true},
        {"SteadyStateSolver", std::string("Newton")},
        {"PseudoTransientInitialStep", 0.25}, {"PseudoTransientMaxStep", 1e6},
        {"PseudoTransientSERRate", 0.5}, {"PseudoTransientSERFloor", 1.5},
        {"NewtonMaxIterations", 7u},
        {"NewtonJacobianReuse", 3u},
        {"NewtonStepTolerance", 1e-9},
        {"NewtonScaling", std::string("ErrorWeights")},
        {"SteadyStateDiagnostics", true},
        {"SteadyStateStepDiagnostics", true},
        {"zeroFlux", true}, {"WriteOutput", false},
        {"SteadyStateTolerance", 1e-5}, {"OutputFilename", std::string("shared")},
        {"FieldSolve", std::string("exact")}, {"FieldSolveTolerance", 1e-10},
        {"FieldSolveMaxSweeps", 7}, {"FieldSolveMaxAdjointSweeps", 31},
    };
    auto fromMap = loadSolverConfig(map_src, ConfigSchema::Reader::Dict);

    BOOST_TEST(fromToml.PolynomialDegree == fromMap.PolynomialDegree);
    BOOST_TEST(fromToml.GridSize == fromMap.GridSize);
    BOOST_TEST(fromToml.delta_t == fromMap.delta_t);
    BOOST_TEST(fromToml.tau == fromMap.tau);
    BOOST_TEST(fromToml.t_initial == fromMap.t_initial);
    BOOST_TEST(fromToml.Relative_tolerance == fromMap.Relative_tolerance);
    BOOST_TEST(fromToml.Absolute_tolerance == fromMap.Absolute_tolerance,
               boost::test_tools::per_element());
    BOOST_TEST(fromToml.OutputPoints == fromMap.OutputPoints);
    BOOST_TEST(fromToml.OutputFilename == fromMap.OutputFilename);
    // Wrapped: std::optional has no operator<< for Boost.Test to print.
    const bool superconvergentAgrees =
        fromToml.Superconvergent == fromMap.Superconvergent;
    BOOST_TEST(superconvergentAgrees);
    BOOST_TEST(fromToml.AggressiveTimesteps == fromMap.AggressiveTimesteps);
    BOOST_TEST(fromToml.SuppressAlgebraicError == fromMap.SuppressAlgebraicError);
    BOOST_TEST(fromToml.ForceConsistentIC == fromMap.ForceConsistentIC);
    BOOST_TEST(fromToml.SteadyStateSolver == fromMap.SteadyStateSolver);
    BOOST_TEST(fromToml.PseudoTransientInitialStep == fromMap.PseudoTransientInitialStep);
    BOOST_TEST(fromToml.PseudoTransientMaxStep == fromMap.PseudoTransientMaxStep);
    BOOST_TEST(fromToml.PseudoTransientSERRate == fromMap.PseudoTransientSERRate);
    BOOST_TEST(fromToml.PseudoTransientSERFloor == fromMap.PseudoTransientSERFloor);
    BOOST_TEST(fromToml.NewtonMaxIterations == fromMap.NewtonMaxIterations);
    BOOST_TEST(fromToml.NewtonJacobianReuse == fromMap.NewtonJacobianReuse);
    BOOST_TEST(fromToml.NewtonStepTolerance == fromMap.NewtonStepTolerance);
    BOOST_TEST(fromToml.NewtonScaling == fromMap.NewtonScaling);
    BOOST_TEST(fromToml.SteadyStateDiagnostics == fromMap.SteadyStateDiagnostics);
    BOOST_TEST(fromToml.SteadyStateStepDiagnostics == fromMap.SteadyStateStepDiagnostics);
    BOOST_TEST(fromToml.zeroFlux == fromMap.zeroFlux);
    BOOST_TEST(fromToml.WriteOutput == fromMap.WriteOutput);
    BOOST_TEST(fromToml.MinStepSize == fromMap.MinStepSize);
    BOOST_TEST(fromToml.initialTimestep == fromMap.initialTimestep);
    BOOST_TEST(fromToml.WriteDatFile == fromMap.WriteDatFile);
    BOOST_TEST(fromToml.WriteDebugDatFiles == fromMap.WriteDebugDatFiles);
    BOOST_TEST(fromToml.LowerBoundary == fromMap.LowerBoundary);
    BOOST_TEST(fromToml.UpperBoundary == fromMap.UpperBoundary);
    BOOST_TEST(fromToml.restart == fromMap.restart);
    BOOST_TEST(fromToml.solveAdjoint == fromMap.solveAdjoint);
    BOOST_TEST(fromToml.FieldSolve == fromMap.FieldSolve);
    BOOST_TEST(fromToml.FieldSolveTolerance == fromMap.FieldSolveTolerance);
    BOOST_TEST(fromToml.FieldSolveMaxSweeps == fromMap.FieldSolveMaxSweeps);
    BOOST_TEST(fromToml.FieldSolveMaxAdjointSweeps == fromMap.FieldSolveMaxAdjointSweeps);
    // FieldModel is ProblemSelection, so it is an error in a dict and cannot be
    // compared across the two -- the same asymmetry TransportSystem has.
    BOOST_TEST(fromToml.FieldModel == "");
    BOOST_REQUIRE(fromToml.SteadyStateTolerance.has_value());
    BOOST_REQUIRE(fromMap.SteadyStateTolerance.has_value());
    BOOST_TEST(*fromToml.SteadyStateTolerance == *fromMap.SteadyStateTolerance);
    BOOST_REQUIRE(fromToml.t_final.has_value());
    BOOST_REQUIRE(fromMap.t_final.has_value());
    BOOST_TEST(*fromToml.t_final == *fromMap.t_final);
}

namespace
{
// A minimal config with the mesh spelled out, since `minimal` already names a
// Grid_size and toml refuses a duplicate key.
std::string meshed(std::string const &mesh, bool restart = true)
{
    return std::string("Polynomial_degree = 2\n")
           + "delta_t = 0.1\n"
             "t_final = 1.0\n"
             "TransportSystem = \"LinearDiffusion\"\n"
           + (restart ? "restart = true\n" : "")
           + mesh;
}

} // namespace

// --- the mesh a restarted run is solved on --------------------------------
//
// restartRunGrid is to Grid_size what restartRunOrder is to Polynomial_degree.
// Both keys are required of every config on both readers; both used to be read,
// validated and then discarded on a restart, because makeGrid took the whole
// discretisation out of the file. The degree was fixed first; this is the mesh.
//
// Why it matters beyond tidiness: a ladder written as "solve coarse, restart
// finer, solve again" silently re-solved the coarse problem at every rung and
// reported it converged, which is indistinguishable from success -- resuming a
// converged state at its own resolution costs one residual evaluation and exits
// at the already-converged test, exactly as a genuine rung would look.

BOOST_AUTO_TEST_CASE(a_restart_onto_the_same_mesh_keeps_it)
{
    // The no-regression half, and the reason the comparison is on the Grid
    // rather than on Grid_size: an equal mesh has to come back equal so that
    // setInitialConditions takes the copy path and every existing restart is
    // bit for bit what it was.
    auto c = load(meshed("Grid_size = 8\nLower_boundary = 0.0\nUpper_boundary = 1.0\n"));
    Grid fileGrid(0.0, 1.0, 8);

    auto run = restartRunGrid(c, fileGrid);
    BOOST_TEST((*run == fileGrid));
    BOOST_TEST(run->getNCells() == 8);
}

BOOST_AUTO_TEST_CASE(a_restart_onto_a_different_mesh_honours_the_configuration)
{
    auto c = load(meshed("Grid_size = 20\nLower_boundary = 0.0\nUpper_boundary = 1.0\n"));
    Grid fileGrid(0.0, 1.0, 5);

    CapturedOutput quiet;
    auto run = restartRunGrid(c, fileGrid);
    BOOST_TEST(run->getNCells() == 20);
    BOOST_TEST(!(*run == fileGrid));
    BOOST_TEST(run->lowerBoundary() == 0.0);
    BOOST_TEST(run->upperBoundary() == 1.0);
}

BOOST_AUTO_TEST_CASE(a_restart_onto_a_coarser_mesh_is_allowed_and_is_the_lossy_direction)
{
    // Refining puts the stored element polynomials inside the new space;
    // coarsening is a genuine approximation. Both are permitted -- a ladder may
    // want either -- and the warning is what distinguishes them, so the test
    // pins only that coarsening is not refused.
    auto c = load(meshed("Grid_size = 4\nLower_boundary = 0.0\nUpper_boundary = 1.0\n"));
    Grid fileGrid(0.0, 1.0, 16);

    CapturedOutput quiet;
    auto run = restartRunGrid(c, fileGrid);
    BOOST_TEST(run->getNCells() == 4);
}

BOOST_AUTO_TEST_CASE(a_restart_onto_a_different_domain_honours_the_configuration)
{
    // The mesh is the cell boundaries, not the cell count, so moving the domain
    // is a mesh change even at the same Grid_size. Worth its own case because
    // Lower_boundary and Upper_boundary are not required keys and default to 0
    // and 1: a restart config that omits them and resumes a run over [-1, 1]
    // will be remeshed onto [0, 1], and the warning is the only thing that says
    // so.
    auto c = load(meshed("Grid_size = 8\nLower_boundary = 0.0\nUpper_boundary = 1.0\n"));
    Grid fileGrid(-1.0, 1.0, 8);

    CapturedOutput quiet;
    auto run = restartRunGrid(c, fileGrid);
    BOOST_TEST(run->getNCells() == 8);
    BOOST_TEST(run->lowerBoundary() == 0.0);
    BOOST_TEST(!(*run == fileGrid));
}

BOOST_AUTO_TEST_CASE(grid_points_supersede_grid_size_on_a_restart_too)
{
    auto c = load(meshed("Grid_size = 8\nGrid_points = [0.0, 0.25, 0.9, 1.0]\n"));
    Grid fileGrid(0.0, 1.0, 8);

    CapturedOutput quiet;
    auto run = restartRunGrid(c, fileGrid);
    BOOST_TEST(run->getNCells() == 3);
}

BOOST_AUTO_TEST_CASE(without_restart_the_file_mesh_is_returned_unchanged)
{
    // Defensive: the callers only reach this on a restart, but a function that
    // silently remeshed a cold start would be a bad one to leave lying about.
    auto c = load(meshed("Grid_size = 20\nLower_boundary = 0.0\nUpper_boundary = 1.0\n", false));
    Grid fileGrid(0.0, 1.0, 5);

    auto run = restartRunGrid(c, fileGrid);
    BOOST_TEST((*run == fileGrid));
}

// --- DegreeLadder / GridLadder --------------------------------------------

BOOST_AUTO_TEST_CASE(a_ladder_reaches_the_same_answer_as_a_direct_solve)
{
    // The property the whole design rests on: the last rung is always the
    // configured resolution, so a ladder is a *route* and not a change of
    // destination. Remove the key and the numbers must not move -- which is
    // what makes it safe to try on a problem you already have an answer for.
    //
    // Exactly, not approximately. Both solves end at the same steady tolerance
    // on the same discretisation, so any difference would be a difference in
    // the state Newton converged from, and the point is that that does not
    // survive to the answer.
    const std::string body =
        "Polynomial_degree = 3\n"
        "Grid_size = 8\n"
        "delta_t = 0.1\n"
        "t_final = 1.0\n"
        "Lower_boundary = 0.0\n"
        "Upper_boundary = 1.0\n"
        "OutputFilename = \"ladder_equal\"\n"
        "WriteOutput = false\n"
        "SteadyStateSolver = \"Newton\"\n"
        "TransportSystem = \"LinearDiffusion\"\n";

    const toml::value diffusion = toml::parse_str(
        "[DiffusionProblem]\nKappa = 1.0\nCentre = 0.0\n");

    auto solve = [&](std::string const &extra)
    {
        auto c = load(body + extra);
        Grid grid(0.0, 1.0, 8);
        TestDiffusion problem(diffusion);
        std::unique_ptr<SystemSolver> sys;
        {
            CapturedOutput quiet;
            if (c.DegreeLadder.empty() && c.GridLadder.empty())
            {
                sys = std::make_unique<SystemSolver>(grid, 3, &problem);
                applySolverConfig(c, *sys);
                sys->runSolver(*c.t_final);
            }
            else
            {
                PhysicsInstance physics(problem, grid);
                sys = runLadder(c, physics, grid, 3, *c.t_final);
            }
        }
        auto Y = sys->stateVector();
        {
            CapturedOutput quiet;
            sys->destroySundials();
        }
        return Y;
    };

    const std::string t = "SteadyStateTolerance = 1.0e-12\n";
    const auto direct = solve(t);
    const auto laddered = solve(t + "DegreeLadder = [1, 2]\nGridLadder = [2, 4]\n");
    BOOST_REQUIRE_EQUAL(direct.size(), laddered.size());

    // Every degree of freedom agrees to round-off, not merely to the steady
    // tolerance: both end on the same discretisation with the residual driven to
    // the same place, so it is the same state.
    //
    // The two Dirichlet trace entries used to be exceptions, and the reason had
    // nothing to do with the ladder. A Dirichlet trace row *and column* are
    // identically zero in K_global, so no solve can move those entries: each
    // keeps whatever setInitialConditions seeded it with. The cold path seeds
    // them from EvaluateLambda's {{u}} and the restart path from the boundary
    // datum, and a ladder ends on a restart -- so the ladder reported 1.0 where
    // a cold solve reported 0.9999991, a discretisation error apart and
    // identical at every tolerance from 1e-8 to 1e-14, because neither was
    // converging to anything.
    //
    // The seeds still differ, deliberately -- see TODO -- and this now passes
    // anyway, because the datum is written back into those entries at every
    // point the state is reported, writeSteadyState() included. So the whole
    // vector is in scope here, and keeping it that way is the point of the test:
    // it is what would notice a reporting path that got missed.
    double worst = 0.0;
    for (size_t i = 0; i < direct.size(); ++i)
        worst = std::max(worst, std::abs(direct[i] - laddered[i]));
    BOOST_TEST_MESSAGE("ladder against direct: worst |dY| = " << worst);
    BOOST_TEST(worst < 1e-12);

    // And both hold the datum at the lower face, rather than merely agreeing
    // with each other on some extrapolation of the interior.
    const size_t lambda0 = 3 * 4 * 8;
    BOOST_TEST(direct[lambda0] == 1.0, boost::test_tools::tolerance(1e-14));
    BOOST_TEST(laddered[lambda0] == 1.0, boost::test_tools::tolerance(1e-14));
}

BOOST_AUTO_TEST_CASE(the_two_ladders_must_describe_the_same_rungs)
{
    BOOST_CHECK_THROW(load(minimal + "DegreeLadder = [1, 2]\nGridLadder = [4]\n"),
                      std::invalid_argument);
    // Either alone is fine: it holds the other quantity at its configured value.
    BOOST_CHECK_NO_THROW(load(minimal + "DegreeLadder = [1, 2]\n"
                                        "SteadyStateSolve = true\n"));
    BOOST_CHECK_NO_THROW(load(minimal + "GridLadder = [2, 4]\n"
                                        "SteadyStateSolve = true\n"));
}

BOOST_AUTO_TEST_CASE(a_ladder_refuses_what_it_cannot_mean)
{
    // Degree zero has no gradient to postprocess and cannot be evaluated off
    // its node; zero cells is not a mesh.
    BOOST_CHECK_THROW(load(minimal + "DegreeLadder = [0, 2]\nSteadyStateSolve = true\n"),
                      std::invalid_argument);
    BOOST_CHECK_THROW(load(minimal + "GridLadder = [0, 4]\nSteadyStateSolve = true\n"),
                      std::invalid_argument);

    // Both choose the sequence of discretisations, from different information.
    BOOST_CHECK_THROW(load(minimal + "DegreeLadder = [1, 2]\nDegreeAdaptation = true\n"
                                     "SteadyStateSolve = true\n"),
                      std::invalid_argument);

    // A transient rung would take the previous rung's final state and integrate
    // the same interval again -- a wrong answer, not a slow one.
    BOOST_CHECK_THROW(load(minimal + "DegreeLadder = [1, 2]\n"
                                     "SteadyStateSolver = \"TimeMarch\"\n"
                                     "SteadyStateSolve = true\n"),
                      std::invalid_argument);

    // And without a steady solve armed at all, SteadyStateSolver is never
    // consulted and every rung time-marches regardless of what it says.
    BOOST_CHECK_THROW(load(minimal + "DegreeLadder = [1, 2]\n"),
                      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(a_ladder_is_a_list_of_whole_numbers)
{
    // UIntList exists rather than reusing DoubleList because these are counts:
    // 2.5 cells is a configuration error and not something to round.
    BOOST_CHECK_THROW(load(minimal + "GridLadder = [2.5]\nSteadyStateSolve = true\n"),
                      std::invalid_argument);
    BOOST_CHECK_THROW(load(minimal + "GridLadder = [-2]\nSteadyStateSolve = true\n"),
                      std::invalid_argument);

    // A bare scalar is a one-rung ladder, matching how DoubleList treats one.
    auto c = load(minimal + "GridLadder = 4\nSteadyStateSolve = true\n");
    BOOST_REQUIRE_EQUAL(c.GridLadder.size(), 1u);
    BOOST_TEST(c.GridLadder[0] == 4u);
}

// ------------------------------------------- geometric mesh grading ----

BOOST_AUTO_TEST_CASE(graded_grid_defaults_are_off_and_harmless)
{
    auto c = load(minimal);
    BOOST_TEST(c.GradedGridBoundary == false);
    BOOST_TEST(c.GradingRatio == 0.3);
    BOOST_TEST(c.GradingCells == 0);      // 0 means "derive from GridSize"
    BOOST_TEST(c.GradingEnd == "Both");

    // Off, so the count is left as the sentinel rather than resolved -- which is
    // the property that keeps a plain config bit for bit what it was.
    unsigned int k = 0;
    auto grid = makeGrid(c, nullptr, k);
    BOOST_TEST(grid->getNCells() == 8u);
    for (Grid::Index i = 0; i < grid->getNCells(); ++i)
        BOOST_TEST((*grid)[i].h() == 0.125, boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(the_layer_fractions_record_whether_they_were_given)
{
    // MeshAdaptation sizes its layer from the sampling mesh unless a fraction is
    // given, so it needs the absent key told apart from the schema default -- a
    // config that writes the default out has asked for it. Either spelling counts.
    auto absent = load(minimal);
    BOOST_TEST(!absent.LowerBoundaryFractionGiven);
    BOOST_TEST(!absent.UpperBoundaryFractionGiven);
    BOOST_TEST(absent.LowerBoundaryFraction == 0.2);

    auto lower = load(minimal + "LowerBoundaryFraction = 0.2\n");
    BOOST_TEST(lower.LowerBoundaryFractionGiven);
    BOOST_TEST(!lower.UpperBoundaryFractionGiven);

    auto alias = load(minimal + "Upper_Boundary_Fraction = 0.1\n");
    BOOST_TEST(!alias.LowerBoundaryFractionGiven);
    BOOST_TEST(alias.UpperBoundaryFractionGiven);
    BOOST_TEST(alias.UpperBoundaryFraction == 0.1);
}

BOOST_AUTO_TEST_CASE(a_graded_grid_config_builds_the_mesh_it_describes)
{
    // The end-to-end path: keys -> SolverConfig -> makeGrid -> Grid. The layer
    // width comes from LowerBoundaryFraction, one key for one meaning rather than
    // a second that would drift from it.
    auto c = load(minimal +
                  "GradedGridBoundary = true\n"
                  "GradingEnd = \"Lower\"\n"
                  "GradingRatio = 0.5\n"
                  "GradingCells = 4\n"
                  "LowerBoundaryFraction = 0.2\n");
    BOOST_TEST(c.GradedGridBoundary == true);
    BOOST_TEST(c.GradingCells == 4);

    unsigned int k = 0;
    auto grid = makeGrid(c, nullptr, k);
    BOOST_TEST(grid->getNCells() == 8u);
    BOOST_TEST(grid->lowerBoundary() == 0.0);
    BOOST_TEST(grid->upperBoundary() == 1.0);

    // h0 = fraction * span * ratio^(cells-1) = 0.2 * 0.5^3
    BOOST_TEST((*grid)[0].h() == 0.2 * 0.125, boost::test_tools::tolerance(1e-12));
    // ...and the four uniform cells beyond the layer
    for (Grid::Index i = 4; i < 8; ++i)
        BOOST_TEST((*grid)[i].h() == 0.8 / 4.0, boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(grading_the_upper_end_reads_the_upper_fraction)
{
    // Which fraction is read depends on GradingEnd, and getting that backwards
    // would still produce a graded mesh -- of the wrong layer width, silently.
    // Distinct fractions here so the two cannot be confused.
    auto c = load(minimal +
                  "GradedGridBoundary = true\n"
                  "GradingEnd = \"Upper\"\n"
                  "GradingRatio = 0.5\n"
                  "GradingCells = 4\n"
                  "LowerBoundaryFraction = 0.4\n"
                  "UpperBoundaryFraction = 0.2\n");

    unsigned int k = 0;
    auto grid = makeGrid(c, nullptr, k);
    BOOST_TEST(grid->getNCells() == 8u);

    // The narrow cell is the last one, and its width is set by 0.2 not 0.4.
    BOOST_TEST((*grid)[7].h() == 0.2 * 0.125, boost::test_tools::tolerance(1e-10));
    BOOST_TEST((*grid)[0].h() == 0.8 / 4.0, boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(grading_both_ends_is_the_default_and_splits_the_grid_in_thirds)
{
    // The default, and what High_Grid_Boundary produced: a third of the cells in
    // each layer. Resolved in loadSolverConfig rather than in the schema, because a
    // schema default cannot see another key.
    auto c = load(minimal +
                  "GradedGridBoundary = true\n"
                  "GradingRatio = 0.5\n"
                  "LowerBoundaryFraction = 0.2\n"
                  "UpperBoundaryFraction = 0.2\n");
    BOOST_TEST(c.GradingEnd == "Both");
    BOOST_TEST(c.GradingCells == 2);      // GridSize is 8, so 8/3

    unsigned int k = 0;
    auto grid = makeGrid(c, nullptr, k);
    BOOST_TEST(grid->getNCells() == 8u);

    // Narrow at both ends, wide in the middle, and symmetric: 2 graded cells per
    // layer with h0 = 0.2 * 0.5 = 0.1, then 4 uniform cells across the middle 60%.
    BOOST_TEST((*grid)[0].h() == 0.1, boost::test_tools::tolerance(1e-12));
    BOOST_TEST((*grid)[7].h() == 0.1, boost::test_tools::tolerance(1e-10));
    for (Grid::Index i = 2; i < 6; ++i)
        BOOST_TEST((*grid)[i].h() == 0.6 / 4.0, boost::test_tools::tolerance(1e-12));

    // Both layers land exactly on the fractions they were given.
    BOOST_TEST((*grid)[1].x_u == 0.2, boost::test_tools::tolerance(1e-12));
    BOOST_TEST((*grid)[6].x_l == 0.8, boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(the_retired_cosine_spelling_still_loads_and_grades_instead)
{
    // High_Grid_Boundary is a deprecated alias of GradedGridBoundary now, so an old
    // config keeps loading -- with two warnings, since both the name *and* the mesh
    // it builds have changed. The alias is the reason this rename touched no
    // .conf file and no driver in the tree.
    auto c = load(minimal + "High_Grid_Boundary = true\n");
    BOOST_TEST(c.GradedGridBoundary == true);
    BOOST_TEST(c.GradingEnd == "Both");
    BOOST_TEST(c.GradingCells == 2);

    unsigned int k = 0;
    auto grid = makeGrid(c, nullptr, k);
    BOOST_TEST(grid->getNCells() == 8u);

    // Finer at the walls than in the middle, which is the property the old flag
    // was for and the only one an old config was entitled to rely on.
    BOOST_TEST((*grid)[0].h() < (*grid)[4].h());
    BOOST_TEST((*grid)[7].h() < (*grid)[4].h());

    // ...and giving both spellings at once is refused rather than resolved.
    BOOST_CHECK_THROW(load(minimal +
                           "GradedGridBoundary = true\n"
                           "High_Grid_Boundary = true\n"),
                      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(a_graded_grid_config_refuses_geometry_it_cannot_build)
{
    BOOST_CHECK_THROW(load(minimal +
                           "GradedGridBoundary = true\n"
                           "GradingEnd = \"Sideways\"\n"),
                      std::invalid_argument);

    // Grading both ends of a 4-cell grid cannot carry two layers of two plus a
    // cell between them; grading one end of it can.
    const std::string tiny =
        "PolynomialDegree = 2\nGridSize = 4\ndelta_t = 0.1\n"
        "t_final = 1.0\nLowerBoundary = 0.0\nUpperBoundary = 1.0\n"
        "TransportSystem = \"LinearDiffusion\"\nGradedGridBoundary = true\n";
    BOOST_CHECK_THROW(load(tiny), std::invalid_argument);
    BOOST_CHECK_NO_THROW(load(tiny + "GradingEnd = \"Lower\"\n"));

    // The rest is gradedMeshPoints's own validation, reached through makeGrid --
    // checked here so the config path is known to surface it rather than to
    // swallow it.
    auto c = load(minimal + "GradedGridBoundary = true\nGradingRatio = 1.5\n");
    unsigned int k = 0;
    BOOST_CHECK_THROW(makeGrid(c, nullptr, k), std::invalid_argument);

    auto c2 = load(minimal + "GradedGridBoundary = true\nGradingCells = 8\n");
    BOOST_CHECK_THROW(makeGrid(c2, nullptr, k), std::invalid_argument);
}

// -------------------------------------- explicit boundaries, and the rename --

BOOST_AUTO_TEST_CASE(explicit_grid_points_need_no_grid_size_or_boundaries)
{
    // GridPoints supersedes GridSize, LowerBoundary and UpperBoundary outright, so
    // none of them is required alongside it. GridSize used to be required of every
    // config regardless, which meant a run driven by explicit boundaries had to
    // carry a number that was then discarded -- every graded-mesh spike in
    // MESH-REFINEMENT.md passed a dummy for exactly that reason.
    auto c = load("PolynomialDegree = 2\ndelta_t = 0.1\nt_final = 1.0\n"
                  "TransportSystem = \"LinearDiffusion\"\n"
                  "GridPoints = [0.0, 0.1, 0.3, 0.7, 1.0]\n");

    unsigned int k = 0;
    auto grid = makeGrid(c, nullptr, k);
    BOOST_TEST(grid->getNCells() == 4u);
    BOOST_TEST(grid->lowerBoundary() == 0.0);
    BOOST_TEST(grid->upperBoundary() == 1.0);
    BOOST_TEST((*grid)[0].h() == 0.1, boost::test_tools::tolerance(1e-12));
    BOOST_TEST((*grid)[3].h() == 0.3, boost::test_tools::tolerance(1e-12));

    // ...and without it all three are still demanded, in one message rather than
    // one at a time. Checked against the *source* rather than the parsed value,
    // because absent and 0 are the same value and must not be the same diagnosis.
    BOOST_CHECK_THROW(load("PolynomialDegree = 2\ndelta_t = 0.1\nt_final = 1.0\n"
                           "TransportSystem = \"LinearDiffusion\"\n"),
                      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(the_old_snake_case_grid_spellings_still_work)
{
    // The whole point of renaming through aliases: every .conf and every
    // Runner.configure dict written against the old names keeps working, with a
    // deprecation warning. That is what kept this rename from touching 35 Python
    // files and every config in the tree.
    // Every one of the seven old spellings, in a config that uses no new one --
    // which is the case a user with an existing file actually has. A blanket
    // rename over this tree once turned these into the new names, at which point
    // the test still passed and covered nothing; hence one config, all old.
    auto c = load("Polynomial_degree = 3\nGrid_size = 6\ndelta_t = 0.1\n"
                  "t_final = 1.0\nLower_boundary = -1.0\nUpper_boundary = 2.0\n"
                  "Lower_Boundary_Fraction = 0.3\nUpper_Boundary_Fraction = 0.4\n"
                  "TransportSystem = \"LinearDiffusion\"\n");

    BOOST_TEST(c.PolynomialDegree == 3u);
    BOOST_TEST(c.GridSize == 6);
    BOOST_TEST(c.LowerBoundary == -1.0);
    BOOST_TEST(c.UpperBoundary == 2.0);
    BOOST_TEST(c.LowerBoundaryFraction == 0.3);
    BOOST_TEST(c.UpperBoundaryFraction == 0.4);

    // Grid_points too, which also has to satisfy the conditional rule above under
    // its old spelling -- so no cell count here at all.
    auto p = load("Polynomial_degree = 2\ndelta_t = 0.1\nt_final = 1.0\n"
                  "TransportSystem = \"LinearDiffusion\"\n"
                  "Grid_points = [0.0, 0.5, 1.0]\n");
    BOOST_TEST(p.GridPoints.size() == 3u);

    // One spelling at a time. `minimal` carries GridSize, so this adds the alias.
    BOOST_CHECK_THROW(load(minimal + "Grid_size = 4\n"), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
