// Reading a [configuration] table against the schema.
//
// The rules under test here are the ones that used to be open-coded twice and
// differently: what is required, what an absent key defaults to, what happens
// to a key nobody recognises, and the two conditional requirements that a flat
// required-list cannot express.

#include <boost/test/unit_test.hpp>

#include "SolverConfig.hpp"

// SolverConfig.hpp only forward-declares Grid, so that it stays cheap to include
// and pybind11-free. The graded-mesh cases below inspect the Grid makeGrid built,
// so they need the definition; taken here rather than by widening that header.
#include "gridStructures.hpp"

#include <map>
#include <stdexcept>
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
    "Polynomial_degree = 2\n"
    "Grid_size = 8\n"
    "delta_t = 0.1\n"
    "t_final = 1.0\n"
    "Lower_boundary = 0.0\n"
    "Upper_boundary = 1.0\n"
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

    BOOST_TEST(c.Polynomial_degree == 2u);
    BOOST_TEST(c.Grid_size == 8);
    BOOST_TEST(c.tau == 1.0);
    BOOST_TEST(c.Relative_tolerance == 1e-3);
    BOOST_REQUIRE(c.Absolute_tolerance.size() == 1u);
    BOOST_TEST(c.Absolute_tolerance[0] == 1e-3);
    BOOST_TEST(c.MinStepSize == 1e-7);
    BOOST_TEST(c.OutputPoints == 301);
    BOOST_TEST(c.WriteOutput);
    BOOST_TEST(!c.WriteDatFile);
    BOOST_TEST(!c.Superconvergent);
    BOOST_TEST(!c.AggressiveTimesteps);
    BOOST_TEST(!c.SuppressAlgebraicError);
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
    BOOST_TEST(c.SteadyStateDiagnostics == false);
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
        load("Grid_size = 8\ndelta_t = 0.1\nTransportSystem = \"X\"\n");
        BOOST_FAIL("expected a throw");
    }
    catch (std::invalid_argument const &e)
    {
        BOOST_TEST(std::string(e.what()).find("Polynomial_degree") != std::string::npos);
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
        BOOST_TEST(msg.find("Polynomial_degree") != std::string::npos);
        BOOST_TEST(msg.find("Grid_size") != std::string::npos);
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
        {"Polynomial_degree", 2u}, {"Grid_size", 8}, {"delta_t", 0.1},
        {"Lower_boundary", 0.0},   {"Upper_boundary", 1.0},
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

BOOST_AUTO_TEST_CASE(boundaries_are_required_unless_grid_points_is_given)
{
    const std::string noBounds =
        "Polynomial_degree = 2\nGrid_size = 8\ndelta_t = 0.1\nt_final = 1.0\n"
        "TransportSystem = \"X\"\n";

    BOOST_CHECK_THROW(load(noBounds), std::invalid_argument);

    auto c = load(noBounds + "Grid_points = [0.0, 0.5, 1.0]\n");
    BOOST_TEST(c.Grid_points.size() == 3u);
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
        "SteadyStateSolver = \"Newton\"\n"
        "PseudoTransientInitialStep = 0.25\n"
        "PseudoTransientMaxStep = 1e6\n"
        "PseudoTransientSERRate = 0.5\n"
        "PseudoTransientSERFloor = 1.5\n"
        "SteadyStateDiagnostics = true\n"
        "zeroFlux = true\n"
        "WriteOutput = false\n"
        "SteadyStateTolerance = 1e-5\n"
        "OutputFilename = \"shared\"\n";

    auto v = toml::parse_str(body);
    TomlConfigSource toml_src(v, "/tmp/ignored.conf");
    auto fromToml = loadSolverConfig(toml_src, ConfigSchema::Reader::Toml);

    // Each entry must hold exactly the alternative the schema declares --
    // MapConfigSource returns the variant as stored rather than converting, so
    // `8` and `8u` are not interchangeable here.
    MapConfigSource map_src;
    map_src.values = {
        {"Polynomial_degree", 2u}, {"Grid_size", 8}, {"delta_t", 0.1},
        {"t_final", 1.0}, {"Lower_boundary", 0.0}, {"Upper_boundary", 1.0},
        {"tau", 2.5}, {"Relative_tolerance", 1e-6},
        {"Absolute_tolerance", std::vector<double>{1e-7, 1e-8}},
        {"t_initial", 0.25}, {"OutputPoints", 51},
        {"Superconvergent", true}, {"AggressiveTimesteps", true},
        {"SuppressAlgebraicError", true},
        {"SteadyStateSolver", std::string("Newton")},
        {"PseudoTransientInitialStep", 0.25}, {"PseudoTransientMaxStep", 1e6},
        {"PseudoTransientSERRate", 0.5}, {"PseudoTransientSERFloor", 1.5},
        {"SteadyStateDiagnostics", true},
        {"zeroFlux", true}, {"WriteOutput", false},
        {"SteadyStateTolerance", 1e-5}, {"OutputFilename", std::string("shared")},
    };
    auto fromMap = loadSolverConfig(map_src, ConfigSchema::Reader::Dict);

    BOOST_TEST(fromToml.Polynomial_degree == fromMap.Polynomial_degree);
    BOOST_TEST(fromToml.Grid_size == fromMap.Grid_size);
    BOOST_TEST(fromToml.delta_t == fromMap.delta_t);
    BOOST_TEST(fromToml.tau == fromMap.tau);
    BOOST_TEST(fromToml.t_initial == fromMap.t_initial);
    BOOST_TEST(fromToml.Relative_tolerance == fromMap.Relative_tolerance);
    BOOST_TEST(fromToml.Absolute_tolerance == fromMap.Absolute_tolerance,
               boost::test_tools::per_element());
    BOOST_TEST(fromToml.OutputPoints == fromMap.OutputPoints);
    BOOST_TEST(fromToml.OutputFilename == fromMap.OutputFilename);
    BOOST_TEST(fromToml.Superconvergent == fromMap.Superconvergent);
    BOOST_TEST(fromToml.AggressiveTimesteps == fromMap.AggressiveTimesteps);
    BOOST_TEST(fromToml.SuppressAlgebraicError == fromMap.SuppressAlgebraicError);
    BOOST_TEST(fromToml.SteadyStateSolver == fromMap.SteadyStateSolver);
    BOOST_TEST(fromToml.PseudoTransientInitialStep == fromMap.PseudoTransientInitialStep);
    BOOST_TEST(fromToml.PseudoTransientMaxStep == fromMap.PseudoTransientMaxStep);
    BOOST_TEST(fromToml.PseudoTransientSERRate == fromMap.PseudoTransientSERRate);
    BOOST_TEST(fromToml.PseudoTransientSERFloor == fromMap.PseudoTransientSERFloor);
    BOOST_TEST(fromToml.SteadyStateDiagnostics == fromMap.SteadyStateDiagnostics);
    BOOST_TEST(fromToml.zeroFlux == fromMap.zeroFlux);
    BOOST_TEST(fromToml.WriteOutput == fromMap.WriteOutput);
    BOOST_TEST(fromToml.MinStepSize == fromMap.MinStepSize);
    BOOST_TEST(fromToml.initialTimestep == fromMap.initialTimestep);
    BOOST_TEST(fromToml.WriteDatFile == fromMap.WriteDatFile);
    BOOST_TEST(fromToml.WriteDebugDatFiles == fromMap.WriteDebugDatFiles);
    BOOST_TEST(fromToml.Lower_boundary == fromMap.Lower_boundary);
    BOOST_TEST(fromToml.Upper_boundary == fromMap.Upper_boundary);
    BOOST_TEST(fromToml.restart == fromMap.restart);
    BOOST_TEST(fromToml.solveAdjoint == fromMap.solveAdjoint);
    BOOST_REQUIRE(fromToml.SteadyStateTolerance.has_value());
    BOOST_REQUIRE(fromMap.SteadyStateTolerance.has_value());
    BOOST_TEST(*fromToml.SteadyStateTolerance == *fromMap.SteadyStateTolerance);
    BOOST_REQUIRE(fromToml.t_final.has_value());
    BOOST_REQUIRE(fromMap.t_final.has_value());
    BOOST_TEST(*fromToml.t_final == *fromMap.t_final);
}

// ------------------------------------------- geometric mesh grading ----

BOOST_AUTO_TEST_CASE(graded_grid_defaults_are_off_and_harmless)
{
    auto c = load(minimal);
    BOOST_TEST(c.Graded_Grid_Boundary == false);
    BOOST_TEST(c.Grading_Ratio == 0.3);
    BOOST_TEST(c.Grading_Cells == 0);      // 0 means "half of Grid_size"
    BOOST_TEST(c.Grading_End == "Lower");

    // Off, so the count is left as the sentinel rather than resolved -- which is
    // the property that keeps a plain config bit for bit what it was.
    unsigned int k = 0;
    auto grid = makeGrid(c, nullptr, k);
    BOOST_TEST(grid->getNCells() == 8u);
    for (Grid::Index i = 0; i < grid->getNCells(); ++i)
        BOOST_TEST((*grid)[i].h() == 0.125, boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(a_graded_grid_config_builds_the_mesh_it_describes)
{
    // The end-to-end path: keys -> SolverConfig -> makeGrid -> Grid. The layer
    // width comes from Lower_Boundary_Fraction, which High_Grid_Boundary also
    // reads -- one key for one meaning rather than two that would drift.
    auto c = load(minimal +
                  "Graded_Grid_Boundary = true\n"
                  "Grading_Ratio = 0.5\n"
                  "Grading_Cells = 4\n"
                  "Lower_Boundary_Fraction = 0.2\n");
    BOOST_TEST(c.Graded_Grid_Boundary == true);
    BOOST_TEST(c.Grading_Cells == 4);

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
    // Which fraction is read depends on Grading_End, and getting that backwards
    // would still produce a graded mesh -- of the wrong layer width, silently.
    // Distinct fractions here so the two cannot be confused.
    auto c = load(minimal +
                  "Graded_Grid_Boundary = true\n"
                  "Grading_End = \"Upper\"\n"
                  "Grading_Ratio = 0.5\n"
                  "Grading_Cells = 4\n"
                  "Lower_Boundary_Fraction = 0.4\n"
                  "Upper_Boundary_Fraction = 0.2\n");

    unsigned int k = 0;
    auto grid = makeGrid(c, nullptr, k);
    BOOST_TEST(grid->getNCells() == 8u);

    // The narrow cell is the last one, and its width is set by 0.2 not 0.4.
    BOOST_TEST((*grid)[7].h() == 0.2 * 0.125, boost::test_tools::tolerance(1e-10));
    BOOST_TEST((*grid)[0].h() == 0.8 / 4.0, boost::test_tools::tolerance(1e-12));
}

BOOST_AUTO_TEST_CASE(the_graded_cell_count_defaults_to_half_the_grid)
{
    // Resolved in loadSolverConfig rather than in the schema, because a schema
    // default cannot see another key.
    auto c = load(minimal + "Graded_Grid_Boundary = true\n");
    BOOST_TEST(c.Grading_Cells == 4);      // Grid_size is 8

    unsigned int k = 0;
    auto grid = makeGrid(c, nullptr, k);
    BOOST_TEST(grid->getNCells() == 8u);
}

BOOST_AUTO_TEST_CASE(the_two_mesh_shaping_flags_are_mutually_exclusive)
{
    // Both reshape the same boundary list from different rules, so a silent
    // precedence would give a mesh nobody asked for with nothing to say so.
    BOOST_CHECK_THROW(load(minimal +
                           "Graded_Grid_Boundary = true\n"
                           "High_Grid_Boundary = true\n"),
                      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(a_graded_grid_config_refuses_geometry_it_cannot_build)
{
    BOOST_CHECK_THROW(load(minimal +
                           "Graded_Grid_Boundary = true\n"
                           "Grading_End = \"Both\"\n"),
                      std::invalid_argument);

    // Fewer than three cells cannot carry two graded plus one outside.
    BOOST_CHECK_THROW(load("Polynomial_degree = 2\nGrid_size = 2\ndelta_t = 0.1\n"
                           "t_final = 1.0\nLower_boundary = 0.0\nUpper_boundary = 1.0\n"
                           "TransportSystem = \"LinearDiffusion\"\n"
                           "Graded_Grid_Boundary = true\n"),
                      std::invalid_argument);

    // The rest is gradedMeshPoints's own validation, reached through makeGrid --
    // checked here so the config path is known to surface it rather than to
    // swallow it.
    auto c = load(minimal + "Graded_Grid_Boundary = true\nGrading_Ratio = 1.5\n");
    unsigned int k = 0;
    BOOST_CHECK_THROW(makeGrid(c, nullptr, k), std::invalid_argument);

    auto c2 = load(minimal + "Graded_Grid_Boundary = true\nGrading_Cells = 8\n");
    BOOST_CHECK_THROW(makeGrid(c2, nullptr, k), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
