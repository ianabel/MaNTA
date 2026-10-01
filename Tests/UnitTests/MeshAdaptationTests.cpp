// Deciding whether to grade a mesh, and the p -> h -> p driver.
//
// Three layers, and the first two are where the content is:
//
//   * gradingDecision -- a pure function of a DGSoln. Driven by *assigning* known
//     functions rather than by solving, so a verdict can be checked against a
//     function whose smoothness is known by construction.
//   * gradedMeshFor -- the mesh a verdict asks for, at the same cell count.
//   * the config rules, which refuse the combinations that cannot mean anything.
//
// The end-to-end sequence is exercised by python/Tests/test_mesh_adaptation.py,
// where a physics case with a closed-form solution is available.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "MeshAdaptation.hpp"
#include "SolverConfig.hpp"
#include "SystemSolver.hpp"
#include "TestDiffusion.hpp"
#include "gridStructures.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
const toml::value mesh_config = toml::parse_str(
    "[DiffusionProblem]\nKappa = 1.0\nCentre = 0.5\n");

// A DGSoln over memory we own, with u set to a given function. The sensor reads
// nodal values, and for this basis those *are* the coefficients, so assigning a
// function is enough -- no solve, and therefore no dependence on a physics case
// having the smoothness we want to test.
struct Field
{
    // Declaration order is the lifetime order: `soln` holds a Grid reference and a
    // pointer into `memory`, so both must be declared -- and so constructed --
    // before it, and destroyed after.
    Grid grid;
    std::vector<double> memory;
    DGSoln soln;

    Field(Grid g, Index k, std::function<double(double)> f)
        // Not `soln(1, grid, k, 0, 0)`: a literal 0 is both an Index and a null
        // double*, so that is ambiguous against the mapping constructor. The
        // defaults are what we want anyway.
        : grid(std::move(g)), soln(1, grid, k)
    {
        memory.assign(soln.getDoF(), 0.0);
        soln.Map(memory.data());
        soln.AssignU([&](Index, Position x) { return f(x); });
    }
};

double sk(double x) { return std::pow(std::abs(x), 4.0 / 3.0); }
} // namespace

BOOST_AUTO_TEST_SUITE(mesh_adaptation_tests, *boost::unit_test::tolerance(1e-9))

// ------------------------------------------------- the grading decision ----

BOOST_AUTO_TEST_CASE(a_singularity_at_the_lower_end_asks_for_the_lower_end)
{
    // x^{4/3} at the axis: the sensor's own coverage measures its decay rate
    // rising monotonically away from cell 0, so the *decision* should name Lower.
    Field f(Grid(0.0, 1.0, 10), 4, sk);
    auto d = gradingDecision(f.soln, 0, 2.0);

    BOOST_TEST_MESSAGE("lower " << d.lowerRate << ", median " << d.interiorMedian
                       << ", upper " << d.upperRate << "; ratios "
                       << d.lowerRatio << " / " << d.upperRatio);

    BOOST_TEST((d.verdict == GradingVerdict::GradeLower));
    BOOST_TEST(d.lowerRate < d.interiorMedian);
    BOOST_TEST(d.lowerRatio > d.upperRatio);
}

BOOST_AUTO_TEST_CASE(a_singularity_at_the_upper_end_asks_for_the_upper_end)
{
    // The mirror, which is the half that a rule keying off the wrong index would
    // still pass the case above. Reflected function, reflected verdict.
    Field f(Grid(0.0, 1.0, 10), 4, [](double x) { return sk(1.0 - x); });
    auto d = gradingDecision(f.soln, 0, 2.0);

    BOOST_TEST((d.verdict == GradingVerdict::GradeUpper));
    BOOST_TEST(d.upperRate < d.interiorMedian);
    BOOST_TEST(d.upperRatio > d.lowerRatio);
}

BOOST_AUTO_TEST_CASE(a_smooth_function_is_left_uniform)
{
    // The false-positive test, and the one MESH-REFINEMENT.md section 7 never ran:
    // a rule that grades a smooth problem is useless however well it localises a
    // singular one. exp(x) is entire, so no end is special.
    Field f(Grid(0.0, 1.0, 10), 4, [](double x) { return std::exp(x); });
    auto d = gradingDecision(f.soln, 0, 2.0);

    BOOST_TEST_MESSAGE("smooth ratios: lower " << d.lowerRatio << ", upper "
                       << d.upperRatio);
    BOOST_TEST((d.verdict == GradingVerdict::Uniform));
    BOOST_TEST(d.lowerRatio < 2.0);
    BOOST_TEST(d.upperRatio < 2.0);
}

BOOST_AUTO_TEST_CASE(a_function_the_space_holds_exactly_is_left_uniform)
{
    // Jardin's case: a steady state the space represents outright. Every cell's
    // spectrum is at round-off above the top retained mode, so every decay rate is
    // infinite and there is nothing to compare. The rule must read that as
    // "smooth", not divide two infinities and grade on the result.
    Field f(Grid(0.0, 1.0, 10), 4, [](double x) { return 3.0 - 2.0 * x; });
    auto d = gradingDecision(f.soln, 0, 2.0);

    BOOST_TEST_MESSAGE("exactly-representable field: lower " << d.lowerRate
                       << ", median " << d.interiorMedian << ", upper "
                       << d.upperRate << "; ratios " << d.lowerRatio << " / "
                       << d.upperRatio);

    BOOST_TEST((d.verdict == GradingVerdict::Uniform));

    // Asserted as "no end stands out", not as "every rate is infinite". The
    // stronger claim was written here first and is wrong: the modal transform of
    // an exactly-linear field does not put the higher coefficients at the
    // round-off floor exactly, so the rates come out large and finite rather than
    // infinite, and what makes the verdict right is that they are *all* large.
    BOOST_TEST(d.lowerRatio < 2.0);
    BOOST_TEST(d.upperRatio < 2.0);
}

BOOST_AUTO_TEST_CASE(the_verdict_is_stable_across_the_degrees_it_is_allowed_at)
{
    // The reason MeshAdaptation refuses k < 3 is that the verdict *changes* below
    // it. Above it the verdict must not: a rule that flipped between k = 3 and
    // k = 5 would make the driver's answer depend on its starting degree.
    for (Index k : {3u, 4u, 5u, 6u})
    {
        BOOST_TEST_CONTEXT("k = " << k)
        {
            Field rough(Grid(0.0, 1.0, 10), k, sk);
            Field smooth(Grid(0.0, 1.0, 10), k, [](double x) { return std::exp(x); });

            auto r = gradingDecision(rough.soln, 0, 2.0);
            auto s = gradingDecision(smooth.soln, 0, 2.0);

            BOOST_TEST_MESSAGE("rough " << r.lowerRatio << " vs smooth "
                               << s.lowerRatio);
            BOOST_TEST((r.verdict == GradingVerdict::GradeLower));
            BOOST_TEST((s.verdict == GradingVerdict::Uniform));

            // ...and separated, not merely on opposite sides of 2.0 by a hair.
            BOOST_TEST(r.lowerRatio > 1.5 * s.lowerRatio);
        }
    }
}

BOOST_AUTO_TEST_CASE(the_threshold_is_what_decides_and_is_checked)
{
    Field f(Grid(0.0, 1.0, 10), 4, sk);

    // A threshold above the measured roughness leaves the mesh alone, which is
    // what makes this a tunable rather than a hardcoded rule.
    auto d = gradingDecision(f.soln, 0, 2.0);
    BOOST_TEST((gradingDecision(f.soln, 0, 100.0).verdict == GradingVerdict::Uniform));
    BOOST_TEST((d.verdict == GradingVerdict::GradeLower));

    // At or below 1 every mesh grades, including one whose ends are its smoothest
    // cells, so it is refused rather than accepted as an aggressive setting.
    BOOST_CHECK_THROW(gradingDecision(f.soln, 0, 1.0), std::invalid_argument);
    BOOST_CHECK_THROW(gradingDecision(f.soln, 0, 0.5), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(deciding_needs_an_interior_to_compare_against)
{
    // Two cells are two ends and no interior, so there is no baseline. Refused
    // rather than compared against one of the two cells being judged.
    Field two(Grid(0.0, 1.0, 2), 4, sk);
    BOOST_CHECK_THROW(gradingDecision(two.soln, 0, 2.0), std::invalid_argument);

    Field three(Grid(0.0, 1.0, 3), 4, sk);
    BOOST_CHECK_NO_THROW(gradingDecision(three.soln, 0, 2.0));
}

// ------------------------------------------------- the mesh it asks for ----

BOOST_AUTO_TEST_CASE(the_graded_mesh_keeps_the_cell_count_and_the_domain)
{
    // The whole point of section 9: this is a redistribution, so the budget must
    // not move. A version that refined would pass every other assertion here.
    Grid uniform(-2.0, 3.0, 12);
    Field f(Grid(-2.0, 3.0, 12), 4, [](double x) { return sk(x + 2.0); });
    auto d = gradingDecision(f.soln, 0, 2.0);
    BOOST_TEST((d.verdict == GradingVerdict::GradeLower));

    auto points = gradedMeshFor(d, uniform, 0, 0.2, 0.2, 0.3);
    Grid graded(points);

    BOOST_TEST(graded.getNCells() == uniform.getNCells());
    BOOST_TEST(graded.lowerBoundary() == -2.0);
    BOOST_TEST(graded.upperBoundary() == 3.0);

    // gradedCells = 0 means "as many as the budget allows" here -- 11 of 12 --
    // which is *not* what it means on the manual path. h0 follows from that.
    BOOST_TEST(graded[0].h() == 0.2 * 5.0 * std::pow(0.3, 10.0),
               boost::test_tools::tolerance(1e-9));
    BOOST_TEST(graded[0].h() < graded[11].h());
}

BOOST_AUTO_TEST_CASE(the_graded_mesh_follows_the_end_the_decision_named)
{
    Grid uniform(0.0, 1.0, 10);
    Field lower(Grid(0.0, 1.0, 10), 4, sk);
    Field upper(Grid(0.0, 1.0, 10), 4, [](double x) { return sk(1.0 - x); });

    Grid lo(gradedMeshFor(gradingDecision(lower.soln, 0, 2.0), uniform, 0, 0.2, 0.2, 0.3));
    Grid hi(gradedMeshFor(gradingDecision(upper.soln, 0, 2.0), uniform, 0, 0.2, 0.2, 0.3));

    // Narrow at opposite ends, and mirror images of each other.
    BOOST_TEST(lo[0].h() < lo[9].h());
    BOOST_TEST(hi[9].h() < hi[0].h());
    BOOST_TEST(lo[0].h() == hi[9].h(), boost::test_tools::tolerance(1e-9));
}

BOOST_AUTO_TEST_CASE(a_uniform_verdict_cannot_be_asked_for_a_mesh)
{
    // A logic error rather than a silent uniform mesh: the caller has branched
    // wrongly, and returning something plausible would hide it.
    Grid uniform(0.0, 1.0, 10);
    Field f(Grid(0.0, 1.0, 10), 4, [](double x) { return std::exp(x); });
    auto d = gradingDecision(f.soln, 0, 2.0);
    BOOST_TEST((d.verdict == GradingVerdict::Uniform));
    BOOST_CHECK_THROW(gradedMeshFor(d, uniform, 0, 0.2, 0.2, 0.3), std::logic_error);
}

// --------------------------------------------------- the configuration ----

namespace
{
// `degree` is a parameter rather than something `extra` can override, because
// toml11 rejects a duplicated key outright rather than letting the later one win
// -- so appending "PolynomialDegree = 2" to a body that already sets it raises a
// toml syntax error, not the invalid_argument the test is looking for.
SolverConfig loadMesh(std::string const &extra, unsigned degree = 4)
{
    const std::string body =
        std::format("PolynomialDegree = {}\n", degree) +
        "GridSize = 10\ndelta_t = 0.1\nt_final = 1.0\n"
        "LowerBoundary = 0.0\nUpperBoundary = 1.0\n"
        "TransportSystem = \"LinearDiffusion\"\nSteadyStateSolve = true\n"
        "MeshAdaptation = true\n" + extra;
    auto v = toml::parse_str(body);
    TomlConfigSource src(v);
    return loadSolverConfig(src, ConfigSchema::Reader::Toml);
}
} // namespace

BOOST_AUTO_TEST_CASE(mesh_adaptation_implies_the_degree_loop_and_superconvergence)
{
    // MeshAdaptation *is* p -> h -> p, and the last p is the degree loop, so it
    // turns that on rather than making the user ask for both -- and inherits its
    // Superconvergent requirement rather than repeating it.
    auto c = loadMesh("");
    BOOST_TEST(c.MeshAdaptation == true);
    BOOST_TEST(c.DegreeAdaptation == true);
    BOOST_TEST(c.Superconvergent.has_value());
    BOOST_TEST(*c.Superconvergent == true);
    BOOST_TEST(c.MeshAdaptationThreshold == 2.0);
    BOOST_TEST(c.MeshAdaptationAttempts == 4u);
}

BOOST_AUTO_TEST_CASE(mesh_adaptation_refuses_a_degree_it_cannot_decide_at)
{
    // The load-bearing refusal. At k = 2 the verdict is *inverted*, not merely
    // uncertain, so proceeding would confidently grade the wrong problem.
    BOOST_CHECK_THROW(loadMesh("", 2), std::invalid_argument);
    BOOST_CHECK_THROW(loadMesh("", 1), std::invalid_argument);
    BOOST_CHECK_NO_THROW(loadMesh("", 3));
}

BOOST_AUTO_TEST_CASE(mesh_adaptation_refuses_deciding_the_mesh_twice)
{
    // Both of these already determine the mesh, so one of them would silently lose.
    BOOST_CHECK_THROW(loadMesh("GradedGridBoundary = true\n"), std::invalid_argument);
    BOOST_CHECK_THROW(loadMesh("GridPoints = [0.0, 0.5, 1.0]\n"), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(mesh_adaptation_checks_its_own_bounds)
{
    BOOST_CHECK_THROW(loadMesh("MeshAdaptationThreshold = 1.0\n"), std::invalid_argument);
    BOOST_CHECK_THROW(loadMesh("MeshAdaptationThreshold = 0.0\n"), std::invalid_argument);
    BOOST_CHECK_THROW(loadMesh("MeshAdaptationAttempts = 0\n"), std::invalid_argument);
    BOOST_CHECK_NO_THROW(loadMesh("MeshAdaptationThreshold = 1.5\n"));
}

BOOST_AUTO_TEST_CASE(mesh_adaptation_needs_a_steady_solve)
{
    // Inherited from the degree loop's rule, and it matters for the same reason:
    // each stage would take the previous stage's final state as its initial
    // condition and integrate the interval again.
    const std::string body =
        "PolynomialDegree = 4\nGridSize = 10\ndelta_t = 0.1\nt_final = 1.0\n"
        "LowerBoundary = 0.0\nUpperBoundary = 1.0\n"
        "TransportSystem = \"LinearDiffusion\"\nMeshAdaptation = true\n";
    auto v = toml::parse_str(body);
    TomlConfigSource src(v);
    BOOST_CHECK_THROW(loadSolverConfig(src, ConfigSchema::Reader::Toml),
                      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(the_driver_refuses_a_low_degree_even_if_the_config_did_not)
{
    // Belt and braces, and not redundant: runAdaptiveMesh is reachable directly
    // from C++ without going through loadSolverConfig at all, which is how the
    // unit tests and any embedding caller reach it.
    Grid grid(0.0, 1.0, 10);
    TestDiffusion problem(mesh_config);
    SolverConfig cfg{};
    BOOST_CHECK_THROW(runAdaptiveMesh(cfg, problem, nullptr, grid, 2, 1.0),
                      std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
