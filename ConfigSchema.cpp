#include "ConfigSchema.hpp"

#include <algorithm>
#include <cctype>

namespace ConfigSchema
{
namespace
{

const std::vector<Entry> &table()
{
    static const std::vector<Entry> t = {
        {"restart", {}, Type::Bool, Category::Solver, false, false, false,
         "Resume from a restart file instead of building an initial condition."},
        {"RestartFile", {}, Type::String, Category::Solver, false, false, std::string{},
         "Restart file to resume from; defaults to <stem>.restart.nc."},
        {"High_Grid_Boundary", {}, Type::Bool, Category::Solver, false, false, false,
         "Concentrate cells near both ends of the domain, on a cosine spacing."},
        {"Lower_Boundary_Fraction", {}, Type::Double, Category::Solver, false, false, 0.2,
         "Fraction of the domain in the dense lower region; read by High_Grid_Boundary, and "
         "by Graded_Grid_Boundary when Grading_End is \"Lower\"."},
        {"Upper_Boundary_Fraction", {}, Type::Double, Category::Solver, false, false, 0.2,
         "Fraction of the domain in the dense upper region; read by High_Grid_Boundary, and "
         "by Graded_Grid_Boundary when Grading_End is \"Upper\"."},
        {"Graded_Grid_Boundary", {}, Type::Bool, Category::Solver, false, false, false,
         "Grade the mesh geometrically into one end of the domain: Grading_Cells cells "
         "over the dense layer, each Grading_Ratio times the width of its outward "
         "neighbour, then the rest uniform. For a solution with a singularity at that "
         "end this is worth orders of magnitude at a fixed cell count -- see "
         "docs/configuration.rst. Mutually exclusive with High_Grid_Boundary."},
        {"Grading_Ratio", {}, Type::Double, Category::Solver, false, false, 0.3,
         "Width ratio between neighbouring cells in the graded layer, strictly between "
         "0 and 1. Smaller grades harder: the cell touching the graded end has width "
         "fraction * span * ratio^(Grading_Cells - 1), which is what sets the error."},
        {"Grading_Cells", {}, Type::Int, Category::Solver, false, false, 0,
         "Cells in the graded layer; at least 2, and below Grid_size so something is "
         "left outside it. 0 means half of Grid_size."},
        {"Grading_End", {}, Type::String, Category::Solver, false, false, std::string{"Lower"},
         "Which end Graded_Grid_Boundary refines into: \"Lower\" (default) or \"Upper\". "
         "For both ends at once use High_Grid_Boundary or give Grid_points outright."},
        {"Polynomial_degree", {}, Type::UInt, Category::Solver, true, true, 1u,
         "Degree k of the nodal basis in each cell."},
        {"Grid_size", {}, Type::Int, Category::Solver, true, true, 0,
         "Number of cells."},
        {"Grid_points", {}, Type::DoubleList, Category::Solver, false, false, std::vector<double>{},
         "Explicit cell boundaries; supersedes Lower_boundary/Upper_boundary/Grid_size."},
        {"Lower_boundary", {}, Type::Double, Category::Solver, false, false, 0.0,
         "Lower end of the domain; required unless Grid_points is given."},
        {"Upper_boundary", {}, Type::Double, Category::Solver, false, false, 1.0,
         "Upper end of the domain; required unless Grid_points is given."},
        {"tau", {}, Type::Double, Category::Solver, false, false, 1.0,
         "HDG stabilisation parameter."},
        {"delta_t", {}, Type::Double, Category::Solver, true, true, 0.0,
         "Interval between output timeslices."},
        {"t_initial", {"tZero"}, Type::Double, Category::Solver, false, false, 0.0,
         "Time the integration starts from."},
        {"t_final", {}, Type::Double, Category::Solver, true, false, 0.0,
         "Time the integration ends at; Runner.run(tFinal) overrides it."},
        {"Relative_tolerance", {}, Type::Double, Category::Solver, false, false, 1e-3,
         "IDA relative error tolerance."},
        {"Absolute_tolerance", {}, Type::DoubleList, Category::Solver, false, false,
         std::vector<double>{1e-3},
         "IDA absolute error tolerance; one value, or one per variable."},
        {"MinStepSize", {}, Type::Double, Category::Solver, false, false, 1e-7,
         "Smallest timestep IDA may take before giving up."},
        {"initialTimestep", {}, Type::Double, Category::Solver, false, false, 0.0,
         "First timestep to attempt; zero lets IDA choose."},
        {"OutputPoints", {}, Type::Int, Category::Solver, false, false, 301,
         "Number of spatial points written to the output files."},
        {"OutputFilename", {}, Type::String, Category::Solver, false, true, std::string{},
         "Base name for output files; defaults to the config file's stem."},
        {"solveAdjoint", {}, Type::Bool, Category::Solver, false, false, false,
         "Build the adjoint problem and solve for dG/dp after the integration."},
        {"SteadyStateTolerance", {}, Type::Double, Category::Solver, false, false, 1e-3,
         "Stop once the solution stops changing by this much; presence arms it. "
         "Compared against dy/dt under TimeMarch, and against a mesh-independent "
         "weighted norm of the steady residual otherwise."},
        {"SteadyStateSolver", {}, Type::String, Category::Solver, false, false, std::string{"PseudoTransient"},
         "How a steady state is reached: PseudoTransient (default), TimeMarch (integrate to it), "
         "or Newton (pseudo-transient with an infinite first step). See docs/running.rst."},
        {"PseudoTransientInitialStep", {}, Type::Double, Category::Solver, false, false, 0.0,
         "First pseudo-time step for SteadyStateSolver = PseudoTransient; 0 means use delta_t."},
        {"PseudoTransientMaxStep", {}, Type::Double, Category::Solver, false, false, 0.0,
         "Cap on the pseudo-time step; 0 means uncapped."},
        {"PseudoTransientSERRate", {}, Type::Double, Category::Solver, false, false, 1.0,
         "Exponent on the residual ratio in the SER schedule: dt *= max((|F_prev|/|F_now|)^rate, floor). "
         "0 grows at the floor alone, 1 is plain SER. Must not be negative."},
        {"PseudoTransientSERFloor", {}, Type::Double, Category::Solver, false, false, 2.0,
         "Least the pseudo-time step may grow on a step that reduced the residual. "
         "1 means no floor. Must not be below 1."},
        {"SteadyStateDiagnostics", {}, Type::Bool, Category::Solver, false, false, false,
         "Report the work a steady solve did: continuation steps, Newton iterations, "
         "residual evaluations, Jacobian builds and solves. Printed on failure too."},
        {"ObjectiveDecreaseTolerance", {}, Type::Double, Category::Solver, false, false, 0.0,
         "Abandon a run whose dG/dt is already below -this at t0; zero is off."},
        {"WriteOutput", {}, Type::Bool, Category::Solver, false, false, true,
         "Write <stem>.nc and <stem>.restart.nc."},
        {"WriteDatFile", {}, Type::Bool, Category::Solver, false, false, false,
         "Also write the plain-text gnuplot output <stem>.dat."},
        {"WriteDebugDatFiles", {}, Type::Bool, Category::Solver, false, false, false,
         "Also write <stem>.dydt.dat and <stem>.res.dat; needs a PHYSICS_DEBUG build."},
        {"Superconvergent", {}, Type::Bool, Category::Solver, false, false, false,
         "Use the superconvergent interpolatory scheme; needs k >= 1."},
        {"zeroFlux", {}, Type::Bool, Category::Solver, false, false, false,
         "Apply a Neumann boundary value to sigma rather than to q. Equivalent to "
         "declaring those ends Mixed with d = 1; Dirichlet ends are unaffected."},
        {"AggressiveTimesteps", {"aggressiveTimesteps"}, Type::Bool, Category::Solver, false, false, false,
         "Let IDA grow the step by 10x rather than 2x between steps."},
        {"SuppressAlgebraicError", {}, Type::Bool, Category::Solver, false, false, false,
         "Drop sigma, q, lambda and phi from IDA's local error test (IDASetSuppressAlg). "
         "Costs restart fidelity and aux-variable accuracy; see docs/running.rst."},
        {"TransportSystem", {}, Type::String, Category::ProblemSelection, true, false, std::string{},
         "Name of the registered physics case to run."},
        {"PhysicsPlugins", {}, Type::StringList, Category::ProblemSelection, false, false,
         std::vector<std::string>{},
         "Shared objects to dlopen for their physics-case registrations."},
        {"PythonModule", {}, Type::String, Category::Cli, false, false, std::string{},
         "Read by the manta command: module to import for its registrations."},
        {"PythonModuleFile", {}, Type::String, Category::Cli, false, false, std::string{},
         "Read by the manta command: module file, resolved beside the config."},
        {"PythonModuleName", {}, Type::String, Category::Cli, false, false, std::string{},
         "Read by the manta command: name to register PythonModuleFile under."},
    };
    return t;
}

// Levenshtein, case-insensitive. Small inputs, so the simple O(nm) table is
// fine and clearer than the banded version.
std::size_t editDistance(std::string_view a, std::string_view b)
{
    auto lower = [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); };
    std::vector<std::size_t> prev(b.size() + 1), curr(b.size() + 1);
    for (std::size_t j = 0; j <= b.size(); ++j)
        prev[j] = j;
    for (std::size_t i = 1; i <= a.size(); ++i)
    {
        curr[0] = i;
        for (std::size_t j = 1; j <= b.size(); ++j)
            curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1,
                                prev[j - 1] + (lower(a[i - 1]) == lower(b[j - 1]) ? 0 : 1)});
        std::swap(prev, curr);
    }
    return prev[b.size()];
}

} // namespace

std::span<const Entry> schema() { return table(); }

const Entry *findEntry(std::string_view key)
{
    for (auto const &e : table())
    {
        if (e.name == key)
            return &e;
        for (auto const &a : e.aliases)
            if (a == key)
                return &e;
    }
    return nullptr;
}

bool isRequired(Entry const &e, Reader r)
{
    return r == Reader::Toml ? e.requiredToml : e.requiredDict;
}

std::string_view nearestKey(std::string_view key)
{
    std::string_view best;
    std::size_t bestDistance = std::string_view::npos;
    for (auto const &e : table())
    {
        auto d = editDistance(key, e.name);

        // Half the longer of the two names, and no more: past that the "did you
        // mean" is noise, and pointing at an unrelated option is worse than
        // saying nothing. The threshold has to scale with the *candidate* as
        // well as with what was written -- `Poly_degree` for `Polynomial_degree`
        // is six edits and is still obviously the key that was meant, while
        // sixteen characters of nothing in particular is sixteen edits from
        // every entry in the table and must suggest none of them. A flat cap
        // cannot separate those two, because the first number is larger.
        const std::size_t limit = std::max(key.size(), e.name.size()) / 2;

        if (d <= limit && d < bestDistance)
        {
            bestDistance = d;
            best = e.name;
        }
    }
    return best;
}

const char *typeName(Type t)
{
    switch (t)
    {
    // The article is part of the name: the callers splice this straight into
    // "Configuration key 'X' must be ...", and picking a/an at the call site
    // got "a integer" wrong.
    case Type::Bool:       return "a boolean";
    case Type::Int:        return "an integer";
    case Type::UInt:       return "a non-negative integer";
    case Type::Double:     return "a number";
    case Type::String:     return "a string";
    case Type::DoubleList: return "a number, or an array of numbers";
    case Type::StringList: return "an array of strings";
    }
    return "of an unrecognised type";
}

} // namespace ConfigSchema
