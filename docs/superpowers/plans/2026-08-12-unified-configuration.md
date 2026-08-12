# Unified Configuration Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two hand-written configuration readers — `runManta`'s open-coded TOML parsing and `PyRunner::configure`'s separate `params` table — with one declarative schema consumed by two thin sources and a single applier, so an option cannot exist on one surface and not the other.

**Architecture:** A pybind11-free `ConfigSchema` table declares every key once: canonical name, deprecated aliases, type, per-reader requiredness, default, category and doc. `TomlConfigSource` and `DictConfigSource` implement a three-method `ConfigSource` interface; `loadSolverConfig` validates against the schema and produces a `SolverConfig` struct; `makeGrid` and `applySolverConfig` are the only consumers of it.

**Tech Stack:** C++23, toml11, pybind11, Boost.Test, netCDF, GNU make.

Spec: `docs/superpowers/specs/2026-08-12-unified-configuration-design.md`.

## Global Constraints

- **`ConfigSchema.hpp`, `SolverConfig.hpp` and their `.cpp` files must not include pybind11.** They link into `MaNTA`, `libmanta.so` and `Tests/UnitTests`. Only `DictConfigSource` is python-side, in a header included by `PyRunner.cpp` alone.
- **`Category` and `Reader` are two different enums.** `Category` says what a key *is* (`Solver`, `ProblemSelection`, `Cli`); `Reader` says who is asking (`Toml`, `Dict`). They were one name in an earlier draft and it read fine until both appeared in one signature.
- **The regression suite must be bit-identical.** No config in the tree uses a key whose default or meaning changes. Any movement in `Tests/RegressionTests` is a defect in this change.
- **Naming style is not unified.** Only the two genuine conflicts — `t_initial`/`tZero` and `AggressiveTimesteps`/`aggressiveTimesteps` — are resolved. `delta_t`, `MinStepSize` and `solveAdjoint` keep their inconsistent styles; regularising them would churn all 68 config files for no functional gain.
- **`Absolute_tolerance` default is `1e-3`** — the Python value. All 68 in-tree configs set it explicitly, so the TOML default is unreachable in practice.
- Canonical spellings: **`t_initial`** (alias `tZero`), **`AggressiveTimesteps`** (alias `aggressiveTimesteps`).
- New `.cpp` files go in `SOURCES` (`Makefile:12`) and `REQUIRED_OBJECTS` (`Tests/UnitTests/Makefile:23`); new test files go in `TEST_SOURCES` (`Tests/UnitTests/Makefile:6`).
- Build and test with the venv on `PATH`: `export PATH="$PWD/.venv/bin:$PATH"`.

---

## File Structure

**Created**

| Path | Responsibility |
|---|---|
| `ConfigSchema.hpp` / `.cpp` | The table. `Entry`, `Type`, `Category`, `Reader`, `Value`; `schema()`, `findEntry()`, `nearestKey()`. No I/O, no toml, no pybind. |
| `SolverConfig.hpp` / `.cpp` | `SolverConfig` struct; `ConfigSource` interface; `TomlConfigSource`; `loadSolverConfig()`; `validate()`; `makeGrid()`; `applySolverConfig()`. |
| `PyConfigSource.hpp` | `DictConfigSource` only. Included by `PyRunner.cpp`. |
| `Tests/UnitTests/ConfigSchemaTests.cpp` | Schema lookup, alias resolution, unknown-key suggestions, requiredness, defaults. |
| `Tests/UnitTests/ConfigSourceTests.cpp` | `TomlConfigSource` + `loadSolverConfig` behaviour, and the conditional-requirement rules. |

**Modified**

| Path | Change |
|---|---|
| `MaNTA.cpp:29-260` | The open-coded reader is replaced by `loadSolverConfig` + `makeGrid` + `applySolverConfig`. |
| `PyRunner.cpp:11-297` | The `params` table and `getValueWithDefault` are deleted; `configure` uses the same three calls. |
| `SystemSolver.hpp` | `writeOutput` member + `setWriteOutput`. |
| `Solver.cpp` | Gate the netCDF and restart writes on it. |
| `python/Tests/*.py` (9 files) | `WriteOutput: False` starts working — check each. |
| `Makefile`, `Tests/UnitTests/Makefile` | New sources. |
| `docs/configuration.rst`, `CLAUDE.md` | The divergence section goes; the `OutputFilename` trap stops being true. |

---

### Task 1: The schema table

**Files:**
- Create: `ConfigSchema.hpp`, `ConfigSchema.cpp`, `Tests/UnitTests/ConfigSchemaTests.cpp`
- Modify: `Makefile:12` (`SOURCES`), `Tests/UnitTests/Makefile:6` (`TEST_SOURCES`), `Tests/UnitTests/Makefile:23` (`REQUIRED_OBJECTS`)

**Interfaces:**
- Consumes: nothing.
- Produces: `ConfigSchema::Type`, `Category`, `Reader`, `Value`, `Entry`; `std::span<const Entry> schema()`; `const Entry *findEntry(std::string_view)`; `std::string_view nearestKey(std::string_view)`; `bool isRequired(Entry const&, Reader)`. Tasks 2-4 use all of these.

- [ ] **Step 1: Write the failing test**

Create `Tests/UnitTests/ConfigSchemaTests.cpp`:

```cpp
// The configuration schema: one declaration per option, shared by the TOML
// reader and the dict reader.
//
// These tests are about the table itself -- lookup, aliases, requiredness,
// defaults -- not about reading a config. ConfigSourceTests.cpp covers that.

#include <boost/test/unit_test.hpp>

#include "ConfigSchema.hpp"

#include <set>
#include <string>

using namespace ConfigSchema;

BOOST_AUTO_TEST_SUITE(config_schema_tests)

BOOST_AUTO_TEST_CASE(every_canonical_name_is_unique)
{
    std::set<std::string_view> seen;
    for (auto const &e : schema())
        BOOST_TEST(seen.insert(e.name).second,
                   "duplicate schema entry: " << std::string(e.name));
}

BOOST_AUTO_TEST_CASE(no_alias_collides_with_a_name_or_another_alias)
{
    std::set<std::string_view> seen;
    for (auto const &e : schema())
        seen.insert(e.name);
    for (auto const &e : schema())
        for (auto const &a : e.aliases)
            BOOST_TEST(seen.insert(a).second,
                       "alias collides: " << std::string(a));
}

BOOST_AUTO_TEST_CASE(find_entry_resolves_canonical_names)
{
    auto const *e = findEntry("Polynomial_degree");
    BOOST_REQUIRE(e != nullptr);
    BOOST_TEST(e->type == Type::UInt);
    BOOST_TEST(isRequired(*e, Reader::Toml));
    BOOST_TEST(isRequired(*e, Reader::Dict));
}

BOOST_AUTO_TEST_CASE(find_entry_resolves_the_deprecated_aliases)
{
    // The two genuine name conflicts between the old readers. Both old
    // spellings must keep working; both must resolve to the canonical entry.
    BOOST_REQUIRE(findEntry("tZero") != nullptr);
    BOOST_TEST(findEntry("tZero")->name == "t_initial");

    BOOST_REQUIRE(findEntry("aggressiveTimesteps") != nullptr);
    BOOST_TEST(findEntry("aggressiveTimesteps")->name == "AggressiveTimesteps");
}

BOOST_AUTO_TEST_CASE(find_entry_returns_null_for_an_unknown_key)
{
    BOOST_TEST(findEntry("Superconvergnet") == nullptr);
}

BOOST_AUTO_TEST_CASE(nearest_key_suggests_the_obvious_typo)
{
    BOOST_TEST(nearestKey("Superconvergnet") == "Superconvergent");
    BOOST_TEST(nearestKey("Poly_degree") == "Polynomial_degree");
    BOOST_TEST(nearestKey("delta_T") == "delta_t");
}

BOOST_AUTO_TEST_CASE(nearest_key_gives_up_on_something_unrelated)
{
    // A suggestion that is nothing like the input is worse than none: it sends
    // the reader off to check a key they never wrote.
    BOOST_TEST(nearestKey("qqqqqqqqqqqqqqqq").empty());
}

BOOST_AUTO_TEST_CASE(transport_system_is_required_of_toml_only)
{
    // The one key whose requiredness genuinely differs by reader: PyRunner is
    // handed the physics object, so a dict has nothing to name.
    auto const *e = findEntry("TransportSystem");
    BOOST_REQUIRE(e != nullptr);
    BOOST_TEST(e->category == Category::ProblemSelection);
    BOOST_TEST(isRequired(*e, Reader::Toml));
    BOOST_TEST(!isRequired(*e, Reader::Dict));
}

BOOST_AUTO_TEST_CASE(the_cli_keys_are_recognised_but_are_not_solver_options)
{
    // manta.cli reads these; the solver never does. They are in the schema so
    // that unknown-key rejection does not fire on the eight .conf files that
    // carry them.
    for (auto const *k : {"PythonModule", "PythonModuleFile", "PythonModuleName"})
    {
        auto const *e = findEntry(k);
        BOOST_REQUIRE_MESSAGE(e != nullptr, "missing schema entry: " << k);
        BOOST_TEST(e->category == Category::Cli);
    }
}

BOOST_AUTO_TEST_CASE(every_default_matches_its_declared_type)
{
    for (auto const &e : schema())
    {
        switch (e.type)
        {
        case Type::Bool:   BOOST_TEST(std::holds_alternative<bool>(e._default),        std::string(e.name)); break;
        case Type::Int:    BOOST_TEST(std::holds_alternative<int>(e._default),         std::string(e.name)); break;
        case Type::UInt:   BOOST_TEST(std::holds_alternative<unsigned>(e._default),    std::string(e.name)); break;
        case Type::Double: BOOST_TEST(std::holds_alternative<double>(e._default),      std::string(e.name)); break;
        case Type::String: BOOST_TEST(std::holds_alternative<std::string>(e._default), std::string(e.name)); break;
        case Type::DoubleList:
            BOOST_TEST(std::holds_alternative<std::vector<double>>(e._default), std::string(e.name)); break;
        case Type::StringList:
            BOOST_TEST(std::holds_alternative<std::vector<std::string>>(e._default), std::string(e.name)); break;
        }
    }
}

BOOST_AUTO_TEST_CASE(every_entry_has_a_doc_line)
{
    // The doc string is what `manta --list-options` prints and what the
    // configuration.rst table is written from. An entry without one is an
    // option nobody can find out about.
    for (auto const &e : schema())
        BOOST_TEST(!e.doc.empty(), "no doc for " << std::string(e.name));
}

BOOST_AUTO_TEST_SUITE_END()
```

- [ ] **Step 2: Add the files to the build**

In `Makefile:12`, append `ConfigSchema.cpp` to `SOURCES`:

```make
SOURCES = Config.cpp ConfigSchema.cpp SystemSolver.cpp SunLinSolWrapper.cpp SunMatrixWrapper.cpp ErrorChecker.cpp Solver.cpp Matrices.cpp DGStatic.cpp PhysicsCases.cpp NetCDFIO.cpp AdjointVectors.cpp Postprocessing.cpp
```

In `Tests/UnitTests/Makefile:6`, append `ConfigSchemaTests.cpp` to `TEST_SOURCES`, and in `:23` append `../../ConfigSchema.o` to `REQUIRED_OBJECTS`.

- [ ] **Step 3: Run the test to verify it fails**

```sh
export PATH="$PWD/.venv/bin:$PATH"
make test 2>&1 | tail -20
```

Expected: a compile error, `ConfigSchema.hpp: No such file or directory`.

- [ ] **Step 4: Write `ConfigSchema.hpp`**

```cpp
#ifndef CONFIGSCHEMA_HPP
#define CONFIGSCHEMA_HPP

// The single declaration of every key MaNTA accepts in a [configuration] table
// or in a Runner.configure dict.
//
// There used to be two: runManta open-coded its TOML reading in MaNTA.cpp and
// PyRunner.cpp carried a separate `params` table. They drifted -- two names for
// the initial time, two defaults for Absolute_tolerance, four keys that existed
// on one side only -- and nothing reported it. One table means a key is on both
// surfaces or on neither.
//
// This header must not include pybind11: it is linked into the standalone
// solver, into libmanta.so and into the unit tests.

#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ConfigSchema
{

enum class Type { Bool, Int, UInt, Double, String, DoubleList, StringList };

// What a key *is*. Not every schema entry is a solver option: TransportSystem
// selects the physics case and has no dict equivalent, and the PythonModule
// keys are read by manta.cli rather than by the solver at all. Recording that
// here is what lets unknown-key rejection be strict without firing on keys
// that are legitimately present in a config file.
enum class Category { Solver, ProblemSelection, Cli };

// Who is asking. Distinct from Category -- one says what a key is, the other
// says which reader is loading it. Only requiredness differs between them, and
// only for TransportSystem.
enum class Reader { Toml, Dict };

using Value = std::variant<bool, int, unsigned, double, std::string,
                           std::vector<double>, std::vector<std::string>>;

struct Entry
{
    std::string_view              name;          // canonical spelling
    std::vector<std::string_view> aliases;       // deprecated; accepted with a warning
    Type                          type;
    Category                      category;
    bool                          requiredToml;
    bool                          requiredDict;
    Value                         _default;
    std::string_view              doc;           // one line; printed by --list-options
};

std::span<const Entry> schema();

// Canonical name or any alias; nullptr if the key is not in the schema.
const Entry *findEntry(std::string_view key);

bool isRequired(Entry const &e, Reader r);

// The closest canonical name to `key` by edit distance, for a did-you-mean.
// Empty when nothing is close enough to be worth suggesting -- a suggestion
// that is nothing like the input sends the reader to check a key they never
// wrote.
std::string_view nearestKey(std::string_view key);

const char *typeName(Type t);

} // namespace ConfigSchema

#endif // CONFIGSCHEMA_HPP
```

- [ ] **Step 5: Write `ConfigSchema.cpp`**

The table, verbatim from the spec. Note `t_final` and `SteadyStateTolerance`
carry ordinary defaults here; it is `SolverConfig` that records whether they
were *present*, which is what arms the corresponding behaviour.

```cpp
#include "ConfigSchema.hpp"

#include <algorithm>
#include <array>

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
         "Concentrate cells near both ends of the domain."},
        {"Lower_Boundary_Fraction", {}, Type::Double, Category::Solver, false, false, 0.2,
         "Fraction of the domain in the dense lower region; ignored unless High_Grid_Boundary."},
        {"Upper_Boundary_Fraction", {}, Type::Double, Category::Solver, false, false, 0.2,
         "Fraction of the domain in the dense upper region; ignored unless High_Grid_Boundary."},
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
        {"t_final", {}, Type::Double, Category::Solver, false, false, 0.0,
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
        {"OutputFilename", {}, Type::String, Category::Solver, false, false, std::string{},
         "Base name for output files; defaults to the config file's stem."},
        {"solveAdjoint", {}, Type::Bool, Category::Solver, false, false, false,
         "Build the adjoint problem and solve for dG/dp after the integration."},
        {"SteadyStateTolerance", {}, Type::Double, Category::Solver, false, false, 1e-3,
         "Stop once the solution stops changing by this much; presence arms it."},
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
         "Impose zero flux at both boundaries."},
        {"AggressiveTimesteps", {"aggressiveTimesteps"}, Type::Bool, Category::Solver, false, false, false,
         "Let IDA grow the step by 10x rather than 2x between steps."},
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
        if (d < bestDistance)
        {
            bestDistance = d;
            best = e.name;
        }
    }

    // A third of the key's length, and never more than four edits: past that
    // the "did you mean" is noise, and pointing at an unrelated option is worse
    // than saying nothing.
    const std::size_t limit = std::min<std::size_t>(4, std::max<std::size_t>(1, key.size() / 3));
    return bestDistance <= limit ? best : std::string_view{};
}

const char *typeName(Type t)
{
    switch (t)
    {
    case Type::Bool:       return "boolean";
    case Type::Int:        return "integer";
    case Type::UInt:       return "non-negative integer";
    case Type::Double:     return "number";
    case Type::String:     return "string";
    case Type::DoubleList: return "number, or array of numbers";
    case Type::StringList: return "array of strings";
    }
    return "unknown";
}

} // namespace ConfigSchema
```

Add `#include <cctype>` if the compiler asks for `std::tolower`.

- [ ] **Step 6: Run the tests to verify they pass**

```sh
make test 2>&1 | tail -6
Tests/UnitTests/UnitTests --run_test=config_schema_tests --log_level=message
```

Expected: no errors. If `nearest_key_gives_up_on_something_unrelated` fails,
the limit in `nearestKey` is too generous — do not loosen the test to match.

- [ ] **Step 7: Commit**

```bash
git add ConfigSchema.hpp ConfigSchema.cpp Tests/UnitTests/ConfigSchemaTests.cpp Makefile Tests/UnitTests/Makefile
git commit -m "Declare every configuration key once, in a schema table"
```

---

### Task 2: `SolverConfig` and the TOML source

**Files:**
- Create: `SolverConfig.hpp`, `SolverConfig.cpp`, `Tests/UnitTests/ConfigSourceTests.cpp`
- Modify: `Makefile:12`, `Tests/UnitTests/Makefile:6,23`

**Interfaces:**
- Consumes: everything from Task 1.
- Produces: `struct SolverConfig` (fields named exactly as the canonical keys, lower-cased first letter kept as-is); `class ConfigSource` with `bool contains(std::string_view) const`, `ConfigSchema::Value get(std::string_view, ConfigSchema::Type) const`, `std::vector<std::string> keys() const`; `class TomlConfigSource : public ConfigSource`; `SolverConfig loadSolverConfig(ConfigSource const&, ConfigSchema::Reader)`. Tasks 3 and 4 call `loadSolverConfig`.

- [ ] **Step 1: Write the failing test**

Create `Tests/UnitTests/ConfigSourceTests.cpp`:

```cpp
// Reading a [configuration] table against the schema.
//
// The rules under test here are the ones that used to be open-coded twice and
// differently: what is required, what an absent key defaults to, what happens
// to a key nobody recognises, and the two conditional requirements that a flat
// required-list cannot express.

#include <boost/test/unit_test.hpp>

#include "SolverConfig.hpp"

#include <map>
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

BOOST_AUTO_TEST_SUITE_END()
```

- [ ] **Step 2: Add to the build and run to verify failure**

Append `SolverConfig.cpp` to `SOURCES` (`Makefile:12`), `ConfigSourceTests.cpp`
to `TEST_SOURCES` and `../../SolverConfig.o` to `REQUIRED_OBJECTS`.

```sh
make test 2>&1 | tail -20
```

Expected: `SolverConfig.hpp: No such file or directory`.

- [ ] **Step 3: Write `SolverConfig.hpp`**

```cpp
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
    // fallback", which makes the key effectively required for that source.
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
```

- [ ] **Step 4: Write `SolverConfig.cpp` — the source and the loader**

`makeGrid` and `applySolverConfig` come in Task 3; write them as declared-but-
unimplemented only if the linker demands it, otherwise leave them out until then.

```cpp
#include "SolverConfig.hpp"

#include "Logging.hpp"

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
                                     "' must be a " + wanted + ".");
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
    throw bad("recognised type");
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

    if (c.OutputFilename.empty())
        throw std::invalid_argument(
            "Missing required configuration key: OutputFilename -- there is no "
            "config file to take a name from.");

    return c;
}
```

- [ ] **Step 5: Run the tests**

```sh
make test 2>&1 | tail -6
Tests/UnitTests/UnitTests --run_test=config_source_tests --log_level=message
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add SolverConfig.hpp SolverConfig.cpp Tests/UnitTests/ConfigSourceTests.cpp Makefile Tests/UnitTests/Makefile
git commit -m "Read a configuration against the schema, from any source"
```

---

### Task 3: `makeGrid`, `applySolverConfig`, and `runManta` on top of them

**Files:**
- Modify: `SolverConfig.cpp` (add the two functions), `SolverConfig.hpp` (already declares them), `MaNTA.cpp:29-260`

**Interfaces:**
- Consumes: `loadSolverConfig` from Task 2.
- Produces: `makeGrid(SolverConfig const&, netCDF::NcFile*, unsigned&)` and `applySolverConfig(SolverConfig const&, SystemSolver&)`, both used unchanged by Task 4.

- [ ] **Step 1: Write `makeGrid` and `applySolverConfig` in `SolverConfig.cpp`**

```cpp
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
    // and the grids came out identical either way.
    return std::make_unique<Grid>(config.Lower_boundary, config.Upper_boundary,
                                  config.Grid_size, config.High_Grid_Boundary,
                                  config.Lower_Boundary_Fraction,
                                  config.Upper_Boundary_Fraction);
}

void applySolverConfig(SolverConfig const &config, SystemSolver &system)
{
    // The only place a configuration reaches the solver. A block dropped from
    // here silently un-configures *both* surfaces at once, which is the price
    // of having one of these instead of two -- see solver_plumbing tests.
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

    // Presence arms it; setSteadyStateTolerance also sets TerminateOnSteadyState.
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
```

Add `#include "SystemSolver.hpp"` and `#include "gridStructures.hpp"` to
`SolverConfig.cpp`. `setWriteOutput` does not exist yet — Task 5 adds it; until
then this will not compile, so **do Task 5's Step 1 first** (it is one line in a
header) and leave the rest of Task 5 for its own commit.

- [ ] **Step 2: Rewrite `runManta` to use them**

Replace `MaNTA.cpp:29-260` — everything from the `toml::parse` to
`system->runSolver(tFinal)` — with:

```cpp
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
			                         std::string(std::filesystem::absolute(fileName)));
			return 1;
		}
	}

	unsigned int k = 1;
	std::unique_ptr<Grid> grid = makeGrid(config, config.restart ? &restart_file : nullptr, k);

	// A case registers itself from a static initialiser, so loading the shared
	// object is all that is needed. Must happen before InstantiateProblem.
	for (auto const &path : config.PhysicsPlugins)
	{
		// RTLD_GLOBAL so a plugin can be linked against another plugin's symbols;
		// RTLD_NOW so an unresolved symbol is reported here rather than at the
		// first call into the case.
		if (dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL) == nullptr)
		{
			logmsg<LOG_LEVEL::ERROR>("Could not load physics plugin {}: {}", path, dlerror());
			return 1;
		}
	}

	std::unique_ptr<TransportSystem> pProblem;
	try
	{
		pProblem = PhysicsCases::InstantiateProblem(config.TransportSystem, configFile, *grid);
	}
	catch (std::invalid_argument const &e)
	{
		logmsg<LOG_LEVEL::ERROR>(
		    "Could not instantiate a physics model for TransportSystem = {}\n  {}",
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
		const Index nCells = grid->getNCells();
		const Index nDOF = pProblem->getNumVars() * 3 * nCells * (k + 1) +
		                   pProblem->getNumVars() * (nCells + 1) +
		                   pProblem->getNumScalars() +
		                   pProblem->getNumAux() * nCells * (k + 1);
		if (nDOF_file != nDOF)
			throw std::invalid_argument(
			    "nVars/nAux/nScalars in restart file inconsistent with physics case");
		pProblem->setRestartValues(Y, dYdt, *grid, k);
	}

	auto system = std::make_shared<SystemSolver>(*grid, k, pProblem.get());
	applySolverConfig(config, *system);
	if (config.solveAdjoint)
		system->setAdjointProblem(adjoint.get());

	system->runSolver(*config.t_final);
```

Add `#include "SolverConfig.hpp"` at the top and drop `#include "Config.hpp"` if
nothing else in the file uses it.

**One behaviour change to be aware of and keep:** `setInputFile` now receives
`config.OutputFilename`, which defaults to the config file's stem — the same
string `Solver.cpp:267` was deriving with `inputFilePath.stem()`. Output names
are therefore unchanged, and an explicit `OutputFilename` now works where it
was previously read by nothing.

- [ ] **Step 3: Check `Solver.cpp` is not double-stemming**

`Solver.cpp:267` and `:405` do `std::string baseName = inputFilePath.stem();`.
`inputFilePath` is now already a stem, and `std::filesystem::path("myrun").stem()`
is `"myrun"`, so this is a no-op — **unless** a user passes an `OutputFilename`
containing a dot, where `"a.b".stem()` is `"a"`. Change both lines to:

```cpp
	std::string baseName = inputFilePath.string();
```

and confirm `setInputFile` stores a path built from the base name only.

- [ ] **Step 4: Build and run every suite**

```sh
export PATH="$PWD/.venv/bin:$PATH"
make MaNTA -j$(nproc)
make test 2>&1 | tail -4
make regression_tests 2>&1 | tail -4
```

Expected: unit tests pass; **regression output bit-identical**. If any
regression case differs, stop — no config in the tree uses a key whose default
or meaning changed, so a difference is a defect in this task.

- [ ] **Step 5: Commit**

```bash
git add SolverConfig.cpp SolverConfig.hpp MaNTA.cpp Solver.cpp
git commit -m "Read the TOML configuration through the shared schema"
```

---

### Task 4: `PyRunner` on the same schema

**Files:**
- Create: `PyConfigSource.hpp`
- Modify: `PyRunner.cpp:11-297` (delete `params` and `getValueWithDefault`; rewrite `configure`), `PyRunner.hpp`
- Test: `python/Tests/test_run_config.py`, and a new cross-source test in `Tests/UnitTests/ConfigSourceTests.cpp`

**Interfaces:**
- Consumes: `loadSolverConfig`, `makeGrid`, `applySolverConfig`.
- Produces: nothing later depends on it.

- [ ] **Step 1: Write `PyConfigSource.hpp`**

```cpp
#ifndef PYCONFIGSOURCE_HPP
#define PYCONFIGSOURCE_HPP

// A ConfigSource over a py::dict.
//
// The only python-aware part of the configuration path, and the reason it is a
// header included by PyRunner.cpp alone: SolverConfig.hpp and ConfigSchema.hpp
// link into the standalone solver and the unit tests, neither of which may see
// pybind11.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "SolverConfig.hpp"

namespace py = pybind11;

class DictConfigSource : public ConfigSource
{
public:
    explicit DictConfigSource(py::dict const &d) : dict(d) {}

    bool contains(std::string_view key) const override
    {
        return dict.contains(std::string(key).c_str());
    }

    std::vector<std::string> keys() const override
    {
        std::vector<std::string> out;
        for (auto const &item : dict)
            out.push_back(py::cast<std::string>(item.first));
        return out;
    }

    ConfigSchema::Value get(std::string_view key, ConfigSchema::Type t) const override
    {
        auto obj = dict[std::string(key).c_str()];
        auto bad = [&] {
            return std::invalid_argument("Configuration key '" + std::string(key) +
                                         "' must be a " + ConfigSchema::typeName(t) + ".");
        };
        try
        {
            switch (t)
            {
            case ConfigSchema::Type::Bool:       return obj.cast<bool>();
            case ConfigSchema::Type::Int:        return obj.cast<int>();
            case ConfigSchema::Type::UInt:       return obj.cast<unsigned>();
            case ConfigSchema::Type::Double:     return obj.cast<double>();
            case ConfigSchema::Type::String:     return obj.cast<std::string>();
            case ConfigSchema::Type::DoubleList:
                // A bare number is accepted where a list is wanted, matching
                // what Absolute_tolerance has always allowed in TOML.
                try { return obj.cast<std::vector<double>>(); }
                catch (py::cast_error const &) { return std::vector<double>{obj.cast<double>()}; }
            case ConfigSchema::Type::StringList: return obj.cast<std::vector<std::string>>();
            }
        }
        catch (py::cast_error const &)
        {
            throw bad();
        }
        throw bad();
    }

    // No config file to take a name from, so OutputFilename stays effectively
    // required here -- which is what `.required = true` gave before.
    std::string outputFilenameFallback() const override { return {}; }

private:
    py::dict const &dict;
};

#endif // PYCONFIGSOURCE_HPP
```

- [ ] **Step 2: Add the cross-source equivalence test**

This is the test the whole change exists for. `MapConfigSource` is already
defined at the top of `Tests/UnitTests/ConfigSourceTests.cpp` from Task 2 —
reuse it. Append:

```cpp
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
        "zeroFlux = true\n"
        "WriteOutput = false\n"
        "SteadyStateTolerance = 1e-5\n"
        "OutputFilename = \"shared\"\n";

    auto v = toml::parse_str(body);
    TomlConfigSource toml_src(v, "/tmp/ignored.conf");
    auto fromToml = loadSolverConfig(toml_src, ConfigSchema::Reader::Toml);

    MapConfigSource map_src;
    map_src.values = {
        {"Polynomial_degree", 2u}, {"Grid_size", 8}, {"delta_t", 0.1},
        {"t_final", 1.0}, {"Lower_boundary", 0.0}, {"Upper_boundary", 1.0},
        {"tau", 2.5}, {"Relative_tolerance", 1e-6},
        {"Absolute_tolerance", std::vector<double>{1e-7, 1e-8}},
        {"t_initial", 0.25}, {"OutputPoints", 51},
        {"Superconvergent", true}, {"AggressiveTimesteps", true},
        {"zeroFlux", true}, {"WriteOutput", false},
        {"SteadyStateTolerance", 1e-5}, {"OutputFilename", std::string("shared")},
    };
    map_src.values.erase("TransportSystem");   // ProblemSelection: not a dict key
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
    BOOST_TEST(fromToml.zeroFlux == fromMap.zeroFlux);
    BOOST_TEST(fromToml.WriteOutput == fromMap.WriteOutput);
    BOOST_TEST(fromToml.MinStepSize == fromMap.MinStepSize);
    BOOST_TEST(fromToml.initialTimestep == fromMap.initialTimestep);
    BOOST_TEST(fromToml.WriteDatFile == fromMap.WriteDatFile);
    BOOST_TEST(fromToml.WriteDebugDatFiles == fromMap.WriteDebugDatFiles);
    BOOST_REQUIRE(fromToml.SteadyStateTolerance.has_value());
    BOOST_REQUIRE(fromMap.SteadyStateTolerance.has_value());
    BOOST_TEST(*fromToml.SteadyStateTolerance == *fromMap.SteadyStateTolerance);
    BOOST_REQUIRE(fromToml.t_final.has_value());
    BOOST_REQUIRE(fromMap.t_final.has_value());
    BOOST_TEST(*fromToml.t_final == *fromMap.t_final);
}
```

- [ ] **Step 3: Run it and watch it pass or fail honestly**

```sh
make test 2>&1 | tail -4
Tests/UnitTests/UnitTests --run_test=config_source_tests/both_sources_produce_the_same_solver_config --log_level=all
```

Expected: PASS. It exercises code written in Task 2, so a failure here is a
defect in the loader, not in this task.

- [ ] **Step 4: Rewrite `PyRunner::configure`**

Delete `PyRunner.cpp:10-98` (the `params` table) and `:100-123`
(`getValueWithDefault`), and replace the body of `configure` with:

```cpp
void PyRunner::configure(const py::dict &config) {
  if (!pProblem)
    throw std::runtime_error("Transport system not set. Please set transport "
                             "system before configuring solver.");
  // Set stored problem to null to allow reconfiguration after object creation
  system = nullptr;
  grid = nullptr;

  DictConfigSource source(config);
  SolverConfig cfg = loadSolverConfig(source, ConfigSchema::Reader::Dict);

  netCDF::NcFile restart_file;
  if (cfg.restart) {
    std::string fileName = cfg.RestartFile.empty()
                               ? cfg.OutputFilename + ".restart.nc"
                               : cfg.RestartFile;
    try {
      restart_file.open(fileName, netCDF::NcFile::FileMode::read);
    } catch (...) {
      throw std::runtime_error(
          "Failed to open restart netCDF file at: " +
          std::string(std::filesystem::absolute(std::filesystem::path(fileName))));
    }
  }

  unsigned int k = 1;
  grid = makeGrid(cfg, cfg.restart ? &restart_file : nullptr, k);

  if (cfg.solveAdjoint)
    adjoint = pProblem->createAdjointProblem();

  if (cfg.restart) {
    std::vector<double> Y, dYdt;
    Index nDOF_file = LoadFromFile(restart_file, Y, dYdt);
    const Index nCells = grid->getNCells();
    const Index nDOF = pProblem->getNumVars() * 3 * nCells * (k + 1) +
                       pProblem->getNumVars() * (nCells + 1) +
                       pProblem->getNumScalars() +
                       pProblem->getNumAux() * nCells * (k + 1);
    if (nDOF_file != nDOF)
      throw std::invalid_argument(
          "nVars/nAux/nScalars in restart file inconsistent with physics case");
    pProblem->setRestartValues(Y, dYdt, *grid, k);
  }

  system = std::make_unique<SystemSolver>(*grid, k, pProblem.get());

  applySolverConfig(cfg, *system);
  if (cfg.solveAdjoint)
    system->setAdjointProblem(adjoint.get());

  // run_ss() arms steady-state termination itself, so it needs the value
  // whether or not the key was present.
  steady_state_tolerance = cfg.SteadyStateTolerance.value_or(1e-3);
  // run() with no argument uses the configured end time.
  configured_t_final = cfg.t_final;

  configured = true;
  logmsg<LOG_LEVEL::INFO>("Configuration done.");
}
```

Add `#include "PyConfigSource.hpp"` to `PyRunner.cpp`, and
`std::optional<double> configured_t_final;` beside `steady_state_tolerance` in
`PyRunner.hpp:95`.

Note `applySolverConfig` calls `setSteadyStateTolerance` when the key was
present, which sets `TerminateOnSteadyState`. `PyRunner::run` already clears
that flag with a warning (`PyRunner.cpp:305-310`), so the existing behaviour is
preserved: configure with the key, call `run`, and you get the warning and a
normal run; call `run_ss` and you get steady-state termination.

- [ ] **Step 5: Add the no-argument `run()` overload**

In `PyRunner.hpp` and `PyRunner.cpp`:

```cpp
void PyRunner::run() {
  if (!configured)
    throw std::runtime_error("Error: Runner must be configured before running solver.");
  if (!configured_t_final)
    throw std::runtime_error(
        "run() with no argument needs t_final in the configuration; "
        "pass run(tFinal) instead.");
  run(*configured_t_final);
}
```

Bind it in `Python.cpp` alongside the existing `run`, as an overload taking no
arguments.

- [ ] **Step 6: Run the Python suite**

```sh
export PATH="$PWD/.venv/bin:$PATH"
make python -j$(nproc)
make python_tests 2>&1 | grep -E '^=+ |FAILED|ERROR' | tail -10
```

Expected: failures only from `WriteOutput: False` now taking effect (Task 5) and
from any test passing a key that is not in the schema. Fix the second kind by
adding the key to the schema if it is real, or correcting the test if it is a
typo — which is the whole point of rejection. Do **not** loosen rejection.

- [ ] **Step 7: Regenerate the stub and typecheck**

```sh
make stubs
make stubs-check
make typecheck
git diff --stat python/manta/_manta.pyi
```

The new `run()` overload changes the generated stub; commit the regeneration.

- [ ] **Step 8: Commit**

```bash
git add PyConfigSource.hpp PyRunner.cpp PyRunner.hpp Python.cpp \
        Tests/UnitTests/ConfigSourceTests.cpp python/manta/_manta.pyi
git commit -m "Read the Runner.configure dict through the same schema"
```

---

### Task 5: `WriteOutput`, wired

**Files:**
- Modify: `SystemSolver.hpp`, `Solver.cpp:379,441,463,504,512,514`, and the nine Python call sites
- Test: `python/Tests/test_runner.py`

**Interfaces:**
- Consumes: `SolverConfig::WriteOutput` from Task 2.
- Produces: `SystemSolver::setWriteOutput(bool)`, called by `applySolverConfig`.

- [ ] **Step 1: Add the flag**

In `SystemSolver.hpp`, beside `setWriteDatFile` (`:207`):

```cpp
        // Gates the netCDF output and the restart file. The .dat files have
        // their own flags below: they are opt-in already, and nesting them
        // under this one would change what a config setting only WriteDatFile
        // does.
        void setWriteOutput(bool in) { writeOutput = in; };
```

and beside the other flags in the private section:

```cpp
        bool writeOutput = true;
```

- [ ] **Step 2: Write the failing Python test**

Append to `python/Tests/test_runner.py`:

The file's existing helpers are `LinearDiffusion` (the case class, at the top)
and `base_config(tmp_path, **overrides)` (which already sets `OutputFilename`
to a path under `tmp_path` and `WriteOutput: False`). Reuse both:

```python
def test_write_output_false_leaves_no_netcdf_behind(tmp_path):
    """WriteOutput: False must actually suppress the output files.

    It was read into an unused local in PyRunner and not read at all by the
    TOML reader, while nine call sites in this suite passed False and went on
    writing the files they believed they had turned off.
    """
    runner = MaNTA.Runner(LinearDiffusion())
    runner.configure(base_config(tmp_path,
                                 OutputFilename=str(tmp_path / "suppressed"),
                                 WriteOutput=False))
    runner.run(0.1)

    assert not (tmp_path / "suppressed.nc").exists()
    assert not (tmp_path / "suppressed.restart.nc").exists()


def test_write_output_true_still_writes(tmp_path, monkeypatch):
    # cwd matters: Solver.cpp writes beside the current directory, and the
    # base name is the stem of OutputFilename.
    monkeypatch.chdir(tmp_path)

    runner = MaNTA.Runner(LinearDiffusion())
    runner.configure(base_config(tmp_path,
                                 OutputFilename="written",
                                 WriteOutput=True))
    runner.run(0.1)

    assert (tmp_path / "written.nc").exists()
    assert (tmp_path / "written.restart.nc").exists()
```

Note `base_config` sets `WriteOutput: False` for **every** test in the file.
Once the flag works, no test in `test_runner.py` writes output unless it
overrides that — which is the point, but it means Step 6's audit covers this
file most heavily.

- [ ] **Step 3: Run it and watch it fail**

```sh
pytest python/Tests/test_runner.py -k write_output -v
```

Expected: `test_write_output_false_leaves_no_netcdf_behind` FAILS — the file is
there. The `True` case passes already.

- [ ] **Step 4: Gate the writes**

In `Solver.cpp`, wrap each of these:

* `:379` — `initialiseNetCDF(baseName + ".nc", nOut);`
* `:441` and `:463` — `WriteTimeslice(tret);`
* `:504` — `problem->finaliseDiagnostics(nc_output);`
* `:512` and the other `nc_output.Close();` calls
* `:514` — `WriteRestartFile(baseName + ".restart.nc", Y, dYdt, nOut);`

as `if (writeOutput) { ... }`. `nc_output.Close()` on a never-opened file is
harmless — `NcFile::close()` checks its null-object flag, which is what the
destructor's `filename != ""` guard already relies on — but guard it anyway so
the reason is visible at each site rather than depending on netCDF's behaviour.

- [ ] **Step 5: Run the tests**

```sh
pytest python/Tests/test_runner.py -k write_output -v
make test 2>&1 | tail -4
```

Expected: both new tests pass; C++ unit tests unaffected.

- [ ] **Step 6: Check each of the nine existing call sites individually**

```sh
grep -rn '"WriteOutput"' python/Tests/
```

They are in `test_adjoint_aux.py:245`, `test_adjoint.py:197`, `test_aux.py:101`,
`test_package_api.py:159`, `test_runner.py:80,156,521`, `test_scalars.py:258`
and `test_trampolines.py:154`. For each, check whether the test **reads an
output file after the run**. If one does, it has been reading a file it asked
not to have; fix it by dropping `WriteOutput: False` from that test rather than
by weakening the flag, and say so in the commit message.

```sh
make python_tests 2>&1 | grep -E '^=+ |FAILED' | tail -10
```

- [ ] **Step 7: Commit**

```bash
git add SystemSolver.hpp Solver.cpp python/Tests
git commit -m "Make WriteOutput actually suppress the output"
```

---

### Task 6: `--list-options`, and the documentation

**Files:**
- Modify: `main.cpp`, `docs/configuration.rst:211-266`, `CLAUDE.md`, `README.md`

**Interfaces:**
- Consumes: `ConfigSchema::schema()`.
- Produces: nothing.

- [ ] **Step 1: Add `--list-options`**

In `main.cpp`, before the config file is opened:

```cpp
	if (argc == 2 && std::string(argv[1]) == "--list-options")
	{
		for (auto const &e : ConfigSchema::schema())
		{
			std::print("{:<28} {:<28} ", std::string(e.name), ConfigSchema::typeName(e.type));
			if (!e.aliases.empty())
				std::print("(was: {}) ", std::string(e.aliases.front()));
			std::println("{}", std::string(e.doc));
		}
		return 0;
	}
```

- [ ] **Step 2: Check the output reads well**

```sh
make MaNTA -j$(nproc) && ./MaNTA --list-options
```

Expected: one line per schema entry, aliases marked. This is the raw material
for the docs table below — read it rather than working from memory.

- [ ] **Step 3: Rewrite the divergence section**

Delete `docs/configuration.rst:211-266` — the `_config-divergences` section
documents a divergence that no longer exists — and replace with:

```rst
.. _config-divergences:

The TOML file and ``Runner.configure``
---------------------------------------

Both read the same schema, declared once in ``ConfigSchema.cpp``. Every option
in the table above has the same name, the same type and the same default on
both, and an unrecognised key is an error on both, with the nearest match
suggested.

Two spellings changed when the schemas were merged. The old ones still work and
warn:

.. list-table::
   :header-rows: 1

   * - Canonical
     - Deprecated
   * - ``t_initial``
     - ``tZero``
   * - ``AggressiveTimesteps``
     - ``aggressiveTimesteps``

Two asymmetries remain, and are deliberate:

* ``TransportSystem`` and ``PhysicsPlugins`` select the physics case. A
  ``Runner`` is handed the object instead, so passing either to ``configure``
  is an error rather than being ignored.
* ``t_final`` is required in a config file. ``Runner.run(tFinal)`` overrides it,
  and ``Runner.run()`` with no argument uses it -- a driver legitimately runs
  one configuration to many end times.

``OutputFilename`` defaults to the config file's stem when read from a file, and
has no default in a dict, where there is no file to take a name from.

Run ``MaNTA --list-options`` for the current table straight from the schema.
```

Also correct the `WriteOutput` and `zeroFlux` rows in the main option table:
both are now ordinary options read by both surfaces.

- [ ] **Step 4: Rewrite the `CLAUDE.md` entries that are no longer true**

The trap

```
* **Output filenames come from the config file's *stem*** (`Solver.cpp` uses
  `inputFilePath.stem()`), so `.nc` / `.dat` / `.restart.nc` land in the current
  directory regardless of any path in `OutputFilename`.
```

becomes a description of the current behaviour: `OutputFilename` is honoured,
defaulting to the config file's stem, and output still lands in the current
directory. Add a short entry recording that both surfaces read one schema, that
`ConfigSchema.hpp`/`SolverConfig.hpp` must stay pybind11-free, and that
`applySolverConfig` is the single point where a configuration reaches the solver
— a block dropped from it un-configures both surfaces at once.

- [ ] **Step 5: Build the docs under `-W`**

```sh
SP=/tmp/claude-1000/-home-ian-projects-MaNTA/*/scratchpad
python3 -m venv $SP/docsvenv && $SP/docsvenv/bin/pip install -q -r docs/requirements.txt
$SP/docsvenv/bin/sphinx-build -W -j auto -b html docs docs/_build/html 2>&1 | tail -5
rm -rf docs/_build
```

Expected: build succeeded.

- [ ] **Step 6: Run everything**

```sh
export PATH="$PWD/.venv/bin:$PATH"
make test && make regression_tests && make python_tests && make typecheck && make stubs-check
```

Expected: all pass, regression bit-identical, `test_jax_aux_test` still the known
xfail.

- [ ] **Step 7: Commit**

```bash
git add main.cpp docs CLAUDE.md README.md
git commit -m "Document the single configuration schema"
```

---

## Verification summary

| Claim | How it is checked |
|---|---|
| Every key declared once, consistently | `ConfigSchemaTests.cpp` — unique names, no alias collisions, defaults match declared types, every entry documented |
| The two surfaces cannot diverge | `both_sources_produce_the_same_solver_config` — two sources, same values, field-by-field comparison |
| Aliases keep old configs working | `a_deprecated_alias_is_accepted`, and a warning naming the canonical spelling |
| Unknown keys are caught | `an_unknown_key_is_an_error_suggesting_the_nearest`; `nearest_key_gives_up_on_something_unrelated` stops the suggestion becoming noise |
| Conditional requirements still hold | `boundaries_are_required_unless_grid_points_is_given` |
| `OutputFilename` works on both | `output_filename_falls_back_to_the_config_stem`, `an_explicit_output_filename_wins_over_the_stem` |
| A key meant for the other surface is rejected | `a_problem_selection_key_is_an_error_for_the_dict_reader`, and its TOML counterpart |
| `WriteOutput` suppresses output | `test_write_output_false_leaves_no_netcdf_behind` |
| Nothing else moved | `make regression_tests` bit-identical; `make test`, `make python_tests`, `make typecheck`, `make stubs-check` |

## Known ordering hazard

Task 3's `applySolverConfig` calls `system.setWriteOutput`, which Task 5 adds.
Do **Task 5 Step 1** (one line in `SystemSolver.hpp`, one member) before Task 3
Step 1, and leave the rest of Task 5 for its own commit. The alternative —
reordering the tasks — would put the nine-call-site audit before the surface it
depends on exists.
