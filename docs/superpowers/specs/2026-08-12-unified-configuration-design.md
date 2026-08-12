# One configuration schema for the TOML file and `Runner.configure`

Date: 2026-08-12
Status: approved, ready for an implementation plan

## Why

MaNTA has two configuration surfaces that mean the same thing and are written
twice. `runManta` (`MaNTA.cpp`) open-codes ~120 lines of `toml::find_or` and
hand-rolled type checks; `PyRunner::configure` (`PyRunner.cpp`) has a
declarative `params` table and another ~120 lines reading a `py::dict` against
it. Neither knows about the other, and `docs/configuration.rst` documents both
plus a hand-maintained table of how they differ.

They have drifted, in three distinct ways:

* **The same concept under two names** — `t_initial` / `tZero`,
  `AggressiveTimesteps` / `aggressiveTimesteps`.
* **The same key with two defaults** — `Absolute_tolerance` is `1e-2` on the
  TOML side and `1e-3` on the Python side.
* **Keys on one side only** — `Grid_points`, `initialTimestep`, `zeroFlux` and
  `WriteOutput` are Python-only; `t_final` and `TransportSystem` are TOML-only.

Two of those are silent: nothing reports them, and a configuration that looks
the same through either surface is not the same run.

The damage is already visible in the documentation. `docs/configuration.rst`
lists `WriteOutput` as "supported, default `true`" on the Python side — it is
read into an unused local (`PyRunner.cpp:294`) and does nothing on either
surface, while nine test call sites pass `WriteOutput: False` and go on writing
the files they believe they suppressed.

## Decisions taken

Four choices were settled before designing:

1. **Full parity through one schema.** Every solver option is declared once and
   consumed by both readers, so a key exists on both surfaces or on neither.
2. **Canonical name plus deprecated alias.** Losing spellings keep working and
   log a warning naming their replacement.
3. **Unknown keys are rejected**, with the nearest schema entry suggested.
4. **`WriteOutput` is wired up** on both surfaces rather than deleted.

## 1. The schema

New, pybind11-free: `ConfigSchema.hpp` / `ConfigSchema.cpp`.

```cpp
enum class Type { Bool, Int, UInt, Double, String, DoubleList, StringList };

// Which readers accept a key. A schema entry is not necessarily a solver
// option: TransportSystem selects the physics case and has no dict equivalent,
// and the PythonModule keys are read by manta.cli rather than by the solver at
// all. Recording that here is what lets unknown-key rejection be strict without
// firing on keys that are legitimately present.
enum class Category { Solver, ProblemSelection, Cli };

// Which reader is loading. Distinct from Category: one says what a key *is*,
// the other says who is asking. They were one name in an earlier draft of this
// design, which read fine until the two appeared in the same signature.
enum class Reader { Toml, Dict };

struct Entry {
    std::string_view                 name;        // canonical spelling
    std::vector<std::string_view>    aliases;     // deprecated, warn on use
    Type                             type;
    Category                         category;
    bool                             requiredToml;
    bool                             requiredDict;
    Value                            _default;    // std::variant over Type
    std::string_view                 doc;         // one line, feeds --list-options
};

std::span<const Entry> schema();
const Entry           *findEntry(std::string_view key);   // canonical or alias
std::string_view       nearestKey(std::string_view key);  // edit distance, for "did you mean"
```

`requiredToml` and `requiredDict` are separate because one key genuinely differs:
`TransportSystem` is required in a config file and meaningless in a dict, where
the physics object is passed to `configure` directly. Everything else has the
same value in both.

### The table

| Key | Alias | Type | Required | Default | Category |
|---|---|---|---|---|---|
| `restart` | | Bool | | `false` | Solver |
| `RestartFile` | | String | | `""` → `<stem>.restart.nc` | Solver |
| `High_Grid_Boundary` | | Bool | | `false` | Solver |
| `Lower_Boundary_Fraction` | | Double | | `0.2` | Solver |
| `Upper_Boundary_Fraction` | | Double | | `0.2` | Solver |
| `Polynomial_degree` | | UInt | both | | Solver |
| `Grid_size` | | Int | both | | Solver |
| `Grid_points` | | DoubleList | | `{}` | Solver |
| `Lower_boundary` | | Double | conditional | `0.0` | Solver |
| `Upper_boundary` | | Double | conditional | `1.0` | Solver |
| `tau` | | Double | | `1.0` | Solver |
| `delta_t` | | Double | both | | Solver |
| `t_initial` | `tZero` | Double | | `0.0` | Solver |
| `t_final` | | Double | | *(unset)* | Solver |
| `Relative_tolerance` | | Double | | `1e-3` | Solver |
| `Absolute_tolerance` | | DoubleList | | `{1e-3}` | Solver |
| `MinStepSize` | | Double | | `1e-7` | Solver |
| `initialTimestep` | | Double | | `0.0` | Solver |
| `OutputPoints` | | Int | | `301` | Solver |
| `OutputFilename` | | String | | *(source fallback)* | Solver |
| `solveAdjoint` | | Bool | | `false` | Solver |
| `SteadyStateTolerance` | | Double | | `1e-3`, *presence arms* | Solver |
| `ObjectiveDecreaseTolerance` | | Double | | `0.0` (off) | Solver |
| `WriteOutput` | | Bool | | `true` | Solver |
| `WriteDatFile` | | Bool | | `false` | Solver |
| `WriteDebugDatFiles` | | Bool | | `false` | Solver |
| `Superconvergent` | | Bool | | `false` | Solver |
| `zeroFlux` | | Bool | | `false` | Solver |
| `AggressiveTimesteps` | `aggressiveTimesteps` | Bool | | `false` | Solver |
| `TransportSystem` | | String | TOML only | | ProblemSelection |
| `PhysicsPlugins` | | StringList | | `{}` | ProblemSelection |
| `PythonModule` | | String | | `""` | Cli |
| `PythonModuleFile` | | String | | `""` | Cli |
| `PythonModuleName` | | String | | `""` | Cli |

"conditional" is `Lower_boundary` / `Upper_boundary`: required unless
`Grid_points` is supplied or the run is a restart. The up-front required check
cannot express that, so `validate()` (section 2) carries it as an explicit rule
— which is what `PyRunner.cpp:196-203` already does by hand, and what `MaNTA.cpp`
does not do at all because it has no `Grid_points` branch to need it.

**Naming style is deliberately not unified.** The key set mixes `snake_case`,
`PascalCase` and `camelCase` — `delta_t`, `MinStepSize`, `solveAdjoint` — and
regularising it would churn all 68 config files in the tree for no functional
gain. Only the two genuine name *conflicts* are resolved.

## 2. Two sources, one applier

```
ConfigSchema
     |
     +-- TomlConfigSource(toml::value)   -- Config.cpp,      links everywhere
     +-- DictConfigSource(py::dict)      -- PyConfigSource.hpp, python-only TU
                    |
                    v
       loadSolverConfig(ConfigSource const&, Reader) -> SolverConfig
                    |
                    +--> makeGrid(SolverConfig const&, netCDF::NcFile *restart, unsigned &k)
                    +--> applySolverConfig(SolverConfig const&, SystemSolver &)
```

`ConfigSource` is a three-method interface — `contains(key)`, `get(key, Type)`
returning a `Value`, and `keys()` for the unknown-key sweep. Everything else is
shared.

`SolverConfig` is a plain struct of typed fields. Three of them are
`std::optional`, because **presence** rather than value carries the meaning:

* `t_final` — `runManta` errors when unset; `PyRunner::run()` uses it and
  `run(tFinal)` overrides it.
* `SteadyStateTolerance` — present arms steady-state termination, which is what
  `MaNTA.cpp:281` does today. `run_ss()` arms it regardless, which is what
  `PyRunner` does today. One key, both behaviours.
* `OutputFilename` — the *source* supplies the fallback. `TomlConfigSource`
  seeds it from the config file's stem; `DictConfigSource` has none, so an
  absent key is an error there, exactly as `.required = true` gives today.

`loadSolverConfig` does, in order: sweep the source's keys and reject any not in
the schema (naming the nearest entry); resolve aliases, warning on each; check
required-for-this-surface, reporting all missing keys in one message; read and
type-check each present key; apply defaults for the rest; run `validate()` for
the conditional rules.

`applySolverConfig` is then the only place `system->set*` is called for
config-derived state. It also removes the duplicated `setNOutput` /
`setMinStepSize` pair at `PyRunner.cpp:270-274`.

### The pybind11 boundary

`ConfigSchema`, `SolverConfig`, `loadSolverConfig`, `makeGrid` and
`applySolverConfig` must not include pybind11: they are linked into `MaNTA`,
`libmanta.so` and the unit tests. Only `DictConfigSource` is python-side, and it
lives in a header included by `PyRunner.cpp` alone. `TomlConfigSource` is fine
everywhere — `Config.cpp` already includes `toml.hpp`.

New files go in `SOURCES` in the top-level `Makefile` and in
`REQUIRED_OBJECTS` in `Tests/UnitTests/Makefile`.

## 3. Conflicts, and how each is resolved

| Conflict | Resolution | Evidence |
|---|---|---|
| `t_initial` / `tZero` | `t_initial` canonical | The documented spelling; `tZero` has one call site in the tree and appears in no `.conf` |
| `AggressiveTimesteps` / `aggressiveTimesteps` | `AggressiveTimesteps` canonical | The documented spelling; the camelCase form has zero uses anywhere |
| `Absolute_tolerance` default | `1e-3` | All 68 config files set it explicitly, so the TOML default is unreachable in practice; the Python side does rely on its default. No in-tree behaviour change |
| `OutputFilename` | One key; TOML falls back to the config stem | No `.conf` in the tree sets it, so making it work changes nothing underfoot — and it is currently read by nothing on the TOML side, which `CLAUDE.md` records as a trap |
| `t_final` | Optional in the schema; `run(tFinal)` still overrides | A driver legitimately runs one configuration to many end times |
| `SteadyStateTolerance` | `std::optional`; presence arms | Preserves both current meanings without a second key |
| `WriteOutput` | Real `SystemSolver` flag gating `<stem>.nc` and `<stem>.restart.nc` | Nine call sites already ask for it |

`Grid_points`, `initialTimestep` and `zeroFlux` become TOML keys as well;
`docs/configuration.rst` already documents them.

### The boundary fractions are *not* a divergence

`MaNTA.cpp:82-84` forces `Lower_Boundary_Fraction` and
`Upper_Boundary_Fraction` to `0.0` when `High_Grid_Boundary` is absent, while
`PyRunner` always defaults them to `0.2`. This looks like a defect and is not:
`Grid`'s constructor ignores both fractions entirely when `highGridBoundary` is
false (`gridStructures.hpp:81`), so the two readers build identical grids. The
schema's single `0.2` is therefore behaviour-preserving for both. Worth stating
because it is the kind of difference a reader will "fix" twice.

## 4. `WriteOutput`, wired

`SystemSolver` gains a `WriteOutput` flag, defaulting true, with
`setWriteOutput`. It gates the two writes in `Solver.cpp`:

* `initialiseNetCDF(baseName + ".nc", nOut)` at `Solver.cpp:379`, and the
  per-step writes that follow it
* `WriteRestartFile(baseName + ".restart.nc", Y, dYdt, nOut)` at
  `Solver.cpp:514`

`WriteDatFile` and `WriteDebugDatFiles` keep their own separate gates — they are
opt-in already, and nesting them under `WriteOutput` would change what a config
setting only `WriteDatFile` does.

**Each of the nine call sites must be checked individually.** A test that passes
`WriteOutput: False` and then reads the output file will start failing, and that
is a real defect being surfaced rather than caused — but it has to be looked at
before the change lands, not after.

## 5. What stays asymmetric

`TransportSystem` and `PhysicsPlugins` have no dict equivalent: `PyRunner` is
handed the physics object. Passing either to `configure()` is an error naming
the reason, rather than being silently ignored.

**Risk, stated:** a driver that reads a `.conf` and forwards its whole
`[configuration]` table into `configure()` breaks on that error. No such driver
exists in this tree, and the message names the fix, but it is the one place
strictness could reach an out-of-tree user. The `Cli` keys are exempt for
exactly this reason — they appear in eight `.conf` files in the tree and are
read by `manta.cli`, not by the solver.

## 6. Testing

The load-bearing test is **cross-source equivalence**: a table of configurations
expressed once, fed through `TomlConfigSource` and `DictConfigSource`, with the
resulting `SolverConfig` compared field by field. That is what makes divergence
impossible rather than merely absent, and it is the test that would have caught
every drift listed at the top of this document.

Around it, in a new `Tests/UnitTests/ConfigSchemaTests.cpp`:

* an unknown key is an error naming the nearest schema entry
* an alias works and warns, and the warning names the canonical spelling
* several missing required keys are reported in one message, not the first
* a wrong type names the key, the type given and the type wanted
* every default in the schema is reachable and has the type it declares
* a `ProblemSelection` key in a dict is an error explaining why
* a `Cli` key is accepted by both without complaint
* the conditional `Lower_boundary` / `Upper_boundary` rule fires only when
  `Grid_points` is absent and the run is not a restart

Plus, in the Python suite: `WriteOutput = false` leaves no `.nc` behind.

**The regression suite must be bit-identical.** No config in the tree uses a key
whose default or meaning changes, so any movement in `Tests/RegressionTests` is a
defect in this change, not an expected consequence of it.

## 7. Documentation

* `docs/configuration.rst` — the `_config-divergences` section (lines 211-266)
  documents a divergence that will no longer exist. It is replaced by a short
  note that both surfaces read one schema, plus the alias table and the two
  remaining asymmetries from section 5. The `WriteOutput` and `zeroFlux` rows
  claiming "not read" on the TOML side become ordinary option rows.
* `manta --list-options` dumps the schema — name, type, default, doc — so the
  hand-written option table can be regenerated by eye rather than from memory.
* `CLAUDE.md` — the trap "output filenames come from the config file's stem
  regardless of any path in `OutputFilename`" stops being true and is rewritten;
  the Python-layer section gains the schema and its pybind11 boundary.

## Out of scope

* Regularising the naming *style* of the key set. Only the two genuine conflicts
  are resolved.
* Generating `docs/configuration.rst` from the schema. `--list-options` gives
  the raw material; wiring it into the docs build is a separate change.
* The per-physics-case config tables (`[MirrorPlasma]`, `[DiffusionProblem]`,
  …). Only `[configuration]` is shared between the two surfaces; a case reads
  its own section through its own code, and nothing in this change touches that.
* `Config.hpp`'s `getFloat` / `getFloatWithDefault` / `getIntWithDefault`. They
  are used by `MaNTA.cpp` and by `Tests/UnitTests/ConfigTests.cpp`; once
  `MaNTA.cpp` reads through the schema they have one caller left, but deleting
  them is a separate tidy-up with its own test to update.

## Risks

* **Strictness reaching an out-of-tree config.** Rejection is the point, but it
  is the one irreversible-feeling part. Mitigated by aliases (no spelling
  breaks), by the `Cli` exemption, and by the survey showing all 68 in-tree
  configs use only known keys.
* **`WriteOutput` becoming real changes what nine tests do.** Enumerated above
  as a per-call-site check rather than a bulk edit.
* **`applySolverConfig` is a single point of failure by design.** A block
  dropped from it silently un-configures the solver for *both* surfaces at once,
  where today the same slip would affect one. The cross-source test does not
  catch that — it compares `SolverConfig`, not the solver. A test asserting each
  setter is reached is the guard, and belongs with the applier.
