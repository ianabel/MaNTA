# Configuration and output

The configuration schema, its two surfaces, and what a run writes.

## Configuration

**Every key MaNTA accepts is declared once**, in `ConfigSchema.cpp`: canonical
name, deprecated aliases, type, category, per-reader requiredness, default and a
line of documentation. `build/MaNTA --list-options` prints the table. One
declaration is what keeps the config-file and the dict surfaces from drifting
apart in names, defaults or which keys exist.

The path is `ConfigSource` -> `loadSolverConfig` -> `SolverConfig` -> `makeGrid`
and `applySolverConfig`. `ConfigSource` is the only piece that differs between
the surfaces: `TomlConfigSource` (in `SolverConfig.cpp`) and `DictConfigSource`
(in `PyConfigSource.hpp`). What to know before editing any of it:

* **`ConfigSchema.hpp`, `SolverConfig.hpp` and their `.cpp` files must stay
  pybind11-free.** They link into `MaNTA`, `libmanta.so` and `Tests/UnitTests`.
  `DictConfigSource` is the python-side half and lives in a header `PyRunner.cpp`
  alone includes. Neither header is in `INSTALL_HEADERS`, and neither belongs
  there: an out-of-tree case is handed the parsed `toml::value` and reads its own
  table through toml11, never through the solver's schema.
* **`applySolverConfig` is the single point at which a configuration reaches the
  solver.** That is what stops the two surfaces configuring differently, and it
  is also why a `set*` call dropped from it un-configures *both* at once, where
  the same slip used to cost one. `both_sources_produce_the_same_solver_config`
  compares `SolverConfig`s, not solvers, so it would not notice.
* **`Category` and `Reader` are two different enums.** `Category` says what a key
  *is* (`Solver`, `ProblemSelection`, `FieldSelection`, `Cli`); `Reader` says who is asking
  (`Toml`, `Dict`). Requiredness differs by reader for exactly three keys:
  `t_final` and `TransportSystem` are required of a config file only,
  `OutputFilename` of a dict only — a file falls back to its own stem.
* **Presence, not value, is the signal for `t_final` and `SteadyStateTolerance`**,
  which are `std::optional` in `SolverConfig`. A present `SteadyStateTolerance`
  arms steady-state termination on both surfaces; `run_ss()` arms it regardless,
  falling back to `1e-3`. `Runner.run()` with no argument uses `t_final`, and
  `run(tFinal)` overrides it.
* **An unknown key is an error, with the nearest schema entry suggested.** The
  sweep sees `[configuration]` only — a physics case's own table is read by the
  case and checked against nothing. The three `PythonModule*` keys are in the
  schema as `Category::Cli` purely so the eight config files carrying them are
  not rejected; the solver never reads them. Conversely `TransportSystem` and
  `PhysicsPlugins` are `Category::ProblemSelection` and are an *error* in a dict,
  rather than being accepted and ignored.
* **The geometry source is built in one place, `GeometrySource.cpp`.**
  `FieldModel`, `EquilibriumModel` and `FieldCoupling` are
  `Category::FieldSelection`: read by both surfaces, and turned into a model
  and a coupling by `geometrySourceFromConfig` -- or `geometrySourceFor`, for a
  Python `EquilibriumModel` object -- which runManta and `PyRunner::configure`
  both call, then `attachTo` the solver before `applySolverConfig`. That is
  `applySolverConfig`'s job for these keys, and for the same reason: two copies
  of the coupling choice is how the surfaces would drift.
  `test_a_dict_couples_a_field_exactly_as_a_config_file_does` compares the two
  bit for bit at every tier.
* **Configuration errors propagate.** `loadSolverConfig` throws
  `std::invalid_argument`, which pybind maps to `ValueError` for `manta.run()`;
  `main()` catches it and prints one line for the command line.
  `PyRunner::configure` re-throws it as `std::runtime_error` so that surface goes
  on raising `RuntimeError`, which is what it always has. "Could not start"
  conditions — no such config file, an unknown `TransportSystem`, an unopenable
  restart file — still make `runManta` log and return 1.
* **`UpperCamelCase` is the convention for new configuration keys, and for
  renames of existing ones.** It is what the schema already mostly is — measured
  over `--list-options`, 23 keys are `UpperCamel` against 3 `lowerCamel`
  (`initialTimestep`, `solveAdjoint`, `zeroFlux`), 2 bare lower-case (`restart`,
  `tau`) and the remaining stragglers with underscores. So a new key should be
  `MaxDegreeIncrement`, not `max_degree_increment` or `maxDegreeIncrement`.

  **The rest is not being regularised wholesale.** `delta_t`, `t_initial` and
  `t_final` keep their spellings;
  churning every config file in the tree for no functional gain is a job for
  never. What is regularised is a group of keys that is being *changed anyway* —
  the grid keys went `Grid_size` → `GridSize`, `Lower_boundary` →
  `LowerBoundary`, `Polynomial_degree` → `PolynomialDegree` and so on alongside
  the graded-mesh work, since editing them for another reason is the one moment
  the rename is free, and `Absolute_tolerance` and `Relative_tolerance` →
  `AbsoluteTolerance` and `RelativeTolerance`, as a pair.

  **A rename is a deprecated alias, never a removal.** The schema's `aliases`
  field is what makes that a one-line change: `presentSpelling` warns when the
  old name is used, and refuses a config that gives both. So every existing
  `.conf` and every `Runner.configure` dict in and out of the tree keeps working
  untouched, which is what kept the grid rename from touching 35 Python files.
  Grep `--list-options` before adding a key, because a rename that forgets the
  alias is silent for anyone whose config predates it.

## Output files

**`OutputFilename` names the output, and only its *basename* survives.**
`loadSolverConfig` fills it from the config file's stem when the key is absent,
so a run still defaults to `myrun.conf` -> `myrun.nc`; `Solver.cpp` then takes
`inputFilePath.filename()` of it. `filename()`, **not** `stem()`: the value it
is given is already a base name, and stemming it a second time would turn
`run.v2.conf` into `run.nc`. The directory part is dropped, so `.nc` / `.dat` /
`.restart.nc` still land in the current directory whatever path
`OutputFilename` carries — `OutputFilename = "runs/case7"` writes `./case7.nc`.
That is pinned by `test_output_filename_keeps_only_the_basename`, which records
it as behaviour to change deliberately rather than by accident. An explicit
`RestartFile` is *not* filtered that way and is opened as given.

**Not every `.nc` in the tree is output.** `clean_data` sweeps generated data
from the directories listed in `cmake/MantaCleanData.cmake`, sparing
`*.ref.nc` / `*.ref.dat`.
`Tests/UnitTests` is deliberately absent: its data files are tracked test
*inputs* — `testic.nc` (`AutodiffTest.cpp`) and `MatrixDiffusion.restart.nc`
(`SystemSolverTests.cpp:378`) — and the second has no `.ref.` in its name, so
the keep-pattern would not save it. Check tracked status, not the filename,
before adding a directory there. Nothing it sweeps comes from a build or a
suite, which write only into the build tree; it is for runs started by hand
inside the checkout. `python` is absent for a
different reason: since the drivers moved out to `python-examples/` and
`python-physics/`, nothing writes output there. `.h5` and `.pkl` are not in the
pattern list either — the DESC equilibria under `python-physics/stellarator/`
are expensive, and `python-physics/mirror-plasma/land.pkl` is generated once by
`landreman.py` rather than by a run.

**netCDF is the default output; the `.dat` files are opt-in.** A run writes
`<stem>.nc` and `<stem>.restart.nc` unless `WriteOutput = false`, which gates
every netCDF and restart write in `Solver.cpp`. The plain-text gnuplot output
needs `WriteDatFile = true`, and `<stem>.dydt.dat` / `<stem>.res.dat` need
`WriteDebugDatFiles = true` *and* a `PHYSICS_DEBUG` build. Those two are
deliberately **not** nested under `WriteOutput`: they are opt-in already, so a
config asking only for `WriteDatFile` gets it. All three are accepted by
`runManta` and `PyRunner::configure` alike — `WriteOutput` was read into an
unused local on the Python side and read by nothing at all on the TOML side
until the schema landed, while nine test call sites passed `WriteOutput: False`
and went on writing the files they believed they had suppressed.
