# Self-Consistent B Fields Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Couple MaNTA's HDG transport solve to a pluggable magnetic-field model whose degrees of freedom live in the IDA vector and whose residual is evaluated on every residual call, so the field responds to the plasma inside the DAE rather than beside it.

**Architecture:** A `FieldModel` declares itself as data (`FieldModelSpec`) and owns `nFieldDOF` unknowns appended to the solution vector after the global scalars. Geometry is *derived* — a function of `(psi, x)` evaluated at the physics nodes and cached per residual, reaching physics cases through a new `State::geom(g)` accessor so no existing hook signature changes. The coupling blocks are assembled outside the cell-local `MX` block, and the coupled Jacobian is solved either by an exact Schur complement onto `psi` (verification) or by block Gauss–Seidel (production).

**Tech Stack:** C++23, Eigen 3.4/5.0, SUNDIALS IDA, Boost.Test, pybind11, toml11, netCDF.

**Spec:** `docs/superpowers/specs/2026-08-15-self-consistent-b-fields-design.md`

## Global Constraints

- **Every key MaNTA accepts is declared once**, in `ConfigSchema.cpp`. Never add a `toml::find_or` or a `params` entry anywhere else.
- **`ConfigSchema.hpp`, `SolverConfig.hpp` and their `.cpp` files must stay pybind11-free.** They link into `MaNTA`, `libmanta.so` and `Tests/UnitTests`.
- **Derivative out-parameters arrive zeroed.** `State` and `GlobalState` zero themselves on construction; a hook assigns only its nonzero entries. An absent hook therefore means an identically zero block, which is the correct meaning of "this case does not read geometry".
- **The stored `sigma` is `-sigma_hat`.** `State::sigma(i)` is the stored value; `State::sigmaHat(i)` is what `SigmaFn` returned.
- **New unit-test `.cpp` files must be added to `TEST_SOURCES` in `Tests/UnitTests/Makefile`** (line 22). New non-test `.cpp` files at the repo root must be added to the top-level `Makefile`'s source list.
- **Include `<Eigen/Core>` and `<Eigen/Dense>` before project headers.** The build defines `EIGEN_USE_BLAS`; a header reaching Eigen only through `Basis.hpp` gives its translation unit a different set of definitions and LTO then picks one, surfacing as heap corruption in an unrelated static destructor.
- **Use `EIGEN_MAJOR_VERSION >= 5` to branch on Eigen's major version**, never `EIGEN_VERSION_AT_LEAST`. Spell `Eigen::all` as an explicit `.row()`/`.middleCols()`/`.leftCols()`; neither `all` spelling compiles on both supported versions under `-Werror`.
- **Never slice an Eigen `solve()` result.** `lu.solve(B)` returns a lazy `Solve<>` expression; `lu.solve(B).topRows(n)` compiles and corrupts the heap. Assign to a `Matrix` first, then slice.
- **`-Werror` is on.** Third-party includes use `-isystem`; do not add `-I` for a dependency.
- **Tests reach private `SystemSolver` members through `MANTA_TEST_PRIVATE`**, which a `-DTEST` build widens to `public`. No friend declarations.
- **`make -B` does not work in this tree.** To force a rebuild, delete the target.
- Build and test with `export PATH="$PWD/.venv/bin:$PATH"` on `PATH`.
- **Fix typos on sight**, anywhere in the repo, whether or not the file is otherwise in scope.
- **The magnetic-field spec type is `FieldModelSpec`, in `FieldModelSpec.hpp`, and must not be shortened to `FieldSpec`.** `SystemSpec.hpp:87` already defines a global `struct FieldSpec` — the per-*transport-variable* descriptor, bound to Python as `manta.Field` and asserted by `python/Tests/test_package_api.py`. The two are unrelated types, and the collision is not a compile error at the point of definition: it surfaces at link time as `-Werror=odr`, with nothing built. Note also that "field" in MaNTA already means a transport variable, so the longer name is the accurate one and not merely the available one.

**The zero-coupling invariant, which every task must preserve:** with no `FieldModel` configured, output must be identical to `main` at `07dd6ab` *bit for bit*. This is checked once in Task 12, but a task that breaks it has broken it silently, so run `make test && make regression_tests` at the end of every task.

---

### Task 1: `FieldModelSpec` and the `FieldModel` base class

Declares what a field model *is*, as validated data — the pattern `SystemSpec.hpp` already established, where a part-built description cannot exist.

**Files:**
- Create: `FieldModelSpec.hpp`
- Create: `FieldModel.hpp`
- Create: `FieldModel.cpp`
- Create: `Tests/UnitTests/FieldModelSpecTests.cpp`
- Modify: `Tests/UnitTests/Makefile:22` (add `FieldModelSpecTests.cpp` to `TEST_SOURCES`)
- Modify: `Tests/UnitTests/Makefile:41` (add `../../FieldModel.o` to `REQUIRED_OBJECTS`)
- Modify: `Makefile` (add `FieldModel.o` to the object list beside `Postprocessing.o`)

**Interfaces:**
- Consumes: `Types.hpp` (`Index`, `Value`, `Vector`, `Matrix`, `VectorRef`, `MatrixRef`, `Position`, `Time`), `State.hpp` (`GlobalState`, `GlobalStateMatrix`).
- Produces: `struct FieldDOF`, `struct FieldSlot`, `class FieldModelSpec` with `validate()`, `class FieldModel` with the pure-virtual hooks listed below. Tasks 5–12 all build on these exact names.

- [ ] **Step 1: Write the failing test**

Create `Tests/UnitTests/FieldModelSpecTests.cpp`:

```cpp
// A FieldModelSpec is validated once, in the FieldModel constructor, so a half-built
// field model cannot exist -- the same contract SystemSpec has. These tests pin
// the refusals, because every one of them is a configuration error that would
// otherwise surface much later as an assembly shape mismatch or an IDA failure
// code with nothing pointing back here.
#include <boost/test/unit_test.hpp>

#include "../../FieldModelSpec.hpp"
#include "../../FieldModel.hpp"

BOOST_AUTO_TEST_SUITE(field_model_spec_tests)

static FieldModelSpec twoDofOneSlot()
{
    FieldModelSpec spec;
    spec.dofs = {{"psi0", "flux at the axis", "Wb", false},
                 {"psi1", "flux at the edge", "Wb", false}};
    spec.geometry = {{"Vprime", "flux surface volume derivative", "m^3"}};
    spec.label = "V";
    return spec;
}

BOOST_AUTO_TEST_CASE(a_well_formed_spec_validates)
{
    BOOST_CHECK_NO_THROW(twoDofOneSlot().validate());
    BOOST_CHECK_EQUAL(twoDofOneSlot().nFieldDOF(), 2);
    BOOST_CHECK_EQUAL(twoDofOneSlot().nGeometry(), 1);
}

BOOST_AUTO_TEST_CASE(a_spec_with_no_dofs_is_refused)
{
    FieldModelSpec spec = twoDofOneSlot();
    spec.dofs.clear();
    BOOST_CHECK_THROW(spec.validate(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(a_spec_with_no_geometry_slots_is_refused)
{
    // A field model that exposes no geometry cannot affect the transport at
    // all: geometry is the only channel from psi into the physics.
    FieldModelSpec spec = twoDofOneSlot();
    spec.geometry.clear();
    BOOST_CHECK_THROW(spec.validate(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(duplicate_names_are_refused)
{
    FieldModelSpec spec = twoDofOneSlot();
    spec.dofs[1].name = "psi0";
    BOOST_CHECK_THROW(spec.validate(), std::invalid_argument);

    FieldModelSpec spec2 = twoDofOneSlot();
    spec2.geometry.push_back({"Vprime", "again", "m^3"});
    BOOST_CHECK_THROW(spec2.validate(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(an_empty_name_is_refused)
{
    // Names become netCDF group names in Task 12, where an empty one is a
    // failure a long way from here.
    FieldModelSpec spec = twoDofOneSlot();
    spec.dofs[0].name = "";
    BOOST_CHECK_THROW(spec.validate(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(an_empty_label_is_refused)
{
    FieldModelSpec spec = twoDofOneSlot();
    spec.label = "";
    BOOST_CHECK_THROW(spec.validate(), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
```

- [ ] **Step 2: Run test to verify it fails**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=field_model_spec_tests
```

Expected: compilation failure, `FieldModelSpec.hpp: No such file or directory`.

- [ ] **Step 3: Write `FieldModelSpec.hpp`**

```cpp
#ifndef FIELDSPEC_HPP
#define FIELDSPEC_HPP

#include "Types.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

/*
    What a magnetic-field model *is*, as data.

    Built once, validated once, handed to the FieldModel constructor, immutable
    after -- the contract SystemSpec.hpp already established, so that a
    part-built model cannot exist and nFieldDOF/nGeometry are derived rather
    than assigned.

    Two vectors, and they mean different things:

      * `dofs` are unknowns. They join the IDA vector after the global scalars,
        and each declares whether it is differential or algebraic -- which goes
        into the `id` vector IDASetId receives.

      * `geometry` are *derived* slots. They are not unknowns: they are a
        function of (psi, x) evaluated at the physics nodes and cached per
        residual, in the same standing as sigmaHat. A physics case reads them
        through State::geom(g).

    `label` names the spatial coordinate the model's geometry is expressed
    against. MaNTA does not interpret it -- the provider declares its own label
    and supplies the metric on it -- but it is recorded in the output so a run
    says what its x meant.
 */

struct FieldDOF
{
    std::string name;
    std::string description;
    std::string units;

    // True if this unknown's residual row carries a d/dt. Declaring it true
    // when the row has no time derivative is the IDA_LINESEARCH_FAIL (-13)
    // misdeclaration: IDA_YA_YDP_INIT holds every differential *value* fixed,
    // so a row reaching no unknown it may move is irreducible and the
    // backtracking loop runs to exhaustion. Task 6 refuses it at run time,
    // where the residual can actually be interrogated.
    bool differential = false;
};

struct FieldSlot
{
    std::string name;
    std::string description;
    std::string units;
};

class FieldModelSpec
{
public:
    std::vector<FieldDOF> dofs;
    std::vector<FieldSlot> geometry;
    std::string label;

    Index nFieldDOF() const { return static_cast<Index>(dofs.size()); }
    Index nGeometry() const { return static_cast<Index>(geometry.size()); }

    void validate() const
    {
        if (dofs.empty())
            throw std::invalid_argument("FieldModelSpec: a field model must declare at least one degree of freedom");
        if (geometry.empty())
            throw std::invalid_argument(
                "FieldModelSpec: a field model must declare at least one geometry slot; "
                "geometry is the only channel from the field DOFs into the transport physics");
        if (label.empty())
            throw std::invalid_argument("FieldModelSpec: the spatial label must be named");

        checkNames(dofs, "degree of freedom");
        checkNames(geometry, "geometry slot");
    }

private:
    template <typename T>
    static void checkNames(std::vector<T> const &v, char const *what)
    {
        for (auto const &e : v)
            if (e.name.empty())
                throw std::invalid_argument(std::string("FieldModelSpec: an unnamed ") + what);

        for (size_t i = 0; i < v.size(); ++i)
            for (size_t j = i + 1; j < v.size(); ++j)
                if (v[i].name == v[j].name)
                    throw std::invalid_argument(std::string("FieldModelSpec: duplicate ") + what + " name '" + v[i].name + "'");
    }
};

#endif // FIELDSPEC_HPP
```

- [ ] **Step 4: Write `FieldModel.hpp`**

```cpp
#ifndef FIELDMODEL_HPP
#define FIELDMODEL_HPP

#include <Eigen/Core>
#include <Eigen/Dense>

#include "FieldModelSpec.hpp"
#include "State.hpp"
#include "Types.hpp"

#include <memory>

/*
    A magnetic-field model: a set of unknowns `psi` carried in the IDA vector,
    an algebraic (or differential) residual constraining them, and a geometry
    map from (psi, x) to the metric fields the transport physics reads.

    The residual is evaluated on every residual call. That is affordable
    because it is the *constraint*, not a solve: the model's own Newton is
    subsumed into IDA's, so its Jacobian is applied once per Newton iteration
    rather than being iterated to convergence per call. See the spec's reading
    of refs/NewtonGSMFEM.pdf.
 */
class FieldModel
{
public:
    explicit FieldModel(FieldModelSpec spec_) : spec(std::move(spec_))
    {
        spec.validate();
        B.setZero(spec.nFieldDOF(), spec.nFieldDOF());
    }
    virtual ~FieldModel() = default;

    FieldModelSpec const &getSpec() const { return spec; }
    Index nFieldDOF() const { return spec.nFieldDOF(); }
    Index nGeometry() const { return spec.nGeometry(); }
    bool isFieldDOFDifferential(Index i) const { return spec.dofs[i].differential; }

    // ---- Residual -------------------------------------------------------

    /// The constraint rows. `out` is length nFieldDOF and arrives zeroed.
    /// `states` and `points` are the transport solution sampled on the physics
    /// nodes -- the same GlobalState ScalarG receives -- and `weights` is one
    /// quadrature weight per node, so Int f dx is weights.dot(f_at_nodes).
    ///
    /// A model that cannot evaluate at this state (no x-point, a boundary that
    /// has left the domain) must throw. static_residual catches and returns 1,
    /// which IDA treats as recoverable and retries with a smaller step.
    virtual void FieldResidual(VectorRef out, Vector const &psi, Vector const &dpsidt,
                               GlobalState const &states, std::vector<Position> const &points,
                               Vector const &weights, Time t) = 0;

    // ---- Geometry -------------------------------------------------------

    /// The metric at one point. `out` is length nGeometry and arrives zeroed.
    virtual void Geometry(VectorRef out, Vector const &psi, Position x, Time t) = 0;

    /// d(geometry slot g)/d(psi_m), shape (nGeometry, nFieldDOF), arrives zeroed.
    virtual void dGeometry_dpsi(MatrixRef out, Vector const &psi, Position x, Time t) = 0;

    // ---- Derivatives of the residual ------------------------------------

    /// Every field row's derivative at once, in the shape ScalarGPrime uses.
    ///
    ///   dR    -- indexed by field row, d(row)/d(transport DOF at each node)
    ///   dRdot -- the same against d/dt of the transport DOFs
    ///   dRdpsi, dRddpsidt -- (nFieldDOF, nFieldDOF), the model's own block
    ///
    /// All four arrive zeroed. Reporting every row at once is deliberate: it is
    /// what lets a model that solves a coupled system internally do so once.
    virtual void FieldResidualPrime(GlobalStateMatrix &dR, GlobalStateMatrix &dRdot,
                                    MatrixRef dRdpsi, MatrixRef dRddpsidt,
                                    Vector const &psi, Vector const &dpsidt,
                                    GlobalState const &states, std::vector<Position> const &points,
                                    Vector const &weights, Time t) = 0;

    // ---- Initial condition ----------------------------------------------

    /// The starting guess for psi. Arrives zeroed.
    virtual void InitialFieldValue(VectorRef out) = 0;

    // ---- The model's own Jacobian solve ---------------------------------

    /// Assemble B = dRdpsi + alpha * dRddpsidt and prepare a solve. The default
    /// stores B densely and factorises it with a partial-pivot LU, which is
    /// right for a small block and is what the manufactured clients use. A
    /// model with a large or structured block overrides all four of the
    /// following; this is the seam a real Grad-Shafranov solver plugs into.
    virtual void updateFieldJacobian(MatrixRef dRdpsi, MatrixRef dRddpsidt, double alpha)
    {
        B = dRdpsi + alpha * dRddpsidt;
        Blu.compute(B);
    }

    virtual void applyB(VectorRef out, Vector const &v) const { out = B * v; }
    virtual void applyBTranspose(VectorRef out, Vector const &v) const { out = B.transpose() * v; }

    virtual void solveB(VectorRef out, Vector const &rhs) const
    {
        // Never slice an Eigen solve() result: assign first.
        Vector x = Blu.solve(rhs);
        out = x;
    }

    virtual void solveBTranspose(VectorRef out, Vector const &rhs) const
    {
        Vector x = Blu.transpose().solve(rhs);
        out = x;
    }

    /// Discard anything cached for one run. Called from
    /// SystemSolver::initialize on every run, because initialize() skips
    /// initialiseMatrices() when already initialised -- the RF_cellwise trap,
    /// which made a reused solver take its initial condition from the previous
    /// run's final state.
    virtual void resetForRun() {}

protected:
    FieldModelSpec spec;
    Matrix B;
    Eigen::PartialPivLU<Matrix> Blu;
};

#endif // FIELDMODEL_HPP
```

- [ ] **Step 5: Write `FieldModel.cpp`**

The class is header-only for now, but the translation unit exists so the object file the test Makefile names is real and later tasks have somewhere to put non-inline code:

```cpp
#include "FieldModel.hpp"

// FieldModel is currently header-only. This translation unit exists so that
// FieldModel.o is a real object the link lines can name, and so the registry
// in Task 2 and the out-of-line pieces later tasks add have a home that does
// not force a rebuild of every includer.
```

- [ ] **Step 6: Wire the build**

In `Tests/UnitTests/Makefile:22`, append `FieldModelSpecTests.cpp` to `TEST_SOURCES`. In `Tests/UnitTests/Makefile:41`, append `../../FieldModel.o` to `REQUIRED_OBJECTS`. In the top-level `Makefile`, add `FieldModel.o` to the same list that carries `Postprocessing.o`.

- [ ] **Step 7: Run the tests**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=field_model_spec_tests
```

Expected: 6 test cases, all passing.

- [ ] **Step 8: Run the full suite and commit**

```sh
make test
git add FieldModelSpec.hpp FieldModel.hpp FieldModel.cpp Tests/UnitTests/FieldModelSpecTests.cpp Tests/UnitTests/Makefile Makefile
git commit -m "A field model declares itself as data, and is validated once"
```

---

### Task 2: The field-model registry

**Files:**
- Modify: `FieldModel.hpp` (add the `FieldModels` struct and the two macros)
- Modify: `FieldModel.cpp` (the map, `RegisterFieldModel`, `InstantiateFieldModel`)
- Create: `Tests/UnitTests/FieldRegistryTests.cpp`
- Modify: `Tests/UnitTests/Makefile:22`

**Interfaces:**
- Consumes: `FieldModel`, `FieldModelSpec` from Task 1.
- Produces: `FieldModels::InstantiateFieldModel(std::string const&, toml::value const&, Grid const&) -> std::unique_ptr<FieldModel>`; the macros `REGISTER_FIELD_MODEL_HEADER(T)` and `REGISTER_FIELD_MODEL_IMPL(T)`.

- [ ] **Step 1: Write the failing test**

Create `Tests/UnitTests/FieldRegistryTests.cpp`:

```cpp
// The field-model registry mirrors PhysicsCases::map, including its two
// deliberate behaviours: a duplicate name throws rather than the first
// registration quietly winning, and an unknown name throws with the list of
// what *is* registered rather than returning nullptr for callers to check.
//
// The map is never reset, so every test here uses a throwaway name.
#include <boost/test/unit_test.hpp>

#include "../../FieldModel.hpp"
#include "../../gridStructures.hpp"

#include <toml.hpp>

namespace
{
    class RegistryProbeField : public FieldModel
    {
    public:
        RegistryProbeField(toml::value const &, Grid const &) : FieldModel(makeSpec()) {}

        static FieldModelSpec makeSpec()
        {
            FieldModelSpec s;
            s.dofs = {{"p", "probe", "1", false}};
            s.geometry = {{"g", "probe", "1"}};
            s.label = "x";
            return s;
        }

        void FieldResidual(VectorRef, Vector const &, Vector const &, GlobalState const &,
                           std::vector<Position> const &, Vector const &, Time) override {}
        void Geometry(VectorRef, Vector const &, Position, Time) override {}
        void dGeometry_dpsi(MatrixRef, Vector const &, Position, Time) override {}
        void FieldResidualPrime(GlobalStateMatrix &, GlobalStateMatrix &, MatrixRef, MatrixRef,
                                Vector const &, Vector const &, GlobalState const &,
                                std::vector<Position> const &, Vector const &, Time) override {}
        void InitialFieldValue(VectorRef) override {}
    };
}

BOOST_AUTO_TEST_SUITE(field_registry_tests)

BOOST_AUTO_TEST_CASE(a_registered_model_can_be_instantiated_by_name)
{
    FieldModels::RegisterFieldModel("RegistryProbeFieldA", createFieldModel<RegistryProbeField>);

    Grid grid(0.0, 1.0, 4);
    toml::value config;
    auto model = FieldModels::InstantiateFieldModel("RegistryProbeFieldA", config, grid);

    BOOST_REQUIRE(model != nullptr);
    BOOST_CHECK_EQUAL(model->nFieldDOF(), 1);
    BOOST_CHECK_EQUAL(model->nGeometry(), 1);
}

BOOST_AUTO_TEST_CASE(a_duplicate_name_throws)
{
    FieldModels::RegisterFieldModel("RegistryProbeFieldB", createFieldModel<RegistryProbeField>);
    BOOST_CHECK_THROW(
        FieldModels::RegisterFieldModel("RegistryProbeFieldB", createFieldModel<RegistryProbeField>),
        std::runtime_error);
}

BOOST_AUTO_TEST_CASE(an_unknown_name_throws_and_names_what_is_registered)
{
    FieldModels::RegisterFieldModel("RegistryProbeFieldC", createFieldModel<RegistryProbeField>);

    Grid grid(0.0, 1.0, 4);
    toml::value config;
    try
    {
        FieldModels::InstantiateFieldModel("NoSuchFieldModel", config, grid);
        BOOST_FAIL("expected InstantiateFieldModel to throw");
    }
    catch (std::runtime_error const &e)
    {
        std::string const msg = e.what();
        BOOST_CHECK(msg.find("NoSuchFieldModel") != std::string::npos);
        BOOST_CHECK(msg.find("RegistryProbeFieldC") != std::string::npos);
    }
}

BOOST_AUTO_TEST_SUITE_END()
```

- [ ] **Step 2: Run test to verify it fails**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=field_registry_tests
```

Expected: compilation failure, `'FieldModels' has not been declared`.

- [ ] **Step 3: Add the registry to `FieldModel.hpp`**

Append, after the `FieldModel` class and inside no namespace:

```cpp
#include "gridStructures.hpp"
#include <toml.hpp>

#include <functional>
#include <map>

template <typename T>
std::unique_ptr<FieldModel> createFieldModel(toml::value const &config, Grid const &grid)
{
    return std::make_unique<T>(config, grid);
}

struct FieldModels
{
public:
    typedef std::function<std::unique_ptr<FieldModel>(toml::value const &, Grid const &)> function_type;
    typedef std::map<std::string, function_type> map_type;

    static std::unique_ptr<FieldModel> InstantiateFieldModel(std::string const &s,
                                                             toml::value const &config,
                                                             Grid const &grid);

    // Throws on a duplicate name rather than quietly keeping the first, which
    // is what a bare map::insert would do -- a model whose name collided would
    // simply never be instantiated, with nothing said at build or run time.
    static void RegisterFieldModel(std::string const &s, function_type creator);

protected:
    static map_type *getMap();

public:
    static map_type *map;
};

template <typename T>
struct FieldModelRegister
{
    explicit FieldModelRegister(std::string const &name)
    {
        FieldModels::RegisterFieldModel(name, createFieldModel<T>);
    }
};

// A model only appears if its object file is linked in -- nothing references it
// directly, so a missing entry is a link-line problem with no compile error.
#define REGISTER_FIELD_MODEL_HEADER(T) static FieldModelRegister<T> registerFieldModel_##T;
#define REGISTER_FIELD_MODEL_IMPL(T) FieldModelRegister<T> T::registerFieldModel_##T(#T);
```

- [ ] **Step 4: Implement it in `FieldModel.cpp`**

```cpp
#include "FieldModel.hpp"

#include <stdexcept>

FieldModels::map_type *FieldModels::map = nullptr;

FieldModels::map_type *FieldModels::getMap()
{
    // Never deleted: this runs during static initialisation and the map must
    // outlive every translation unit that registers into it.
    if (!map)
        map = new map_type;
    return map;
}

void FieldModels::RegisterFieldModel(std::string const &s, function_type creator)
{
    auto *m = getMap();
    if (m->count(s) > 0)
        throw std::runtime_error("Duplicate field model name '" + s + "'");
    m->insert(std::make_pair(s, creator));
}

std::unique_ptr<FieldModel> FieldModels::InstantiateFieldModel(std::string const &s,
                                                               toml::value const &config,
                                                               Grid const &grid)
{
    auto *m = getMap();
    auto it = m->find(s);
    if (it == m->end())
    {
        std::string known;
        for (auto const &e : *m)
            known += (known.empty() ? "" : ", ") + e.first;
        throw std::runtime_error("Unknown field model '" + s + "'. Registered field models are: " +
                                 (known.empty() ? "(none)" : known));
    }
    return it->second(config, grid);
}
```

- [ ] **Step 5: Wire the build and run**

Append `FieldRegistryTests.cpp` to `TEST_SOURCES` in `Tests/UnitTests/Makefile:22`, then:

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=field_registry_tests
```

Expected: 3 test cases, all passing.

- [ ] **Step 6: Run the full suite and commit**

```sh
make test
git add FieldModel.hpp FieldModel.cpp Tests/UnitTests/FieldRegistryTests.cpp Tests/UnitTests/Makefile
git commit -m "Register field models by name, and refuse a duplicate"
```

---

### Task 3: Geometry in `State` and `GlobalState`

Geometry reaches a physics case through a new named accessor, so that no existing hook signature changes and every existing case, trampoline and stub is untouched.

**Files:**
- Modify: `State.hpp:69-91` (constructor, `clone`, `zero`), `State.hpp:141-145` (element accessors), `State.hpp:171-179` (whole-vector accessors and members), and the `GlobalState` class from `State.hpp:182`
- Modify: `PyState.hpp` (expose `geom`)
- Modify: `Python.cpp` (the `GlobalState` caster: add the `"Geometry"` key)
- Create: `Tests/UnitTests/StateGeometryTests.cpp`
- Modify: `Tests/UnitTests/Makefile:22`
- Modify: `python/Tests/test_typecasters.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `State::geom(Index) -> double&` / `double`, `State::geom() -> Vector&`; `State(Index nv, Index ns, Index naux, Index ngeom)`; `GlobalState(Index nCells, Index k, Index nv, Index ns, Index naux, Index ngeom)` and `GlobalState::Geometry(Index node) -> Vector`. Tasks 5–11 use these.

- [ ] **Step 1: Write the failing test**

Create `Tests/UnitTests/StateGeometryTests.cpp`:

```cpp
// Geometry is a *derived* quantity carried on State beside u, q, sigma and phi
// -- not an unknown. It reaches a physics case through s.geom(g), which is why
// SigmaFn(i, State, x, t) does not change shape and why no existing case,
// trampoline or stub had to move.
//
// The default of zero geometry slots is what keeps every existing State
// construction compiling and meaning what it did.
#include <boost/test/unit_test.hpp>

#include "../../State.hpp"

BOOST_AUTO_TEST_SUITE(state_geometry_tests)

BOOST_AUTO_TEST_CASE(geometry_defaults_to_empty)
{
    State s(3, 0, 0);
    BOOST_CHECK_EQUAL(s.geom().size(), 0);
}

BOOST_AUTO_TEST_CASE(geometry_is_born_zeroed_like_everything_else)
{
    State s(2, 1, 1, 3);
    BOOST_REQUIRE_EQUAL(s.geom().size(), 3);
    for (Index g = 0; g < 3; ++g)
        BOOST_CHECK_EQUAL(s.geom(g), 0.0);
}

BOOST_AUTO_TEST_CASE(geometry_round_trips_by_index_and_whole)
{
    State s(2, 0, 0, 2);
    s.geom(0) = 1.5;
    s.geom(1) = -2.5;
    BOOST_CHECK_EQUAL(s.geom(0), 1.5);
    BOOST_CHECK_EQUAL(s.geom(1), -2.5);
    BOOST_CHECK_EQUAL(s.geom().sum(), -1.0);
}

BOOST_AUTO_TEST_CASE(zero_clears_geometry_too)
{
    State s(1, 0, 0, 2);
    s.geom(0) = 7.0;
    s.zero();
    BOOST_CHECK_EQUAL(s.geom(0), 0.0);
}

BOOST_AUTO_TEST_CASE(clone_copies_the_geometry_width)
{
    State s(1, 0, 0, 4);
    State t;
    t.clone(s);
    BOOST_CHECK_EQUAL(t.geom().size(), 4);
}

BOOST_AUTO_TEST_CASE(a_global_state_carries_geometry_per_node)
{
    // GlobalState stores (nGeom, nNodes); the per-node accessor returns a
    // column. Orientation matters and is checked here rather than through a
    // Python round trip, because that caster transposes in both directions and
    // so cannot detect a missing transpose.
    const Index nCells = 3, k = 2, nGeom = 2;
    GlobalState gs(nCells, k, 1, 0, 0, nGeom);
    BOOST_REQUIRE_EQUAL(gs.Geometry(0).size(), nGeom);

    gs.setGeometry(4, (Vector(2) << 3.0, 4.0).finished());
    BOOST_CHECK_EQUAL(gs.Geometry(4)(0), 3.0);
    BOOST_CHECK_EQUAL(gs.Geometry(4)(1), 4.0);
    BOOST_CHECK_EQUAL(gs.Geometry(3)(0), 0.0);
}

BOOST_AUTO_TEST_CASE(a_state_extracted_from_a_global_state_carries_its_geometry)
{
    GlobalState gs(2, 1, 1, 0, 0, 2);
    gs.setGeometry(1, (Vector(2) << 5.0, 6.0).finished());
    State s = gs[1];
    BOOST_CHECK_EQUAL(s.geom(0), 5.0);
    BOOST_CHECK_EQUAL(s.geom(1), 6.0);
}

BOOST_AUTO_TEST_SUITE_END()
```

- [ ] **Step 2: Run test to verify it fails**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=state_geometry_tests
```

Expected: compilation failure, `'class State' has no member named 'geom'`.

- [ ] **Step 3: Add geometry to `State`**

In `State.hpp`, extend the constructor at line 69, `clone` at 77 and `zero` at 85, and add the accessors and member. The `nGeom = 0` default is what keeps every existing construction compiling:

```cpp
  explicit State(Index nv, Index ns = 0, Index naux = 0, Index ngeom = 0) {
    m_Variable.setZero(nv);
    m_Derivative.setZero(nv);
    m_Flux.setZero(nv);
    m_Scalars.setZero(ns);
    m_Aux.setZero(naux);
    m_Geometry.setZero(ngeom);
  }

  void clone(const State &other) {
    m_Variable.setZero(other.m_Variable.size());
    m_Derivative.setZero(other.m_Derivative.size());
    m_Flux.setZero(other.m_Flux.size());
    m_Scalars.setZero(other.m_Scalars.size());
    m_Aux.setZero(other.m_Aux.size());
    m_Geometry.setZero(other.m_Geometry.size());
  }

  void zero() {
    m_Variable.setZero();
    m_Derivative.setZero();
    m_Flux.setZero();
    m_Scalars.setZero();
    m_Aux.setZero();
    m_Geometry.setZero();
  }
```

Beside the `phi` accessors at line 141:

```cpp
  /// A derived metric field, not an unknown: geometry is a function of the
  /// field model's psi and of x, evaluated at the physics nodes and cached per
  /// residual, in the same standing as sigmaHat. Read-write because the solver
  /// fills it before handing the State to a physics hook; a case only reads it.
  double &geom(Index i) { return checked(m_Geometry, i, "geometry slot"); }
  double geom(Index i) const { return checked(m_Geometry, i, "geometry slot"); }
```

Beside the whole-vector accessors at line 171:

```cpp
  Vector &geom() { return m_Geometry; }
  Vector const &geom() const { return m_Geometry; }
```

And extend the member declaration at line 178:

```cpp
  Vector m_Variable, m_Derivative, m_Flux, m_Aux, m_Geometry;
  Vector m_Scalars;
```

- [ ] **Step 4: Add geometry to `GlobalState`**

Extend the constructor at `State.hpp:189` and `setWithState` at 199, and add the accessors. Note the storage is `(nGeom, nNodes)`, matching how `m_Aux` is stored:

```cpp
  explicit GlobalState(Index nCells, Index k, Index nv, Index ns = 0,
                       Index naux = 0, Index ngeom = 0) noexcept
      : nCells(nCells), k(k), nVars(nv), nScalars(ns), nAux(naux), nGeom(ngeom) {
    m_Variable.setZero(nVars, nCells * (k + 1));
    m_Derivative.setZero(nVars, nCells * (k + 1));
    m_Flux.setZero(nVars, nCells * (k + 1));
    m_Aux.setZero(nAux, nCells * (k + 1));
    m_Geometry.setZero(nGeom, nCells * (k + 1));
    m_Scalars.setZero(nScalars);
  }

  Vector Geometry(Index node) const { return m_Geometry.col(node); }
  void setGeometry(Index node, Vector const &g) { m_Geometry.col(node) = g; }

  Matrix &GeometryMatrix() { return m_Geometry; }
  Matrix const &GeometryMatrix() const { return m_Geometry; }
```

Add `m_Geometry` to the private members beside `m_Aux`, and `nGeom` beside `nAux`. Then extend `setWithState` to carry it, and the `operator[]`/`State` extraction so an extracted `State` is built with `nGeom` and has its geometry column copied in.

- [ ] **Step 5: Run the C++ test**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=state_geometry_tests
```

Expected: 7 test cases, all passing.

- [ ] **Step 6: Expose it to Python**

In `PyState.hpp`, add `geom` beside the existing `phi` property, indexable by position and by declared name, following exactly the pattern the `phi` property uses. In `Python.cpp`'s `GlobalState` caster, add a `"Geometry"` key carrying an `(nPoints, nGeom)` array — **transposed**, like every other key, because C++ stores `(nGeom, nPoints)`.

Add to `python/Tests/test_typecasters.py`:

```python
def test_geometry_reaches_a_batched_call_with_the_right_orientation():
    """The GlobalState caster transposes in both directions, so a round trip
    cannot detect a missing transpose. Check the orientation from inside a
    batched call, where the (nPoints, nGeom) shape is observable."""
    seen = {}

    class GeometryProbe(manta.TransportSystem):
        variables = [manta.Field("n", "density", "m^-3")]

        def ComputePhysics(self, states, positions, t):
            seen["shape"] = states["Geometry"].shape
            seen["npoints"] = len(positions)
            return [[np.zeros(len(positions))], [np.zeros(len(positions))], []]

        def ComputePhysicsDerivatives(self, states, positions, t):
            return {}

    # ... drive it through the batched entry point as the other tests in this
    # file do, then:
    assert seen["shape"][0] == seen["npoints"]
```

- [ ] **Step 7: Regenerate the stub, typecheck and commit**

```sh
make python && make stubs && make stubs-check && make typecheck && make python_tests
make test
git add State.hpp PyState.hpp Python.cpp python/manta/_manta.pyi \
        Tests/UnitTests/StateGeometryTests.cpp Tests/UnitTests/Makefile \
        python/Tests/test_typecasters.py
git commit -m "State carries geometry, so no physics hook changes shape"
```

---

### Task 4: `DGSoln` grows a field block

**Files:**
- Modify: `DGSoln.hpp:17-19` (constructors), `:27-35` (`getDoF`), `:37-74` (`Map`), and the accessors from `:89`
- Create: `Tests/UnitTests/FieldDoFLayoutTests.cpp`
- Modify: `Tests/UnitTests/Makefile:22`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `DGSolnImpl(..., Index Scalars, Index aux, Index nField)`, `DGSolnImpl::getField() -> VectorWrapper&`, `DGSolnImpl::Field(Index) -> double&`, `DGSolnImpl::getFieldDOF() -> Index`. Tasks 6–12 use these.

- [ ] **Step 1: Write the failing test**

Create `Tests/UnitTests/FieldDoFLayoutTests.cpp`:

```cpp
// The field block is appended after the global scalars, so the DOF layout is
//
//     [ sigma | q | u | aux ] per cell,  then lambda,  then mu,  then psi
//
// Nothing existing shifts. That is the whole point of putting it last, and it
// is what these tests pin: every offset for the pre-existing blocks must be
// unchanged by a nonzero nField, because getting a column index wrong in this
// layout is the most common way to break the solver silently.
#include <boost/test/unit_test.hpp>

#include "../../DGSoln.hpp"
#include "../../gridStructures.hpp"

#include <vector>

BOOST_AUTO_TEST_SUITE(field_dof_layout_tests)

BOOST_AUTO_TEST_CASE(the_field_block_adds_exactly_its_own_width)
{
    Grid grid(0.0, 1.0, 5);
    const Index nVars = 2, k = 2, nScalars = 3, nAux = 1;

    DGSoln without(nVars, grid, k, nScalars, nAux);
    DGSoln with(nVars, grid, k, nScalars, nAux, 4);

    BOOST_CHECK_EQUAL(with.getDoF(), without.getDoF() + 4);
    BOOST_CHECK_EQUAL(with.getFieldDOF(), 4);
    BOOST_CHECK_EQUAL(without.getFieldDOF(), 0);
}

BOOST_AUTO_TEST_CASE(the_field_block_is_last_and_nothing_before_it_moves)
{
    Grid grid(0.0, 1.0, 3);
    const Index nVars = 1, k = 1, nScalars = 2, nAux = 0, nField = 3;

    DGSoln soln(nVars, grid, k, nScalars, nAux, nField);
    std::vector<double> mem(soln.getDoF(), 0.0);
    soln.Map(mem.data());

    // Write a recognisable value into each block and read it back out of the
    // raw memory at the offset the layout promises.
    soln.Scalar(0) = 11.0;
    soln.Scalar(1) = 12.0;
    soln.Field(0) = 21.0;
    soln.Field(2) = 23.0;

    const size_t scalarOffset = (3 * nVars + nAux) * (k + 1) * grid.getNCells() + nVars * (grid.getNCells() + 1);
    const size_t fieldOffset = scalarOffset + nScalars;

    BOOST_CHECK_EQUAL(mem[scalarOffset + 0], 11.0);
    BOOST_CHECK_EQUAL(mem[scalarOffset + 1], 12.0);
    BOOST_CHECK_EQUAL(mem[fieldOffset + 0], 21.0);
    BOOST_CHECK_EQUAL(mem[fieldOffset + 2], 23.0);
}

BOOST_AUTO_TEST_CASE(the_whole_field_vector_is_reachable)
{
    Grid grid(0.0, 1.0, 2);
    DGSoln soln(1, grid, 1, 0, 0, 3);
    std::vector<double> mem(soln.getDoF(), 0.0);
    soln.Map(mem.data());

    soln.getField() = (Vector(3) << 1.0, 2.0, 3.0).finished();
    BOOST_CHECK_EQUAL(soln.Field(1), 2.0);
    BOOST_CHECK_EQUAL(soln.getField().sum(), 6.0);
}

BOOST_AUTO_TEST_CASE(zero_field_dofs_is_the_default_and_costs_nothing)
{
    Grid grid(0.0, 1.0, 4);
    DGSoln soln(2, grid, 2, 1, 1);
    BOOST_CHECK_EQUAL(soln.getFieldDOF(), 0);
    BOOST_CHECK_EQUAL(soln.getField().size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()
```

- [ ] **Step 2: Run test to verify it fails**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=field_dof_layout_tests
```

Expected: compilation failure, no matching constructor taking six arguments.

- [ ] **Step 3: Extend `DGSolnImpl`**

In `DGSoln.hpp`, add `Index nField = 0` as a trailing constructor parameter on both constructors (lines 17 and 19), store it as a member `nField`, and extend `getDoF`:

```cpp
    size_t getDoF() const
    {
        // 3 = u + q + sigma
        // nCells + 1 for lambda because we store values at both ends
        // and we are carrying nScalar scalar variables
        // Auxiliary variables depend on space, so each one carries nCells * (k+1) degrees of freedom
        // The field model's unknowns are appended last, so adding them shifts
        // nothing before them.
        return grid.getNCells() * nVars * (k + 1) * 3 +
               (grid.getNCells() + 1) * nVars + nScalars + grid.getNCells() * nAux * (k + 1) +
               nField;
    };
```

In `Map`, after the `mu_` placement-new at line 70:

```cpp
        size_t field_offset = scalar_offset + nScalars;
        new (&psi_) VectorWrapper(Y + field_offset, nField);
```

Declare `VectorWrapper psi_;` beside `mu_`, initialise it as `psi_(nullptr, 0)` in both constructors' initialiser lists, and add the accessors beside `Scalars()`:

```cpp
    Index getFieldDOF() const { return nField; };

    double Field(Index j) const { return psi_[j]; };
    double &Field(Index j) { return psi_[j]; };

    VectorWrapper const &getField() const { return psi_; };
    VectorWrapper &getField() { return psi_; };
```

Extend `copy`, `zeroCoeffs` and any other whole-object operation in the class to carry `psi_`. Grep for `mu_` in `DGSoln.hpp` and handle `psi_` at every site.

- [ ] **Step 4: Run the test**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=field_dof_layout_tests
```

Expected: 4 test cases, all passing.

- [ ] **Step 5: Run the full suite and commit**

Nothing constructs a `DGSoln` with a nonzero `nField` yet, so the suite must be entirely unaffected.

```sh
make test && make regression_tests
git add DGSoln.hpp Tests/UnitTests/FieldDoFLayoutTests.cpp Tests/UnitTests/Makefile
git commit -m "The field block goes last in the DOF layout, so nothing before it moves"
```

---

### Task 5: The two manufactured field models

Test fixtures, not production models. `ManufacturedField` is the cheapest configuration in which the exact and iterative solves must agree; `ManufacturedFieldVector` is what exercises a `B` that is not a scalar.

**Files:**
- Create: `Tests/UnitTests/ManufacturedFields.hpp`
- Create: `Tests/UnitTests/ManufacturedFieldTests.cpp`
- Modify: `Tests/UnitTests/Makefile:22`

**Interfaces:**
- Consumes: `FieldModel`, `FieldModelSpec` (Task 1); `GlobalState` with geometry (Task 3).
- Produces: `class ManufacturedField`, `class ManufacturedFieldVector`, and the free functions `manufacturedU(Position x, Time t)`, `manufacturedPsiExact(Time t)`. Tasks 6, 8, 9, 10 and 11 all use these.

- [ ] **Step 1: Write the failing test**

Create `Tests/UnitTests/ManufacturedFieldTests.cpp`:

```cpp
// The manufactured field models, checked against their own closed forms before
// anything in the solver depends on them.
//
// ManufacturedField is nFieldDOF = 1 with R = psi - Int u dx and one geometry
// slot g(x; psi) = 1 + psi c(x). With the harness's u = sin(pi x)(1+t) on
// [0,1], Int u dx = (2/pi)(1+t), so psi_exact(t) = (2/pi)(1+t).
//
// The coupling is genuinely two-way: A2 is dense across every u DOF, and A1 is
// nonzero because the flux is sigma_hat = g kappa q.
#include <boost/test/unit_test.hpp>

#include "ManufacturedFields.hpp"

#include <cmath>

BOOST_AUTO_TEST_SUITE(manufactured_field_tests)

BOOST_AUTO_TEST_CASE(psi_exact_is_the_integral_of_u)
{
    // Int_0^1 sin(pi x) dx = 2/pi
    BOOST_CHECK_CLOSE(manufacturedPsiExact(0.0), 2.0 / M_PI, 1e-12);
    BOOST_CHECK_CLOSE(manufacturedPsiExact(1.0), 4.0 / M_PI, 1e-12);
}

BOOST_AUTO_TEST_CASE(the_residual_vanishes_at_the_exact_solution)
{
    ManufacturedField model{toml::value{}, Grid(0.0, 1.0, 16)};
    const Time t = 0.3;

    // Sample u_exact on the nodes of a fine grid and integrate with the
    // solver's own weights, exactly as the residual will.
    auto [states, points, weights] = sampleExactOnNodes(Grid(0.0, 1.0, 16), 3, t);

    Vector psi(1);
    psi(0) = manufacturedPsiExact(t);
    Vector dpsidt(1);
    dpsidt(0) = 0.0;

    Vector out = Vector::Zero(1);
    model.FieldResidual(out, psi, dpsidt, states, points, weights, t);

    // Not exactly zero: the quadrature is interpolatory on a degree-3 basis,
    // so it integrates sin(pi x) to the basis's accuracy, not exactly.
    BOOST_CHECK_SMALL(out(0), 1e-6);
}

BOOST_AUTO_TEST_CASE(dR_dpsi_is_the_identity_and_dR_dstate_is_minus_the_weights)
{
    // R = psi - Int u dx, so dR/dpsi = 1 and dR/du_j = -w_j exactly. A case
    // must use the solver's quadrature weights rather than a rule of its own;
    // ScalarTestLD3 disagreed with its own Jacobian by 8% for doing otherwise.
    ManufacturedField model{toml::value{}, Grid(0.0, 1.0, 8)};
    const Time t = 0.0;
    auto [states, points, weights] = sampleExactOnNodes(Grid(0.0, 1.0, 8), 2, t);

    GlobalStateMatrix dR(1), dRdot(1);
    dR.add(8, 2, 1, 0, 0);
    dRdot.add(8, 2, 1, 0, 0);
    Matrix dRdpsi = Matrix::Zero(1, 1), dRddpsidt = Matrix::Zero(1, 1);

    Vector psi(1); psi(0) = manufacturedPsiExact(t);
    Vector dpsidt = Vector::Zero(1);

    model.FieldResidualPrime(dR, dRdot, dRdpsi, dRddpsidt, psi, dpsidt,
                             states, points, weights, t);

    BOOST_CHECK_CLOSE(dRdpsi(0, 0), 1.0, 1e-12);
    BOOST_CHECK_SMALL(dRddpsidt(0, 0), 1e-15);
    for (Index j = 0; j < weights.size(); ++j)
        BOOST_CHECK_CLOSE(dR[0][j].u(0), -weights(j), 1e-12);
}

BOOST_AUTO_TEST_CASE(the_geometry_and_its_derivative_agree_with_the_closed_form)
{
    ManufacturedField model{toml::value{}, Grid(0.0, 1.0, 4)};
    Vector psi(1); psi(0) = 0.75;

    Vector g = Vector::Zero(1);
    model.Geometry(g, psi, 0.25, 0.0);
    BOOST_CHECK_CLOSE(g(0), 1.0 + 0.75 * manufacturedC(0.25), 1e-12);

    Matrix dg = Matrix::Zero(1, 1);
    model.dGeometry_dpsi(dg, psi, 0.25, 0.0);
    BOOST_CHECK_CLOSE(dg(0, 0), manufacturedC(0.25), 1e-12);
}

BOOST_AUTO_TEST_CASE(the_vector_model_has_a_nonscalar_b_block)
{
    // L is SPD tridiagonal, so B is genuinely a matrix and its solve is not a
    // division. This is what stops the block solve being exercised only in a
    // degenerate case.
    ManufacturedFieldVector model{toml::value{}, Grid(0.0, 1.0, 4)};
    BOOST_CHECK_EQUAL(model.nFieldDOF(), 5);

    Matrix dRdpsi = manufacturedL(5);
    Matrix dRddpsidt = Matrix::Zero(5, 5);
    model.updateFieldJacobian(dRdpsi, dRddpsidt, 0.0);

    Vector rhs = Vector::LinSpaced(5, 1.0, 5.0);
    Vector x = Vector::Zero(5);
    model.solveB(x, rhs);

    Vector back = Vector::Zero(5);
    model.applyB(back, x);
    BOOST_CHECK_SMALL((back - rhs).norm(), 1e-12);
}

BOOST_AUTO_TEST_SUITE_END()
```

- [ ] **Step 2: Run test to verify it fails**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=manufactured_field_tests
```

Expected: compilation failure, `ManufacturedFields.hpp: No such file or directory`.

- [ ] **Step 3: Write `Tests/UnitTests/ManufacturedFields.hpp`**

```cpp
#ifndef MANUFACTUREDFIELDS_HPP
#define MANUFACTUREDFIELDS_HPP

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../../FieldModel.hpp"
#include "../../gridStructures.hpp"

#include <cmath>
#include <tuple>

/*
    Manufactured field models: test fixtures, deliberately not registered for
    production.

    The exact transport solution is the one MMSHarness.hpp already shares,
    u = sin(pi x)(1 + t) on [0,1].

    A note on how the constraint is compensated, because the obvious way is
    wrong. A compensating term written against the *discrete* state can be an
    exact row operation: residual() evaluates the hooks on the same states at
    the same abscissae and pushes them through the same projection, so a
    compensation of that shape cancels identically and the study silently
    measures the uncoupled problem. Everything below is therefore compensated
    against u_exact(x, t), never against the state it is handed.
*/

inline Value manufacturedU(Position x, Time t) { return std::sin(M_PI * x) * (1.0 + t); }

/// Int_0^1 sin(pi x)(1+t) dx = (2/pi)(1+t)
inline Value manufacturedPsiExact(Time t) { return (2.0 / M_PI) * (1.0 + t); }

/// The shape function the single-DOF model's geometry slot carries. Chosen
/// nonconstant so that dGeometry/dpsi varies across a cell -- a constant would
/// be annihilated by exactly the operator that hid the mass-matrix confusion in
/// DerivativeSubVector, so it would fail to distinguish a right answer from a
/// wrong one.
inline Value manufacturedC(Position x) { return std::cos(M_PI * x); }

/// The SPD tridiagonal standing in for a 1-D elliptic operator: the usual
/// second-difference stencil with unit Dirichlet ends.
inline Matrix manufacturedL(Index n)
{
    Matrix L = Matrix::Zero(n, n);
    for (Index i = 0; i < n; ++i)
    {
        L(i, i) = 2.0;
        if (i > 0)
            L(i, i - 1) = -1.0;
        if (i + 1 < n)
            L(i, i + 1) = -1.0;
    }
    return L;
}

/// nFieldDOF = 1, algebraic:  R = psi - Int u dx,  g(x; psi) = 1 + psi c(x).
class ManufacturedField : public FieldModel
{
public:
    ManufacturedField(toml::value const &, Grid const &) : FieldModel(buildSpec()) {}

    static FieldModelSpec buildSpec()
    {
        FieldModelSpec s;
        s.dofs = {{"psi", "the manufactured field unknown", "1", false}};
        s.geometry = {{"g", "metric factor multiplying the diffusivity", "1"}};
        s.label = "x";
        return s;
    }

    void FieldResidual(VectorRef out, Vector const &psi, Vector const &,
                       GlobalState const &states, std::vector<Position> const &,
                       Vector const &weights, Time) override
    {
        double integral = 0.0;
        for (Index j = 0; j < weights.size(); ++j)
            integral += weights(j) * states[j].u(0);
        out(0) = psi(0) - integral;
    }

    void Geometry(VectorRef out, Vector const &psi, Position x, Time) override
    {
        out(0) = 1.0 + psi(0) * manufacturedC(x);
    }

    void dGeometry_dpsi(MatrixRef out, Vector const &, Position x, Time) override
    {
        out(0, 0) = manufacturedC(x);
    }

    void FieldResidualPrime(GlobalStateMatrix &dR, GlobalStateMatrix &, MatrixRef dRdpsi,
                            MatrixRef, Vector const &, Vector const &, GlobalState const &,
                            std::vector<Position> const &, Vector const &weights, Time) override
    {
        dRdpsi(0, 0) = 1.0;
        for (Index j = 0; j < weights.size(); ++j)
            dR[0][j].u(0) = -weights(j);
    }

    void InitialFieldValue(VectorRef out) override { out(0) = manufacturedPsiExact(0.0); }
};

/// nFieldDOF = n:  L psi = f(state), with f_m sampling the transport solution.
/// B = L is tridiagonal and dGeometry/dpsi is dense, because geometry at x
/// interpolates every entry of psi.
class ManufacturedFieldVector : public FieldModel
{
public:
    static constexpr Index N = 5;

    ManufacturedFieldVector(toml::value const &, Grid const &) : FieldModel(buildSpec()), L(manufacturedL(N)) {}

    static FieldModelSpec buildSpec()
    {
        FieldModelSpec s;
        s.geometry = {{"g", "metric factor multiplying the diffusivity", "1"}};
        s.label = "x";
        for (Index m = 0; m < N; ++m)
            s.dofs.push_back({"psi" + std::to_string(m), "manufactured field unknown", "1", false});
        return s;
    }

    /// psi is sampled at N equispaced points in [0,1]; geometry at x is the
    /// piecewise-linear interpolant, which is what makes dGeometry/dpsi dense
    /// in the sense that matters (every x sees more than one psi entry).
    static Position node(Index m) { return static_cast<double>(m) / static_cast<double>(N - 1); }

    void FieldResidual(VectorRef out, Vector const &psi, Vector const &,
                       GlobalState const &states, std::vector<Position> const &points,
                       Vector const &weights, Time t) override
    {
        out = L * psi - f(states, points, weights);
    }

    void Geometry(VectorRef out, Vector const &psi, Position x, Time) override
    {
        out(0) = 1.0 + interpolate(psi, x);
    }

    void dGeometry_dpsi(MatrixRef out, Vector const &, Position x, Time) override
    {
        for (Index m = 0; m < N; ++m)
            out(0, m) = basis(m, x);
    }

    void FieldResidualPrime(GlobalStateMatrix &dR, GlobalStateMatrix &, MatrixRef dRdpsi,
                            MatrixRef, Vector const &, Vector const &, GlobalState const &,
                            std::vector<Position> const &points, Vector const &weights,
                            Time) override
    {
        dRdpsi = L;
        // f_m = Int c_m(x) u(x) dx, so df_m/du_j = -w_j c_m(x_j) in the residual.
        for (Index m = 0; m < N; ++m)
            for (Index j = 0; j < weights.size(); ++j)
                dR[m][j].u(0) = -weights(j) * basis(m, points[j]);
    }

    void InitialFieldValue(VectorRef out) override
    {
        // The exact psi at t = 0, from L psi = f(u_exact).
        Vector rhs = fExact(0.0);
        Vector x = L.partialPivLu().solve(rhs);
        out = x;
    }

    /// The exact psi at time t, for the order study to compare against.
    Vector psiExact(Time t) const
    {
        Vector rhs = fExact(t);
        Vector x = L.partialPivLu().solve(rhs);
        return x;
    }

private:
    /// The hat function centred on node m, evaluated at x.
    static double basis(Index m, Position x)
    {
        const double h = 1.0 / static_cast<double>(N - 1);
        const double d = std::abs(x - node(m)) / h;
        return d >= 1.0 ? 0.0 : 1.0 - d;
    }

    static double interpolate(Vector const &psi, Position x)
    {
        double v = 0.0;
        for (Index m = 0; m < N; ++m)
            v += psi(m) * basis(m, x);
        return v;
    }

    /// The residual's own term: f_m = Int c_m(x) u(x) dx against the *discrete*
    /// state, which is what the constraint L psi = f(state) means.
    Vector f(GlobalState const &states, std::vector<Position> const &points,
             Vector const &weights) const
    {
        Vector out = Vector::Zero(N);
        for (Index m = 0; m < N; ++m)
            for (Index j = 0; j < weights.size(); ++j)
                out(m) += weights(j) * basis(m, points[j]) * states[j].u(0);
        return out;
    }

    /// The same integral against u_exact, for psiExact and the initial value.
    /// This -- not f() above -- is the "compensate against u_exact" the header
    /// comment describes: it is what the order study compares to, so it must
    /// not be a function of the discrete state.
    Vector fExact(Time t) const
    {
        // Int c_m(x) sin(pi x)(1+t) dx, by a fine Simpson rule; the constraint
        // only has to be *consistent*, not analytic, for the order study.
        const Index nq = 4001;
        Vector out = Vector::Zero(N);
        const double h = 1.0 / static_cast<double>(nq - 1);
        for (Index m = 0; m < N; ++m)
        {
            double s = 0.0;
            for (Index j = 0; j < nq; ++j)
            {
                const double x = j * h;
                const double w = (j == 0 || j == nq - 1) ? 1.0 : (j % 2 ? 4.0 : 2.0);
                s += w * basis(m, x) * manufacturedU(x, t);
            }
            out(m) = s * h / 3.0;
        }
        return out;
    }

    Matrix L;
};

/// Sample u_exact on a grid's nodes and return the GlobalState, the abscissae
/// and the integration weights, so a test can call a field hook without a
/// solver. Defined in ManufacturedFieldTests.cpp.
std::tuple<GlobalState, std::vector<Position>, Vector>
sampleExactOnNodes(Grid const &grid, Index k, Time t);

#endif // MANUFACTUREDFIELDS_HPP
```

- [ ] **Step 4: Implement `sampleExactOnNodes` in `ManufacturedFieldTests.cpp`**

Above the test suite, using the same `Integrator::getIntegrationWeights` the residual uses:

```cpp
std::tuple<GlobalState, std::vector<Position>, Vector>
sampleExactOnNodes(Grid const &grid, Index k, Time t)
{
    DGSoln soln(1, grid, k, 0, 0);
    std::vector<double> mem(soln.getDoF(), 0.0);
    soln.Map(mem.data());
    soln.AssignU(0, [t](Position x) { return manufacturedU(x, t); });

    GlobalState states = soln.evalOnNodes();
    std::vector<Position> points = soln.getPoints();
    Vector weights = Integrator::getIntegrationWeights(soln.getBasis(), grid);
    return {states, points, weights};
}
```

- [ ] **Step 5: Wire the build and run**

Append `ManufacturedFieldTests.cpp` to `TEST_SOURCES`, then:

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=manufactured_field_tests
```

Expected: 5 test cases, all passing.

- [ ] **Step 6: Run the full suite and commit**

```sh
make test
git add Tests/UnitTests/ManufacturedFields.hpp Tests/UnitTests/ManufacturedFieldTests.cpp Tests/UnitTests/Makefile
git commit -m "Two manufactured field models, checked against their closed forms"
```

---

### Task 6: The coupled residual, and the first end-to-end run

The largest task, and the one that first produces a running coupled solve. It ends with a coupled run reaching a manufactured solution, on IDA's default Jacobian handling — the coupling blocks arrive in Task 8.

**Files:**
- Modify: `SystemSolver.hpp` (members `fieldModel`, `nField`; the `PhysicsNodes` struct at `:411`; declarations)
- Modify: `SystemSolver.cpp:20` (constructor), `:76` (`setInitialConditions`), `:212` (`initialiseMatrices`), `:623` (`evaluatePhysicsDerivatives`), `:1127` (`residual`)
- Modify: `Solver.cpp:186-199` (the `isDifferential` vector)
- Create: `Tests/UnitTests/CoupledResidualTests.cpp`
- Modify: `Tests/UnitTests/Makefile:22`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: `SystemSolver::setFieldModel(std::shared_ptr<FieldModel>)`, `SystemSolver::evaluateGeometry(DGSoln const&, std::vector<Position> const&, GlobalState&, Time)`, and `SystemSolver::nField`. Tasks 7–12 use these.

- [ ] **Step 1: Write the failing test**

Create `Tests/UnitTests/CoupledResidualTests.cpp`:

```cpp
// The coupled residual: field rows in the residual vector, geometry evaluated
// at the physics nodes and reaching the physics through State::geom.
//
// Three things are pinned here and each has a distinct failure mode.
//
//  * The field rows land in the field block and nowhere else. Getting a column
//    index wrong in this layout is the most common way to break the solver
//    silently.
//  * Geometry reaches a physics case's SigmaFn. Without this the coupling is
//    one-way and every subsequent test would still pass on a decoupled problem.
//  * A field DOF declared differential whose residual carries no d/dt is
//    refused at initialisation. Left to IDA it is IDA_LINESEARCH_FAIL (-13),
//    a message about the linesearch for a defect in the declaration -- which is
//    what kept python-physics/mirror-plasma's voltage controller from ever
//    starting.
#include <boost/test/unit_test.hpp>

#include "ManufacturedFields.hpp"
#include "../../SystemSolver.hpp"

BOOST_AUTO_TEST_SUITE(coupled_residual_tests)

BOOST_AUTO_TEST_CASE(the_field_rows_appear_in_the_field_block)
{
    // Build a solver with ManufacturedField attached, evaluate the residual at
    // a state whose psi is deliberately wrong by a known amount, and check the
    // field row equals that amount.
    auto [solver, y, dydt, res] = makeCoupledSolver(/*nCells=*/8, /*k=*/2);

    DGSoln yMap = mapSoln(*solver, y);
    yMap.AssignU(0, [](Position x) { return manufacturedU(x, 0.0); });
    yMap.Field(0) = manufacturedPsiExact(0.0) + 0.25;

    solver->residual(0.0, y, dydt, res);

    DGSoln resMap = mapSoln(*solver, res);
    BOOST_CHECK_CLOSE(resMap.Field(0), 0.25, 1e-6);
}

BOOST_AUTO_TEST_CASE(geometry_reaches_the_physics_case)
{
    // GeometryProbeCase records the geometry it was handed at each node. With
    // psi set to a known value, g(x) = 1 + psi c(x) is a closed form.
    auto [solver, y, dydt, res] = makeCoupledSolverWithProbe(/*nCells=*/4, /*k=*/1);

    DGSoln yMap = mapSoln(*solver, y);
    yMap.Field(0) = 0.5;

    solver->residual(0.0, y, dydt, res);

    auto const &seen = GeometryProbeCase::lastGeometry();
    auto const &points = GeometryProbeCase::lastPoints();
    BOOST_REQUIRE_EQUAL(seen.size(), points.size());
    for (size_t j = 0; j < seen.size(); ++j)
        BOOST_CHECK_CLOSE(seen[j], 1.0 + 0.5 * manufacturedC(points[j]), 1e-10);
}

BOOST_AUTO_TEST_CASE(a_differential_field_dof_with_no_time_derivative_is_refused)
{
    // The same check ScalarGPrime's dGdot needs: call FieldResidualPrime and
    // require the dRddpsidt row to be nonzero for every DOF declared
    // differential.
    auto solver = makeSolverWithModel(std::make_shared<BadlyDeclaredDifferentialField>());
    BOOST_CHECK_THROW(solver->initialize(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(a_field_model_that_throws_is_a_recoverable_error)
{
    // static_residual catches and returns 1, which IDA treats as recoverable
    // and retries with a smaller step. Throwing out of the residual would abort
    // a run a shorter step would have survived.
    auto [solver, y, dydt, res] = makeCoupledSolverWithThrowingField(4, 1);
    const int retval = static_residual(0.0, y, dydt, res, solver.get());
    BOOST_CHECK_EQUAL(retval, 1);
}

BOOST_AUTO_TEST_CASE(a_coupled_run_reaches_the_manufactured_solution)
{
    // The end-to-end check: integrate to t = 0.5 and compare both u and psi
    // against their closed forms.
    auto solver = runCoupledToTime(/*nCells=*/16, /*k=*/3, /*tFinal=*/0.5);

    BOOST_CHECK_SMALL(uError(*solver, 0.5), 1e-4);
    BOOST_CHECK_SMALL(std::abs(solver->getSolution().Field(0) - manufacturedPsiExact(0.5)), 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
```

The fixture helpers (`makeCoupledSolver`, `mapSoln`, `GeometryProbeCase`, `BadlyDeclaredDifferentialField`, `runCoupledToTime`, `uError`) go at the top of this file. `GeometryProbeCase` is a `TransportSystem` whose `SigmaFn` returns `s.geom(0) * kappa * s.q(0)` and records `s.geom(0)` into a static vector; `BadlyDeclaredDifferentialField` is `ManufacturedField` with `differential = true` on its one DOF and an unchanged residual.

- [ ] **Step 2: Run test to verify it fails**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=coupled_residual_tests
```

Expected: compilation failure, `'class SystemSolver' has no member named 'setFieldModel'`.

- [ ] **Step 3: Give `SystemSolver` a field model**

In `SystemSolver.hpp`, beside the `problem` member:

```cpp
        // Null when no field model is configured, which is every existing run.
        // Held by shared_ptr because the adjoint solve and the Python layer
        // both need to reach it and neither owns the solver.
        std::shared_ptr<FieldModel> fieldModel = nullptr;
        Index nField = 0;
        Index nGeom = 0;

        void setFieldModel(std::shared_ptr<FieldModel> model);
```

and extend the `PhysicsNodes` struct at `:411` so the geometry travels with the states it was evaluated for:

```cpp
        struct PhysicsNodes
        {
            std::vector<Position> points;
            GlobalState states;
        };
```

— no change needed to the struct itself, because `GlobalState` now carries geometry (Task 3). Pass `nGeom` when constructing it.

`setFieldModel` sets `fieldModel`, `nField` and `nGeom`, and must be called before `initialize()`. Every `DGSoln` constructed in `SystemSolver` — `y`, `dydt`, `yJac`, `dydtJac`, `dydtComplete` in the constructor initialiser list at `:21`, and the local ones in `residual`, `solveHDGJac`, `setJacEvalY` and `mapDGtoSundials` — gains `nField` as its trailing argument. Grep for `nScalars, nAux)` in `SystemSolver.cpp` and handle every site.

- [ ] **Step 4: Add geometry evaluation**

In `SystemSolver.cpp`, a new member used by both `residual` and `evaluatePhysicsDerivatives`:

```cpp
// Fill the geometry rows of `states` from the field model, at the points those
// states were sampled on. Called once per residual and once per Jacobian
// update, never per variable: the geometry does not depend on which equation is
// being assembled.
//
// With Superconvergent = true the points are the k+2 star nodes rather than the
// k+1 basis nodes, which needs no special case here -- geometry is a function of
// (psi, x) and star nodes are just more x.
void SystemSolver::evaluateGeometry(DGSoln const &Y, std::vector<Position> const &points,
                                    GlobalState &states, Time t)
{
    if (!fieldModel)
        return;

    const Vector psi = Y.getField();
    Vector g(nGeom);
    for (size_t j = 0; j < points.size(); ++j)
    {
        g.setZero();
        fieldModel->Geometry(g, psi, points[j], t);
        states.setGeometry(static_cast<Index>(j), g);
    }
}
```

Call it in `residual` immediately after the `GlobalState` of physics states is built and **before** `ComputePhysics` is invoked, and in `evaluatePhysicsDerivatives` at `SystemSolver.cpp:634` immediately after `nodes` is constructed.

- [ ] **Step 5: Add the field rows to `residual`**

At the end of `SystemSolver::residual`, after the `nScalars` block at `:1248-1260` and before `return 0;`:

```cpp
    if (fieldModel)
    {
        // Sampled once, like the scalars: every field row sees the same state.
        const GlobalState fieldStates = Y_h.evalOnNodes();
        const GlobalState fieldStates_dt = dYdt_h.evalOnNodes();
        const Vector &weights = Integrator::getIntegrationWeights(Y_h.getBasis(), grid);

        Vector fieldRes = Vector::Zero(nField);
        fieldModel->FieldResidual(fieldRes, Vector(Y_h.getField()), Vector(dYdt_h.getField()),
                                  fieldStates, Y_h.getPoints(), weights, tres);
        res.getField() = fieldRes;
    }
```

- [ ] **Step 6: Extend the `id` vector and add the differential check**

In `Solver.cpp`, after the `nScalars` loop at `:193-199`:

```cpp
	for (Index f = 0; f < nField; ++f)
	{
		if (fieldModel->isFieldDOFDifferential(f))
			isDifferential.Field(f) = 1.0;
	}
```

and in `SystemSolver::initialize`, before `IDACalcIC` is reached, the refusal:

```cpp
    // A field DOF declared differential whose residual carries no d/dt is a
    // row every unknown of which IDA_YA_YDP_INIT holds fixed: no Newton
    // direction touches it, so the backtracking loop runs to exhaustion and
    // IDA reports IDA_LINESEARCH_FAIL (-13) -- a message about the linesearch
    // for a defect in the declaration. Ask which unknowns the row can reach,
    // here, where the answer names the DOF.
    if (fieldModel)
    {
        GlobalStateMatrix dR(nField), dRdot(nField);
        for (Index f = 0; f < nField; ++f)
        {
            dR.add(nCells, k, nVars, nScalars, nAux);
            dRdot.add(nCells, k, nVars, nScalars, nAux);
        }
        Matrix dRdpsi = Matrix::Zero(nField, nField);
        Matrix dRddpsidt = Matrix::Zero(nField, nField);

        fieldModel->FieldResidualPrime(dR, dRdot, dRdpsi, dRddpsidt,
                                       Vector(yJac.getField()), Vector(dydtJac.getField()),
                                       yJac.evalOnNodes(), yJac.getPoints(),
                                       Integrator::getIntegrationWeights(yJac.getBasis(), grid), t0);

        for (Index f = 0; f < nField; ++f)
            if (fieldModel->isFieldDOFDifferential(f) && dRddpsidt.row(f).isZero(0.0))
                throw std::invalid_argument(
                    "Field DOF '" + fieldModel->getSpec().dofs[f].name +
                    "' is declared differential but its residual row carries no time derivative. "
                    "IDACalcIC holds every differential value fixed, so this row is irreducible "
                    "and the initialisation would fail with IDA_LINESEARCH_FAIL.");
    }
```

- [ ] **Step 7: Seed the initial condition and reset per run**

In `setInitialConditions`, after the scalars are seeded:

```cpp
    if (fieldModel)
    {
        Vector psi0 = Vector::Zero(nField);
        fieldModel->InitialFieldValue(psi0);
        Y.getField() = psi0;
    }
```

and at the top of `SystemSolver::initialize`, before anything else:

```cpp
    // initialise() skips initialiseMatrices() when already initialised, so a
    // reused solver would otherwise keep whatever the field model cached from
    // the previous run -- the RF_cellwise trap.
    if (fieldModel)
        fieldModel->resetForRun();
```

- [ ] **Step 8: Run the coupled tests**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=coupled_residual_tests --log_level=message
```

Expected: 5 test cases, all passing. The end-to-end one is slow relative to the others because IDA has no coupling blocks yet and will take extra Newton iterations; that is expected and is what Task 8 fixes.

- [ ] **Step 9: Run the full suite and commit**

```sh
make test && make regression_tests
git add SystemSolver.hpp SystemSolver.cpp Solver.cpp \
        Tests/UnitTests/CoupledResidualTests.cpp Tests/UnitTests/Makefile
git commit -m "The field residual joins the DAE, and geometry reaches the physics"
```

---

### Task 7: Geometry derivative hooks on `TransportSystem`

**Files:**
- Modify: `TransportSystem.hpp` (three new virtuals beside `dSources_dScalars` at `:381`, and the batched `dSources` at `:277`)
- Modify: `PhysicsCases/AutodiffTransportSystem.{hpp,cpp}`
- Modify: `PyTransportSystem.hpp:35` (the override-name list) and the dispatchers
- Modify: `Python.cpp` (bind the three)
- Modify: `python/manta/__init__.pyi`
- Create: `Tests/UnitTests/GeometryDerivativeTests.cpp`
- Modify: `Tests/UnitTests/Makefile:22`

**Interfaces:**
- Consumes: `State::geom` (Task 3).
- Produces: `TransportSystem::dSigmaFn_dGeometry(Index, VectorRef, State const&, Position, Time)`, `dSources_dGeometry(...)`, `dAuxG_dGeometry(...)`, each with a default that leaves the out-parameter zeroed. Task 8 assembles from them.

- [ ] **Step 1: Write the failing test**

Create `Tests/UnitTests/GeometryDerivativeTests.cpp`:

```cpp
// The geometry derivative hooks. An absent hook is an identically zero block,
// which is the correct meaning of "this case does not read geometry" -- the
// same convention every other derivative out-parameter follows.
#include <boost/test/unit_test.hpp>

#include "../../PhysicsCases/AutodiffTransportSystem.hpp"
#include "../../TransportSystem.hpp"

BOOST_AUTO_TEST_SUITE(geometry_derivative_tests)

BOOST_AUTO_TEST_CASE(the_default_hooks_leave_the_block_zero)
{
    MinimalCase sys;               // implements only SigmaFn and Sources
    State s(1, 0, 0, 2);
    Vector out = Vector::Constant(2, 99.0);
    out.setZero();

    sys.dSigmaFn_dGeometry(0, out, s, 0.5, 0.0);
    BOOST_CHECK_EQUAL(out.norm(), 0.0);

    sys.dSources_dGeometry(0, out, s, 0.5, 0.0);
    BOOST_CHECK_EQUAL(out.norm(), 0.0);
}

BOOST_AUTO_TEST_CASE(a_case_that_overrides_them_is_dispatched_to)
{
    // GeometryDependentCase has sigma_hat = g0 * kappa * q, so
    // d(sigma_hat)/d(g0) = kappa * q and d/d(g1) = 0.
    GeometryDependentCase sys(/*kappa=*/2.5);
    State s(1, 0, 0, 2);
    s.q(0) = 3.0;

    Vector out = Vector::Zero(2);
    sys.dSigmaFn_dGeometry(0, out, s, 0.5, 0.0);
    BOOST_CHECK_CLOSE(out(0), 2.5 * 3.0, 1e-12);
    BOOST_CHECK_EQUAL(out(1), 0.0);
}

BOOST_AUTO_TEST_CASE(the_autodiff_layer_derives_them)
{
    // AutodiffTransportSystem widens its RealVector over the geometry slots, so
    // a case that writes Flux() in terms of them gets the derivative for free.
    // Checked against a central difference of the case's own Flux.
    AutodiffGeometryCase sys;
    State s(1, 0, 0, 1);
    s.q(0) = 1.25;
    s.geom(0) = 0.8;

    Vector analytic = Vector::Zero(1);
    sys.dSigmaFn_dGeometry(0, analytic, s, 0.5, 0.0);

    const double h = std::cbrt(std::numeric_limits<double>::epsilon());
    State sp = s, sm = s;
    sp.geom(0) += h;
    sm.geom(0) -= h;
    const double fd = (sys.SigmaFn(0, sp, 0.5, 0.0) - sys.SigmaFn(0, sm, 0.5, 0.0)) / (2 * h);

    BOOST_CHECK_CLOSE(analytic(0), fd, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
```

- [ ] **Step 2: Run test to verify it fails**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=geometry_derivative_tests
```

Expected: compilation failure, `'class TransportSystem' has no member named 'dSigmaFn_dGeometry'`.

- [ ] **Step 3: Add the hooks to `TransportSystem.hpp`**

Beside `dSources_dScalars` at `:381`:

```cpp
  /*
      Derivatives with respect to the geometry slots a field model supplies.

      Each `out` is length nGeometry and arrives zeroed, so a case that does not
      read geometry may leave all three unimplemented and contributes an
      identically zero coupling block -- which is exactly right, because it does
      not couple.

      These are the first factor of A1; the second, dGeometry/dpsi, is the field
      model's. Note that q has no geometry dependence (q = d_x u is a definition,
      not a physical relation) and neither do the trace rows, so there is no
      hook for either -- a geometry-dependent boundary condition is out of scope.
  */
  virtual void dSigmaFn_dGeometry(Index, VectorRef, const State &, Position, Time) {}
  virtual void dSources_dGeometry(Index, VectorRef, const State &, Position, Time) {}
  virtual void dAuxG_dGeometry(Index, VectorRef, const State &, Position, Time) {}
```

- [ ] **Step 4: Extend the autodiff layer**

In `AutodiffTransportSystem`, add the geometry slots to the `RealVector` the `Flux`/`Source` wrappers are built from, and implement the three hooks by differentiating with respect to that segment — the same pattern the existing `dSigmaFn_du`/`dSigmaFn_dq` implementations use.

- [ ] **Step 5: Extend the Python trampoline**

In `PyTransportSystem.hpp:35`, add `"dSigmaFn_dGeometry"`, `"dSources_dGeometry"` and `"dAuxG_dGeometry"` to the override-name list, and add three dispatchers following the `dSources_dScalars` pattern at `:631`. **Look overrides up with `override_for(name)`, never `method_overrides[name]`** — the latter default-constructs a null `py::function` and calling it segfaults. Bind them in `Python.cpp` beside `dSources_dScalars` at `:338`, and declare them in `python/manta/__init__.pyi` with `# type: ignore[override]`.

- [ ] **Step 6: Run everything**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=geometry_derivative_tests
cd ../.. && make test && make python && make stubs && make stubs-check && make typecheck && make python_tests
```

Expected: 3 new test cases passing; `stubs-check` clean.

- [ ] **Step 7: Commit**

```sh
git add TransportSystem.hpp PhysicsCases/AutodiffTransportSystem.hpp PhysicsCases/AutodiffTransportSystem.cpp \
        PyTransportSystem.hpp Python.cpp python/manta/__init__.pyi python/manta/_manta.pyi \
        Tests/UnitTests/GeometryDerivativeTests.cpp Tests/UnitTests/Makefile
git commit -m "Physics cases may differentiate with respect to geometry"
```

---

### Task 8: The coupling blocks and the exact Schur solve

**Files:**
- Modify: `Matrices.cpp` (add `dPhysics_dField_Mat` and `dPhysics_dField_StarMat`)
- Modify: `SystemSolver.hpp` (declarations, and the `A1`/`A2` storage)
- Modify: `SystemSolver.cpp:853` (`updateMatricesForJacSolve`), `:916` (`solveJacEq`)
- Modify: `ConfigSchema.cpp` (three keys)
- Modify: `SolverConfig.{hpp,cpp}`, `PyRunner.cpp` (the parameter table)
- Create: `Tests/UnitTests/FieldJacobianTests.cpp`
- Modify: `Tests/UnitTests/Makefile:22`
- Modify: `Tests/UnitTests/SolveJacTests.cpp`

**Interfaces:**
- Consumes: Tasks 5, 6, 7.
- Produces: `SystemSolver::assembleFieldCoupling(...)`, `SystemSolver::solveCoupledJacExact(N_Vector, N_Vector)`, and the config keys `FieldModel`, `FieldSolve`, `FieldSolveTolerance`, `FieldSolveMaxSweeps`.

- [ ] **Step 1: Write the failing test**

Create `Tests/UnitTests/FieldJacobianTests.cpp`:

```cpp
// The coupled Jacobian solve, checked the way SolveJacTests checks the
// uncoupled one: finite-difference the whole residual and require J dy = g.
//
// This is the ONLY test that can catch a wrong coupling block. A sign error in
// A1 or A2 leaves the answer correct -- the Jacobian is never assembled, so an
// error in it costs Newton speed and nothing else. MMS will not see it, and the
// regression suite will not see it.
#include <boost/test/unit_test.hpp>

#include "ManufacturedFields.hpp"
#include "../../SystemSolver.hpp"

BOOST_AUTO_TEST_SUITE(field_jacobian_tests)

BOOST_AUTO_TEST_CASE(the_exact_solve_inverts_a_finite_differenced_coupled_jacobian)
{
    auto solver = makeCoupledSolverAtState(/*nCells=*/6, /*k=*/2, ManufacturedFieldTag{});

    // J by central differences of residual(), including the field rows and
    // columns. Rank-deficient by exactly the number of Dirichlet boundaries,
    // because residual() does not write those rows -- the same caveat
    // SolveJacTests already carries.
    const Matrix J = finiteDifferenceCoupledJacobian(*solver);

    Vector dy = Vector::Random(J.cols());
    zeroDirichletRows(*solver, dy);
    const Vector g = J * dy;

    Vector recovered = Vector::Zero(J.cols());
    solver->solveCoupledJacExact(toNVector(g), toNVector(recovered));

    BOOST_CHECK_SMALL((recovered - dy).norm() / dy.norm(), 1e-7);
}

BOOST_AUTO_TEST_CASE(the_same_holds_for_a_multi_dof_field_block)
{
    auto solver = makeCoupledSolverAtState(6, 2, ManufacturedFieldVectorTag{});
    const Matrix J = finiteDifferenceCoupledJacobian(*solver);

    Vector dy = Vector::Random(J.cols());
    zeroDirichletRows(*solver, dy);
    const Vector g = J * dy;

    Vector recovered = Vector::Zero(J.cols());
    solver->solveCoupledJacExact(toNVector(g), toNVector(recovered));

    BOOST_CHECK_SMALL((recovered - dy).norm() / dy.norm(), 1e-7);
}

BOOST_AUTO_TEST_CASE(a_sign_error_in_a1_would_be_caught)
{
    // Guard against the test passing vacuously: perturb A1 and require the
    // check to fail. If this does not fail, the coupling is not being exercised
    // and neither of the two tests above means anything.
    //
    // The perturbation reaches A1_cellwise directly rather than through a
    // production mutator: this is a -DTEST build, where MANTA_TEST_PRIVATE has
    // widened SystemSolver's private members to public. Nothing test-only is
    // added to the shipped class.
    auto solver = makeCoupledSolverAtState(6, 2, ManufacturedFieldTag{});
    solver->updateMatricesForJacSolve();
    for (auto &block : solver->A1_cellwise)
        block *= -1.0;

    const Matrix J = finiteDifferenceCoupledJacobian(*solver);
    Vector dy = Vector::Random(J.cols());
    zeroDirichletRows(*solver, dy);
    const Vector g = J * dy;

    Vector recovered = Vector::Zero(J.cols());
    solver->solveCoupledJacExact(toNVector(g), toNVector(recovered));

    BOOST_CHECK_GT((recovered - dy).norm() / dy.norm(), 1e-3);
}

BOOST_AUTO_TEST_CASE(selecting_the_exact_solve_warns)
{
    CapturedOutput out;
    auto solver = makeCoupledSolverWithFieldSolve("exact");
    solver->initialize();
    BOOST_CHECK(out.stderrContains("FieldSolve = exact"));
    BOOST_CHECK(out.stderrContains("verification"));
}

BOOST_AUTO_TEST_SUITE_END()
```

- [ ] **Step 2: Run test to verify it fails**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=field_jacobian_tests
```

Expected: compilation failure, no `solveCoupledJacExact`.

- [ ] **Step 3: Assemble A1**

In `Matrices.cpp`, beside `dSources_dScalars_Mat` at `:140`, following its structure exactly — including the `InterpolateOntoBasis` form, which is what the residual builds and therefore what the derivative must match:

```cpp
// One cell's block of A1: the derivative of the sigma, u and aux residual rows
// with respect to the field DOFs, via the chain rule
//
//     d(row)/d(psi_m) = sum_g d(row)/d(geometry_g) * d(geometry_g)/d(psi_m)
//
// The first factor comes from the case's geometry hooks, the second from the
// field model. q rows and trace rows have no geometry dependence.
//
// Shape: ( (3 nVars + nAux) (k+1), nField ), laid out [ sigma | q | u | aux ]
// to match assembleCellMatrix's row ordering, with the q block left zero.
void SystemSolver::dPhysics_dField_Mat(Matrix &mat, DGSoln const &Y, Index intervalIndex, Time tEval)
{
	Interval const &I(grid[intervalIndex]);

	assert(mat.rows() == (3 * nVars + nAux) * (k + 1));
	assert(mat.cols() == nField);

	mat.setZero();

	const Vector psi = Y.getField();

	Values dXdG(nGeom);
	Matrix dGdPsi(nGeom, nField);
	Matrix nodal(nField, k + 1);

	// sigma rows: res.sigma = A sigma + Pi(sigmaHat), so the derivative is
	// Pi( d sigmaHat / d psi ).
	for (Index XVar = 0; XVar < nVars; XVar++)
	{
		for (Index j = 0; j < k + 1; ++j)
		{
			dXdG.setZero();
			dGdPsi.setZero();
			double x_j = I.fromRef(Y.getBasis().Nodes(j));
			State s = Y.evalOnNode(intervalIndex, j);
			problem->dSigmaFn_dGeometry(XVar, dXdG, s, x_j, tEval);
			fieldModel->dGeometry_dpsi(dGdPsi, psi, x_j, tEval);
			nodal.col(j) = dGdPsi.transpose() * dXdG;
		}
		for (Index m = 0; m < nField; ++m)
		{
			Vector vals = nodal.row(m).transpose();
			mat.block(XVar * (k + 1), m, k + 1, 1) = Y.getBasis().InterpolateOntoBasis(I, vals);
		}
	}

	// u rows, from dSources_dGeometry, offset by 2 * nVars * (k+1); and aux
	// rows, from dAuxG_dGeometry, offset by 3 * nVars * (k+1). Both follow the
	// same three-step shape as the sigma block above: evaluate at the nodes,
	// contract with dGeometry/dpsi, project with InterpolateOntoBasis.
	// [repeat the loop for each, with the appropriate hook and row offset]
}
```

Add a `dPhysics_dField_StarMat` alongside it, following `dSources_dScalars_StarMat` at `:189` — same body with `A9 * Vector(...)` in place of `InterpolateOntoBasis`, the star node count `k + 2`, and `states[g]`/`points[g]` in place of `Y.evalOnNode`.

- [ ] **Step 4: Assemble A2 and B**

In `SystemSolver.cpp`, beside `assembleScalarCoupling` at `:794`:

```cpp
// The field coupling blocks. A2 comes back from FieldResidualPrime in the shape
// ScalarGPrime uses -- one GlobalStateMatrix row per field row -- and is folded
// into a flat (nField, DOF) operator here; B is the model's own block, which the
// model factorises for itself.
void SystemSolver::assembleFieldCoupling(DGSoln const &Y, DGSoln const &Ydot,
                                         PhysicsNodes const &nodes, Time tEval,
                                         double alphaValue)
{
    GlobalStateMatrix dR(nField), dRdot(nField);
    for (Index f = 0; f < nField; ++f)
    {
        dR.add(nCells, k, nVars, nScalars, nAux);
        dRdot.add(nCells, k, nVars, nScalars, nAux);
    }
    Matrix dRdpsi = Matrix::Zero(nField, nField);
    Matrix dRddpsidt = Matrix::Zero(nField, nField);

    fieldModel->FieldResidualPrime(dR, dRdot, dRdpsi, dRddpsidt,
                                   Vector(Y.getField()), Vector(Ydot.getField()),
                                   Y.evalOnNodes(), Y.getPoints(),
                                   Integrator::getIntegrationWeights(Y.getBasis(), grid), tEval);

    // A2 rows, as DGSoln views over the N_Vectors a2[f], with the alpha
    // weighting the scalar path already uses for its w vectors.
    for (Index f = 0; f < nField; ++f)
    {
        DGSoln row(nVars, grid, k, N_VGetArrayPointer(a2[f]), nScalars, nAux, nField);
        row.zeroCoeffs();
        for (Index i = 0; i < nCells; ++i)
            for (Index l = 0; l < k + 1; ++l)
            {
                const Index g = i * (k + 1) + l;
                for (Index v = 0; v < nVars; ++v)
                {
                    row.sigma(v).getCoeff(i).second(l) = dR[f][g].sigma(v) + alphaValue * dRdot[f][g].sigma(v);
                    row.q(v).getCoeff(i).second(l)     = dR[f][g].q(v)     + alphaValue * dRdot[f][g].q(v);
                    row.u(v).getCoeff(i).second(l)     = dR[f][g].u(v)     + alphaValue * dRdot[f][g].u(v);
                }
                for (Index a = 0; a < nAux; ++a)
                    row.Aux(a).getCoeff(i).second(l) = dR[f][g].phi(a) + alphaValue * dRdot[f][g].phi(a);
            }
    }

    fieldModel->updateFieldJacobian(dRdpsi, dRddpsidt, alphaValue);

    // A1 columns, cell by cell.
    for (Index i = 0; i < nCells; ++i)
    {
        if (superconvergent)
            dPhysics_dField_StarMat(A1_cellwise[i], nodes.states, nodes.points, i, tEval);
        else
            dPhysics_dField_Mat(A1_cellwise[i], yJac, i, tEval);
    }
}
```

Call it from `updateMatricesForJacSolve` at `:877`, in a `if (fieldModel)` block beside the `nScalars` one. Allocate `a2` (an `N_Vector[nField]`) and `A1_cellwise` (a `std::vector<Matrix>`) in `initialiseMatrices` beside `v` and `w` at `:46`, and free `a2` in `destroySundials` beside `:67`.

- [ ] **Step 5: Implement the exact Schur solve**

```cpp
// Exact Schur complement onto psi:
//
//     ( B - A2 A^-1 A1 ) dpsi = r2 - A2 A^-1 r1
//     A dx = r1 - A1 dpsi
//
// with A the existing transport operator -- HDG condensation plus the scalar
// bordering, i.e. the whole of solveJacEq's uncoupled path.
//
// Costs nField applications of A^-1, so it is affordable only for a small field
// block. It is here because SolveJacTests' style -- finite-difference the
// residual, require J dy = g -- only extends to the coupled system if an exact
// solve exists, and because it is the oracle the iterative path in Task 9 is
// checked against.
void SystemSolver::solveCoupledJacExact(N_Vector res_g, N_Vector delY)
{
    DGSoln rhs(nVars, grid, k, N_VGetArrayPointer(res_g), nScalars, nAux, nField);
    DGSoln out(nVars, grid, k, N_VGetArrayPointer(delY), nScalars, nAux, nField);

    // AinvA1: one transport solve per field DOF.
    N_Vector col = N_VClone(delY);
    Matrix S = Matrix::Zero(nField, nField);
    std::vector<N_Vector> AinvA1(nField);

    for (Index m = 0; m < nField; ++m)
    {
        AinvA1[m] = N_VClone(delY);
        scatterA1Column(m, col);              // col <- A1(:, m), zero in the field block
        solveTransportJac(col, AinvA1[m]);    // the uncoupled path, unchanged
        for (Index f = 0; f < nField; ++f)
            S(f, m) = N_VDotProd(a2[f], AinvA1[m]);
    }

    Matrix Bdense = Matrix::Zero(nField, nField);
    for (Index m = 0; m < nField; ++m)
    {
        Vector e = Vector::Unit(nField, m), Be = Vector::Zero(nField);
        fieldModel->applyB(Be, e);
        Bdense.col(m) = Be;
    }
    const Matrix Schur = Bdense - S;

    N_Vector Ainv_r1 = N_VClone(delY);
    solveTransportJac(res_g, Ainv_r1);

    Vector r2 = rhs.getField();
    for (Index f = 0; f < nField; ++f)
        r2(f) -= N_VDotProd(a2[f], Ainv_r1);

    Vector dpsi = Schur.partialPivLu().solve(r2);

    // dx = A^-1 r1 - sum_m dpsi_m (A^-1 A1)(:, m)
    N_VScale(1.0, Ainv_r1, delY);
    for (Index m = 0; m < nField; ++m)
        N_VLinearSum(1.0, delY, -dpsi(m), AinvA1[m], delY);
    out.getField() = dpsi;

    for (Index m = 0; m < nField; ++m)
        N_VDestroy(AinvA1[m]);
    N_VDestroy(col);
    N_VDestroy(Ainv_r1);
}
```

Rename the existing body of `solveJacEq` to `solveTransportJac` and have `solveJacEq` dispatch: no field model, call `solveTransportJac`; `FieldSolve = exact`, call `solveCoupledJacExact`.

**Delete Task 6's block-Jacobi field solve as part of this rename — do not carry it across.** Task 6 added the field model's own diagonal block (`updateFieldBlock`, and a `solveB` call inside `solveJacEq`) because without it `dpsi` is *structurally* zero: `solveHDGJac` begins with `delYVec.setZero()` over the whole increment and writes nothing past `lambda`, so no Newton direction for `psi` exists at all — and IDA cannot report it, because `acor = y_n - y_pred` is identically zero for an unknown whose predictor is constant. That made it a prerequisite for Task 6 running, and it makes it an obstacle here. Two concrete reasons it must go rather than stay:

* `solveTransportJac` is called *inside* `solveCoupledJacExact` (once per field DOF for the `A1` columns, once for `r1`). A block-Jacobi `psi` solve living in there would contaminate every inner solve. It happens to be harmless today — `a2`'s field entries are zeroed by `row.zeroCoeffs()` and `out.getField() = dpsi` overwrites the result — but harmless-by-accident is not a property to ship into a Schur complement.
* `updateFieldBlock` duplicates the whole prologue of `assembleFieldCoupling`: the same `dR`/`dRdot` construction, the same `FieldResidualPrime` call, the same `updateFieldJacobian`. Keeping both evaluates `FieldResidualPrime` twice per Jacobian update.

So `solveTransportJac` must be the *uncoupled* operator — HDG condensation plus the scalar bordering, and nothing about the field.

- [ ] **Step 6: Add the config keys and the warning**

In `ConfigSchema.cpp`, beside the entries at `:57`:

```cpp
        {"FieldModel", {}, Type::String, Category::ProblemSelection, false, false, std::string{},
         "Name of a registered magnetic-field model to couple to; absent means no coupling."},
        {"FieldSolve", {}, Type::String, Category::Solver, false, false, std::string{"iterative"},
         "How the coupled Jacobian is solved: iterative (block Gauss-Seidel, the default) "
         "or exact (Schur complement onto the field block; a verification tool, see docs)."},
        {"FieldSolveTolerance", {}, Type::Double, Category::Solver, false, false, 1e-8,
         "Convergence tolerance for FieldSolve = iterative."},
        {"FieldSolveMaxSweeps", {}, Type::Int, Category::Solver, false, false, 20,
         "Sweep cap for FieldSolve = iterative."},
```

Carry all four through `SolverConfig` and `applySolverConfig` — **the single point at which a configuration reaches the solver**, so a `set*` call dropped from it un-configures both surfaces at once. Add them to `PyRunner.cpp`'s declarative parameter table.

The warning, emitted once from `applySolverConfig`:

```cpp
    if (config.fieldSolve == "exact")
        logmsg<LOG_LEVEL::WARNING>(
            "FieldSolve = exact costs one full transport solve per field degree of freedom on "
            "every Jacobian solve, so the linear algebra is {} times the iterative path's. It is "
            "a verification tool and is not intended for production runs.",
            solver.getFieldDOF());
```

- [ ] **Step 7: Extend `SolveJacTests.cpp`**

Add a coupled case to the existing suite so the uncoupled and coupled checks live together and share the finite-difference helper.

- [ ] **Step 8: Run and commit**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=field_jacobian_tests
cd ../.. && make test && make regression_tests && ./MaNTA --list-options | grep -i field
```

Expected: 4 new cases passing, including the guard that a flipped `A1` *fails*; four new keys listed.

```sh
git add Matrices.cpp SystemSolver.hpp SystemSolver.cpp ConfigSchema.cpp SolverConfig.hpp SolverConfig.cpp PyRunner.cpp \
        Tests/UnitTests/FieldJacobianTests.cpp Tests/UnitTests/SolveJacTests.cpp Tests/UnitTests/Makefile
git commit -m "Assemble the field coupling, and solve it exactly for verification"
```

---

### Task 9: Block Gauss–Seidel, the production path

**Files:**
- Modify: `SystemSolver.cpp` (`solveCoupledJacIterative`, and the dispatch in `solveJacEq`)
- Modify: `SystemSolver.hpp`
- Modify: `Tests/UnitTests/FieldJacobianTests.cpp`

**Interfaces:**
- Consumes: Task 8's `solveTransportJac`, `a2`, `A1_cellwise`, `solveCoupledJacExact`.
- Produces: `SystemSolver::solveCoupledJacIterative(N_Vector, N_Vector)`.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UnitTests/FieldJacobianTests.cpp`:

```cpp
BOOST_AUTO_TEST_CASE(the_iterative_solve_agrees_with_the_exact_one)
{
    // Not to roundoff: the iterative path stops at FieldSolveTolerance, so the
    // agreement is to that tolerance and no better. What this pins is that the
    // two are solving the same system -- a sign error in either path's use of
    // A1 or A2 separates them immediately.
    auto solver = makeCoupledSolverAtState(6, 2, ManufacturedFieldTag{});
    N_Vector g = randomRHS(*solver);

    N_Vector exact = N_VClone(g), iterative = N_VClone(g);
    solver->solveCoupledJacExact(g, exact);
    solver->solveCoupledJacIterative(g, iterative);

    BOOST_CHECK_SMALL(relativeDifference(exact, iterative), 1e-6);
}

BOOST_AUTO_TEST_CASE(the_iterative_solve_agrees_for_a_multi_dof_block_too)
{
    auto solver = makeCoupledSolverAtState(6, 2, ManufacturedFieldVectorTag{});
    N_Vector g = randomRHS(*solver);

    N_Vector exact = N_VClone(g), iterative = N_VClone(g);
    solver->solveCoupledJacExact(g, exact);
    solver->solveCoupledJacIterative(g, iterative);

    BOOST_CHECK_SMALL(relativeDifference(exact, iterative), 1e-6);
}

BOOST_AUTO_TEST_CASE(a_coupled_run_on_the_iterative_path_reaches_the_manufactured_solution)
{
    // The iterative solve is an *approximation to the Jacobian*, so an
    // under-converged sweep costs Newton iterations and nothing else. This
    // checks the answer is unmoved: same tolerance as the exact path's run.
    auto solver = runCoupledToTime(16, 3, 0.5, /*fieldSolve=*/"iterative");
    BOOST_CHECK_SMALL(uError(*solver, 0.5), 1e-4);
    BOOST_CHECK_SMALL(std::abs(solver->getSolution().Field(0) - manufacturedPsiExact(0.5)), 1e-5);
}
```

- [ ] **Step 2: Run test to verify it fails**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=field_jacobian_tests/the_iterative_solve_agrees_with_the_exact_one
```

Expected: compilation failure, no `solveCoupledJacIterative`.

- [ ] **Step 3: Implement it**

```cpp
// Block Gauss-Seidel on the coupled Jacobian:
//
//     A dx^{k+1}   = r1 - A1 dpsi^k
//     B dpsi^{k+1} = r2 - A2 dx^{k+1}
//
// One transport solve and one field solve per sweep, against the exact path's
// nField + 1 transport solves per Jacobian solve.
//
// This is safe in a way a lagged *residual* would not be. The Jacobian is never
// assembled and IDA tolerates an inexact linear solve, so an error here costs
// Newton speed rather than correctness -- which is why Serino et al.'s
// block-triangular preconditioners can drop the Schur complement outright and
// still converge. Accuracy comes from the residual, which is exact.
void SystemSolver::solveCoupledJacIterative(N_Vector res_g, N_Vector delY)
{
    DGSoln rhs(nVars, grid, k, N_VGetArrayPointer(res_g), nScalars, nAux, nField);
    DGSoln out(nVars, grid, k, N_VGetArrayPointer(delY), nScalars, nAux, nField);

    N_Vector work = N_VClone(delY);
    Vector dpsi = Vector::Zero(nField);
    Vector dpsiPrev = Vector::Zero(nField);

    for (Index sweep = 0; sweep < fieldSolveMaxSweeps; ++sweep)
    {
        // work <- r1 - A1 dpsi
        N_VScale(1.0, res_g, work);
        subtractA1Times(dpsi, work);
        DGSoln workMap(nVars, grid, k, N_VGetArrayPointer(work), nScalars, nAux, nField);
        workMap.getField().setZero();

        solveTransportJac(work, delY);

        // dpsi <- B^-1 ( r2 - A2 dx )
        Vector r2 = rhs.getField();
        for (Index f = 0; f < nField; ++f)
            r2(f) -= N_VDotProd(a2[f], delY);

        dpsiPrev = dpsi;
        fieldModel->solveB(dpsi, r2);

        if ((dpsi - dpsiPrev).norm() <= fieldSolveTolerance * std::max(1.0, dpsi.norm()))
            break;
    }

    out.getField() = dpsi;
    N_VDestroy(work);
}
```

`subtractA1Times(dpsi, work)` walks the cells and subtracts `A1_cellwise[i] * dpsi` from the corresponding `[sigma | q | u | aux]` segment of `work`. Note the iteration does **not** fail on non-convergence: it returns its last iterate, which for a Jacobian is legitimate. Task 10 makes the adjoint's version do the opposite.

- [ ] **Step 4: Dispatch on the config key**

In `solveJacEq`: no field model → `solveTransportJac`; `exact` → `solveCoupledJacExact`; otherwise → `solveCoupledJacIterative`.

- [ ] **Step 5: Run and commit**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=field_jacobian_tests
cd ../.. && make test && make regression_tests
git add SystemSolver.hpp SystemSolver.cpp Tests/UnitTests/FieldJacobianTests.cpp
git commit -m "Iterate the coupled Jacobian solve, which is the production path"
```

---

### Task 10: The adjoint transposes

**Files:**
- Modify: `SystemSolver.cpp` (`initializeMatricesForAdjointSolve`, `solveAdjointState`, `computeAdjointGradients`)
- Modify: `AdjointVectors.cpp`
- Create: `Tests/UnitTests/FieldAdjointTests.cpp`
- Modify: `Tests/UnitTests/Makefile:22`
- Modify: `python/Tests/test_adjoint.py`

**Interfaces:**
- Consumes: Tasks 8 and 9.
- Produces: `SystemSolver::solveCoupledAdjointExact`, `SystemSolver::solveCoupledAdjointIterative`.

- [ ] **Step 1: Write the failing test**

Create `Tests/UnitTests/FieldAdjointTests.cpp`:

```cpp
// The adjoint through the coupling.
//
// This is the one place a missing block is *silently wrong* rather than merely
// slow: initializeMatricesForAdjointSolve stores the forward blocks'
// transpose, so a coupling block present in the forward Jacobian and absent
// from the adjoint gives a wrong gradient with a perfectly good G. The
// dSigma/dPhi block was missing here for exactly that reason and cost nothing
// visible until test_adjoint_aux.py was written.
#include <boost/test/unit_test.hpp>

#include "ManufacturedFields.hpp"
#include "../../SystemSolver.hpp"

BOOST_AUTO_TEST_SUITE(field_adjoint_tests)

BOOST_AUTO_TEST_CASE(the_gradient_matches_a_finite_difference_through_the_coupling)
{
    // The objective depends on the state only through quantities the field
    // model influences, so a zero coupling block gives a visibly wrong answer
    // rather than a slightly wrong one.
    auto solver = makeCoupledAdjointProblem(/*nCells=*/8, /*k=*/2);
    const Vector analytic = solver->getAdjointGradients();
    const Vector fd = finiteDifferenceGradient(*solver, /*h=*/1e-6);

    BOOST_CHECK_SMALL((analytic - fd).norm() / fd.norm(), 1e-5);
}

BOOST_AUTO_TEST_CASE(dropping_the_coupling_block_makes_the_gradient_wrong)
{
    // The guard: without it, a zero A1^T would pass the test above if the
    // objective happened not to see the coupling.
    //
    // Reached directly through MANTA_TEST_PRIVATE, as in field_jacobian_tests:
    // this is a -DTEST build, so no test-only method is added to SystemSolver.
    auto solver = makeCoupledAdjointProblem(8, 2);
    solver->initializeMatricesForAdjointSolve();
    for (auto &block : solver->A1_transpose_cellwise)
        block.setZero();
    const Vector analytic = solver->getAdjointGradients();
    const Vector fd = finiteDifferenceGradient(*solver, 1e-6);

    BOOST_CHECK_GT((analytic - fd).norm() / fd.norm(), 1e-2);
}

BOOST_AUTO_TEST_CASE(an_unconverged_adjoint_iteration_throws_rather_than_returning)
{
    // The asymmetry with the forward path, and the reason it exists: an
    // under-converged forward Jacobian costs Newton iterations, an
    // under-converged adjoint returns a wrong gradient with a good G. So the
    // adjoint sweep may not silently return its last iterate.
    auto solver = makeCoupledAdjointProblem(8, 2);
    solver->setFieldSolveMaxSweeps(1);
    solver->setFieldSolveTolerance(1e-14);

    BOOST_CHECK_THROW(solver->getAdjointGradients(), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
```

- [ ] **Step 2: Run test to verify it fails**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=field_adjoint_tests
```

Expected: the first test fails — a wrong gradient, because the coupling blocks are absent from the adjoint matrix.

- [ ] **Step 3: Add the transposes**

In `initializeMatricesForAdjointSolve`, alongside the existing `M.transpose()` storage, store `A1_cellwise[i].transpose()` and keep the `a2` rows available as columns. Add the adjoint solves:

```cpp
// The transpose of solveCoupledJacExact. The block elimination runs in the
// other order, so the Schur complement is
//
//     ( B^T - A1^T A^-T A2^T ) dpsi = ...
//
// which is why FieldModel requires applyBTranspose and solveBTranspose rather
// than only the forward pair: a model supplying one direction cannot be
// silently accommodated here.
void SystemSolver::solveCoupledAdjointExact(N_Vector res_g, N_Vector delY);

// The transpose sweep. Unlike the forward iteration, this one THROWS on
// non-convergence instead of returning its last iterate.
void SystemSolver::solveCoupledAdjointIterative(N_Vector res_g, N_Vector delY)
{
    // ... sweeps as in solveCoupledJacIterative, with A1^T and A2^T swapped and
    // solveBTranspose in place of solveB ...
    if (!converged)
        throw std::runtime_error(
            "The coupled adjoint solve did not converge in " + std::to_string(fieldSolveMaxSweeps) +
            " sweeps (residual " + std::to_string(residualNorm) + " against a tolerance of " +
            std::to_string(fieldSolveTolerance) + "). Unlike the forward Jacobian, an "
            "under-converged adjoint returns a wrong gradient with a correct objective, so this "
            "is an error rather than a warning. Raise FieldSolveMaxSweeps or use FieldSolve = exact.");
}
```

Extend `computeAdjointGradients` so `G_p` picks up the field block's contribution, and `AdjointVectors.cpp` so anything sized per-DOF accounts for `nField`.

- [ ] **Step 4: Add a Python-level check**

Add a coupled case to `python/Tests/test_adjoint.py` mirroring the existing gradient-versus-finite-difference test.

- [ ] **Step 5: Run and commit**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=field_adjoint_tests
cd ../.. && make test && make python_tests
git add SystemSolver.cpp SystemSolver.hpp AdjointVectors.cpp \
        Tests/UnitTests/FieldAdjointTests.cpp Tests/UnitTests/Makefile python/Tests/test_adjoint.py
git commit -m "The adjoint transposes the coupling, and refuses to guess"
```

---

### Task 11: The order study

**Files:**
- Modify: `Tests/UnitTests/MMSHarness.hpp` (a coupled sweep)
- Create: `Tests/UnitTests/MMSFieldTests.cpp`
- Modify: `Tests/UnitTests/Makefile:22`
- Modify: `Tests/README.md`

**Interfaces:**
- Consumes: Tasks 5–10.
- Produces: measured orders, recorded in `Tests/README.md`.

- [ ] **Step 1: Write the failing test**

Create `Tests/UnitTests/MMSFieldTests.cpp`:

```cpp
// Order of accuracy with the field coupled. This is the test that catches a
// sign error in the *equations* -- a wrong A1 or A2 in the residual converges
// at the right rate to the wrong function, so only a closed-form comparison
// sees it. A Jacobian error is invisible here by construction.
//
// Read LOCAL orders, not the least-squares slope: a fit averages a changing
// rate away, which is how the nonlinear-flux superconvergence breakdown stayed
// invisible to n <= 32.
#include <boost/test/unit_test.hpp>

#include "MMSHarness.hpp"
#include "ManufacturedFields.hpp"

BOOST_AUTO_TEST_SUITE(mms_field_tests)

BOOST_AUTO_TEST_CASE(the_coupled_problem_converges_at_k_plus_one_in_u)
{
    for (Index k = 1; k <= 3; ++k)
    {
        const Rates r = solveCoupledAndMeasure<ManufacturedField>(k, {4, 8, 16, 32});
        BOOST_TEST_MESSAGE("k = " << k << " local orders in u: " << format(r.localU));
        for (double o : r.localU)
            BOOST_CHECK_GT(o, k + 1 - 0.25);
    }
}

BOOST_AUTO_TEST_CASE(psi_converges_too)
{
    // psi = Int u dx, so its error is the quadrature error of an O(h^{k+1})
    // function and should fall at least as fast.
    const Rates r = solveCoupledAndMeasure<ManufacturedField>(2, {4, 8, 16, 32});
    BOOST_TEST_MESSAGE("local orders in psi: " << format(r.localExtra));
    for (double o : r.localExtra)
        BOOST_CHECK_GT(o, 2.75);
}

BOOST_AUTO_TEST_CASE(the_multi_dof_field_converges_the_same_way)
{
    const Rates r = solveCoupledAndMeasure<ManufacturedFieldVector>(2, {4, 8, 16, 32});
    BOOST_TEST_MESSAGE("multi-DOF local orders in u: " << format(r.localU));
    for (double o : r.localU)
        BOOST_CHECK_GT(o, 2.75);
}

BOOST_AUTO_TEST_CASE(superconvergent_coupling_reaches_k_plus_two)
{
    // Geometry is a function of (psi, x) and star nodes are just more x, so
    // this should work through ComputePhysics's states.size() loop with no
    // special case. If it does not, the flag must throw with a field model
    // attached rather than silently losing the extra order -- see Step 5.
    const Rates r = solveCoupledAndMeasure<ManufacturedField>(2, {4, 8, 16, 32}, /*superconvergent=*/true);
    BOOST_TEST_MESSAGE("local orders in u*: " << format(r.localStarOn));
    for (double o : r.localStarOn)
        BOOST_CHECK_GT(o, 3.75);
}

BOOST_AUTO_TEST_SUITE_END()
```

- [ ] **Step 2: Run test to verify it fails**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=mms_field_tests
```

Expected: compilation failure, no `solveCoupledAndMeasure`.

- [ ] **Step 3: Extend the harness**

Add `solveCoupledAndMeasure<FieldModelType>(Index k, std::vector<Index> const &nCells, bool superconvergent = false)` to `MMSHarness.hpp`, returning the existing `Rates` struct with `localExtra` carrying the field error. It builds a `SystemSolver`, attaches the field model, integrates to a fixed `t`, and measures `||u - u_exact||` and `|psi - psi_exact|`.

- [ ] **Step 4: Write the manufactured source**

The transport equation solved is `u_t - d_x[ g(x; psi) kappa u_x ] = S`, with the minus sign the stored-sigma convention gives. With `u = sin(pi x)(1+t)`, `g = 1 + psi c(x)` and `c(x) = cos(pi x)`:

```
S(x,t) = sin(pi x)
       + kappa (1+t) [ (1 + psi cos(pi x)) pi^2 sin(pi x)
                     + psi pi^2 sin(pi x) cos(pi x)   ... ]
```

Derive it symbolically and check it: a unit test that evaluates `u_t - d_x[g kappa u_x] - S` at a scatter of `(x,t)` and requires it below `1e-10` costs nothing and catches an algebra slip that would otherwise appear as a convergence rate one lower than expected.

- [ ] **Step 5: Handle the superconvergent case honestly**

Run the fourth test. If `u*` reaches `k+2`, keep the assertion. If it does not, replace the test with one asserting that `Superconvergent = true` together with a field model **throws**, and record the measured rates in `Tests/README.md` — following the precedent that spatial adjoint parameters with `Superconvergent = true` throw rather than guessing.

- [ ] **Step 6: Record the numbers and commit**

Add a section to `Tests/README.md` with the measured local orders, exactly as the existing superconvergence tables do, and say which assertions are pinned to them.

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=mms_field_tests --log_level=message
cd ../.. && make test
git add Tests/UnitTests/MMSHarness.hpp Tests/UnitTests/MMSFieldTests.cpp Tests/UnitTests/Makefile Tests/README.md
git commit -m "Measure the coupled problem's order of accuracy, and record it"
```

---

### Task 12: Output, restart, docs, and the zero-coupling invariant

**Files:**
- Modify: `NetCDFIO.{hpp,cpp}`, `Solver.cpp` (output and restart)
- Modify: `docs/formulation.rst`, `docs/physics_interface.rst`, `docs/configuration.rst`, `docs/running.rst`
- Modify: `FEATURES.md`, `CLAUDE.md`
- Create: `Tests/RegressionTests/coupled-field.conf` and its reference
- Modify: `Tests/UnitTests/SolverLifecycleTests.cpp`

**Interfaces:**
- Consumes: everything.
- Produces: the documented, serialising feature.

- [ ] **Step 1: Write the failing test**

Add to `Tests/UnitTests/SolverLifecycleTests.cpp`:

```cpp
BOOST_AUTO_TEST_CASE(a_coupled_solver_reused_matches_a_fresh_one_bit_for_bit)
{
    // At exactly zero tolerance, like the uncoupled version. The tolerance is
    // the point: the last defect here left the second run completing,
    // plausible, and wrong in the eleventh digit. A field model that caches
    // anything across runs and does not reset it fails here and nowhere else.
    auto reused = integrateTwice(makeCoupledSolver(8, 2));
    auto fresh = integrateOnce(makeCoupledSolver(8, 2));
    BOOST_CHECK_EQUAL(maxAbsDifference(reused, fresh), 0.0);
}

BOOST_AUTO_TEST_CASE(psi_round_trips_through_a_restart)
{
    auto solver = runCoupledToTime(8, 2, 0.25);
    writeRestart(*solver, "coupled-restart-test.restart.nc");

    auto restarted = restartFrom("coupled-restart-test.restart.nc");
    BOOST_CHECK_SMALL(std::abs(restarted->getSolution().Field(0) - solver->getSolution().Field(0)), 1e-12);
}
```

- [ ] **Step 2: Run test to verify it fails**

```sh
cd Tests/UnitTests && make -j && ./UnitTests --run_test=solver_lifecycle_tests
```

Expected: the restart test fails — `psi` is not serialised.

- [ ] **Step 3: Write `psi` and the geometry to netCDF**

Add a group named for the field model carrying one variable per `FieldDOF` and per `FieldSlot`, with the names, descriptions and units the spec declares, plus the declared `label` as an attribute so a run records what its `x` meant. Gate every write on `WriteOutput`, as `Solver.cpp` already does for the rest.

- [ ] **Step 4: Extend the restart format**

Serialise the field block in `<stem>.restart.nc` and read it back. Restarting is already fragile at tight tolerances with `nAux > 0` and `psi` adds to that, so the regression round-trip case runs at the tightest tolerance that completes and records which that is.

- [ ] **Step 5: Prove the zero-coupling invariant**

The strongest guard available, and the one that must be run by hand rather than asserted:

```sh
git stash
make MaNTA && cp MaNTA /tmp/MaNTA-baseline
git stash pop
make MaNTA

for conf in Tests/RegressionTests/*.conf; do
    /tmp/MaNTA-baseline "$conf" && mv "$(basename "${conf%.conf}").nc" /tmp/baseline-$(basename "${conf%.conf}").nc
    ./MaNTA "$conf"
    cmp "/tmp/baseline-$(basename "${conf%.conf}").nc" "$(basename "${conf%.conf}").nc" \
      || echo "DIFFERS: $conf"
done
```

Expected: no output. Not "within 1e-2" — the regression suite's tolerance is far too loose to see a change of this kind, which is exactly how the `zeroFlux` reimplementation was verified.

- [ ] **Step 6: Record that there is no coupled regression case**

There is deliberately **no** `Tests/RegressionTests/coupled-field.conf`. A regression case would need a field model registered in `PhysicsCases/` and therefore present in the shipped binary, and the only model that exists is a manufactured fixture with no physics in it. Add a paragraph to `Tests/README.md`'s coverage section saying so, and saying what does cover the coupled path instead: the MMS order study (Task 11), the coupled Jacobian check (Task 8), the adjoint gradient check (Task 10) and the restart round trip (Step 4 above).

Name the gap this leaves, rather than implying there is none: nothing exercises the coupled path through a `.conf` file, so the config plumbing added in Task 8 and the netCDF group added in Step 3 are covered by unit tests only.

- [ ] **Step 7: Write the documentation**

- `docs/formulation.rst`: the coupled system, the DOF layout with the field block last, and the sign conventions.
- `docs/physics_interface.rst`: the three geometry derivative hooks, and that an absent one is a zero block.
- `docs/configuration.rst`: the four keys, and that `FieldSolve = exact` is a verification tool.
- `docs/running.rst`: what a coupled run writes.
- A new `docs/field_coupling.rst` for the `FieldModel` interface itself, added to `docs/index.rst`.
- `FEATURES.md`: replace the roadmap entry with a pointer to the implemented feature and to what remains (2-D Grad–Shafranov, DESC).
- `CLAUDE.md`: a section on the coupling, the three-way split of which test catches which failure class, and the adjoint asymmetry.

- [ ] **Step 8: Run everything and commit**

```sh
make test && make regression_tests && make python_tests && make docs && make stubs-check && make typecheck
git add -A
git commit -m "Serialise the coupled field, and document what it is"
```

---

### Task 13: An accelerated sweep that cannot be wrong

**Run this task after Task 10 and before Task 11.** It changes the iteration
Task 11's order study will measure, and it removes the failure mode Task 12
would otherwise have to document. The number is 13 rather than 11 because
Tasks 11 and 12 were already numbered when this was added; the ledger records
the execution order.

**Why this exists.** Task 10's review measured the spectral radius of the block
Gauss–Seidel iteration matrix `M = B⁻¹ A2 A⁻¹ A1` on `RichGeometricDiffusion`:
**ρ = 1.611 at `cj = 0` and 1.571 at `cj = 1e8`** — it moves 2% across eight
decades, so it is a property of the *coupling*, not of the time step. A field
residual that reads `sigma`, `q` and `phi` as well as `u` makes the plain sweep
divergent at every step size, and the adjoint sweep always runs at `cj = 0`,
where ρ is largest. The dominant eigenvalue there is real and **negative**
(−1.611), so a damped sweep with iteration matrix `(1−ω)I + ωM` has eigenvalues
`1 − 2.611ω` and any `ω ∈ (0, 0.766)` would converge — but for a real `λ > 1`
no `ω > 0` works at all, and no one can pick ω in advance. That rules out SOR
as the primary fix.

**Files:**
- Modify: `SystemSolver.hpp` (`ironsTuck`, the adjoint cap, the counters, `FieldSweepStats`)
- Modify: `SystemSolver.cpp` (`solveCoupledJacIterative`, `solveCoupledAdjointIterative`)
- Modify: `ConfigSchema.cpp`, `SolverConfig.hpp`, `SolverConfig.cpp`
- Modify: `Solver.cpp` (the `initialize()` warning text, the end-of-run report)
- Modify: `Tests/UnitTests/FieldJacobianTests.cpp`, `Tests/UnitTests/FieldAdjointTests.cpp`, `Tests/UnitTests/ConfigSourceTests.cpp`
- Modify: `TODO`

**Interfaces:**
- Consumes: Task 8's `solveCoupledJacExact`, `subtractA1Times`, `a2`,
  `solveTransportJac`; Task 9's `solveCoupledJacIterative`; Task 10's
  `solveCoupledAdjointExact`, `solveCoupledAdjointIterative`,
  `A1_transpose_cellwise`, `A2_transpose_cellwise`, `solveTransportAdjoint`.
- Produces: `SystemSolver::ironsTuck` (private static),
  `SystemSolver::setFieldSolveMaxAdjointSweeps(int)` /
  `getFieldSolveMaxAdjointSweeps()`, `SystemSolver::FieldSweepStats` and
  `getFieldSweepStats()`, config key `FieldSolveMaxAdjointSweeps`.

#### The three invariants this task must preserve

1. **Acceleration changes only the *next* iterate, never the accepted one.**
   Both sweeps' stopping tests are derived from the structure of a plain
   Gauss–Seidel step, and both derivations survive verbatim if and only if the
   iterate that gets accepted is the unaccelerated `G(p_k)`. Forwards, the
   returned pair `(dx, dpsi)` is consistent because `dx` was solved against the
   `p_k` that produced the accepted `G(p_k)`, so row one is off by `A1·Δ`, i.e.
   by the tolerance. Backwards, Task 10's identity
   `Aᵀ z_x + A2ᵀ z_psi − G_y = A2ᵀ (z_psi^{n+1} − z_psi^n)` is an *exact*
   backward error only because row two holds exactly — which it does when
   `z_psi^{n+1}` came from `solveBTranspose` against this sweep's `z_x`, and
   does **not** if an extrapolated value is returned instead. Accepting an
   accelerated iterate would silently turn a backward-error test into a proxy.
   Cost: one extra sweep to certify, which is what certification always costs.

2. **The fallback makes agreement tests vacuous, so every test must assert the
   sweep converged on its own.** Once `solveCoupledJacIterative` escalates to
   `solveCoupledJacExact` on failure, "iterative agrees with exact" passes
   whether or not the iteration works at all — including with the acceleration
   deleted. Every test in this task therefore asserts `fieldSweepFallbacks`
   explicitly, and the mutation experiments in Step 8 are what prove it.

3. **Scale equivariance survives.** Task 9's fix made the stopping test
   scale-equivariant — `solve(c·g) == c·solve(g)` to 8 digits at `c = 1e-9`.
   Irons–Tuck is homogeneous in the affine problem's data (μ is a ratio of
   inner products of quantities that all scale by `c`), so the property should
   hold unchanged. The existing test must be re-run, not relaxed.

- [ ] **Step 1: Add the accelerator, with its own unit tests first**

Declare it as a **private static member** of `SystemSolver` so
`MANTA_TEST_PRIVATE` exposes it directly — the guards below are much easier to
test on the function than through a solve.

```cpp
// Irons-Tuck vector acceleration: Aitken's Delta^2 generalised to vectors.
//
// Both coupled sweeps are affine fixed-point iterations on the field block
// alone. Writing G for one sweep, G(p) = c + M p with M = B^-1 A2 A^-1 A1
// forwards and its transpose backwards, so the plain sweep converges only for
// rho(M) < 1 -- and rho is a property of the coupling rather than of the time
// step: 1.611 at cj = 0 against 1.571 at cj = 1e8 for RichGeometricDiffusion.
//
// Given the two most recent increments D_k = G(p_k) - p_k and D_{k-1}:
//
//     mu  = D_k . (D_k - D_{k-1}) / |D_k - D_{k-1}|^2
//     p*  = G(p_k) - mu D_k
//
// which is the secant (rank-one quasi-Newton) step on F(p) = G(p) - p. For
// nField == 1 and an affine G it lands on the fixed point *exactly*, in one
// step, for every m -- including m > 1, where no relaxation parameter helps at
// all: D_k = m D_{k-1} gives mu = m/(m-1) and p* = p_k + D_k/(1-m), and
// D_k = (1-m)(p_fix - p_k), so p* = p_fix. That exactness for a divergent
// scalar map is why this rather than SOR, whose eigenvalues 1 - w + w*lambda
// cannot be brought inside the unit circle by any w > 0 when lambda > 1.
//
// For nField > 1 it is a rank-one approximation: it removes the dominant
// eigendirection and leaves the rest, which is why the caller still needs the
// exact fallback. Anderson acceleration -- equivalently GMRES on the Schur
// complement, since the map is affine -- is the depth-m generalisation and is
// in TODO.
static Vector ironsTuck(const Vector &g, const Vector &delta, const Vector &deltaPrev);
```

```cpp
Vector SystemSolver::ironsTuck(const Vector &g, const Vector &delta, const Vector &deltaPrev)
{
    const Vector secant = delta - deltaPrev;

    // Relative, not absolute. The secant is compared against the increments it
    // was differenced from, so this fires when the two increments agree to
    // rounding -- for an affine map, m == 1, a pure translation with no fixed
    // point -- and not merely when both are small, which is the regime the
    // sweep spends its last iterations in. An absolute floor here would
    // reintroduce exactly the scale-dependence Task 9's review removed from the
    // stopping criterion. Written as !(x > y) so a NaN secant takes this branch.
    const double scale = std::max(delta.norm(), deltaPrev.norm());
    if (!(secant.norm() > 1e-12 * scale))
        return g;

    const Vector accelerated = g - (delta.dot(secant) / secant.squaredNorm()) * delta;

    // The guard above bounds the denominator relative to the numerator, so this
    // is the residual case -- an overflow in the dot product, or a non-finite g
    // handed in. Returning g is the plain sweep, which the caller may take.
    return accelerated.allFinite() ? accelerated : g;
}
```

Tests, in `Tests/UnitTests/FieldJacobianTests.cpp`:

```cpp
BOOST_AUTO_TEST_CASE(irons_tuck_is_exact_for_a_scalar_affine_map)
{
    // p_{k+1} = c + m p_k, fixed point c/(1-m). Two plain steps from p_0 = 0,
    // then one accelerated step must land on it exactly -- for a contraction
    // and, the point of the exercise, for a divergent map too. m = -1.611 is
    // the measured dominant eigenvalue of RichGeometricDiffusion's coupling.
    for (double m : {0.33, -1.611, 2.5})
    {
        const double c = 0.7;
        auto G = [&](const Vector &p) { return Vector::Constant(1, c + m * p(0)); };

        Vector p0 = Vector::Zero(1);
        Vector g0 = G(p0), d0 = g0 - p0;
        Vector p1 = g0;
        Vector g1 = G(p1), d1 = g1 - p1;

        Vector acc = SystemSolver::ironsTuck(g1, d1, d0);
        BOOST_CHECK_CLOSE(acc(0), c / (1.0 - m), 1e-10);
    }
}

BOOST_AUTO_TEST_CASE(irons_tuck_declines_when_the_secant_vanishes)
{
    // Equal increments mean m == 1: a translation, no fixed point, and a zero
    // denominator. The plain iterate must come back, not a NaN.
    Vector g = Vector::Constant(3, 2.0), d = Vector::Constant(3, 0.5);
    Vector out = SystemSolver::ironsTuck(g, d, d);
    BOOST_CHECK(out.isApprox(g));

    // ...and the guard is relative: increments of 1e-30 that differ by 1e-31
    // are a perfectly good secant, not a vanishing one.
    Vector small = Vector::Constant(3, 1e-30), smaller = Vector::Constant(3, 9e-31);
    Vector accelerated = SystemSolver::ironsTuck(Vector::Constant(3, 1e-30), small, smaller);
    BOOST_CHECK(accelerated.allFinite());
    BOOST_CHECK(!accelerated.isApprox(Vector::Constant(3, 1e-30)));
}
```

Run: `Tests/UnitTests/UnitTests --run_test=field_jacobian_tests --log_level=message`
Expected: FAIL to compile (`ironsTuck` not declared), then PASS.

- [ ] **Step 2: Restructure the forward sweep**

`solveCoupledJacIterative` becomes: apply the map, test the *unaccelerated*
iterate, accept it or accelerate the next starting point; on exhausting the
cap, escalate to the exact solve. Replace the body's loop and tail with:

```cpp
    N_Vector work = N_VClone(delY);
    Vector dpsi = Vector::Zero(nField);
    Vector delta = Vector::Zero(nField), deltaPrev = Vector::Zero(nField);
    bool haveDeltaPrev = false, converged = false;

    ++fieldSweepSolves;

    for (Index sweep = 0; sweep < fieldSolveMaxSweeps; ++sweep)
    {
        ++fieldSweepIterations;

        // work <- r1 - A1 dpsi, then dx <- A^-1 work. The pair (dx, dpsi) is
        // consistent here and stays so: everything below changes only what the
        // *next* sweep starts from. solveTransportJac zeroes the whole
        // increment (see solveHDGJac), so delY's field entries come back zero
        // regardless of what work's field segment held.
        N_VScale(1.0, res_g, work);
        subtractA1Times(dpsi, work);
        solveTransportJac(work, delY);

        // g <- B^-1 ( r2 - A2 dx ): one application of the fixed-point map.
        Vector r2 = rhs.getField();
        for (Index f = 0; f < nField; ++f)
            r2(f) -= N_VDotProd(a2[f], delY);
        Vector g(nField);
        fieldModel->solveB(g, r2);

        deltaPrev = delta;
        delta = g - dpsi;

        // The acceptance test is on the UNACCELERATED iterate, and is the same
        // relative test Task 9's review arrived at -- not `tol * max(1, |g|)`,
        // whose absolute floor stopped the sweep after the first iterate
        // whenever |g| < 1, i.e. throughout the small-correction regime Newton
        // lives in. Accepting only g is also what keeps the returned pair
        // consistent: dx was solved against the dpsi that produced this g, so
        // row one of the coupled system is off by A1 . delta, which is the
        // tolerance. An extrapolated dpsi would be off by the length of the
        // extrapolation instead, and nothing downstream would report it.
        //
        // isZero(0.0), not the default: Eigen's dummy_precision is ~1e-12, so
        // the bare call was an absolute magnitude test wearing a degenerate
        // case's clothing. This fires only when the increment is exactly zero,
        // which is the 0/0 the relative test genuinely cannot handle.
        if (delta.isZero(0.0) || delta.norm() <= fieldSolveTolerance * g.norm())
        {
            dpsi = g;
            converged = true;
            break;
        }

        dpsi = haveDeltaPrev ? ironsTuck(g, delta, deltaPrev) : g;
        haveDeltaPrev = true;
    }

    N_VDestroy(work);

    if (converged)
    {
        out.getField() = dpsi;
        return;
    }

    // Escalate rather than return the last iterate. Task 9 returned it, on the
    // argument that an under-converged Jacobian solve is merely a worse search
    // direction -- true, and still true. What changed is the price of being
    // right: the exact Schur solve costs nField + 1 transport solves, against
    // the cap's fieldSolveMaxSweeps, and for the large nField this path exists
    // to serve the escalation is *cheaper* than the sweeps already spent. So
    // there is no longer a reason to hand back a direction we know is bad, and
    // with the escalation in place the iterative mode can no longer be wrong at
    // all -- only slower. That is what makes it a safe default.
    //
    // Counted, not warned: a warning here would fire once per Jacobian solve.
    // The count is reported once per run, in Solver.cpp.
    ++fieldSweepFallbacks;
    solveCoupledJacExact(res_g, delY);
```

Note `solveCoupledJacExact` overwrites the whole of `delY` including the field
block, so nothing written above needs undoing.

- [ ] **Step 3: Restructure the adjoint sweep, and delete its throw**

Same shape. `solveCoupledAdjointIterative` keeps its backward-error test
verbatim — see invariant 1 for why that is legitimate — and replaces the
`throw` with an escalation to `solveCoupledAdjointExact()`.

```cpp
        Vector g(nField);
        fieldModel->solveBTranspose(g, r);

        deltaPrev = delta;
        delta = g - zpsi;

        double residual2 = 0.0;
        for (Index i = 0; i < nCells; ++i)
            residual2 += (A2_transpose_cellwise[i] * delta).squaredNorm();
        residualNorm = std::sqrt(residual2);

        if (residualNorm <= fieldSolveTolerance * rhsNorm)
        {
            zpsi = g;
            converged = true;
            break;
        }

        zpsi = haveDeltaPrev ? ironsTuck(g, delta, deltaPrev) : g;
        haveDeltaPrev = true;
```

and the tail:

```cpp
    if (converged)
    {
        adjoint_field = zpsi;
        return;
    }

    // Task 10 threw here, on the argument that an under-converged adjoint is a
    // wrong gradient beside a correct objective and there is no "close enough"
    // to fall back on. The argument was right and the remedy is now better: the
    // exact transposed Schur solve gives the same guarantee without failing the
    // run. Escalating is strictly stronger than throwing -- the caller gets a
    // correct gradient instead of an exception -- so the exception_ptr guard in
    // Solver.cpp that stops a refusal from eating the forward run's output is
    // now unreachable from this path. Leave it: solveCoupledAdjointExact can
    // still throw out of the field model, and the guard is what keeps a netCDF
    // file from dying with it.
    //
    // Warned, not counted: unlike the forward Jacobian this runs once per run,
    // so once per occurrence *is* once per run, and the cost of the escalation
    // is worth a line.
    fieldAdjointFellBack = true;
    logmsg<LOG_LEVEL::WARNING>(
        "The coupled adjoint sweep did not converge in {} sweeps (backward error {:g} "
        "against a tolerance of {:g} times a right-hand side of norm {:g}); falling back "
        "to the exact transposed Schur solve, which costs {} transposed transport solves. "
        "The gradient is correct either way. Raise FieldSolveMaxAdjointSweeps to try "
        "longer, or set FieldSolve = exact to skip the sweep.",
        fieldSolveMaxAdjointSweeps, residualNorm, fieldSolveTolerance, rhsNorm, nField + 1);
    solveCoupledAdjointExact();
```

Verify that `solveCoupledAdjointExact()` writes `adjoint_squ`, `adjoint_lambdas`
*and* `adjoint_field` — if it leaves any of them holding the sweep's last
iterate, the gradient is a mixture of the two solves. Say in the report which
you checked and how.

Keep `std::format`'s `{:g}` and the reason recorded in Task 10's comment (
`std::to_string` is fixed to six decimals, so `1e-8` prints as `0.000000`).

- [ ] **Step 4: The separate adjoint cap**

The adjoint sweep always runs at `cj = 0`, where ρ is largest — it is strictly
the harder direction, so it gets its own cap rather than inheriting the
forward one. `ConfigSchema.cpp`, beside the other three field keys:

```cpp
{"FieldSolveMaxAdjointSweeps", {}, Type::Int, Category::Solver, false, false, 100,
 "Sweep cap for the coupled adjoint solve. Separate from FieldSolveMaxSweeps, and "
 "larger, because the adjoint always runs at cj = 0 where the coupling is stiffest. "
 "Exceeding it falls back to the exact transposed solve, not to a wrong gradient."},
```

Mirror `FieldSolveMaxSweeps` exactly: a field on `SolverConfig`, a `READ` in
`loadSolverConfig`, a `setFieldSolveMaxAdjointSweeps` call in
`applySolverConfig`, and the setter/getter pair on `SystemSolver` with the same
`n < 1` `std::logic_error`. Default `100` in both the schema and the member
initialiser, for the reason the existing comment gives: one default, not two
that can drift.

`Tests/UnitTests/ConfigSourceTests.cpp` needs the key in **four** places — the
defaults check near line 260, the explicit-value TOML near line 282, and the
TOML/dict pair plus the comparison in `both_sources_produce_the_same_solver_config`
near lines 384–440. That last test compares `SolverConfig`s field by field, so
a key added to the struct and not to the comparison is silently unchecked.

- [ ] **Step 5: Counters and the once-per-run report**

On `SystemSolver`, private:

```cpp
// Diagnostics for the coupled sweeps, zeroed per run. Nothing here feeds the
// answer, so none of it can disturb the bit-for-bit reuse invariant that
// a_second_integration_on_one_solver_matches_a_fresh_one pins -- but it is
// zeroed per run all the same, because a cumulative count reported as a
// per-run one is a lie a second run would tell silently.
long fieldSweepSolves = 0;
long fieldSweepIterations = 0;
long fieldSweepFallbacks = 0;
long fieldAdjointSweeps = 0;
bool fieldAdjointFellBack = false;
```

Zero them beside the `fieldModel->resetForRun()` call at `Solver.cpp:127`. If
that call sits inside a branch a reused solver skips, zero them in the
unconditional part of `initialize()` instead and say so in a comment — the
`initialised` guard around `initialiseMatrices` is exactly the trap that broke
solver reuse before (see `RF_cellwise` in `CLAUDE.md`).

Public, so `Solver.cpp` can read them without `MANTA_TEST_PRIVATE`:

```cpp
struct FieldSweepStats
{
    long solves, iterations, fallbacks, adjointSweeps;
    bool adjointFellBack;
};
FieldSweepStats getFieldSweepStats() const
{
    return {fieldSweepSolves, fieldSweepIterations, fieldSweepFallbacks,
            fieldAdjointSweeps, fieldAdjointFellBack};
};
```

In `Solver.cpp`, in the block that prints the IDA totals, matching its
`std::println` style:

```cpp
if (nField > 0 && system.getFieldSolveMode() == SystemSolver::FieldSolveMode::Iterative)
{
    auto fs = system.getFieldSweepStats();
    std::println("Coupled field sweeps                  :{} over {} solves ({} exact fallbacks)",
                 fs.iterations, fs.solves, fs.fallbacks);
    // This is the only signal a user has that the coupling is not converging.
    // The run is still correct -- the fallback is the exact solve -- so this is
    // a cost report, not an error, and it says what to do about it.
    if (fs.fallbacks > 0)
        logmsg<LOG_LEVEL::WARNING>(
            "{} of {} coupled Jacobian solves exhausted FieldSolveMaxSweeps = {} and fell "
            "back to the exact Schur solve, at {} transport solves each. The answers are "
            "correct; the run is paying for both. Raise FieldSolveMaxSweeps, or set "
            "FieldSolve = exact and skip the sweeps.",
            fs.fallbacks, fs.solves, fieldSolveMaxSweeps, nField + 1);
}
```

This closes the Task 9 deferred minor "nothing reports that the sweep cap was
hit; a once-per-run summary is the only signal a user would have that the
coupling is not converging."

- [ ] **Step 6: Correct the `initialize()` warning**

The iterative branch of the warning at `Solver.cpp:222` currently ends
"...and returns its last iterate rather than failing if that cap is reached
first, costing Newton speed, not correctness." That is no longer what happens.
Replace that clause with the escalation, and name the accelerator:

```cpp
logmsg<LOG_LEVEL::WARNING>(
    "FieldSolve = iterative: block Gauss-Seidel between the transport and field "
    "blocks with Irons-Tuck acceleration, one transport solve per sweep against "
    "exact's {} per Jacobian solve. Stops once the relative change in psi is below "
    "FieldSolveTolerance = {}, up to FieldSolveMaxSweeps = {} sweeps ({} for the "
    "adjoint); a sweep that reaches its cap falls back to the exact solve, so this "
    "mode costs more than exact in the worst case and never less accuracy.",
    nField + 1, fieldSolveTolerance, fieldSolveMaxSweeps, fieldSolveMaxAdjointSweeps);
```

- [ ] **Step 7: A fixture whose divergence is measured, not assumed**

`FieldJacobianTests.cpp` has no fixture known to be divergent — the ρ = 1.611
measurement was made on `RichGeometricDiffusion` in `FieldAdjointTests.cpp`.
Rather than move that fixture or assume a pairing is hard enough, give the
coupling a **strength dial** and have the test *measure* ρ, so a fixture that
silently stops being divergent fails loudly instead of passing vacuously.

Add a coupling-strength constructor parameter to a copy of
`ManufacturedFieldEverySlot` (nField == 1) and of `ManufacturedFieldVector`
(nField == 5) — `R = psi − c·∫(u + σ/2 + q/4 + 3φ/4) dx` and its vector
analogue. `B` is unchanged by `c` and `A2 ∝ c`, so `ρ(M) ∝ c` and the dial is
linear and predictable. Pair each with `GeometricDiffusion`, which has a
nonzero `dSigmaFn_dGeometry` (Task 9's fix — confirm before relying on it, and
say in the report what `max|A1_cellwise|` came out as).

The measurement, in the test itself — no production code needed, since
`subtractA1Times`, `solveTransportJac`, `a2` and `fieldModel` are all reachable
under `MANTA_TEST_PRIVATE`:

```cpp
// Power-iterate M = B^-1 A2 A^-1 A1. With a zero right-hand side the affine
// map's constant term vanishes and one sweep is exactly one application of M,
// so the ratio of successive norms converges to |lambda_max|. This is the same
// operator solveCoupledJacIterative iterates, reached the same way, rather than
// a reimplementation of it that could agree with a wrong original.
double spectralRadius(CoupledSolver &solver, int iterations = 60)
{
    const Index nField = solver->nField;
    N_Vector zero = N_VClone(solver->y_vec()), work = N_VClone(zero), dx = N_VClone(zero);
    N_VConst(0.0, zero);

    Vector p = Vector::Ones(nField).normalized();
    double ratio = 0.0;
    for (int it = 0; it < iterations; ++it)
    {
        N_VScale(1.0, zero, work);
        solver->subtractA1Times(p, work);
        solver->solveTransportJac(work, dx);

        Vector r2 = Vector::Zero(nField);
        for (Index f = 0; f < nField; ++f)
            r2(f) -= N_VDotProd(solver->a2[f], dx);

        Vector next(nField);
        solver->fieldModel->solveB(next, r2);
        ratio = next.norm() / p.norm();
        p = next.normalized();
    }
    N_VDestroy(zero); N_VDestroy(work); N_VDestroy(dx);
    return ratio;
}
```

Adapt the vector accessors to whatever `CoupledSolver` (`FieldJacobianTests.cpp:255`)
actually exposes; the algebra above is the specification.

- [ ] **Step 8: The tests that discriminate**

Each of these asserts `fieldSweepFallbacks` explicitly. Per invariant 2, one
that only checks agreement with the exact solve proves nothing now.

```cpp
BOOST_AUTO_TEST_CASE(acceleration_converges_a_sweep_that_plainly_diverges)
{
    // First: prove the fixture is what it claims. If a later change makes the
    // coupling weak, this fails here rather than turning the test below into a
    // statement about a contraction.
    auto solver = makeCoupledSolverAtState(6, 2, EverySlotStrengthTag{2.0});
    const double rho = spectralRadius(*solver);
    BOOST_TEST_MESSAGE("rho(M) = " << rho);
    BOOST_CHECK_GT(rho, 1.2);

    N_Vector g = randomRHS(*solver);
    N_Vector exact = N_VClone(g), iterative = N_VClone(g);
    solver->solveCoupledJacExact(g, exact);
    solver->solveCoupledJacIterative(g, iterative);

    // The agreement is necessary but not sufficient -- solveCoupledJacIterative
    // now escalates to solveCoupledJacExact, so this line passes with the
    // acceleration deleted. The fallback count is the assertion that matters.
    BOOST_CHECK_SMALL(relativeDifference(exact, iterative), 1e-6);
    BOOST_CHECK_EQUAL(solver->fieldSweepFallbacks, 0);
    BOOST_CHECK_LE(solver->fieldSweepIterations, 12);
}

BOOST_AUTO_TEST_CASE(the_fallback_returns_the_exact_solve_bit_for_bit)
{
    // A cap of one sweep cannot converge anything that needs two, so this
    // reaches the escalation deterministically without needing a pathological
    // fixture. Bit-for-bit, not to a tolerance: the escalation does not blend
    // the sweep's last iterate with the exact answer, it discards it.
    auto solver = makeCoupledSolverAtState(6, 2, EverySlotStrengthTag{2.0});
    solver->setFieldSolveMaxSweeps(1);

    N_Vector g = randomRHS(*solver);
    N_Vector exact = N_VClone(g), iterative = N_VClone(g);
    solver->solveCoupledJacExact(g, exact);
    solver->solveCoupledJacIterative(g, iterative);

    BOOST_CHECK_EQUAL(solver->fieldSweepFallbacks, 1);
    BOOST_CHECK_EQUAL(relativeDifference(exact, iterative), 0.0);
}

BOOST_AUTO_TEST_CASE(acceleration_does_not_disturb_a_contraction)
{
    // The risk in adding an extrapolation is that it destabilises the cases
    // that were already fine. rho < 1 here, and the sweep must still converge,
    // still without falling back, and in no more sweeps than the plain
    // iteration needed (2, recorded in Task 9's review).
    auto solver = makeCoupledSolverAtState(6, 2, EverySlotStrengthTag{0.25});
    BOOST_CHECK_LT(spectralRadius(*solver), 1.0);

    N_Vector g = randomRHS(*solver);
    N_Vector out = N_VClone(g);
    solver->solveCoupledJacIterative(g, out);

    BOOST_CHECK_EQUAL(solver->fieldSweepFallbacks, 0);
    BOOST_CHECK_LE(solver->fieldSweepIterations, 4);
}

BOOST_AUTO_TEST_CASE(the_accelerated_sweep_is_still_scale_equivariant)
{
    // Task 9's property, re-run rather than assumed: mu is a ratio of inner
    // products of quantities that all scale with the right-hand side, so
    // Irons-Tuck is homogeneous and solve(c g) == c solve(g) should survive it.
    // The scale that broke the old criterion was 1e-9; use it again.
    auto solver = makeCoupledSolverAtState(6, 2, EverySlotStrengthTag{2.0});
    N_Vector g = randomRHS(*solver), gSmall = N_VClone(g);
    N_VScale(1e-9, g, gSmall);

    N_Vector big = N_VClone(g), small = N_VClone(g);
    solver->solveCoupledJacIterative(g, big);
    solver->solveCoupledJacIterative(gSmall, small);
    N_VScale(1e9, small, small);

    BOOST_CHECK_SMALL(relativeDifference(big, small), 1e-8);
}
```

In `FieldAdjointTests.cpp`, the two that matter for the gradient. The existing
non-convergence test — the one that asserted the throw at
`FieldAdjointTests.cpp:981` with `setFieldSolveMaxAdjointSweeps`'s predecessor
set to 1 — must be **rewritten, not deleted**: the same forcing condition now
has to produce a correct gradient rather than an exception.

```cpp
BOOST_AUTO_TEST_CASE(the_adjoint_sweep_converges_on_the_rich_fixture)
{
    // RichGeometricDiffusion is where rho = 1.611 was measured, at cj = 0,
    // which is where every adjoint solve runs. Before this task the sweep
    // needed ~42 iterations against a cap of 20 and threw.
    auto run = makeRichAdjointRun();
    run->integrate();
    auto stats = run->solver().getFieldSweepStats();
    BOOST_CHECK(!stats.adjointFellBack);
    checkGradientAgainstFiniteDifferences(*run, 1e-6);
}

BOOST_AUTO_TEST_CASE(an_exhausted_adjoint_sweep_still_gives_the_right_gradient)
{
    // The replacement for the throw. One sweep cannot converge, the escalation
    // fires, and the gradient is the exact one -- which is the whole point of
    // escalating rather than failing: the caller loses time, never accuracy.
    auto run = makeRichAdjointRun();
    run->solver().setFieldSolveMaxAdjointSweeps(1);
    run->integrate();
    BOOST_CHECK(run->solver().getFieldSweepStats().adjointFellBack);
    checkGradientAgainstFiniteDifferences(*run, 1e-6);
}
```

Adapt the fixture-construction calls to whatever `AdjointRun` /
`AdjointStateFixture` (`FieldAdjointTests.cpp:480`, `:522`) provide.

**Mutation experiments — run these and report what happened.** A test that
passes under the mutation is not testing what it says.

1. Replace `haveDeltaPrev ? ironsTuck(...) : g` with plain `g` in the forward
   sweep. `acceleration_converges_a_sweep_that_plainly_diverges` must fail on
   the fallback count, and must still pass its `relativeDifference` line —
   that contrast is the demonstration of invariant 2.
2. Accept the accelerated iterate instead of `g` (invariant 1): set
   `dpsi = ironsTuck(...)` and break on it. Report whether any test notices.
   If none does, that is a gap to close before the task is finished, because
   the returned pair is then inconsistent by the length of the extrapolation.
3. In the adjoint, do the same as (2) and check the backward-error test: it
   should now be a proxy rather than an exact residual, so
   `the_adjoint_sweep_converges_on_the_rich_fixture`'s gradient check should
   degrade. Report the measured gradient error either way.
4. Change `secant.norm() > 1e-12 * scale` to `secant.norm() > 1e-12`.
   `irons_tuck_declines_when_the_secant_vanishes`'s second half must fail.

- [ ] **Step 9: `TODO`, and the note for Task 12**

Replace the Task 9/10 Gauss–Seidel entry with what is now true, and add the
successor:

```
- The coupled sweep uses Irons-Tuck, a rank-one secant acceleration. It is
  exact in one step for nField == 1 and any spectrum, and removes the dominant
  eigendirection for nField > 1 -- but only that one, so a coupling with two
  comparable eigenvalues outside the unit circle still falls back to the exact
  Schur solve. The depth-m generalisation is Anderson acceleration, which for
  an affine map is GMRES on the Schur complement; that is the right answer for
  the large nField this path exists to serve, and is a separate piece of work.
  rho(M) was measured at 1.611 (cj = 0) and 1.571 (cj = 1e8) on
  RichGeometricDiffusion: a property of the coupling, not of the time step.

- Nothing latches the fallback. A genuinely divergent coupling pays
  FieldSolveMaxSweeps sweeps *and* the exact solve on every Jacobian solve for
  the whole run. Latching after N consecutive fallbacks would fix it, but N is
  a number nobody can presently justify, and rho does drift with cj even if
  only by 2% in the one case measured. The end-of-run count is the signal for
  now.
```

Add a line to Task 12's documentation step (`docs/running.rst` and the
`CLAUDE.md` section): `FieldSolve = iterative` is a *cost* choice, not an
accuracy one — the sweep escalates to the exact solve rather than returning an
under-converged answer, in both directions — and `FieldSolveMaxAdjointSweeps`
exists because the adjoint runs at `cj = 0` where the coupling is stiffest.

- [ ] **Step 10: Run everything and commit**

```sh
export PATH="$PWD/.venv/bin:$PATH"
make test && make regression_tests && make python_tests && make stubs-check && make typecheck
git add -A
git commit -m "Accelerate the coupled sweep, and escalate rather than guess"
```

`make test` must show the full case count, not a subset. State the count in the
report, along with the ρ measured for each new fixture and the sweep counts
observed.

---

## Self-Review

**Spec coverage.** Every section of the spec maps to a task: the `FieldModelSpec`/`FieldModel` abstraction → Tasks 1–2; geometry derived and reaching cases through `State` → Task 3; the DOF layout → Task 4; the manufactured clients → Task 5; the coupled residual, `id` vector and recoverable failure → Task 6; the geometry derivative hooks → Task 7; `A1`/`A2`/`B` and the exact Schur, plus the config keys and the exact-solve warning → Task 8; block Gauss–Seidel → Task 9; the adjoint transposes and the loud-failure asymmetry → Task 10; the order study including the superconvergent case → Task 11; output, restart, docs and the zero-coupling invariant → Task 12. The two named traps (the differential-without-`d/dt` refusal, per-run state in `initialiseMatrices`) are Task 6 Steps 6–7, checked in Tasks 6 and 12.

Task 13 was added during execution, after Task 10's review measured the block
Gauss–Seidel spectral radius at 1.611 for a coupling that reads `sigma`, `q`
and `phi`. It is not new scope: it is the spec's "iterate, assuming the
coupling is weak" made safe for a coupling that is not. **Execution order is
1–10, 13, 11, 12** — Task 13 changes the iteration Task 11 measures and removes
the failure mode Task 12 would document.

**Two known soft spots, called out rather than hidden.**

*Task 6 is large.* It is the first task that produces a running coupled solve, and its pieces — field rows, geometry evaluation, `id` wiring, the initial condition — cannot be separately tested, because none of them does anything observable alone. Splitting it would produce tasks whose "test" is that the build still compiles. If the reviewer wants it split anyway, the natural cut is after Step 5 (residual rows, tested by the first test case) with Steps 6–8 as a second task.

*Task 11 Step 5 is conditional by design.* Whether the superconvergent path keeps `k+2` under coupling is not knowable without running it, and the plan says what to do in each case rather than guessing. That is deliberate, not a placeholder: the precedent is that spatial adjoint parameters with `Superconvergent = true` throw rather than silently redefining what the answer means.

**Type consistency.** `nField`/`nGeom` are used with those names from Task 4 onward; `getField()`/`Field(Index)` on `DGSoln`; `geom()`/`geom(Index)` on `State`; `Geometry(Index)`/`setGeometry(Index, Vector)` on `GlobalState`; `FieldResidual`/`FieldResidualPrime`/`Geometry`/`dGeometry_dpsi`/`InitialFieldValue`/`updateFieldJacobian`/`applyB`/`applyBTranspose`/`solveB`/`solveBTranspose`/`resetForRun` on `FieldModel`; `solveTransportJac`/`solveCoupledJacExact`/`solveCoupledJacIterative`/`solveCoupledAdjointExact`/`solveCoupledAdjointIterative` on `SystemSolver`. `manufacturedU`/`manufacturedPsiExact`/`manufacturedC`/`manufacturedL` are defined once in Task 5 and used unchanged in Tasks 6, 8, 9, 10 and 11.
