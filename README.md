# MaNTA -- The Maryland Nonlinear Transport Analyzer

MaNTA is a code developed for the prediction and analysis of transport processes in a strongly-magnetized plasma.
It solves one-dimensional nonlinear reaction-diffusion and transport systems with a hybridizable discontinuous
Galerkin (HDG) spatial discretisation, integrated in time as an index-1 DAE by SUNDIALS IDA.

**Documentation: [manta-docs.readthedocs.io](https://manta-docs.readthedocs.io)** — the equations being solved,
every configuration key, the output format, the adjoint interface, and how to write a transport system in C++ or
Python. The sources live in [`docs/`](docs/index.rst); build them locally with

```sh
cmake --build build --target docs      # -> docs/_build/html/index.html
```

which creates `.venv-docs` from `docs/requirements.txt` the first time and reuses it
after. It builds with `-W`, the same as Read the Docs, so a local build that passes is
one that will publish.

Configuration keys are declared once, in `ConfigSchema.cpp`, and read from there by
both the config file and `Runner.configure`; `build/MaNTA --list-options` prints the
current list with types, defaults and a line of description for each.

## Getting Started

You will need to download this codebase and compile it in order to run MaNTA

### Prerequisites

To compile and use MaNTA you will need a system with the following

 - A C++23 compliant C++ compiler: **g++ 15 or newer**, or **clang++ 18 or newer**.
   Verified by hand on clang++ 18, clang++ 19 and clang++ 21 -- each builds the
   solver, the pybind11 module and all three test suites clean under
   `-Wall -Werror`. Pick one when you configure the build directory:
   `cmake -B build -DCMAKE_CXX_COMPILER=clang++-19`.

   > **Health warning: avoid g++ 14 if you can.** Something in it breaks on this
   > tree at the release flags (`-O3 -flto -march=native`), and the symptom is a
   > wrong number rather than an error: adding any member to `SystemSolver` makes
   > one of the Jacobian tests fail about half the time, with an O(1) error, and no
   > sanitiser reports anything. g++ 15, g++ 16 and every clang tested are clean.
   > The build prints a warning if it sees g++ 14. It still compiles, and the
   > solver's own output was reproducible in the case measured, so this is a
   > recommendation rather than a hard floor -- but the numbers from a g++ 14
   > release build are not ones this project can vouch for. `CLAUDE.md` and `TODO`
   > have the reproduction; the root cause is not yet known.

   CI runs seven build legs: g++ 15, g++ 16, clang++ 19, clang++ 20, clang++ 21,
   and then g++ 15 and clang++ 19 again against Eigen 5.0.1. **g++ 14 and clang++ 18
   are deliberately not among them** -- the first for the reason above, the second
   because guarding it would mean maintaining clang-18 `-Werror` compliance and its
   C++23 support has gaps. Both are therefore unguarded and can regress without
   anyone noticing. Note what dropping g++ 14 costs: it is the compiler in Ubuntu
   noble's archive, so the version most people will have by default is now the one
   least tested here.

   The lower bound on gcc was measured, not guessed: **g++ 13 cannot build MaNTA at
   all**, because libstdc++ 13 has no `<print>` and the output layer uses
   `std::print` throughout. g++ 14 builds it -- the bound moved to 15 on the
   miscompile above, not on a language feature. clang++ 18 took one source change to admit --
   `PyGrid.hpp` used to declare `constexpr Vector getNodes(...)`, and a dynamic
   Eigen vector is not a literal type; C++23 permits that under P2448R2 provided the
   function is never constant-evaluated, which clang 18 does not implement. That
   `constexpr` could never have done anything and is now `inline`.
 - The Boost C++ Template Library
 - The Eigen linear algebra template library, either 3.4.x or 5.0.x. Both are
   supported and tested; point `EIGEN_DIR` at whichever you have. Note that
   Eigen 5.0 needs the `extern/autodiff` submodule this tree pins, which
   carries a patch upstream does not have yet -- see `.gitmodules`.
 - The SUNDIALS library, Version 7.1.0 or newer. Not 6.x: MaNTA links
   `sundials_core` and uses `SUNContext`, neither of which exists before v7.
 - NETCDF C and NETCDF C++ 4.3 or newer (depends upon netcdf C interface 4.6.0 or newer)
 - CMake 3.22 or newer, and a generator for it — GNU make or Ninja. (SUNDIALS is
   built with CMake too, so you almost certainly have one already.)

Three further libraries are **bundled as git submodules** under `extern/`, so
there is nothing to install and no path to configure for them -- you only need to
initialise them, which step 1 below covers:

 - [toml11](http://github.com/toruniina/toml11) -- parses the configuration files
 - [autodiff](https://autodiff.github.io) -- forward-mode automatic
   differentiation, which is how `AutodiffTransportSystem` derives every Jacobian
   entry from a physics case's `Flux` and `Source`
 - [pybind11](https://github.com/pybind/pybind11) -- needed only for the Python
   module, i.e. the `_manta` target and the `python` test

Precise dependencies have not been exhaustively tested, bug reports are welcome. Running on Windows requires the installation of [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install) (WSL) and proceeding as for linux.

MacOS has yet to be tested.

### Building MaNTA

MaNTA builds with CMake, out of source. There is no file you have to create
first: if every dependency is somewhere the compiler and `pkg-config` already
look, this is the whole of it.

```sh
cmake --preset default        # configure  -> build/
cmake --build build -j        # compile
ctest --test-dir build        # all three suites
```

`--preset default` is a Release build: `-O3 -flto=auto -march=native`, with
assertions left **on** — CMake would normally add `-DNDEBUG` to a Release build
and this one deliberately does not, because Eigen's own assertions are the
diagnostic of record for a whole class of defect here. The other presets are
`debug`, `coverage` and `portable` (Release without `-march=native`, for a binary
that has to run on a different machine).

In practice one dependency usually does need naming, because distributions rarely
package SUNDIALS 7:

```sh
cmake --preset default -DSUNDIALS_ROOT=/path/to/sundials/install
```

...unless you used the bundled `./build_sundials`, which installs into
`./sundials/install`; the configure step looks there on its own, so a checkout
that ran it needs no arguments at all.

Anything else that is somewhere unusual is named the same way — `-DEigen3_ROOT`,
`-DBOOST_ROOT`, or `-DCMAKE_PREFIX_PATH="/opt/a;/opt/b"` for several at once (the
quotes matter — that semicolon is CMake's list separator, not the shell's).
Naming a prefix the compiler already searches is harmless; CMake filters those
out itself, and CI checks that it does.

To avoid retyping them, copy `CMakeUserPresets.json.example` to
`CMakeUserPresets.json` and put them there. That file is gitignored and is the
direct replacement for the old `Makefile.local`.

> **Upgrading a checkout that predates CMake?** Delete `Makefile.local`, and run
> `git clean -xn` to see the in-source `.o`, `.d` and binaries the Makefile left
> behind; nothing reads them any more. Your `Makefile.local` maps to
> `CMakeUserPresets.json` roughly as `SUNDIALS_DIR` → `SUNDIALS_ROOT`, `CXX` →
> `CMAKE_CXX_COMPILER`, `EIGEN_DIR` → `MANTA_EIGEN_INCLUDE_DIR` for a source
> checkout of Eigen (or nothing at all for a packaged one), `BOOST_DIR` →
> `BOOST_ROOT`, `NETCDF_DIR` → `CMAKE_PREFIX_PATH`, and `DEBUG=on` →
> `--preset debug`.

Step by step, for a first build:

 1. Clone this repository into your chosen location, **with its submodules**:

    ```sh
    git clone --recurse-submodules https://github.com/ianabel/MaNTA.git
    ```

    If you have already cloned without that flag, `extern/` will contain three
    empty directories and the configure step stops with a message naming the
    submodule and the command to populate it:

    ```sh
    git submodule update --init
    ```

    That is the whole of the toml11, autodiff and pybind11 installation — none of
    the three needs a path configured.
 2. Install the Boost library, either using your system package manager or manually by downloading from [here](https://www.boost.org). If this is a system-wide install,
 proceed to the next step. If you downloaded the Boost libraries, configure with `-DBOOST_ROOT=/path/to/boost`.
 Boost is only needed for the unit tests; `-DMANTA_TESTS=OFF` builds without it.
 3. Install [SUNDIALS](https://computing.llnl.gov/projects/sundials) and pass
 `-DSUNDIALS_ROOT=` the location you installed it in. If you are only using SUNDIALS
 for MaNTA, a quick intro to installing it is included below.
 4. Install [NETCDF C and NETCDF C++](https://www.unidata.ucar.edu/software/netcdf/). On Ubuntu or Debian these can be installed from the package manager: `apt-get install libnetcdf-dev libnetcdff-dev libnetcdf-c++4-dev libnetcdf-c++4-1`.
 On MacOS, you can use either `brew install netcdf` or `conda install -c anaconda netcdf4` to install the C version, and `conda install -c conda-forge netcdf-cxx4` to install the C++ version.

    A package-manager install needs nothing configured: it lands in `/usr`, and
    CMake asks `pkg-config`. For an install elsewhere — a `brew`/`conda` pair on
    MacOS, say — add the prefixes to `CMAKE_PREFIX_PATH`:
    `-DCMAKE_PREFIX_PATH="/usr/local/Cellar/netcdf/4.8.0_2;$HOME/miniconda3"`.
 5. Set any other options. `cmake -B build -LH` lists every one with its
 description; the MaNTA-specific ones all begin `MANTA_`.
 6. Run `cmake --preset default` and then `cmake --build build -j`.
 7. Check the unit tests with `ctest --test-dir build -R unit`.

#### The build targets

Everything the old Makefile had a target for still has one, built with
`cmake --build build --target <name>`:

| Target | What it does |
|---|---|
| `MaNTA` | the solver binary, at `build/MaNTA` |
| `_manta` | the Python extension, into `python/manta/` |
| `UnitTests` | the Boost.Test binary, at `build/Tests/UnitTests/UnitTests` |
| `manta` | `libmanta.so`, for embedding the solver in another program |
| `unit_tests`, `regression_tests`, `python_tests` | run one suite (`ctest` runs all three) |
| `stubs`, `stubs-check`, `typecheck` | regenerate / verify `_manta.pyi`, and mypy the package |
| `docs` | Sphinx, into `docs/_build/html` |
| `coverage` | the gcovr reports; needs a `coverage` build directory |
| `venv` | create `.venv` from `requirements.txt` |
| `clean_data`, `clean_coverage` | sweep run output and instrumentation data |
| `install`, `uninstall` | headers, `libmanta.so` and `manta.pc` under a prefix |

`cmake --build build` with no target builds the solver, the library, the
extension and the unit tests.

### Testing

MaNTA has three test suites, all registered with CTest:

```sh
ctest --test-dir build                 # all three
ctest --test-dir build -R unit         # just one
ctest --test-dir build --output-on-failure
```

| Test | What it runs |
|---|---|
| `unit` | Boost.Test C++ unit tests (`Tests/UnitTests`) |
| `regression` | Runs the solver over `Tests/RegressionTests/*.conf` and compares against checked-in `.ref.nc` references |
| `python` | pytest suite for the `manta` package (`python/Tests`); needs the `_manta` target built |

The regression and Python suites need the Python dependencies. On distributions
where the system Python is externally managed (Debian, Ubuntu), that means a
virtualenv, which the `venv` target will build for you:

```sh
cmake --build build --target venv
cmake -B build -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
```

That installs `requirements.txt` plus `gcovr`, so every target is then runnable.
There is no need to put it on `PATH`: CMake records which interpreter to use, and
runs the regression driver and pytest with it — where the Makefile relied on
`PATH` and the regression driver's `env python3` shebang. A `.venv` in the
repository root is picked up automatically on a fresh configure, so the second
line above is only needed if you already configured without one.

It builds the environment with a *versioned* interpreter (`python3.13` by
default) on purpose: a venv created by plain `python3 -m venv` records the
unversioned `/usr/bin/python3`, so when your distribution moves that symlink to a
new release the environment's packages are stranded in the old
`lib/python3.X/site-packages` and every import fails. Pick a different one with
`-DMANTA_VENV_PYTHON=python3.12`, or a different location with `-DMANTA_VENV=...`.

All three suites can be run from any working directory.

### Writing a physics case in Python

A case and the driver that runs it do not have to live in this repository:

```sh
cmake --build build --target _manta   # builds python/manta/_manta<abi>.so
pip install .                         # the `manta` package and the `manta` command
pip install .[jax]                    # ...and manta.jax, for cases written as JAX functions
```

`pip install .` runs CMake itself, reusing `build/` if it is already configured,
so a machine whose SUNDIALS needed naming does not have to name it again here.

A case is then a subclass of `manta.TransportSystem` in your own package, run
with `manta myrun.conf`. `python-examples/` holds a worked directory per
example — the case, its config and a README — each importing `manta` exactly as
code outside this tree would, so one can be copied out and still run. Start with
`python-examples/linear-diffusion`; `docs/out_of_tree.rst` covers both languages,
including C++ cases built as `PhysicsPlugins` shared objects.

`python-physics/` follows the same convention for the systems that are run to
get physics rather than to demonstrate the framework — the centrifugal mirror
and the DESC/yancc stellarator.

#### Coverage

```sh
cmake --preset coverage
cmake --build build-coverage --target coverage
```

Coverage is a build *type*, in a build directory of its own: `--coverage -O0`,
no LTO (it destroys line attribution) and no `-Werror`. The target runs all three
suites and writes, under `build-coverage/coverage/`:

  * `index.html` -- the numerical core and the Python binding layer
  * `physics.html` -- `PhysicsCases/`, reported separately

There is no percentage threshold; it fails only if the build or a suite does.
`clean_coverage` removes the instrumentation data and the reports.

Note that both build directories write the Python extension to the same place,
`python/manta/`, because that is where `import manta` has to find it. Each build
directory records what it linked there and replaces anything it does not
recognise, so switching between `build/` and `build-coverage/` needs nothing from
you — the first build after the switch relinks the module, and the `coverage`
target additionally refuses to start if what is in place is not instrumented.

#### Installing SUNDIALS

If you are only building a version of SUNDIALS for use with MaNTA the included script `build_sundials` should provide
the minimal needed installation of SUNDIALS. If using MacOS, `coreutils` and `cmake` must be installed to run the build script.

MaNTA needs **IDA and KINSOL** — the second drives the steady-state solver, so a
SUNDIALS built without it fails at `kinsol/kinsol.h`, three files into the build,
rather than at the link step. `build_sundials` enables both. If you have an
install from a copy of that script predating the steady-state solver, rerun it;
a SUNDIALS built by hand from the instructions below has every solver already,
that being the cmake default.

If this is your first use of SUNDIALS, and you want a custom install, a quick guide to installing the base libraries follows here.

Pick where you want the sundials sources / build tree / compiled libraries to go. We will call these directories
SUNDIALS_SOURCE, SUNDIALS_BUILD, and SUNDIALS_INSTALL in the following. One suggestion would be
```
SUNDIALS_SOURCE = ~/sundials/source
SUNDIALS_BUILD  = ~/sundials/build
SUNDIALS_INSTALL = ~/sundials/
```

With these directories picked, we can download and compile SUNDIALS.

 1. Download the SUNDIALS source from [here](https://computing.llnl.gov/projects/sundials) or [here](https://github.com/LLNL/sundials) into `SUNDIALS_SOURCE`
 2. Move to `SUNDIALS_BUILD`. Configure the SUNDIALS build with
 ```
 cmake $SUNDIALS_SOURCE -DCMAKE_INSTALL_PREFIX=$SUNDIALS_INSTALL -DEXAMPLES_INSTALL=off
 ```
	   If this gives you any errors (lack of C compiler, etc), refer to the SUNDIALS documentation.
 3. Compile SUNDIALS with `make -j install`.
 4. You now have sundials installed into the `SUNDIALS_INSTALL` directory. This is the path to pass as `-DSUNDIALS_ROOT=` when you configure MaNTA.

### Providing new Transport Systems

The core MaNTA algorithm solves a generic set of reaction-diffusion equations. The
physics models are all contained in the `PhysicsCases/` directory, and every
`.cpp` in it is compiled and linked in — there is no list to add a new one to.

### Example Configurations

Example configurations live in the `examples/` subdirectory. 

### Superconvergence

Every run with `Polynomial_degree >= 1` reports an extra field per variable,
`u_star`, in its netCDF output and in the `.dat` files: the element-local
postprocessed solution in `P_{k+1}`, reconstructed from `u` and `q`.

Setting

```toml
[configuration]
Superconvergent = true
```

additionally switches the discretisation to the superconvergent interpolatory HDG
method of Chen, Cockburn, Singler and Zhang (*J. Sci. Comput.* **81**, 2188), in
which the physics is evaluated at the postprocessed solution. `u_star` then
converges one order faster than `u` — order `k+2` against `k+1`. The option
defaults to `false`, and with it off the discretisation is exactly what it was
before the option existed.

`Tests/README.md` has the measured convergence orders and the cases that are not
covered (notably `k = 0` and spatial adjoint parameters, both of which are
rejected with a clear error).

## Built With

* [Boost](http://boost.org) - C++ Template library that radically extends the STL
* [TOML11](http://github.com/toruniina/toml11) - For parsing configuration files written in [TOML](https://github.com/toml-lang/toml). Submodule, `extern/toml11`
* [autodiff](https://autodiff.github.io) - Forward-mode automatic differentiation, used by `AutodiffTransportSystem` to derive Jacobians from a physics case's flux and source. Submodule, `extern/autodiff`
* [pybind11](https://github.com/pybind/pybind11) - Builds the `MaNTA` Python module. Submodule, `extern/pybind11`
* [Eigen](https://eigen.tuxfamily.org) - Dense linear algebra throughout the solver
* [Sundials](https://computing.llnl.gov/projects/sundials) - Suite of libraries from Lawrence Livermore National Laboratory for numerical solution of Nonlinear Algebraic Equations, ODEs and DAEs
* [NETCDF C and NETCDF C++](https://www.unidata.ucar.edu/software/netcdf/) - A set of software libraries and machine-independent data formats that support the creation, access, and sharing of array-oriented scientific data.

## Known Issues


Specific known issues are listed here.


## Contributing

Contributions to this project through the github interface are welcome. Please email the authors to help out

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/ianabel/MaNTA/tags).

## Authors

* **Myles Kelly** - *Original author*
* **Ian Abel** - *Maintainer* - [Ian Abel at UMD](https://ireap.umd.edu/faculty/abel)
* **Eddie Tocco** - *Autodiff interfaces*

For full copyright attribution, see the [COPYRIGHT](COPYRIGHT) file.
For a summary of contributors, see the [contributors](http://github.com/ianabel/MCTrans/contributors) page.

## License

This project is licensed under the 3-Clause BSD Licence - see the [LICENSE.md](LICENSE.md) file for details
