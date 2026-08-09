# MaNTA -- The Maryland Nonlinear Transport Analyzer

MaNTA is a code developed for the prediction and analysis of transport processes in a strongly-magnetized plasma.
It solves one-dimensional nonlinear reaction-diffusion and transport systems with a hybridizable discontinuous
Galerkin (HDG) spatial discretisation, integrated in time as an index-1 DAE by SUNDIALS IDA.

**Documentation** lives in [`docs/`](docs/index.rst) and is built with Sphinx for Read the Docs: the equations
being solved, every configuration key, the output format, the adjoint interface, and how to write a transport
system in C++ or Python. Build it locally with

```sh
python3 -m venv /tmp/docsvenv
/tmp/docsvenv/bin/pip install -r docs/requirements.txt
/tmp/docsvenv/bin/sphinx-build -W -b html docs docs/_build/html
```

## Getting Started

You will need to download this codebase and compile it in order to run MaNTA

### Prerequisites

To compile and use MaNTA you will need a system with the following

 - A C++23 compliant C++ compiler: **g++ 14 or newer**, or **clang++ 18 or newer**.
   Verified by hand on g++ 14, clang++ 18, clang++ 19 and clang++ 21 -- each builds
   the solver, the pybind11 module and all three test suites clean under
   `-Wall -Werror`. Pick one by setting `CXX` in `Makefile.local`, or on the make
   command line (`make CXX=clang++-19`), which overrides it.

   CI runs five, one matrix leg each: g++ 14, g++ 15, clang++ 19, clang++ 20 and
   clang++ 21. **clang++ 18 is deliberately not among them**, so it is verified
   rather than guarded, and can regress without anyone noticing -- adding it would
   mean maintaining clang-18 `-Werror` compliance, and its C++23 support has gaps.

   The lower bound on gcc was measured, not guessed: **g++ 13 cannot build MaNTA at
   all**, because libstdc++ 13 has no `<print>` and the output layer uses
   `std::print` throughout. clang++ 18 took one source change to admit --
   `PyGrid.hpp` used to declare `constexpr Vector getNodes(...)`, and a dynamic
   Eigen vector is not a literal type; C++23 permits that under P2448R2 provided the
   function is never constant-evaluated, which clang 18 does not implement. That
   `constexpr` could never have done anything and is now `inline`.
 - The Boost C++ Template Library
 - The Eigen linear algebra template library
 - The SUNDIALS library, Version 7.1.0 or newer. Not 6.x: MaNTA links
   `sundials_core` and uses `SUNContext`, neither of which exists before v7.
 - NETCDF C and NETCDF C++ 4.3 or newer (depends upon netcdf C interface 4.6.0 or newer)

Three further libraries are **bundled as git submodules** under `extern/`, so
there is nothing to install and no path to configure for them -- you only need to
initialise them, which step 1 below covers:

 - [toml11](http://github.com/toruniina/toml11) -- parses the configuration files
 - [autodiff](https://autodiff.github.io) -- forward-mode automatic
   differentiation, which is how `AutodiffTransportSystem` derives every Jacobian
   entry from a physics case's `Flux` and `Source`
 - [pybind11](https://github.com/pybind/pybind11) -- needed only for the Python
   module, i.e. `make python` and `make python_tests`

Precise dependencies have not been exhaustively tested, bug reports are welcome. Running on Windows requires the installation of [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install) (WSL) and proceeding as for linux.

MacOS has yet to be tested.

### Building MaNTA

All the build options are set in the file `Makefile.local`, which you need to provide for your system.
An example is provided in `Makefile.local.example` -- copy this file to `Makefile.local` and make any edits needed.
This file is in GNU-compatible Makefile format, and you can set and override all the compilation options here.
For example, if you are not using the default compiler (g++), then you can add a line to `Makefile.local` that reads `CXX = /path/to/my/c++/compiler`.

If you're happy with this, let's proceed!

 1. Clone this repository into your chosen location, **with its submodules**:

    ```sh
    git clone --recurse-submodules https://github.com/ianabel/MaNTA.git
    ```

    If you have already cloned without that flag, `extern/` will contain three
    empty directories and the build will stop at a missing `toml.hpp`. Populate
    them with

    ```sh
    git submodule update --init
    ```

    That is the whole of the toml11, autodiff and pybind11 installation.
    `Makefile.config` already defaults `TOML11_DIR` and `AUTODIFF_DIR` into
    `extern/`, and the pybind11 include path is not configurable at all, so none
    of the three needs a `Makefile.local` entry -- set `TOML11_DIR` or
    `AUTODIFF_DIR` only if you deliberately want to build against your own copy
    somewhere else. (`git submodule update --init --recursive` is what CI runs and
    also works; it additionally fetches toml11's own documentation and test
    submodules, which MaNTA does not use.)
 2. Install the Boost library, either using your system package manager or manually by downloading from [here](https://www.boost.org). If this is a system-wide install,
 proceed to the next step. If you downloaded the Boost libraries, add a line to `Makefile.local` which sets `BOOST_DIR = /path/to/boost`.
 3. Install [SUNDIALS](https://computing.llnl.gov/projects/sundials) and edit Makefile.local to set `SUNDIALS_DIR` to the location you have installed the Sundials library in. If you are only using SUNDIALS for MaNTA, a quick intro to installing SUNDIALS is included below.
 4. Install [NETCDF C and NETCDF C++](https://www.unidata.ucar.edu/software/netcdf/). On Ubuntu or Debian these can be installed from the package manager: `apt-get install libnetcdf-dev libnetcdff-dev libnetcdf-c++4-dev libnetcdf-c++4-1`. 
 On MacOS, you can use either `brew install netcdf` or `conda install -c anaconda netcdf4` to install the C version, and `conda install -c conda-forge netcdf-cxx4` to install the C++ version. 
 Please specify in `Makefile.local` where these libraries are installed. For example, `NETCDF_DIR = /usr/local/Cellar/netcdf/4.8.0_2` and `NETCDF_CXX_DIR = /Users/<username>/miniconda3` if you used `brew` and `conda` to install on MacOS.
 5. Set any other options, e.g. setting the variable `DEBUG` to any value will build a version that you can use to develop MaNTA and that includes debug information.
 6. Run `make`.
 7. Check the unit tests with `make test`. 

### Testing

MaNTA has three test suites, all driven from the top-level Makefile:

| Command | What it runs |
|---|---|
| `make test` | Boost.Test C++ unit tests (`Tests/UnitTests`) |
| `make regression_tests` | Runs the solver over `Tests/RegressionTests/*.conf` and compares against checked-in `.ref.nc` references |
| `make python_tests` | pytest suite for the pybind11 module (`python/Tests`); needs `make python` first |

The regression and Python suites need the Python dependencies. On distributions
where the system Python is externally managed (Debian, Ubuntu), that means a
virtualenv, which `make venv` will build for you:

```sh
make venv
export PATH="$PWD/.venv/bin:$PATH"   # the regression driver uses `env python3`
```

That installs `requirements.txt` plus `gcovr`, so every target in the Makefile is
then runnable. It builds the environment with a *versioned* interpreter
(`python3.13` by default) on purpose: a venv created by plain `python3 -m venv`
records the unversioned `/usr/bin/python3`, so when your distribution moves that
symlink to a new release the environment's packages are stranded in the old
`lib/python3.X/site-packages` and every import fails. Pick a different one with
`make venv VENV_PYTHON=python3.12`, or a different location with
`make venv VENV=/path/to/env`.

All three suites can be run from any working directory.

#### Coverage

```sh
make coverage
```

This rebuilds everything with `--coverage -O0` (no LTO -- it destroys line
attribution), runs all three suites, and writes:

  * `coverage/index.html` -- the numerical core and the Python binding layer
  * `coverage/physics.html` -- `PhysicsCases/`, reported separately

`make clean_coverage` removes the instrumentation data and the report.

#### Installing SUNDIALS

If you are only building a version of SUNDIALS for use with MaNTA the included script `build_sundials` should provide
the minimal needed installation of SUNDIALS. If using MacOS, `coreutils` and `cmake` must be installed to run the build script.

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
 4. You now have sundials installed into the `SUNDIALS_INSTALL` directory. This is the path you should set `SUNDIALS_DIR` to in your MCTrans `Makefile.local`

### Providing new Transport Systems

The core MaNTA algorithm solves a generic set of reaction-diffusion equations. The physics models are all contained in the `PhysicsCases/` directory.

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
