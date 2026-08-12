Building MaNTA
==============

Prerequisites
-------------

A **C++23** compiler: g++ 14 or newer, or clang++ 18 or newer.

Both floors are measured rather than assumed. g++ 13 cannot build MaNTA at all —
libstdc++ 13 has no ``<print>``, and the output layer uses ``std::print``
throughout. clang++ 18 needed one source change to admit: ``PyGrid.hpp`` used to
declare ``constexpr Vector getNodes(...)``, and a dynamically allocating Eigen
vector is not a literal type, which C++23 permits only under P2448R2 —
implemented by clang 19 but not clang 18. That ``constexpr`` was decorative and
is now ``inline``.

g++ 14, clang++ 18, clang++ 19 and clang++ 21 have each been verified by hand to
build the solver, the Python module and all three test suites clean under
``-Wall -Werror``. CI runs five compilers, one matrix leg each: g++ 14, g++ 15,
clang++ 19, clang++ 20 and clang++ 21. clang++ 18 is deliberately not among them,
so it is verified rather than guarded.

.. note::

   With clang, **the standard library matters as much as the compiler.** clang
   uses the newest GCC installation it finds, so a local clang build and CI's
   clang legs can compile against different versions of libstdc++ — CI's get the
   ``ubuntu-24.04`` image's libstdc++ 14, while a developer box with g++ 15
   installed gets libstdc++ 15.

   This has bitten once already: three ``std::bind_front`` conversions built fine
   under clang++ 21 with libstdc++ 15 and were rejected by the same compiler with
   libstdc++ 14.2, because ``_Bind_front::operator()`` has two implementations
   selected by ``__cpp_explicit_this_parameter`` — a macro clang defines only from
   version 20. If you are checking portability locally, pin the library too:
   ``--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/14``.

System libraries, which you install yourself:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Library
     - Notes
   * - Boost
     - Headers only. Boost.Test is used in header-only mode, so there is no
       ``unit_test_framework`` to link.
   * - Eigen
     - Dense linear algebra throughout. Set ``EIGEN_DIR`` if it is not in
       ``/usr/include``.
   * - SUNDIALS
     - **7.1.0 or newer.** Not 6.x — MaNTA links ``sundials_core`` and uses
       ``SUNContext``, neither of which exists before v7.
   * - netCDF C and netCDF C++
     - 4.3 or newer for the C++ interface, which needs netCDF C 4.6.0 or newer.
   * - BLAS
     - Eigen is built with ``-DEIGEN_USE_BLAS``.

Three more libraries are **git submodules** under ``extern/``, so there is
nothing to install and no path to configure — see below:

* `toml11 <https://github.com/ToruNiina/toml11>`_ parses the configuration files.
* `autodiff <https://autodiff.github.io>`_ provides the forward-mode automatic
  differentiation that ``AutodiffTransportSystem`` uses to derive Jacobian
  entries from a flux and a source.
* `pybind11 <https://github.com/pybind/pybind11>`_ builds the Python module. Only
  needed for ``make python``.

Getting the source
------------------

Clone with submodules:

.. code-block:: sh

   git clone --recurse-submodules https://github.com/ianabel/MaNTA.git

If you have already cloned without that flag, ``extern/`` will contain three
empty directories and the build will stop at
``fatal error: toml.hpp: No such file or directory``. Populate them with

.. code-block:: sh

   git submodule update --init

That is the whole of the toml11, autodiff and pybind11 installation.
``Makefile.config`` already defaults ``TOML11_DIR`` and ``AUTODIFF_DIR`` into
``extern/``, and the pybind11 include path is not configurable at all, so none
of the three needs a ``Makefile.local`` entry. Set ``TOML11_DIR`` or
``AUTODIFF_DIR`` only if you deliberately want to build against your own copy
elsewhere.

``Makefile.local``
------------------

Every build option lives in ``Makefile.local``, which you must provide.
``Makefile.local.example`` is a starting point; the Makefile errors out with a
pointer to it if the file is missing.

.. code-block:: make

   CXX = g++-14
   SUNDIALS_DIR = ../sundials/install
   EIGEN_DIR = /usr/include/eigen3
   # BOOST_DIR = /path/to/boost          # only for a non-system install
   # NETCDF_DIR = /path/to/netcdf        # ditto
   # NETCDF_CXX_DIR = /path/to/netcdf-cxx

Anything set here can be overridden on the make command line, which takes
precedence — ``make CXX=clang++-19`` for instance.

Installing SUNDIALS
-------------------

The included ``build_sundials`` script fetches and builds a minimal SUNDIALS
(IDA and the serial ``N_Vector`` only, no examples) into ``./sundials/install``:

.. code-block:: sh

   ./build_sundials
   # then in Makefile.local:  SUNDIALS_DIR = ./sundials/install

Override the version with ``SUNDIALS_VERSION=7.4.0 ./build_sundials``. On macOS
the script needs ``coreutils`` and ``cmake``.

Build targets
-------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Command
     - What it does
   * - ``make MaNTA``
     - The solver binary, and nothing else.
   * - ``make``
     - The solver, the Python module, and then the test suites.
   * - ``make python``
     - The pybind11 extension, ``python/manta/_manta<suffix>.so``.
   * - ``make test``
     - The Boost.Test C++ unit tests.
   * - ``make regression_tests``
     - The solver over ``Tests/RegressionTests/*.conf``, compared against
       checked-in references.
   * - ``make python_tests``
     - The pytest suite for the Python module.
   * - ``make coverage``
     - Rebuilds instrumented, runs all three suites, writes ``coverage/``.
   * - ``make venv``
     - Creates ``.venv`` and installs the Python dependencies into it. See below.
   * - ``make clean``
     - Also sweeps orphaned ``PhysicsCases/*.o`` and ``.d`` files, every
       ABI-suffixed module under ``python/manta``, the bytecode and pytest
       caches, and — via ``clean_data`` — the run output.
   * - ``make clean_data``
     - Just the run output: ``.nc``, ``.restart.nc`` and ``.dat`` at the repo
       root and in ``Tests/RegressionTests``, ``python/Tests`` and each
       directory under ``python-examples``.

.. warning::

   ``clean_data`` spares files with ``.ref.`` in the name — the references the
   regression and pytest suites compare against — and it does not descend into
   subdirectories, so scratch and archive directories like ``runs/`` are left
   alone.

   It also skips ``Tests/UnitTests`` entirely. Every ``.nc`` in there is a
   tracked test *input* rather than output, and one of them —
   ``MatrixDiffusion.restart.nc``, read by ``SystemSolverTests.cpp`` — has no
   ``.ref.`` in its name, so the keep-pattern would not save it. Nothing is lost
   by the omission: ``make test`` runs the binary from the repo root, so
   unit-test output lands there. Keep both facts in mind before adding a
   directory to ``CLEAN_DATA_DIRS``.

Build variants are set on the command line, for example ``make DEBUG=on test``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Variant
     - Effect
   * - ``DEBUG``
     - ``-O0 -g -DDEBUG -DPHYSICS_DEBUG``. Also makes ``State.hpp``'s
       ``checkShapeAndSet`` shape-checking rather than a plain assignment, and is
       required for the debug ``.dat`` output.
   * - ``OMP``
     - Enables the ``#pragma omp parallel for`` in the batched physics wrappers.
   * - ``COVERAGE``
     - ``--coverage -O0``, no LTO.
   * - ``VERBOSE``
     - Extra logging.
   * - ``XLA_FFI`` / ``CUDA``
     - The JAX FFI interface. Needs jaxlib headers.

Python dependencies
-------------------

The regression and Python suites need the packages in ``requirements.txt``. On
distributions where the system Python is externally managed that means a
virtualenv, and ``make venv`` builds one:

.. code-block:: sh

   make venv
   export PATH="$PWD/.venv/bin:$PATH"

It installs ``requirements.txt`` plus ``gcovr``, so ``make coverage`` works too.
Putting it on ``PATH`` matters because the regression driver's shebang is
``env python3``, so it takes whichever ``python3`` comes first.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Override
     - Effect
   * - ``make venv VENV_PYTHON=python3.12``
     - Build against a different interpreter.
   * - ``make venv VENV=/path/to/env``
     - Put the environment somewhere else.
   * - ``make venv VENV_CREATE_FLAGS=--clear``
     - Rebuild an existing environment from scratch.
   * - ``make venv VENV_EXTRA=``
     - Skip ``gcovr``.

.. note::

   ``make venv`` uses a **versioned** interpreter (``python3.13`` by default)
   deliberately. A virtualenv records the interpreter it was created with, and
   ``python3 -m venv`` records the unversioned ``/usr/bin/python3``. When the
   distribution later moves that symlink to a new release, the environment's
   ``bin/python3`` follows it while the installed packages stay behind in
   ``lib/python3.<old>/site-packages``, and every import in the environment fails
   with ``No module named pytest``. Naming ``python3.13`` records
   ``python3.13``, and the environment survives the upgrade.

   The same upgrade catches ``make python``, which asks ``python3-config`` for the
   ABI suffix: after the system default moves, it starts building
   ``MaNTA.cpython-3<new>-*.so`` while the virtualenv still runs the old release
   and imports whichever stale module is left over — so the tests exercise old
   code. The Makefile therefore prefers the ``pythonX.Y-config`` matching
   ``$(VENV)`` when both exist, and falls back to plain ``python3-config``
   otherwise. Override it with ``make python PYTHON_CONFIG=python3.12-config``.

All three suites can be run from any working directory.
