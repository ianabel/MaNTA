Building MaNTA
==============

Prerequisites
-------------

**CMake 3.22 or newer**, with a generator — GNU make or Ninja — and a **C++23**
compiler: g++ 15 or newer, or clang++ 18 or newer.

.. warning::

   **Avoid g++ 14 if you can.** Something in it breaks on this tree at the release
   flags (``-O3 -flto -march=native``), and the symptom is a wrong number rather
   than an error: adding any member to ``SystemSolver`` makes one of the Jacobian
   tests fail about half the time, with an O(1) error, and no sanitiser reports
   anything. g++ 15, g++ 16 and every clang tested are clean, and the build prints
   a warning if it sees g++ 14.

   It still compiles, and the solver's own output was reproducible in the case
   measured, so this is a recommendation rather than a hard floor — but the
   numbers from a g++ 14 release build are not ones this project can vouch for.
   The root cause is not yet known; ``CLAUDE.md`` and ``TODO`` carry the
   reproduction.

The clang floor is measured rather than assumed, and so was the old gcc one:
g++ 13 cannot build MaNTA at all — libstdc++ 13 has no ``<print>``, and the output
layer uses ``std::print`` throughout. g++ 14 does build it; the floor moved to 15
on the miscompile above, not on a language feature. clang++ 18 needed one source
change to admit: ``PyGrid.hpp`` used to declare ``constexpr Vector
getNodes(...)``, and a dynamically allocating Eigen vector is not a literal type,
which C++23 permits only under P2448R2 — implemented by clang 19 but not clang 18.
That ``constexpr`` was decorative and is now ``inline``.

clang++ 18, clang++ 19 and clang++ 21 have each been verified by hand to build the
solver, the Python module and all three test suites clean under ``-Wall -Werror``.
CI runs seven build legs: g++ 15, g++ 16, clang++ 19, clang++ 20 and clang++ 21
against the distribution's Eigen, then g++ 15 and clang++ 19 again against Eigen
5.0.1. g++ 14 and clang++ 18 are deliberately not among them, so both are verified
rather than guarded — and since g++ 14 is what Ubuntu noble ships, the gcc most
people have by default is now the one this project tests least.

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
     - Dense linear algebra throughout. Either 3.4.x or 5.0.x. A packaged Eigen
       is found with nothing configured; for an unpacked *source* tree, which
       carries no ``Eigen3Config.cmake``, use ``MANTA_EIGEN_INCLUDE_DIR``.
   * - SUNDIALS
     - **7.1.0 or newer**, built with **IDA and KINSOL**. Not 6.x — MaNTA links
       ``sundials_core`` and uses ``SUNContext``, neither of which exists before
       v7. KINSOL drives the steady-state solver; without it the build stops at
       ``kinsol/kinsol.h`` rather than at the link step.
   * - netCDF C and netCDF C++
     - 4.3 or newer for the C++ interface, which needs netCDF C 4.6.0 or newer.
   * - BLAS
     - Eigen is built with ``-DEIGEN_USE_BLAS``. A plain ``-lblas`` by default;
       see ``MANTA_BLAS_VENDOR`` below before changing that.

Three more libraries are **git submodules** under ``extern/``, so there is
nothing to install and no path to configure — see below:

* `toml11 <https://github.com/ToruNiina/toml11>`_ parses the configuration files.
* `autodiff <https://autodiff.github.io>`_ provides the forward-mode automatic
  differentiation that ``AutodiffTransportSystem`` uses to derive Jacobian
  entries from a flux and a source.
* `pybind11 <https://github.com/pybind/pybind11>`_ builds the Python module. Only
  needed for the ``_manta`` target; configure with ``-DMANTA_PYTHON=OFF`` to skip it.

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

That is the whole of the toml11, autodiff and pybind11 installation. None of the
three needs a path configured — the include directories are wired into the build.
If one is missing, the configure step stops with a message naming the submodule
and the command above, rather than at ``fatal error: toml.hpp``.

Configuring
-----------

MaNTA builds with CMake, out of source. There is no file you have to write first:

.. code-block:: sh

   cmake --preset default        # configure  -> build/
   cmake --build build -j        # compile
   ctest --test-dir build        # all three suites

In practice one dependency usually needs naming, because distributions rarely
package SUNDIALS 7 — unless you used the bundled ``build_sundials`` script, which
installs into ``./sundials/install``, where the configure step looks on its own:

.. code-block:: sh

   cmake --preset default -DSUNDIALS_ROOT=/somewhere/else/sundials/install

Anything else somewhere unusual is named the same way: ``-DEigen3_ROOT``,
``-DBOOST_ROOT``, or ``-DCMAKE_PREFIX_PATH="/opt/a;/opt/b"`` for several at once.
The quotes on that last one matter — the semicolon is CMake's list separator, not
the shell's.

.. note::

   Naming a prefix the compiler already searches is a **no-op**, and that is
   worth stating because it used not to be. Passing ``-isystem /usr/include``
   makes gcc and clang drop ``/usr/include`` from the end of the system chain and
   search it where the flag appeared — ahead of the libstdc++ headers — so
   ``<cstdlib>``'s ``#include_next <stdlib.h>`` finds nothing and every
   translation unit fails. ``Makefile.config`` carried a compiler probe and a
   hand-written filter for exactly that. CMake strips implicit include
   directories from the flags it generates, so the filter is gone; CI still
   checks the property holds on both compiler families.

Presets
~~~~~~~

``CMakePresets.json`` defines four, each with its own build directory so they can
coexist:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Preset
     - Effect
   * - ``default``
     - Release: ``-O3 -flto=auto -march=native``, into ``build/``.
   * - ``debug``
     - ``-O0 -g -DDEBUG -DPHYSICS_DEBUG``, into ``build-debug/``.
   * - ``coverage``
     - ``--coverage -O0``, no LTO, no ``-Werror``, into ``build-coverage/``.
   * - ``portable``
     - Release without ``-march=native``, into ``build-portable/``.

.. note::

   The Release configuration deliberately does **not** define ``NDEBUG``, which
   CMake would normally add. ``NDEBUG`` disables ``assert()`` and takes Eigen's
   own assertions with it, and those are the diagnostic of record for a class of
   defect here: when the adjoint's spatial-parameter branch wrote a
   ``(np, nPoints)`` block into an ``(nPoints, np)`` destination, the only thing
   that reported it was Eigen's ``resize()`` assertion. Under ``NDEBUG`` that run
   would have silently transposed a gradient instead.

``CMakeUserPresets.json``
~~~~~~~~~~~~~~~~~~~~~~~~~

Machine-specific paths belong in ``CMakeUserPresets.json``, which is gitignored
and is the direct replacement for the old ``Makefile.local``. Copy
``CMakeUserPresets.json.example`` and edit it; its presets ``inherit`` the ones
above, so you add only what your machine needs.

.. code-block:: json

   {
     "version": 3,
     "configurePresets": [
       {
         "name": "local",
         "inherits": "default",
         "cacheVariables": {
           "SUNDIALS_ROOT": "$env{HOME}/sundials/install",
           "CMAKE_CXX_COMPILER": "g++-15"
         }
       }
     ]
   }

Then ``cmake --preset local``. Anything on the command line still wins.

Options
~~~~~~~

``cmake -B build -LH`` lists every option with its description. The MaNTA-specific
ones:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Option
     - Effect
   * - ``MANTA_TESTS``
     - Build the Boost.Test unit tests. ``OFF`` also removes the Boost
       requirement. Default ``ON``.
   * - ``MANTA_PYTHON``
     - Build the ``manta`` Python extension. Default ``ON``.
   * - ``MANTA_OPENMP``
     - Enable the ``#pragma omp parallel for`` in the batched physics wrappers.
   * - ``MANTA_VERBOSE``
     - Define ``VERBOSE`` — extra solver logging.
   * - ``MANTA_PHYSICS_DEBUG``
     - Define ``PHYSICS_DEBUG`` without a full Debug build. Implied by ``Debug``.
   * - ``MANTA_NATIVE_ARCH``
     - ``-march=native`` in Release. Default ``ON``.
   * - ``MANTA_XLA_FFI`` / ``MANTA_CUDA``
     - The JAX FFI interface. Needs jaxlib headers.
   * - ``MANTA_EIGEN_INCLUDE_DIR``
     - Eigen headers, for a source checkout with no ``Eigen3Config.cmake``.
   * - ``MANTA_BLAS_VENDOR``
     - ``BLA_VENDOR`` for ``FindBLAS``. ``Generic`` (the default) is a plain
       ``-lblas``; ``Any`` lets CMake choose.
   * - ``MANTA_VENV`` / ``MANTA_VENV_PYTHON``
     - Where the ``venv`` target builds its environment, and with which
       interpreter.

.. warning::

   ``MANTA_BLAS_VENDOR`` defaults to ``Generic`` — a plain ``-lblas``, which is
   what the Makefile linked and what the distribution's alternatives symlink
   points at — rather than letting ``FindBLAS`` choose freely. On a box with
   Intel's libraries installed, an unconstrained ``FindBLAS`` takes the *layered*
   MKL link (``mkl_gf_lp64`` + ``mkl_gnu_thread`` + ``mkl_core`` + ``libgomp``).
   That combination is unsafe to ``dlopen``, and importing a C extension is a
   ``dlopen``: the Python module built that way dies mid-solve and takes the
   interpreter with it, while the standalone solver — linked from the very same
   objects — runs the whole regression suite. Change this deliberately, and test
   the Python suite if you do.

Installing SUNDIALS
-------------------

The included ``build_sundials`` script fetches and builds a minimal SUNDIALS
(IDA, KINSOL and the serial ``N_Vector`` only, no examples) into
``./sundials/install``:

.. code-block:: sh

   ./build_sundials
   # then, when configuring:  -DSUNDIALS_ROOT=./sundials/install

Override the version with ``SUNDIALS_VERSION=7.4.0 ./build_sundials``. On macOS
the script needs ``coreutils`` and ``cmake``.

.. note::

   An install from a copy of that script predating the steady-state solver has
   no KINSOL — it passed ``-DBUILD_KINSOL=OFF`` — so rerun the script if the
   build stops at ``kinsol/kinsol.h``. A SUNDIALS built by hand has every solver
   already, that being the cmake default.

Build targets
-------------

Built with ``cmake --build build --target <name>``.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Target
     - What it does
   * - ``MaNTA``
     - The solver binary, at ``build/MaNTA``.
   * - *(none)*
     - The solver, ``libmanta.so``, the Python module and the unit tests.
   * - ``_manta``
     - The pybind11 extension, ``python/manta/_manta<suffix>.so``.
   * - ``UnitTests``
     - The Boost.Test binary, at ``build/Tests/UnitTests/UnitTests``.
   * - ``manta``
     - ``libmanta.so``, for embedding the solver in another program.
   * - ``unit_tests`` / ``regression_tests`` / ``python_tests``
     - Run one suite. ``ctest --test-dir build`` runs all three.
   * - ``stubs`` / ``stubs-check`` / ``typecheck``
     - Regenerate ``_manta.pyi``, fail if the committed one is stale, and run
       mypy over the package.
   * - ``docs``
     - Sphinx, into ``docs/_build/html``.
   * - ``coverage``
     - Runs all three suites instrumented and writes the gcovr reports. Only
       does anything in a ``Coverage`` build directory.
   * - ``venv``
     - Creates ``.venv`` and installs the Python dependencies into it. See below.
   * - ``install`` / ``uninstall``
     - Headers, ``libmanta.so`` and ``manta.pc`` under a prefix.
   * - ``clean_data``
     - Just the run output: ``.nc``, ``.restart.nc`` and ``.dat`` at the repo
       root and in ``Tests/RegressionTests``, ``python/Tests`` and each
       directory under ``python-examples`` and ``python-physics``.
   * - ``clean_coverage``
     - Instrumentation data and the reports, in both the build and source trees.

There is no ``clean`` target to describe, and that is the point of an
out-of-source build: ``rm -rf build`` is the whole of it.

.. warning::

   ``clean_data`` spares files with ``.ref.`` in the name — the references the
   regression and pytest suites compare against — and it does not descend into
   subdirectories, so scratch and archive directories like ``runs/`` are left
   alone.

   It also skips ``Tests/UnitTests`` entirely. Every ``.nc`` in there is a
   tracked test *input* rather than output, and one of them —
   ``MatrixDiffusion.restart.nc``, read by ``SystemSolverTests.cpp`` — has no
   ``.ref.`` in its name, so the keep-pattern would not save it. Keep both facts
   in mind before adding a directory to the list in
   ``cmake/MantaCleanData.cmake``.

   The repo root is still swept even though the unit tests now run from the build
   directory, so a tree carrying output from the Makefile era is tidied rather
   than stranded.

Python dependencies
-------------------

The regression and Python suites need the packages in ``requirements.txt``. On
distributions where the system Python is externally managed that means a
virtualenv, and the ``venv`` target builds one:

.. code-block:: sh

   cmake --build build --target venv
   cmake -B build -DPython3_EXECUTABLE="$PWD/.venv/bin/python"

It installs ``requirements.txt`` plus ``gcovr``, so ``coverage`` works too.

There is no need to put it on ``PATH``. CMake records which interpreter to use
and runs the regression driver and pytest with that one, where the Makefile
relied on ``PATH`` and the regression driver's ``env python3`` shebang. A
``.venv`` in the repository root is picked up automatically on a fresh configure,
so the second line above is only needed if you configured before creating it.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Option
     - Effect
   * - ``-DMANTA_VENV_PYTHON=python3.12``
     - Build the environment against a different interpreter.
   * - ``-DMANTA_VENV=/path/to/env``
     - Put the environment somewhere else.
   * - ``-DMANTA_VENV_EXTRA=``
     - Skip ``gcovr``.
   * - ``-DPython3_EXECUTABLE=...``
     - Which interpreter the extension is built for and the tools run under.

.. note::

   The ``venv`` target uses a **versioned** interpreter (``python3.13`` by
   default) deliberately. A virtualenv records the interpreter it was created
   with, and ``python3 -m venv`` records the unversioned ``/usr/bin/python3``.
   When the distribution later moves that symlink to a new release, the
   environment's ``bin/python3`` follows it while the installed packages stay
   behind in ``lib/python3.<old>/site-packages``, and every import in the
   environment fails with ``No module named pytest``. Naming ``python3.13``
   records ``python3.13``, and the environment survives the upgrade.

.. note::

   The **ABI-mismatch trap is gone**, and it is worth knowing what it was. The
   Makefile named the module from ``python3-config --extension-suffix`` and took
   its headers from the same program, so the two matched each other but not
   necessarily the interpreter that would import them. After the system default
   moved, ``make python`` built ``_manta.cpython-3<new>-*.so`` while the
   virtualenv still ran the old release — and ``python_tests``, ``stubs-check``
   and ``typecheck`` then failed with three messages pointing somewhere else.
   ``stubs-check`` was the worst: regenerating the stub needs the import too, so
   it failed to write one and then reported a perfectly good committed file
   stale.

   CMake finds **one** interpreter and derives the headers, the ABI suffix and
   every tool command from it, so the three cannot disagree. If you want a
   different one, name it once with ``-DPython3_EXECUTABLE``.

All three suites can be run from any working directory.
