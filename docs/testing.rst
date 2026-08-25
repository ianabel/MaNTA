Testing
=======

MaNTA has three suites, all registered with CTest:

.. code-block:: sh

   ctest --test-dir build                      # all three
   ctest --test-dir build -R unit              # just one
   ctest --test-dir build --output-on-failure

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Test
     - What it runs
   * - ``unit``
     - Boost.Test C++ unit tests (``Tests/UnitTests``).
   * - ``regression``
     - The solver over ``Tests/RegressionTests/*.conf``, comparing each result
       against a checked-in ``.ref.nc``.
   * - ``python``
     - pytest over the Python module (``python/Tests``). Needs the ``_manta``
       target built first.

Each also has a build target of its own — ``unit_tests``, ``regression_tests``,
``python_tests`` — which builds what it needs and then runs that one suite.

The regression and Python suites need the packages in ``requirements.txt``:

.. code-block:: sh

   cmake --build build --target venv           # once

Nothing needs to go on ``PATH``. CMake records which interpreter to use and runs
the regression driver and pytest with that one, where the Makefile relied on
``PATH`` and the driver's ``env python3`` shebang. See :doc:`install` for what the
``venv`` target does and why it pins a versioned interpreter.

All three run from any working directory.

Running one test
----------------

.. code-block:: sh

   build/Tests/UnitTests/UnitTests --run_test=solve_jac_tests --log_level=all
   build/Tests/UnitTests/UnitTests --run_test=mms_convergence_tests --log_level=message
   pytest python/Tests/test_adjoint.py
   SOLVER=$PWD/build/MaNTA Tests/RegressionTests/TestSolutions.py --tolerance 1e-2

``--log_level=message`` is what shows ``BOOST_TEST_MESSAGE`` output, which is how
the convergence tests report measured orders.

``TestSolutions.py`` resolves the solver from ``$SOLVER``, falling back to
``<repo>/MaNTA`` — which an out-of-source build does not produce, hence the
variable above. CTest sets it for you.

New unit-test files must be added to ``MANTA_TEST_SOURCES`` in
``Tests/UnitTests/CMakeLists.txt``. That stays an explicit list rather than a
glob: these are named files with a reason to be built, where ``PhysicsCases/`` is
a directory whose whole contents are cases and is globbed.

What the suites are actually for
--------------------------------

This is worth stating explicitly, because the obvious reading — "the regression
suite protects correctness" — is wrong in an important way.

Because the Jacobian is never assembled (see :doc:`formulation`), **an error in
it does not produce a wrong answer, only slow Newton convergence.** A regression
suite that compares output against reference files will therefore pass with a
badly wrong Jacobian. Several defects in that area survived a passing regression
suite for months.

The tests that carry real weight are consequently:

* **``SolveJacTests.cpp``** — finite-differences the residual and requires the
  linear solve to satisfy :math:`J \, \delta y = g`. This is the only check that
  can catch an error in the Jacobian or in the chain rule, and it is the one to
  extend when you add a coupling. Note that ``residual`` does not write the
  Dirichlet boundary rows, so a finite-differenced Jacobian is rank-deficient by
  exactly the number of Dirichlet boundaries; the tests account for that.
* **``MMSConvergenceTests.cpp``** — measures observed order of accuracy against
  manufactured solutions. This is what catches a discretisation error that is
  consistent but wrong.
* **A closed-form comparison** — the only thing that catches a sign error in the
  flux, which converges at the right rate to the wrong function. See
  :ref:`the sign convention <sign-convention>`.

The regression suite's job is different and still valuable: it detects
*unintended* change. If a refactor is meant to be behaviour-preserving, the
``.ref.nc`` files are what prove it.

Coverage
--------

.. code-block:: sh

   cmake --preset coverage
   cmake --build build-coverage --target coverage

Coverage is a build *type*, in a build directory of its own: ``--coverage -O0`` —
no LTO, which destroys line attribution — and no ``-Werror``. The target runs all
three suites and writes, under ``build-coverage/coverage/``:

* ``index.html`` — the numerical core and the Python binding layer;
* ``physics.html`` — ``PhysicsCases/``, reported separately because it is
  exercised as test fixtures rather than gated on.

There is no percentage threshold: it fails only if the build or a suite does,
which is the same thing the other CI legs gate on.

``clean_coverage`` removes the instrumentation data and the reports. A separate
build directory is what makes that all there is to it — under the Makefile this
was one tree that recursed with ``COVERAGE=on``, needed ``env -u CXXFLAGS -u
LDFLAGS`` to stop the parent's release flags leaking in, and had to end with a
``make clean`` or the next ordinary build failed to link with undefined
references to ``__gcov_init``.

.. warning::

   Both build directories write the Python extension to the same place,
   ``python/manta/``, because that is where ``import manta`` has to find it. So
   whichever you built last owns the module: rebuild the ``_manta`` target in
   ``build/`` when you go back to Release work.

.. note::

   **gcov counts a templated line once per instantiation**, which makes
   header-heavy files look far worse than they are. Judge those by distinct
   uncovered lines rather than by the headline percentage. ``Tests/README.md`` has
   the numbers and the discussion.

   Whatever reads the ``.gcno``/``.gcda`` files must come from the same toolchain
   version that wrote them. ``cmake/MantaCompilerFlags.cmake`` derives
   ``MANTA_GCOV`` from ``CMAKE_CXX_COMPILER_VERSION`` for exactly this reason — a
   bare ``gcov`` is whatever the system default compiler provides, which need not
   match the compiler you built with. Override it with ``-DMANTA_GCOV=...``.

Continuous integration
----------------------

CI builds and runs all three suites under five compilers, one matrix leg each:
g++ 14, g++ 15, clang++ 19, clang++ 20 and clang++ 21, plus a separate coverage
job. ``fail-fast`` is off, so when one compiler breaks the others' results are
still there to tell you whether it is a real bug or that compiler's opinion.

Running the regression suite under clang doubles as a cross-compiler check on the
numerics, since the reference files were generated with gcc.

Further reading
---------------

``Tests/README.md`` is the detailed companion to this page: test conventions,
what each suite does and does not cover, the measured coverage numbers, and the
current known gaps. Read it before adding tests or interpreting a coverage
number.
