Testing
=======

MaNTA has three suites, all driven from the top-level Makefile:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - What it runs
   * - ``make test``
     - Boost.Test C++ unit tests (``Tests/UnitTests``).
   * - ``make regression_tests``
     - The solver over ``Tests/RegressionTests/*.conf``, comparing each result
       against a checked-in ``.ref.nc``.
   * - ``make python_tests``
     - pytest over the Python module (``python/Tests``). Needs ``make python``
       first.

The regression and Python suites need the packages in ``requirements.txt`` and
the virtualenv on ``PATH`` — the regression driver's shebang is ``env python3``:

.. code-block:: sh

   make venv                            # once
   export PATH="$PWD/.venv/bin:$PATH"

See :doc:`install` for what ``make venv`` does and why it pins a versioned
interpreter.

All three run from any working directory.

Running one test
----------------

.. code-block:: sh

   Tests/UnitTests/UnitTests --run_test=solve_jac_tests --log_level=all
   Tests/UnitTests/UnitTests --run_test=mms_convergence_tests --log_level=message
   pytest python/Tests/test_adjoint.py
   Tests/RegressionTests/TestSolutions.py --tolerance 1e-2

``--log_level=message`` is what shows ``BOOST_TEST_MESSAGE`` output, which is how
the convergence tests report measured orders.

New unit-test files must be added to ``TEST_SOURCES`` in
``Tests/UnitTests/Makefile``.

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

   make coverage

This rebuilds everything with ``--coverage -O0`` — no LTO, which destroys line
attribution — runs all three suites, and writes:

* ``coverage/index.html`` — the numerical core and the Python binding layer;
* ``coverage/physics.html`` — ``PhysicsCases/``, reported separately because it is
  exercised as test fixtures rather than gated on.

``make clean_coverage`` removes the instrumentation data and the report. The
``coverage`` target ends with a ``make clean`` so the tree is left buildable;
without it the instrumented objects remain and the next ordinary build fails to
link with undefined references to ``__gcov_init``.

.. note::

   **gcov counts a templated line once per instantiation**, which makes
   header-heavy files look far worse than they are. Judge those by distinct
   uncovered lines rather than by the headline percentage. ``Tests/README.md`` has
   the numbers and the discussion.

   Whatever reads the ``.gcno``/``.gcda`` files must come from the same toolchain
   version that wrote them. ``Makefile.config`` derives ``GCOV`` from ``CXX`` for
   exactly this reason — a bare ``gcov`` is whatever the system default compiler
   provides, which need not match the compiler you built with.

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
