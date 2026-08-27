# The Boost.Test C++ unit tests.
#
# Two properties of this target are load-bearing and easy to lose.
#
# 1. These sources are compiled with -DTEST; the core objects they link are NOT.
#    -DTEST widens MANTA_TEST_PRIVATE from private to public, which is how the
#    tests reach SystemSolver's internals without friend declarations. An object
#    from the core compiled with -DTEST is not the object the solver wants, and
#    nothing would say so -- under the Makefile this needed a hand-written rule
#    whose whole job was to refuse to build ../../Solver.o from this directory.
#    Linking $<TARGET_OBJECTS:manta_objects> makes it structural instead: those
#    objects belong to another target and carry that target's flags.
#
#    The Makefile linked a hand-maintained subset of the core objects here; this
#    links all of them, as the solver binary does. A physics case registers itself
#    by static-initialisation side effect and a duplicate name throws, so the
#    extra cases are not inert -- but the throw would take the whole binary down
#    at startup, which is about as visible as a failure gets.
#
# 2. One object per source, not one $(CXX) over all forty-one. The Boost.Test
#    arrangement already suits it -- <boost/test/included/unit_test.hpp> in
#    main.cpp alone, plain <boost/test/unit_test.hpp> in the rest, which is the
#    documented header-only-across-several-translation-units usage. CMake does
#    this by construction; it is worth noting because the single-invocation
#    version cost 4m23s at an average of 1.6 cores where the split costs 1m02s
#    at 11.6, and recompiled all forty-one whenever any one of them changed.
#
# A new test .cpp must be added to the list below. That is the same rule
# TEST_SOURCES in the old Makefile had, kept rather than replaced with a glob:
# these are named files with a reason to be built, where PhysicsCases/ is a
# directory whose whole contents are cases.

set(MANTA_TEST_SOURCES
  AFnTests.cpp
  AdjointProblemTests.cpp
  AdjointTests.cpp
  AutodiffTest.cpp
  ConfigSchemaTests.cpp
  ConfigSourceTests.cpp
  ConfigTests.cpp
  CoupledResidualTests.cpp
  DGSolnTests.cpp
  DGTests.cpp
  DegreeAdaptationTests.cpp
  FieldAdjointTests.cpp
  FieldDoFLayoutTests.cpp
  FieldJacobianTests.cpp
  FieldModelSpecTests.cpp
  FieldRegistryTests.cpp
  GeometryDerivativeTests.cpp
  GridTests.cpp
  IntegratorTest.cpp
  MMSAuxScalarTests.cpp
  MMSConvergenceTests.cpp
  MMSFieldTests.cpp
  MagneticFieldTest.cpp
  ManufacturedFieldTests.cpp
  MixedBoundaryTests.cpp
  NetCDFIOTests.cpp
  PhysicsRegistryTests.cpp
  PostprocessingTests.cpp
  PyIntegratorTests.cpp
  ResidualTests.cpp
  ScalarJacobianTests.cpp
  SolveJacTests.cpp
  SolverLifecycleTests.cpp
  SolverPlumbingTests.cpp
  StateGeometryTests.cpp
  SystemSolverMatrixTests.cpp
  SystemSolverTests.cpp
  TransportSystemTests.cpp
  UtilityTests.cpp
  main.cpp
)

add_executable(UnitTests ${MANTA_TEST_SOURCES})
target_link_libraries(UnitTests PRIVATE manta_objects ${MANTA_BOOST_TARGET})

# TEST_DATA_DIR lets the tests find their .nc fixtures whatever directory they
# are launched from -- testic.nc (AutodiffTest.cpp) and
# MatrixDiffusion.restart.nc (SystemSolverTests.cpp) are tracked test *inputs*
# living beside these sources, not build output.
target_compile_definitions(UnitTests PRIVATE
  TEST
  TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")

# Run from the build directory, so the .nc files a run drops land there rather
# than in the source tree. This is the one behavioural difference from
# `make test`, which ran the binary from the repo root and left its output beside
# the sources. `clean_data` still sweeps the repo root, so an existing tree
# carrying those files gets tidied rather than stranded.
add_test(NAME unit
         COMMAND UnitTests
         WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")
set_tests_properties(unit PROPERTIES LABELS "unit" TIMEOUT 3600)

add_custom_target(unit_tests
  COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${PROJECT_BINARY_DIR}" -R "^unit$" --output-on-failure
  USES_TERMINAL
  COMMENT "Running the C++ unit tests")
add_dependencies(unit_tests UnitTests)
