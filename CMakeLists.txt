# MaNTA -- the Maryland Nonlinear Transport Analyzer.
#
#   cmake --preset default          # or: cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ctest --test-dir build
#
# Layout of the build, and why it is shaped this way:
#
#   manta_objects   every core .cpp plus PhysicsCases/, compiled once, -fPIC
#   MaNTA           the solver binary            = main.cpp + those objects
#   manta           libmanta.so, for embedding   =             those objects
#   _manta          the Python extension         = Python.cpp, PyRunner.cpp + those
#   UnitTests       Boost.Test suite             = the test .cpp files + those
#
# One OBJECT library rather than a static library, because all four consumers
# want the same objects and none of them wants --whole-archive semantics
# negotiated for it. It also preserves the property the Makefile had: the
# registry in PhysicsCases.cpp is populated by static-initialisation side
# effects, so a case that is dropped by the linker is not a compile error but a
# case that silently does not exist.
#
# The one asymmetry worth knowing: UnitTests compiles its OWN sources with -DTEST
# (which widens MANTA_TEST_PRIVATE from private to public) and links the core
# objects built WITHOUT it. That is exactly what the Makefile did, and it is
# deliberate -- an object compiled here with -DTEST is not the object the solver
# wants, and nothing would say so.

cmake_minimum_required(VERSION 3.22)

project(MaNTA
        VERSION 0.1.0
        DESCRIPTION "Maryland Nonlinear Transport Analyzer"
        HOMEPAGE_URL "https://manta-docs.readthedocs.io"
        LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)

# In-source builds would drop the object tree on top of the sources the way the
# Makefile did, and the point of moving to CMake is that they no longer have to.
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not supported. Use a build directory:\n"
    "    cmake -B build && cmake --build build -j")
endif()

option(MANTA_TESTS   "Build the Boost.Test C++ unit tests" ON)
option(MANTA_PYTHON  "Build the manta Python extension"    ON)
option(MANTA_OPENMP  "Enable the OpenMP pragmas in the batched physics wrappers" OFF)
option(MANTA_XLA_FFI "Build the JAX XLA FFI bindings (needs jaxlib headers)" OFF)
option(MANTA_CUDA    "With MANTA_XLA_FFI, also build the CUDA FFI targets"   OFF)

include(MantaCompilerFlags)
include(MantaDependencies)

# Generated compile_commands.json, for clangd and friends. On by default because
# it costs nothing and the tree already gitignores it.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------- the core --

set(MANTA_CORE_SOURCES
  AdjointVectors.cpp
  Config.cpp
  ConfigSchema.cpp
  DGStatic.cpp
  DegreeAdaptation.cpp
  ErrorChecker.cpp
  FieldModel.cpp
  MaNTA.cpp
  Matrices.cpp
  NetCDFIO.cpp
  PhysicsCases.cpp
  Postprocessing.cpp
  Solver.cpp
  SolverConfig.cpp
  SteadyState.cpp
  SunLinSolWrapper.cpp
  SunMatrixWrapper.cpp
  SystemSolver.cpp
)

# A glob, matching the $(wildcard PhysicsCases/*.cpp) it replaces, so adding a
# physics case stays a matter of dropping a file in. CONFIGURE_DEPENDS makes the
# generator re-run the glob, which is what stops the usual objection to globbing
# from applying: a new case is picked up by `cmake --build` without a manual
# re-configure.
#
# PhysicsCases/ is flat again -- the MirrorPlasma subdirectory went with the C++
# mirror plasma. If a case ever needs a subdirectory, add its glob here; a source
# that is not matched produces no compile error, only an unregistered case and a
# link failure at the point something names it.
file(GLOB MANTA_PHYSICS_SOURCES CONFIGURE_DEPENDS
     "${PROJECT_SOURCE_DIR}/PhysicsCases/*.cpp")

add_library(manta_objects OBJECT ${MANTA_CORE_SOURCES} ${MANTA_PHYSICS_SOURCES})
add_library(manta::objects ALIAS manta_objects)
target_include_directories(manta_objects PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(manta_objects PUBLIC manta::flags manta::deps)

# ------------------------------------------------------------- the binaries --

add_executable(MaNTA main.cpp)
target_link_libraries(MaNTA PRIVATE manta_objects)

# -rdynamic, which is what lets a dlopen'd physics plugin resolve MaNTA's own
# symbols against the running solver.
#
# A plugin must NOT link libmanta.so. The solver links the objects directly, so a
# plugin that also pulled in libmanta would get a second copy of
# PhysicsCases::map and register itself into a map the solver never reads --
# silently, with the case simply missing at run time. Compile a plugin against the
# headers alone and let the loader bind its MaNTA symbols to the host process.
set_target_properties(MaNTA PROPERTIES ENABLE_EXPORTS ON)

# libmanta.so is for embedding the solver in another program, which is a
# different job from writing a plugin.
#
# The generator expression is this target's source list: a SHARED library with no
# sources of its own has no linker language and CMake refuses it. Linking
# manta_objects as well is not a second copy -- CMake de-duplicates the objects
# against the list above (checked: 37 objects on the link line, none repeated) --
# it is how the usage requirements arrive.
add_library(manta SHARED $<TARGET_OBJECTS:manta_objects>)
target_link_libraries(manta PUBLIC manta_objects)
set_target_properties(manta PROPERTIES
  OUTPUT_NAME manta
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})

# --------------------------------------------------------------- Python --
#
# Before the tests, because the regression suite is a Python script and wants the
# same interpreter -- and after the core targets, because pybind11's subdirectory
# is only added when MANTA_PYTHON is on.
include(MantaPython)

# ----------------------------------------------------------------- tests --
#
# include(CTest) comes first, and the ordering is load-bearing: add_test() before
# enable_testing() is silently ignored, so a suite registered in a subdirectory
# added above this line simply does not exist and `ctest` reports the smaller
# number without comment. That is exactly how the Python suite went missing the
# first time this file was written.
include(CTest)

if(MANTA_TESTS)
  add_subdirectory(Tests/UnitTests)
endif()
add_subdirectory(Tests/RegressionTests)

# ------------------------------------------------------- the Python module --

if(MANTA_PYTHON)
  add_subdirectory(python)
endif()

# --------------------------------------------------------------- install --
#
# Enough for a physics case to be built outside this tree: the headers it
# includes, a shared library with the solver and the registry in it, and a
# pkg-config file so the case's own build can find both.
#
#     cmake --install build --prefix /where/you/want
#     g++ -shared -fPIC $(pkg-config --cflags manta) MyCase.cpp -o libmycase.so
#
# and name libmycase.so in the config's PhysicsPlugins array.
#
# Deliberately pkg-config only, with no exported CMake package. An installed
# MaNTAConfig.cmake would invite `target_link_libraries(mycase manta)`, and
# linking libmanta is precisely the mistake that gives a plugin its own copy of
# the registry. The rule is easier to state when there is no target to link.

# Every header a physics case can reach transitively, keeping the directory
# layout so their quoted includes of each other still resolve.
#
# FieldModel.hpp and FieldModelSpec.hpp are here for the same reason
# PhysicsCases.hpp is: the field-model registry, the (toml::value, Grid)
# constructor and the REGISTER_FIELD_MODEL_* macros exist for out-of-tree models,
# so a model cannot be built at all without them.
#
# ConfigSchema.hpp, SolverConfig.hpp and PyConfigSource.hpp are deliberately
# absent. An out-of-tree case is handed the parsed toml::value and reads its own
# table through toml11, never through the solver's schema.
set(MANTA_INSTALL_HEADERS
  AdjointProblem.hpp
  Basis.hpp
  Config.hpp
  DGApprox.hpp
  DGSoln.hpp
  ErrorChecker.hpp
  FieldModel.hpp
  FieldModelSpec.hpp
  Logging.hpp
  NetCDFIO.hpp
  PhysicsCases.hpp
  Postprocessing.hpp
  PyGrid.hpp
  PyIntegrator.hpp
  State.hpp
  SunLinSolWrapper.hpp
  SunMatrixWrapper.hpp
  SystemSolver.hpp
  SystemSpec.hpp
  TransportSystem.hpp
  Types.hpp
  gridStructures.hpp
)

install(FILES ${MANTA_INSTALL_HEADERS}
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/manta")
install(DIRECTORY util/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/manta/util"
        FILES_MATCHING PATTERN "*.hpp")
install(TARGETS manta LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

set(MANTA_PC_STD "c++${MANTA_CXX_STANDARD}")
set(MANTA_PC_MARCH "")
if(MANTA_ABI_MARCH)
  set(MANTA_PC_MARCH "-march=${MANTA_ABI_MARCH}")
endif()
# Written at configure time as well, so `pkg-config --cflags` can be pointed at
# the build tree during development, and re-written at install time so it records
# the prefix the files actually went to. See cmake/MantaInstallPc.cmake.in.
configure_file(cmake/manta.pc.in "${PROJECT_BINARY_DIR}/manta.pc" @ONLY)
configure_file(cmake/MantaInstallPc.cmake.in
               "${PROJECT_BINARY_DIR}/MantaInstallPc.cmake" @ONLY)
install(SCRIPT "${PROJECT_BINARY_DIR}/MantaInstallPc.cmake")

# CMake has no built-in uninstall. This is the conventional install-manifest
# implementation, kept because `make uninstall` existed and dropping it would be
# a regression rather than a simplification. It removes exactly what the last
# install wrote, so an install to a different prefix leaves the earlier one alone.
add_custom_target(uninstall
  COMMAND ${CMAKE_COMMAND} -DMANIFEST=${PROJECT_BINARY_DIR}/install_manifest.txt
          -P "${PROJECT_SOURCE_DIR}/cmake/MantaUninstall.cmake"
  COMMENT "Removing the files listed in install_manifest.txt"
  VERBATIM)

# --------------------------------------------------- housekeeping targets --

include(MantaTools)

# ------------------------------------------------------------- the summary --

message(STATUS "")
message(STATUS "MaNTA ${PROJECT_VERSION}")
message(STATUS "  build type      ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler        ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CXX_COMPILER})")
message(STATUS "  SUNDIALS        ${SUNDIALS_VERSION}")
message(STATUS "  Eigen           ${Eigen3_VERSION} (${EIGEN3_INCLUDE_DIR})")
message(STATUS "  netCDF          ${MANTA_NETCDF_ORIGIN}")
message(STATUS "  BLAS            ${BLAS_LIBRARIES} [vendor: ${MANTA_BLAS_VENDOR_USED}]")
if(MANTA_PYTHON)
  message(STATUS "  Python          ${Python3_VERSION} (${Python3_EXECUTABLE})")
else()
  message(STATUS "  Python          off")
endif()
message(STATUS "  unit tests      ${MANTA_TESTS}")
if(MANTA_ABI_MARCH)
  message(STATUS "  plugin -march   ${MANTA_ABI_MARCH}")
endif()
message(STATUS "")
