# Everything MaNTA links against, gathered into one INTERFACE target.
#
# Four come from the system -- SUNDIALS, Eigen, netCDF and a BLAS -- and three
# are vendored as git submodules under extern/. The submodules need no
# configuration at all; the system ones are found the way CMake finds things, so
# an install somewhere unusual is named with -DCMAKE_PREFIX_PATH or the
# package's own <Name>_ROOT rather than through a Makefile.local.
#
# A note that applies to every -isystem in here. Passing the compiler a directory
# it already searches is NOT a no-op: gcc and clang both de-duplicate it, dropping
# the directory from its proper place at the end of the system chain and
# searching it where the flag appeared -- ahead of the libstdc++ headers.
# <cstdlib> then does `#include_next <stdlib.h>`, finds nothing, and every
# translation unit dies with a message about stdlib.h. Makefile.config carried a
# hand-written `sysinclude` filter and a compiler probe for exactly this. CMake
# does the filtering itself: it strips anything in
# CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES from the -I/-isystem flags it generates.
# The CI step that checks a system prefix stays a no-op is kept, because that is
# a property worth testing rather than trusting.

add_library(manta_deps INTERFACE)
add_library(manta::deps ALIAS manta_deps)

# --------------------------------------------------------------------- BLAS --
#
# First in this file, and pinned rather than left to CMake. Both of those are
# deliberate; see the two notes below.
#
# Which BLAS, and why this is pinned rather than left to CMake.
#
# The Makefile linked a bare -lblas, i.e. whatever the distribution's BLAS
# alternative resolves to. FindBLAS, asked freely, does something else: it walks
# its own vendor list and takes the first match, which on a box with Intel's
# libraries installed is the LAYERED MKL link -- libmkl_gf_lp64 + libmkl_gnu_thread
# + libmkl_core, plus libgomp. Both are MKL on such a box; -lblas there is
# libmkl_rt, the single dispatch layer that initialises itself.
#
# The difference is not academic and it is not about speed. The layered link with
# the GNU threading layer is unsafe to dlopen, and importing a C extension is a
# dlopen: the Python module built that way died mid-solve, taking the interpreter
# with it and leaving pytest to report a bare exit 2 with no traceback -- while
# the standalone solver, linked from the very same objects, ran the whole
# regression suite. That asymmetry is the signature, and it cost an hour: the
# obvious reading of "CMake found MKL, the Makefile found -lblas" is
# reference-versus-vendor, and it is wrong.
#
# So: ask for what -lblas asked for. Generic means FindBLAS looks for a library
# called `blas`, which is the alternatives symlink on Debian and Fedora alike.
# Any other value is passed straight through as BLA_VENDOR (OpenBLAS, Intel10_64lp,
# ...), and MANTA_BLAS_VENDOR=Any lets CMake choose freely again.
set(MANTA_BLAS_VENDOR "Generic" CACHE STRING
    "BLA_VENDOR for FindBLAS. 'Generic' is a plain -lblas; 'Any' lets CMake choose.")

set(_manta_blas_vendors "${MANTA_BLAS_VENDOR}")
if(MANTA_BLAS_VENDOR STREQUAL "Generic")
  # A box with no libblas at all -- an OpenBLAS-only install, say -- should still
  # configure rather than being told to go and read this file.
  list(APPEND _manta_blas_vendors "Any")
endif()

foreach(_vendor IN LISTS _manta_blas_vendors)
  if(_vendor STREQUAL "Any")
    unset(BLA_VENDOR)
  else()
    set(BLA_VENDOR "${_vendor}")
  endif()
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    set(MANTA_BLAS_VENDOR_USED "${_vendor}")
    break()
  endif()
  # FindBLAS caches its result and short-circuits on the next call, so a failed
  # attempt has to be cleared before the fallback can look for anything else.
  unset(BLAS_LIBRARIES CACHE)
  unset(BLAS_LINKER_FLAGS CACHE)
  unset(BLAS_${_vendor}_LIBRARY CACHE)
endforeach()

if(NOT BLAS_FOUND)
  message(FATAL_ERROR
    "No BLAS found. Install one (apt: libblas-dev; dnf: blas-devel), or name a "
    "vendor with -DMANTA_BLAS_VENDOR=OpenBLAS (see CMake's FindBLAS for the list).")
endif()

# Pin the answer into the cache, and this is the second half of the trap above.
#
# FindBLAS leaves BLAS_LIBRARIES a NORMAL variable. SUNDIALSConfig.cmake -- when
# SUNDIALS was built with LAPACK on -- writes the BLAS *its own* configure
# resolved straight into the cache, hardcoded as a literal path list, before
# calling find_dependency(LAPACK):
#
#   set(BLAS_LIBRARIES "/usr/lib/.../libmkl_gf_lp64.so;..." CACHE "FILEPATH" "BLAS libraries")
#
# A set(... CACHE ...) with no FORCE is a no-op when the entry already exists --
# but when it does not, CMake *removes the normal variable of the same name* from
# the calling scope -- so a BLAS chosen here is silently replaced by whatever
# machine SUNDIALS was built on, and BLAS::BLAS follows it.
#
# So both halves are needed, and neither alone is enough.
#
#   * This block runs FIRST, above find_package(SUNDIALS), so BLAS::BLAS is
#     created from our choice rather than from SUNDIALS's. Set BLA_VENDOR after
#     that point and it is ignored outright -- FindBLAS short-circuits on the
#     cached result, and the configure summary goes on reporting the vendor you
#     asked for while the link line carries the other one.
#   * And the answer is written into the CACHE with FORCE, which makes
#     SUNDIALSConfig's own non-FORCE set a no-op instead of a silent overwrite.
#
# The summary in CMakeLists.txt reads BLAS_LIBRARIES *after* find_package(SUNDIALS)
# on purpose, so a recurrence of this is visible rather than inferred.
set(BLAS_LIBRARIES "${BLAS_LIBRARIES}" CACHE FILEPATH "BLAS libraries" FORCE)
set(BLAS_LINKER_FLAGS "${BLAS_LINKER_FLAGS}" CACHE STRING "BLAS linker flags" FORCE)

# ------------------------------------------------------------------- LAPACK --
#
# Optional, and only for one thing: the banded factorisation of the HDG trace
# matrix (dgbtrf/dgbtrs), which util/BandedMatrix.hpp calls when this is found
# and reimplements when it is not. The reimplementation is a fallback rather than
# a preference -- it exists so a box with a BLAS and no LAPACK still builds -- so
# a found LAPACK is always used.
#
# Everything the BLAS block above says applies here unchanged, because FindLAPACK
# reads the same BLA_VENDOR and can resolve to the same layered MKL link that is
# unsafe to dlopen. So it is asked for the vendor that BLAS actually resolved to,
# not left free: mixing a Generic BLAS with a layered-MKL LAPACK would put both in
# the Python module's link line, which is the exact configuration that killed the
# interpreter mid-solve.
#
# Not REQUIRED. Debian splits liblapack-dev from libblas-dev and a minimal
# container often has only the latter; the fallback is there precisely so that
# still works, and `cmake -B build -LH` reports which one a given build got.
option(MANTA_LAPACK "Use LAPACK's banded solver (dgbtrf/dgbtrs) when available" ON)

set(MANTA_HAVE_LAPACK OFF)
if(MANTA_LAPACK)
  if(MANTA_BLAS_VENDOR_USED STREQUAL "Any")
    unset(BLA_VENDOR)
  else()
    set(BLA_VENDOR "${MANTA_BLAS_VENDOR_USED}")
  endif()
  # Discovered afresh every configure. FindLAPACK short-circuits on a cached
  # LAPACK_LIBRARIES, and the entry it would find is not necessarily one this
  # file wrote -- SUNDIALSConfig.cmake writes its own configure's answer into the
  # cache too. Without this unset, one bad configure is permanent: the poisoned
  # value is read back, captured, and re-pinned every time afterwards, and no
  # amount of reconfiguring heals it. Re-running the find is two find_library
  # calls.
  unset(LAPACK_LIBRARIES CACHE)
  unset(LAPACK_FOUND CACHE)
  find_package(LAPACK QUIET)
  if(LAPACK_FOUND)
    set(MANTA_HAVE_LAPACK ON)
    # Copied out by value, immediately, and linked as a list of paths rather than
    # as LAPACK::LAPACK. That is not fussiness -- it was measured, and the
    # imported target is NOT safe to hold on to.
    #
    # CMake's FindLAPACK creates the target under `if(NOT TARGET LAPACK::LAPACK)`
    # but sets INTERFACE_LINK_LIBRARIES on it *outside* that guard. So the next
    # find_package(LAPACK) anywhere in the project -- and SUNDIALSConfig.cmake
    # runs one, via find_dependency(LAPACK), a few lines below -- silently
    # rewrites the contents of the target this project already linked.
    #
    # That is the BLAS trap above, one level deeper: not the variable this time
    # but the target. Measured on the development box, where SUNDIALS was built
    # against Intel's libraries: MANTA_LAPACK=ON put
    # libmkl_gf_lp64 + libmkl_gnu_thread + libmkl_core + libgomp on the link line
    # of *libmanta and the Python module*, while an isolated
    # find_package(LAPACK) with the same BLA_VENDOR resolved cleanly to
    # liblapack.so + libblas.so. The layered MKL link is the one that is unsafe
    # to dlopen -- see the BLAS block -- so this would have reintroduced the
    # interpreter crash that block exists to prevent, on a knob whose whole
    # purpose is to make one factorisation faster.
    set(MANTA_LAPACK_LIBRARIES "${LAPACK_LIBRARIES}")
    target_link_libraries(manta_deps INTERFACE ${MANTA_LAPACK_LIBRARIES})
    target_compile_definitions(manta_deps INTERFACE MANTA_HAVE_LAPACK)
  else()
    message(STATUS
      "No LAPACK found; the banded trace solve will use the built-in "
      "factorisation. Install one (apt: liblapack-dev; dnf: lapack-devel) to use "
      "dgbtrf/dgbtrs instead.")
  endif()
endif()

# Pinned with FORCE for the same reason BLAS_LIBRARIES is, and it is the *second*
# line of defence rather than the first: what actually protects the link line is
# MANTA_LAPACK_LIBRARIES above, copied out by value before anything else can run
# a find_package(LAPACK).
#
# The configure summary prints MANTA_LAPACK_LIBRARIES -- the value that is really
# linked -- and not LAPACK_LIBRARIES, which SUNDIALS may well have rewritten by
# the time it is read. If the two ever need comparing, print both.
set(LAPACK_LIBRARIES "${LAPACK_LIBRARIES}" CACHE FILEPATH "LAPACK libraries" FORCE)

# ------------------------------------------------------------------ SUNDIALS --
#
# v7.1.0 or newer, not 6.x: MaNTA links sundials_core and uses SUNContext,
# neither of which exists before v7. The component list must stay in step with
# build_sundials, which switches the rest off -- KINSOL in particular is what
# SteadyState drives for the PseudoTransient and Newton paths, and a SUNDIALS
# built without it fails three files into the build rather than at the link.
# ./sundials/install is where the bundled build_sundials script puts one, so a
# checkout that ran it configures with no arguments at all. SUNDIALS_ROOT is a
# *preferred* prefix rather than an exclusive one, so setting it when the
# directory happens to exist cannot hide a system installation -- and it is only
# set when the caller named nothing.
if(NOT DEFINED SUNDIALS_ROOT AND NOT DEFINED ENV{SUNDIALS_ROOT}
   AND IS_DIRECTORY "${PROJECT_SOURCE_DIR}/sundials/install")
  set(SUNDIALS_ROOT "${PROJECT_SOURCE_DIR}/sundials/install")
  message(STATUS "Using the SUNDIALS built by ./build_sundials: ${SUNDIALS_ROOT}")
endif()

find_package(SUNDIALS 7.1.0 REQUIRED
             COMPONENTS core ida kinsol nvecserial)
target_link_libraries(manta_deps INTERFACE
  SUNDIALS::core SUNDIALS::ida SUNDIALS::kinsol SUNDIALS::nvecserial)

# --------------------------------------------------------------------- Eigen --
#
# Either 3.4.x or 5.0.x. The two differ enough to need guarding in the sources
# -- Eigen::all moved namespace and internal::SingleRange became a template --
# and note that EIGEN_VERSION_AT_LEAST cannot tell them apart, because Eigen 5
# kept EIGEN_WORLD_VERSION at 3 and renumbered the rest underneath it. Branch on
# EIGEN_MAJOR_VERSION >= 5 in C++, never on that macro.
#
# Two ways in, because both are in use. A packaged Eigen -- the distribution's
# libeigen3-dev, or one built and installed from source -- ships Eigen3Config.cmake
# and is found by find_package with nothing configured. An *unpacked source tree*
# does not: eigen's tarball carries only Eigen3Config.cmake.in, so find_package
# cannot see it however the path is spelled. That second form is exactly what
# Makefile.local.example recommended ("EIGEN_DIR = ../eigen", the easiest way to
# have Eigen 5 today) and what CI's Eigen 5.0.1 legs unpack, so dropping it would
# quietly break both.
set(MANTA_EIGEN_INCLUDE_DIR "" CACHE PATH
    "Eigen headers, for a source checkout with no installed Eigen3Config.cmake")

if(MANTA_EIGEN_INCLUDE_DIR)
  if(NOT EXISTS "${MANTA_EIGEN_INCLUDE_DIR}/Eigen/Core")
    message(FATAL_ERROR
      "MANTA_EIGEN_INCLUDE_DIR=${MANTA_EIGEN_INCLUDE_DIR} does not hold Eigen/Core. "
      "It wants the directory *containing* Eigen/, e.g. an unpacked eigen tarball "
      "or /usr/include/eigen3.")
  endif()
  add_library(manta_eigen INTERFACE)
  target_include_directories(manta_eigen SYSTEM INTERFACE "${MANTA_EIGEN_INCLUDE_DIR}")
  target_link_libraries(manta_deps INTERFACE manta_eigen)

  # Read the version out of the headers, because which Eigen this is matters here
  # and a bare path says nothing. 3.4 keeps its macros in
  # Eigen/src/Core/util/Macros.h; 5.0 moved them to Eigen/Version.
  set(Eigen3_VERSION "unknown")
  foreach(_hdr "Eigen/Version" "Eigen/src/Core/util/Macros.h")
    if(EXISTS "${MANTA_EIGEN_INCLUDE_DIR}/${_hdr}")
      file(STRINGS "${MANTA_EIGEN_INCLUDE_DIR}/${_hdr}" _v
           REGEX "#define EIGEN_(WORLD|MAJOR|MINOR)_VERSION")
      string(REGEX MATCH "WORLD_VERSION ([0-9]+)" _ "${_v}")
      set(_w ${CMAKE_MATCH_1})
      string(REGEX MATCH "MAJOR_VERSION ([0-9]+)" _ "${_v}")
      set(_mj ${CMAKE_MATCH_1})
      string(REGEX MATCH "MINOR_VERSION ([0-9]+)" _ "${_v}")
      set(_mn ${CMAKE_MATCH_1})
      if(_w AND DEFINED _mj AND DEFINED _mn)
        set(Eigen3_VERSION "${_w}.${_mj}.${_mn}")
        break()
      endif()
    endif()
  endforeach()
  set(EIGEN3_INCLUDE_DIR "${MANTA_EIGEN_INCLUDE_DIR}")
else()
  # No version in the request, and the check made below instead. That looks lax
  # and is the opposite: `find_package(Eigen3 3.4)` excludes half of what this
  # project supports.
  #
  # MaNTA builds against Eigen 3.4.x and 5.0.x alike -- CLAUDE.md says so and the
  # CI matrix has legs for both -- but Eigen ships an Eigen3ConfigVersion.cmake
  # generated with SameMajorVersion compatibility. To that file a requested "3.4"
  # is not a *minimum*: it is a major version to match, so 5.0.1 is rejected
  # outright rather than accepted as newer. CMake then keeps looking, which is
  # why this stayed invisible on any box that also had a 3.4 installed -- it
  # silently fell through to the older one.
  #
  # Where it is not invisible is a box that has only Eigen 5. fedora:latest
  # rolled to fc44, whose eigen3-devel is 5.0.1, and the `Compile (fedora:latest)`
  # leg -- a required check -- went red with nothing in this tree having changed:
  #
  #     Could not find a configuration file for package "Eigen3" that is
  #     compatible with requested version "3.4".
  #       /usr/share/cmake/eigen3/Eigen3Config.cmake, version: 5.0.1
  #         The version found is not compatible with the version requested.
  #
  # Any distribution moving to Eigen 5 does the same, so this is a floor that
  # arrives on its own rather than one anybody chooses.
  #
  # The MANTA_EIGEN_INCLUDE_DIR branch above has never had the problem: it reads
  # the version out of the headers and does not consult a config file at all,
  # which is why the two Eigen 5.0.1 CI legs are green while fedora is not. They
  # go through that branch.
  find_package(Eigen3 REQUIRED NO_MODULE)
  if(Eigen3_VERSION AND Eigen3_VERSION VERSION_LESS 3.4)
    message(FATAL_ERROR
      "Eigen ${Eigen3_VERSION} is too old: MaNTA needs 3.4 or newer. 3.4.x and "
      "5.0.x are both supported and both tested; nothing between them exists.")
  endif()
  target_link_libraries(manta_deps INTERFACE Eigen3::Eigen)
endif()

# EIGEN_USE_BLAS swaps in BLAS-backed product specialisations. It is part of the
# ABI, not a tuning flag: a plugin compiled without it inlines different Eigen
# expression templates from the core's and the two disagree about layout. It is
# also why every header must reach <Eigen/Core> and <Eigen/Dense> before the
# project headers -- see the note in Postprocessing.hpp.
target_compile_definitions(manta_deps INTERFACE EIGEN_USE_BLAS)

find_package(Threads REQUIRED)
target_link_libraries(manta_deps INTERFACE BLAS::BLAS Threads::Threads ${CMAKE_DL_LIBS} m)

# ------------------------------------------------------------------- netCDF --
#
# The C library and the C++4 binding. pkg-config first, because it knows the
# multiarch libdir and reports -lnetcdf_c++4 ahead of -lnetcdf -- the order
# static linking needs, and the one a hand-written fallback tends to get
# backwards because only shared linking forgives it. Both are asked for in a
# single pkg_check_modules call so that ordering survives.
#
# For an install pkg-config does not know about -- an HPC module, a brew/conda
# pair under two different prefixes -- point CMAKE_PREFIX_PATH at it, or set
# netCDF_ROOT / netCDFCxx_ROOT, and the find_path/find_library fallback below
# picks it up.
find_package(PkgConfig QUIET)
set(_manta_netcdf_found OFF)
if(PkgConfig_FOUND)
  pkg_check_modules(MANTA_NETCDF QUIET IMPORTED_TARGET netcdf netcdf-cxx4)
  if(MANTA_NETCDF_FOUND)
    target_link_libraries(manta_deps INTERFACE PkgConfig::MANTA_NETCDF)
    set(_manta_netcdf_found ON)
    set(MANTA_NETCDF_ORIGIN "pkg-config (netcdf ${MANTA_NETCDF_netcdf_VERSION}, netcdf-cxx4 ${MANTA_NETCDF_netcdf-cxx4_VERSION})")
  endif()
endif()

if(NOT _manta_netcdf_found)
  find_path(NETCDF_INCLUDE_DIR netcdf.h)
  find_library(NETCDF_LIBRARY NAMES netcdf)
  find_path(NETCDF_CXX_INCLUDE_DIR ncFile.h)
  # NETCDF_CXX_LIBRARY is a cache variable so an install that calls the binding
  # something other than netcdf_c++4 can be named without patching this file.
  find_library(NETCDF_CXX_LIBRARY NAMES netcdf_c++4 netcdf-cxx4 netcdf_c++)
  if(NOT NETCDF_LIBRARY OR NOT NETCDF_CXX_LIBRARY
     OR NOT NETCDF_INCLUDE_DIR OR NOT NETCDF_CXX_INCLUDE_DIR)
    message(FATAL_ERROR
      "netCDF not found. Install the C library and the C++4 binding "
      "(apt: libnetcdf-dev libnetcdf-c++4-dev; dnf: netcdf-devel netcdf-cxx4-devel), "
      "or point CMAKE_PREFIX_PATH at the prefix holding them.")
  endif()
  # The C++ binding first, then the C library it depends on.
  target_link_libraries(manta_deps INTERFACE
    ${NETCDF_CXX_LIBRARY} ${NETCDF_LIBRARY})
  target_include_directories(manta_deps SYSTEM INTERFACE
    ${NETCDF_CXX_INCLUDE_DIR} ${NETCDF_INCLUDE_DIR})
  set(MANTA_NETCDF_ORIGIN "${NETCDF_CXX_LIBRARY}")
  mark_as_advanced(NETCDF_INCLUDE_DIR NETCDF_LIBRARY
                   NETCDF_CXX_INCLUDE_DIR NETCDF_CXX_LIBRARY)
endif()

# -------------------------------------------------------------------- Boost --
#
# Header-only: the unit tests use <boost/test/included/unit_test.hpp>, so there
# is no unit_test_framework library to link. CMake 4 removed the FindBoost
# module, so this is Boost's own BoostConfig.cmake when it is installed, and a
# plain header search when it is not -- which keeps a hand-unpacked Boost
# (BOOST_ROOT, the old Makefile.local's BOOST_DIR) working.
find_package(Boost CONFIG QUIET)
if(Boost_FOUND)
  set(MANTA_BOOST_TARGET Boost::headers)
else()
  find_path(BOOST_INCLUDE_DIR boost/test/included/unit_test.hpp
            HINTS ${BOOST_ROOT} ENV BOOST_ROOT ENV BOOST_DIR
            PATH_SUFFIXES include)
  if(NOT BOOST_INCLUDE_DIR)
    message(FATAL_ERROR
      "Boost headers not found (looking for boost/test/included/unit_test.hpp). "
      "Install Boost (apt: libboost-dev; dnf: boost-devel), or set -DBOOST_ROOT=/path/to/boost. "
      "Configure with -DMANTA_TESTS=OFF to build without the unit tests.")
  endif()
  add_library(manta_boost_headers INTERFACE)
  target_include_directories(manta_boost_headers SYSTEM INTERFACE ${BOOST_INCLUDE_DIR})
  set(MANTA_BOOST_TARGET manta_boost_headers)
  mark_as_advanced(BOOST_INCLUDE_DIR)
endif()

# ------------------------------------------------- vendored, under extern/ --
#
# toml11, autodiff and pybind11 are git submodules, so there is nothing to
# install and no path to configure. A checkout without them leaves three empty
# directories and the build stops at a missing toml.hpp, which reads as a broken
# dependency rather than as the missing `--recurse-submodules` it is -- hence the
# check.
foreach(_sub toml11 autodiff)
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/${_sub}/CMakeLists.txt")
    message(FATAL_ERROR
      "extern/${_sub} is empty. This is a git submodule; populate it with\n"
      "    git submodule update --init\n"
      "(or clone with --recurse-submodules next time).")
  endif()
endforeach()

# Included as plain header directories rather than through each project's own
# CMake, which would build its tests and examples for no benefit here.
#
# SYSTEM, and not merely as a tidiness: -Werror is on, and Eigen's headers trip
# -Wunused-but-set-variable under clang from the one translation unit that pulls
# in SparseCore. A dependency added with a non-SYSTEM include re-arms that.
add_library(manta_vendored INTERFACE)
target_include_directories(manta_vendored SYSTEM INTERFACE
  "${PROJECT_SOURCE_DIR}/extern/toml11/include"
  "${PROJECT_SOURCE_DIR}/extern/autodiff")
target_link_libraries(manta_deps INTERFACE manta_vendored)

# extern/autodiff is a FORK, tracking ianabel/autodiff branch eigen-5-singlerange
# for one patch: upstream specialises VectorTraits on Eigen::internal::SingleRange
# as a plain type, which Eigen 5.0 made a template, so upstream does not compile
# against Eigen 5 at all -- and its main has not moved since January 2025. A plain
# `git submodule update --remote` would silently revert to a commit that cannot
# build here, and the failure appears as a wall of template errors inside a
# third-party header rather than as anything about submodules. .gitmodules
# records it; watch autodiff/autodiff#397.
