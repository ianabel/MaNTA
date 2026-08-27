# The compile and link flags MaNTA is built with, as one INTERFACE target.
#
# This is the CMake half of what Makefile.config used to do. Everything here was
# carried across deliberately; where the CMake default differs from what the
# Makefile did, the difference is called out rather than accepted silently,
# because several of these flags are load-bearing for reasons that are not
# obvious from the flag itself.

include(CheckCXXCompilerFlag)

# --------------------------------------------------------------- build types --
#
# Coverage is a build type rather than an option. `make coverage` used to rebuild
# the tree through a recursive $(MAKE) COVERAGE=on, which needed `env -u CXXFLAGS
# -u LDFLAGS` to stop the parent's release flags leaking into the child and
# leaving -flto=auto on top of -O0 -- silently ruining line attribution. A build
# type has no such parent to inherit from: configure a separate build directory
# with -DCMAKE_BUILD_TYPE=Coverage (or `cmake --preset coverage`) and the two
# cannot mix.
get_property(_manta_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_manta_multi_config)
  if(NOT "Coverage" IN_LIST CMAKE_CONFIGURATION_TYPES)
    list(APPEND CMAKE_CONFIGURATION_TYPES Coverage)
  endif()
elseif(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
             Release Debug RelWithDebInfo Coverage)

# -------------------------------------------------------------------- NDEBUG --
#
# CMake puts -DNDEBUG in every optimised configuration's flags. This build takes
# it back out, and the reason is that the trade is entirely one-sided: the
# assertions cost nothing measurable and they are the diagnostic of record for a
# whole class of defect here.
#
# What they cost depends on the polynomial degree, and it is small: measured
# +2.5% at k = 3, +8.5% at k = 8 and +7.4% at k = 10 (paired A/B, 5 reps,
# medians, MKL pinned). CLAUDE.md carries the table.
#
# The shape of that is worth knowing before anyone re-derives it. Eigen's
# assertions guard its *API* -- operator(), resize, block construction -- not its
# inner kernels, which address raw pointers; MaNTA's own are 26 shape checks in
# Matrices.cpp, one or two per assembled block. All O(1) per call, so what
# matters is the ratio of calls to arithmetic, and raising k grows the blocks
# without growing the call count. Hence near-free at the degrees the fixtures
# use, and a few percent at k = 8+.
#
# What the assertions buy: State.hpp's checkShapeAndSet is a plain assignment
# outside a DEBUG build, so when the adjoint's spatial-parameter branch wrote a
# (np, nPoints) block into an (nPoints, np) destination, the *only* thing that
# reported it was Eigen's resize() assertion firing inside Block<>::operator=.
# With NDEBUG that run transposes a gradient silently instead.
#
# MANTA_ASSERTS=OFF defines NDEBUG anyway -- worth having for a high-k production
# run, where it is a real if modest few percent, and for a build that has to match
# some other toolchain's expectations.
#
# Coverage does not appear below and does not want to: CMAKE_CXX_FLAGS_COVERAGE
# is set to "-O0 -g" outright, so an instrumented build asserts whatever this
# option says -- the right default for the build that exists to run every suite.
option(MANTA_ASSERTS
       "Keep assert() -- and so Eigen's assertions -- in optimised builds" ON)

# CMake's own -DNDEBUG is stripped out of the cache *unconditionally*, and when
# MANTA_ASSERTS is off the definition is made by manta::flags instead. Two
# reasons, and the second is the one that bites.
#
# One: a single source of truth. NDEBUG then follows MANTA_ASSERTS in one place
# rather than being partly CMake's business and partly ours.
#
# Two: `set(... CACHE ... FORCE)` writes a value that outlives the code that
# wrote it, so a *conditional* strip cannot be undone. A build directory
# configured while the strip was running holds a CMAKE_CXX_FLAGS_RELEASE with no
# NDEBUG in it, and reconfiguring it with the condition now false would leave
# that alone -- reporting one thing and building another, silently and forever.
# Stripping every time and adding the definition back from the target makes the
# cache's history irrelevant, which is what lets MANTA_ASSERTS be flipped in an
# existing directory.
foreach(_cfg RELEASE RELWITHDEBINFO MINSIZEREL)
  string(REGEX REPLACE "(^| )-DNDEBUG( |$)" " " CMAKE_CXX_FLAGS_${_cfg}
         "${CMAKE_CXX_FLAGS_${_cfg}}")
  string(STRIP "${CMAKE_CXX_FLAGS_${_cfg}}" CMAKE_CXX_FLAGS_${_cfg})
  set(CMAKE_CXX_FLAGS_${_cfg} "${CMAKE_CXX_FLAGS_${_cfg}}" CACHE STRING
      "Flags used by the C++ compiler during ${_cfg} builds." FORCE)
endforeach()

set(CMAKE_CXX_FLAGS_COVERAGE "-O0 -g" CACHE STRING
    "Flags used by the C++ compiler during Coverage builds.")
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "" CACHE STRING "" )
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "" CACHE STRING "")
mark_as_advanced(CMAKE_CXX_FLAGS_COVERAGE
                 CMAKE_EXE_LINKER_FLAGS_COVERAGE
                 CMAKE_SHARED_LINKER_FLAGS_COVERAGE)

set(MANTA_COVERAGE_BUILD OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
  set(MANTA_COVERAGE_BUILD ON)
endif()

# ------------------------------------------------------------------- the target --
add_library(manta_flags INTERFACE)
add_library(manta::flags ALIAS manta_flags)

# NDEBUG when MANTA_ASSERTS is off, per the block above -- made here rather than
# left to CMake so that one option decides it and the cache cannot outvote us.
if(NOT MANTA_ASSERTS)
  target_compile_definitions(manta_flags INTERFACE
    $<$<CONFIG:Release,RelWithDebInfo,MinSizeRel>:NDEBUG>)
endif()

# cxx_std_23 plus CXX_EXTENSIONS OFF is -std=c++23, which is what the Makefile
# passed. Left to itself CMake asks for -std=gnu++23, and a GNU-extensions build
# is not the one -pedantic below is meant to police.
#
# The number is a variable because manta.pc has to record the same one: a plugin
# compiled at a different standard is an ABI mismatch, and two places to change
# is how they come to disagree.
set(MANTA_CXX_STANDARD 23)
target_compile_features(manta_flags INTERFACE cxx_std_${MANTA_CXX_STANDARD})
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -fPIC everywhere. The core objects are linked into the solver, libmanta.so and
# the Python extension alike, so they have to be position-independent whether or
# not the consumer is.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(_manta_is_gnu "$<CXX_COMPILER_ID:GNU>")
set(_manta_is_clang "$<CXX_COMPILER_ID:Clang,AppleClang>")
set(_manta_coverage "$<CONFIG:Coverage>")
set(_manta_release "$<CONFIG:Release>")
set(_manta_debug "$<CONFIG:Debug>")

target_compile_options(manta_flags INTERFACE
  -Wall -pedantic
  # -Wno-unused-variable in every configuration the Makefile built, but as
  # -Wno-error=unused-variable under DEBUG, where an unused local is usually
  # something half-written rather than noise.
  $<$<NOT:${_manta_debug}>:-Wno-unused-variable>
  $<${_manta_debug}:-Wno-error=unused-variable>
)

# -Werror everywhere except Coverage, which the Makefile also left off it: an
# instrumented build turns up warnings about the instrumentation rather than
# about MaNTA.
option(MANTA_WERROR "Treat compiler warnings as errors" ON)
if(MANTA_WERROR)
  target_compile_options(manta_flags INTERFACE $<$<NOT:${_manta_coverage}>:-Werror>)
endif()

# Every `#pragma omp` in the tree is inside util/ParallelFor.hpp, behind
# `#ifdef _OPENMP`, so a build without OpenMP does not compile one at all. That
# is what retired the `-Wno-unknown-pragmas` this branch used to carry: it was
# added for the pragmas the batched physics wrappers wrote out unconditionally,
# it applied to the whole project, and it silenced every mistyped pragma
# anywhere in the tree for the sake of eight known ones.
if(MANTA_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(manta_flags INTERFACE OpenMP::OpenMP_CXX)
endif()

# ------------------------------------------------------------------ Release --
#
# -O3 -flto=auto -march=native, the flags the Makefile's default branch used.
#
# -flto=auto rather than CMake's INTERPROCEDURAL_OPTIMIZATION, which spells it
# plain -flto and so serialises the link. clang has accepted -flto=auto as a
# spelling of full LTO since clang 17, and the CI matrix starts at 19.
# -O3 itself comes from CMAKE_CXX_FLAGS_RELEASE (with -DNDEBUG stripped out
# above), so only the flags CMake does not supply are added here.
target_compile_options(manta_flags INTERFACE $<${_manta_release}:-flto=auto>)
target_link_options(manta_flags INTERFACE $<${_manta_release}:-flto=auto>)

# -march=native is not a performance knob you can turn off freely: Eigen aligns
# its fixed-size types to the widest vector unit the compiler knows about and
# inlines its expression templates across every boundary, so the core and
# anything sharing Eigen objects with it must agree. That is why manta.pc records
# the *concrete* architecture rather than the word "native" -- see cmake/manta.pc.in.
option(MANTA_NATIVE_ARCH "Build for the host CPU (-march=native)" ON)
if(MANTA_NATIVE_ARCH)
  check_cxx_compiler_flag(-march=native MANTA_HAVE_MARCH_NATIVE)
  if(MANTA_HAVE_MARCH_NATIVE)
    target_compile_options(manta_flags INTERFACE $<${_manta_release}:-march=native>)
  endif()
endif()

# The concrete -march the compiler resolves `native` to, for manta.pc. Asked of
# the compiler rather than guessed, so a plugin built on a different machine is
# rejected at compile time instead of faulting inside an aligned AVX-512 load the
# first time the solver touches its state.
set(MANTA_ABI_MARCH "")
if(MANTA_NATIVE_ARCH AND MANTA_HAVE_MARCH_NATIVE)
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -march=native -Q --help=target
    OUTPUT_VARIABLE _manta_march_probe ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_manta_march_probe MATCHES "[\r\n][ \t]*-march=[ \t]+([^ \t\r\n]+)")
    set(MANTA_ABI_MARCH "${CMAKE_MATCH_1}")
  endif()
endif()

# -------------------------------------------------------------------- Debug --
# -g comes from CMAKE_CXX_FLAGS_DEBUG; -O0 is not in it, and is the point.
target_compile_options(manta_flags INTERFACE $<${_manta_debug}:-O0>)
target_compile_definitions(manta_flags INTERFACE $<${_manta_debug}:DEBUG>)

# PHYSICS_DEBUG gates the extra .dat debug output and a handful of solver checks.
# The Makefile tied it to DEBUG and nothing else -- its top-level `PHYSICS_DEBUG=on`
# was a make variable no recipe ever turned into a -D, so a release build never
# had it. Same default here, but reachable on its own.
option(MANTA_PHYSICS_DEBUG "Define PHYSICS_DEBUG (implied by a Debug build)" OFF)
target_compile_definitions(manta_flags INTERFACE
  $<$<OR:${_manta_debug},$<BOOL:${MANTA_PHYSICS_DEBUG}>>:PHYSICS_DEBUG>)

option(MANTA_VERBOSE "Define VERBOSE (extra solver logging)" OFF)
if(MANTA_VERBOSE)
  target_compile_definitions(manta_flags INTERFACE VERBOSE)
endif()

# ----------------------------------------------------------------- Coverage --
#
# -O0 and no LTO: optimisation and -flto destroy the mapping from machine code
# back to source lines, and a gcov report built from either is meaningless.
if(MANTA_COVERAGE_BUILD)
  target_compile_options(manta_flags INTERFACE --coverage -fno-inline)
  target_link_options(manta_flags INTERFACE --coverage)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # gcc-only. -fprofile-abs-path keeps .gcno references absolute so gcovr can
    # resolve sources from anywhere; clang rejects it outright. The two inline
    # knobs clang accepts and ignores, with a warning apiece.
    target_compile_options(manta_flags INTERFACE
      -fprofile-abs-path -fno-inline-small-functions -fno-default-inline)
  endif()
endif()

# ------------------------------------------------------- compiler-specific --
#
# clang only, and about the *target* rather than about anything in this tree. On
# a host advertising AVX10.1-256, -march=native makes clang 19 emit
#
#   invalid feature combination: +avx10.1-256; will be promoted to avx10.1-512
#
# once per translation unit, and -Werror turns that into a hard failure before a
# line of MaNTA is compiled. clang 20 and 21 no longer diagnose it and gcc never
# did, so it surfaces as a single red matrix leg that depends on which runner
# GitHub allocated rather than on the commit.
#
# CLAUDE.md argues against blanket -Wno- flags and is right to: the two this
# build used to carry outlived their reason and went on hiding defects in our own
# code. This one is different in kind -- it reports how the compiler resolved
# -march=native into a feature set, so there is no MaNTA-side defect it could
# conceal. Drop it when clang 19 leaves the CI matrix.
#
# Probed rather than assumed, so it cannot itself become an unknown-warning error
# on a clang that has retired the name.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  check_cxx_compiler_flag(-Wno-invalid-feature-combination
                          MANTA_HAVE_WNO_INVALID_FEATURE_COMBINATION)
  if(MANTA_HAVE_WNO_INVALID_FEATURE_COMBINATION)
    target_compile_options(manta_flags INTERFACE -Wno-invalid-feature-combination)
  endif()
endif()

# g++-14 miscompiles this tree at exactly the release flags -- all three of
# g++-14, -flto and -march=native are required, and the trigger is SystemSolver's
# member layout, so whether a given tree is affected is luck rather than something
# a build can test for. See CLAUDE.md, "g++-14 miscompiles this tree".
#
# Warn rather than refuse: it is the compiler Ubuntu noble ships as its archive
# default, this is the one place a person reliably meets the fact, and a hard
# error would stop work that is very often unaffected. Asked of the compiler
# version rather than matched against its name, because a distribution whose
# plain `g++` *is* 14 is the case nobody would think to check.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14
   AND CMAKE_BUILD_TYPE STREQUAL "Release"
   AND MANTA_NATIVE_ARCH)
  message(WARNING
    "g++-14 miscompiles this tree at -O3 -flto -march=native. Numbers from this "
    "build are not trustworthy if SystemSolver's members have changed. See "
    "CLAUDE.md. Prefer g++-15 or clang: cmake -DCMAKE_CXX_COMPILER=g++-15 ...")
endif()

# gcov must match the compiler that wrote the .gcno files, or gcovr exits 3 with
# `AdjointVectors.gcno:version 'B42*', prefer 'B33*'` -- an error that looks
# nothing like a coverage problem and that the Makefile hit on CI while passing
# locally.
#
# Derived from the compiler's VERSION rather than from its name. The Makefile
# substituted `gcov` for `g++` in $(CXX), which works for `g++-15` and does
# nothing at all for `/usr/bin/c++` or `/opt/gcc/bin/cc1plus` -- and the fallback
# was a bare `gcov`, i.e. whatever the distribution's default happens to be. On
# the box this was written on that is gcov 15 and matches by luck; on ubuntu-24.04
# the image ships gcc 12/13/14 with 13 as the default, so a build with g++-15
# would have paired it with gcov-13.
#
# clang emits gcov-format data too, but it has to be read with `llvm-cov gcov`.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  string(REGEX MATCH "^[0-9]+" _manta_gcc_major "${CMAKE_CXX_COMPILER_VERSION}")
  find_program(MANTA_GCOV_PROGRAM NAMES "gcov-${_manta_gcc_major}" gcov)
  if(MANTA_GCOV_PROGRAM)
    set(_manta_gcov "${MANTA_GCOV_PROGRAM}")
  else()
    set(_manta_gcov gcov)
  endif()
  mark_as_advanced(MANTA_GCOV_PROGRAM)
else()
  set(_manta_gcov "llvm-cov gcov")
endif()
set(MANTA_GCOV "${_manta_gcov}" CACHE STRING "gcov executable matching CMAKE_CXX_COMPILER")
mark_as_advanced(MANTA_GCOV)
