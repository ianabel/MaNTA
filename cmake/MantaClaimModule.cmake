# Keep python/manta/_manta<abi>.so owned by the build directory that is building.
#
# The extension is written into the *source* tree, because that is where
# `import manta` has to find it, so every build directory writes to the same
# path -- and CMake calls its own target up to date whenever that file is newer
# than the target's objects, with no way to ask which build put it there. The
# result is a build directory quietly running against another one's module:
# measured here as an instrumented coverage run whose Python suite imported the
# Release module (133s against 748s, with PyRunner.cpp.gcda untouched), and, in
# the other direction, a Release `--target _manta` that reports "Built target"
# and leaves a 57MB instrumented module in place.
#
# So each directory records what it linked, and checks before linking again.
#
#   RECORD=ON   write STAMP describing MODULE as this directory has just left it
#   otherwise   delete MODULE unless STAMP says this directory is what wrote it,
#               which makes the link run and the ownership return here
#
# Size and mtime rather than a hash: the linker sets both, and for a collision
# two directories would have to link byte-identically-sized modules inside one
# second. A hash would cost ~0.2s of every build to close that.

if(NOT MODULE OR NOT STAMP)
  message(FATAL_ERROR "MantaClaimModule.cmake needs -DMODULE= and -DSTAMP=")
endif()

function(manta_describe path out)
  file(SIZE "${path}" _size)
  file(TIMESTAMP "${path}" _mtime "%Y-%m-%dT%H:%M:%S" UTC)
  set(${out} "${_size} ${_mtime}" PARENT_SCOPE)
endfunction()

if(RECORD)
  # EXPECT is the path CMake actually linked, passed as $<TARGET_FILE:_manta>
  # from the POST_BUILD step. MODULE is assembled from the target's properties,
  # because the genex cannot be used in the command that runs *before* the link.
  # If those two ever disagree the claim would be checking a file nothing writes,
  # which is the original silent bug with extra machinery on top.
  if(EXPECT AND NOT MODULE STREQUAL EXPECT)
    message(FATAL_ERROR
      "The module path this build checks is not the one it links:\n"
      "  assembled: ${MODULE}\n"
      "  linked:    ${EXPECT}\n"
      "MANTA_MODULE_FILE in python/CMakeLists.txt is built from _manta's "
      "OUTPUT_NAME, PREFIX and SUFFIX; one of those no longer says what it "
      "used to.")
  endif()
  # Nothing to record if the link did not happen -- leave any previous claim
  # alone rather than writing one that describes a file that is not there.
  if(EXISTS "${MODULE}")
    manta_describe("${MODULE}" _desc)
    file(WRITE "${STAMP}" "${_desc}\n")
  endif()
  return()
endif()

# No module: the link is about to produce one, and will claim it on the way out.
if(NOT EXISTS "${MODULE}")
  return()
endif()

# A module with no claim from this directory belongs to some other build (or to
# a `pip install`), so it goes.
if(NOT EXISTS "${STAMP}")
  file(REMOVE "${MODULE}")
  return()
endif()

manta_describe("${MODULE}" _now)
file(READ "${STAMP}" _claimed)
string(STRIP "${_claimed}" _claimed)

if(NOT _now STREQUAL _claimed)
  file(REMOVE "${MODULE}")
endif()
