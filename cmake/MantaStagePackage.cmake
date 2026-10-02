# cmake -DSOURCE=<repo>/python/manta -DDEST=<build>/python/manta -P MantaStagePackage.cmake
#
# Make DEST the importable `manta` package: a symlink there for every file of
# SOURCE that belongs to the package, and nothing else that points into SOURCE.
# DEST also holds what the build puts there itself -- the extension, and the
# __pycache__ directories Python writes beside the links -- and those are left
# alone. Run on every build, so a file added to or deleted from SOURCE is picked
# up without a reconfigure; it touches only links that are missing or wrong.

# -P scripts start with no policies set, so the project's minimum has to be
# restated here: without it CMake 3.x reads IN_LIST and friends by their pre-3.3
# rules, where CMake 4 has already dropped those and quietly gets it right.
cmake_minimum_required(VERSION 3.22)

if(NOT SOURCE OR NOT DEST)
  message(FATAL_ERROR "MantaStagePackage.cmake needs -DSOURCE= and -DDEST=")
endif()

file(GLOB_RECURSE _wanted RELATIVE "${SOURCE}"
     "${SOURCE}/*.py" "${SOURCE}/*.pyi" "${SOURCE}/py.typed")
list(FILTER _wanted EXCLUDE REGEX "(^|/)__pycache__/")

foreach(_rel ${_wanted})
  set(_link "${DEST}/${_rel}")
  set(_target "${SOURCE}/${_rel}")
  if(IS_SYMLINK "${_link}")
    file(READ_SYMLINK "${_link}" _points_at)
    if(_points_at STREQUAL _target)
      continue()
    endif()
  endif()
  get_filename_component(_dir "${_link}" DIRECTORY)
  file(MAKE_DIRECTORY "${_dir}")
  file(REMOVE "${_link}")
  file(CREATE_LINK "${_target}" "${_link}" SYMBOLIC)
endforeach()

# A link to a file that has since left SOURCE -- deleted, or renamed -- would
# otherwise go on being importable from here and nowhere else.
file(GLOB_RECURSE _present LIST_DIRECTORIES false RELATIVE "${DEST}" "${DEST}/*")
foreach(_rel ${_present})
  if(IS_SYMLINK "${DEST}/${_rel}" AND NOT _rel IN_LIST _wanted)
    file(REMOVE "${DEST}/${_rel}")
  endif()
endforeach()
