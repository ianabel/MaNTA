# cmake -DMANIFEST=<build>/install_manifest.txt -P MantaUninstall.cmake

if(NOT EXISTS "${MANIFEST}")
  message(FATAL_ERROR
    "No install manifest at ${MANIFEST} -- nothing has been installed from this "
    "build directory, so there is nothing to remove.")
endif()

file(STRINGS "${MANIFEST}" _files)
set(_removed 0)
foreach(_f ${_files})
  # EXISTS is false for a dangling symlink, which the versioned libmanta.so links
  # become as soon as the real file goes; IS_SYMLINK catches those.
  if(EXISTS "$ENV{DESTDIR}${_f}" OR IS_SYMLINK "$ENV{DESTDIR}${_f}")
    file(REMOVE "$ENV{DESTDIR}${_f}")
    message(STATUS "Removed $ENV{DESTDIR}${_f}")
    math(EXPR _removed "${_removed} + 1")
  else()
    message(STATUS "Already gone: $ENV{DESTDIR}${_f}")
  endif()
endforeach()
message(STATUS "uninstall: removed ${_removed} file(s)")
