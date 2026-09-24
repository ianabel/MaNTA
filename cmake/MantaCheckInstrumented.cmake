# Fail unless MODULE was compiled with --coverage.
#
# The forced relink in python/CMakeLists.txt is what keeps this true; this is
# what notices if it ever stops being. Worth the second it costs, because the
# failure it guards against is silent in both directions: an uninstrumented
# module gives a passing Python suite *and* a report that does not mention the
# binding layer at all, so neither the run nor the numbers look wrong.
#
# gcov instrumentation writes the absolute path of each .gcda file into the
# object, so those strings are present in anything built with --coverage and in
# nothing built without it. LIMIT_COUNT stops the scan at the first one.

if(NOT EXISTS "${MODULE}")
  message(FATAL_ERROR
    "The Python extension is missing:\n"
    "    ${MODULE}\n"
    "Build it first: cmake --build <this build directory> --target _manta")
endif()

file(STRINGS "${MODULE}" _manta_gcda REGEX "\\.gcda" LIMIT_COUNT 1)

if(NOT _manta_gcda)
  message(FATAL_ERROR
    "${MODULE}\n"
    "carries no coverage instrumentation, so the Python suite would run against "
    "it and be measured as though it had never run.\n"
    "\n"
    "The extension is written into the source tree -- python/manta/ is where "
    "`import manta` has to find it -- so another build directory can have put "
    "this one there. Remove it and rebuild from here:\n"
    "\n"
    "    rm -f ${MODULE}\n"
    "    cmake --build <this build directory> --target _manta\n")
endif()
