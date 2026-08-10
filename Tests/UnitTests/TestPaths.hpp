#ifndef TEST_PATHS_HPP
#define TEST_PATHS_HPP

#include <string>

// Resolve a netCDF fixture that lives alongside the unit-test sources.
//
// TEST_DATA_DIR is baked in by Tests/UnitTests/Makefile as an absolute path, so
// the test binary can be run from any working directory -- including from
// Tests/UnitTests itself, and from wherever `make coverage` invokes it. The
// fallback keeps the file usable if someone compiles a test by hand without it.
inline std::string testDataPath(std::string const &filename)
{
#ifdef TEST_DATA_DIR
    return std::string(TEST_DATA_DIR) + filename;
#else
    return "./Tests/UnitTests/" + filename;
#endif
}

#endif // TEST_PATHS_HPP
