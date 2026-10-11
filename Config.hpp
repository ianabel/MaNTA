#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <toml.hpp>

// Small helpers for reading scalars out of the [configuration] table.
//
// These live in their own translation unit rather than in MaNTA.cpp so that
// they can be linked into the unit tests. They were also duplicated verbatim in
// the now-deleted TestHarness.cpp.
//
// Each throws std::invalid_argument if the key is specified more than once or
// holds the wrong type; the *WithDefault variants return the default when the
// key is absent, the others throw.

double getFloat(std::string const &name, toml::value const &config);
double getFloatWithDefault(std::string const &name, toml::value const &config, double defaultValue);
int getIntWithDefault(std::string const &name, toml::value const &config, int defaultValue);

#endif // CONFIG_HPP
