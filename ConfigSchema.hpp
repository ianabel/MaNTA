#ifndef CONFIGSCHEMA_HPP
#define CONFIGSCHEMA_HPP

// The single declaration of every key MaNTA accepts in a [configuration] table
// or in a Runner.configure dict.
//
// There used to be two: runManta open-coded its TOML reading in MaNTA.cpp and
// PyRunner.cpp carried a separate `params` table. They drifted -- two names for
// the initial time, two defaults for Absolute_tolerance, four keys that existed
// on one side only -- and nothing reported it. One table means a key is on both
// surfaces or on neither.
//
// This header must not include pybind11: it is linked into the standalone
// solver, into libmanta.so and into the unit tests.

#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ConfigSchema
{

enum class Type { Bool, Int, UInt, Double, String, DoubleList, StringList };

// What a key *is*. Not every schema entry is a solver option: TransportSystem
// selects the physics case and has no dict equivalent, and the PythonModule
// keys are read by manta.cli rather than by the solver at all. Recording that
// here is what lets unknown-key rejection be strict without firing on keys
// that are legitimately present in a config file.
enum class Category { Solver, ProblemSelection, Cli };

// Who is asking. Distinct from Category -- one says what a key is, the other
// says which reader is loading it. Only requiredness differs between them, and
// only for TransportSystem.
enum class Reader { Toml, Dict };

using Value = std::variant<bool, int, unsigned, double, std::string,
                           std::vector<double>, std::vector<std::string>>;

struct Entry
{
    std::string_view              name;          // canonical spelling
    std::vector<std::string_view> aliases;       // deprecated; accepted with a warning
    Type                          type;
    Category                      category;
    bool                          requiredToml;
    bool                          requiredDict;
    Value                         _default;
    std::string_view              doc;           // one line; printed by --list-options
};

std::span<const Entry> schema();

// Canonical name or any alias; nullptr if the key is not in the schema.
const Entry *findEntry(std::string_view key);

bool isRequired(Entry const &e, Reader r);

// The closest canonical name to `key` by edit distance, for a did-you-mean.
// Empty when nothing is close enough to be worth suggesting -- a suggestion
// that is nothing like the input sends the reader to check a key they never
// wrote.
std::string_view nearestKey(std::string_view key);

const char *typeName(Type t);

} // namespace ConfigSchema

#endif // CONFIGSCHEMA_HPP
