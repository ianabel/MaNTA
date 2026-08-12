#ifndef PYCONFIGSOURCE_HPP
#define PYCONFIGSOURCE_HPP

// A ConfigSource over a py::dict.
//
// The only python-aware part of the configuration path, and the reason it is a
// header included by PyRunner.cpp alone: SolverConfig.hpp and ConfigSchema.hpp
// link into the standalone solver, into libmanta.so and into the unit tests,
// none of which may see pybind11.
//
// Everything above this class -- validation, aliases, defaults, the conditional
// rules, applying to the solver -- is shared with the TOML reader. That is what
// stops the two surfaces drifting, as they had.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "SolverConfig.hpp"

namespace py = pybind11;

class DictConfigSource : public ConfigSource
{
public:
    explicit DictConfigSource(py::dict const &d) : dict(d) {}

    bool contains(std::string_view key) const override
    {
        return dict.contains(std::string(key).c_str());
    }

    std::vector<std::string> keys() const override
    {
        std::vector<std::string> out;
        for (auto const &item : dict)
            out.push_back(py::cast<std::string>(item.first));
        return out;
    }

    ConfigSchema::Value get(std::string_view key, ConfigSchema::Type t) const override
    {
        auto obj = dict[std::string(key).c_str()];
        auto bad = [&] {
            return std::invalid_argument("Configuration key '" + std::string(key) +
                                         "' must be " + ConfigSchema::typeName(t) + ".");
        };
        try
        {
            switch (t)
            {
            case ConfigSchema::Type::Bool:   return obj.cast<bool>();
            case ConfigSchema::Type::Int:    return obj.cast<int>();
            case ConfigSchema::Type::UInt:   return obj.cast<unsigned>();
            case ConfigSchema::Type::Double: return obj.cast<double>();
            case ConfigSchema::Type::String: return obj.cast<std::string>();
            case ConfigSchema::Type::DoubleList:
                // A bare number is accepted where a list is wanted, which is
                // what Absolute_tolerance has always allowed on the TOML side.
                try
                {
                    return obj.cast<std::vector<double>>();
                }
                catch (py::cast_error const &)
                {
                    return std::vector<double>{obj.cast<double>()};
                }
            case ConfigSchema::Type::StringList:
                return obj.cast<std::vector<std::string>>();
            }
        }
        catch (py::cast_error const &)
        {
            throw bad();
        }
        throw bad();
    }

    // There is no config file to take a name from, so OutputFilename stays
    // effectively required on this surface -- which is what the old table's
    // `.required = true` gave.
    std::string outputFilenameFallback() const override { return {}; }

private:
    py::dict const &dict;
};

#endif // PYCONFIGSOURCE_HPP
