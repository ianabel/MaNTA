#ifndef PYTOML_HPP
#define PYTOML_HPP

// Conversions between a Python object and a toml::value, in both directions.
//
// They are inverses of each other and were about to be written a translation
// unit apart -- cast_toml backs manta.TomlValue.__getitem__ in Python.cpp,
// tomlFromPy is how a C++ physics case's own table reaches it from a
// Runner.configure dict in PyRunner.cpp -- so they live together.
//
// pybind11-aware, so this is a header those two include and nothing that links
// into MaNTA, libmanta.so or the unit tests does. Same rule as
// PyConfigSource.hpp, and the same reason.

#include <cstdint>
#include <pybind11/pybind11.h>
#include <string>
#include <toml.hpp>

#include "ConfigSchema.hpp"

namespace py = pybind11;

// toml -> Python. Returns None for a node of no recognised type, which is what
// TomlValue.__getitem__'s subtable search uses as its "not here" signal.
inline py::object cast_toml(toml::value v)
{
    if (v.is_boolean())
        return py::bool_(v.as_boolean());
    else if (v.is_integer())
        return py::int_(v.as_integer());
    else if (v.is_floating())
        return py::float_(v.as_floating());
    else if (v.is_string())
        return py::str(v.as_string());
    else if (v.is_array())
    {
        py::list lst;
        for (const auto &elem : v.as_array())
            lst.append(cast_toml(elem));
        return lst;
    }
    else if (v.is_table())
    {
        py::dict d;
        for (const auto &[key, val] : v.as_table())
            d[py::str(key)] = cast_toml(val);
        return d;
    }
    else
        return py::none();
}

// Python -> toml, recursively.
//
// bool is tested before int deliberately: in Python `bool` is a subclass of
// `int`, so the other order turns True into the integer 1 and a case doing
// toml::find<bool> on it dies with a toml::type_error naming a type the caller
// never wrote.
//
// A float that happens to be integral stays a float. TOML distinguishes 1 from
// 1.0 and toml::find<double> on an integer node throws, so preserving Python's
// own distinction is the only choice that lets `{"Kappa": 1.0}` and
// `Kappa = 1.0` behave identically.
inline toml::value tomlFromPy(py::handle obj)
{
    if (py::isinstance<py::bool_>(obj))
        return toml::value(py::cast<bool>(obj));
    if (py::isinstance<py::int_>(obj))
        return toml::value(py::cast<std::int64_t>(obj));
    if (py::isinstance<py::float_>(obj))
        return toml::value(py::cast<double>(obj));
    if (py::isinstance<py::str>(obj))
        return toml::value(py::cast<std::string>(obj));

    if (py::isinstance<py::dict>(obj))
    {
        toml::table t;
        for (auto const &item : py::reinterpret_borrow<py::dict>(obj))
            t.emplace(py::cast<std::string>(py::str(item.first)),
                      tomlFromPy(item.second));
        return toml::value(std::move(t));
    }

    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj))
    {
        toml::array a;
        for (auto const &item : py::reinterpret_borrow<py::sequence>(obj))
            a.push_back(tomlFromPy(item));
        return toml::value(std::move(a));
    }

    // A numpy scalar or 0-d array, unwrapped to the Python type it holds and
    // re-tested above. This is not an exotic input: the caller for whom driving
    // a C++ case from Python is worth anything is an optimiser sweeping a
    // parameter, and `arr[i]` off a numpy array is a numpy scalar.
    //
    // np.float64 happens to subclass Python's float and so is caught above
    // already -- which is the trap, because it makes the common case work and
    // leaves np.float32 and every np.int* (which subclass nothing) throwing.
    // Going through item() rather than casting to double keeps the integer /
    // float distinction TOML cares about.
    if (py::hasattr(obj, "item"))
        return tomlFromPy(obj.attr("item")());

    throw std::invalid_argument(
        "Cannot express a Python " +
        py::cast<std::string>(py::str(py::type::of(obj).attr("__name__"))) +
        " as a TOML value. A physics case's configuration table may hold "
        "booleans, integers, floats, strings, and lists or dicts of those.");
}

// Whether a Runner.configure entry is a physics case's own table rather than a
// solver key: a dict under a name the schema does not know.
//
// The schema check is what keeps this from swallowing a mistake. A dict given
// where a solver key is wanted -- {"Grid_size": {...}} -- is still reported as
// a type error by DictConfigSource, because that key *is* in the schema; only
// names the schema has never heard of are treated as physics.
inline bool isPhysicsTable(py::handle key, py::handle value)
{
    if (!py::isinstance<py::dict>(value))
        return false;
    if (!py::isinstance<py::str>(key))
        return false;
    return ConfigSchema::findEntry(py::cast<std::string>(key)) == nullptr;
}

// A Runner.configure dict, in the shape a physics case already reads a config
// file in.
//
// A .conf file puts the solver's keys in [configuration] and the case's own
// keys in a sibling table, and runManta hands InstantiateProblem the *whole*
// parsed file -- so a case says config.at("DiffusionProblem"). This rebuilds
// that value from the dict: a table-valued entry stays top level, and
// everything else goes under `configuration`. A case therefore reads the same
// table whichever surface drove it, and one that reaches into [configuration]
// for the grid or the degree still finds it there.
inline toml::value physicsConfigFromDict(py::dict const &d)
{
    toml::table top;
    toml::table configuration;

    for (auto const &item : d)
    {
        auto const key = py::cast<std::string>(py::str(item.first));

        // `configuration` is where the solver's keys go, so a physics table of
        // that name would be put in the same slot -- and since toml::table is a
        // map, the emplace below would then quietly do nothing and a case
        // reading [configuration] would see the caller's table instead of the
        // run's settings. Refused rather than merged: there is no reading of
        // "my case's table is called configuration" that is worth guessing at.
        if (key == "configuration")
            throw std::invalid_argument(
                "'configuration' cannot be a physics case's table name: it is "
                "where a Runner.configure dict's solver keys go, which is what "
                "makes the dict look to a physics case like the config file it "
                "would otherwise have read. Give the table the name the case "
                "asks for.");

        if (isPhysicsTable(item.first, item.second))
            top.emplace(key, tomlFromPy(item.second));
        else
            configuration.emplace(key, tomlFromPy(item.second));
    }

    top.emplace("configuration", toml::value(std::move(configuration)));
    return toml::value(std::move(top));
}

#endif // PYTOML_HPP
