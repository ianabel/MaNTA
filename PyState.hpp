#ifndef PYSTATE_HPP
#define PYSTATE_HPP

#include "State.hpp"
#include "SystemSpec.hpp"
#include "Types.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <string>
#include <vector>

namespace py = pybind11;

/*
    What a physics case sees when a hook hands it a State.

    This used to be a dict of five 1-D arrays keyed "Variable", "Derivative",
    "Flux", "Aux", "Scalars", so a case read `state["Derivative"][0]`. Three
    problems with that, in ascending order of seriousness:

      * it named storage rather than meaning, and the reader had to know that
        "Flux" holds the *negated* flux;
      * a mistyped key was a KeyError deep inside the first residual evaluation
        rather than anything earlier;
      * building the dict copied all five vectors on every call, once per point
        per hook.

    A StateView is a non-owning window onto the C++ State, carrying the case's
    SystemSpec so its fields can be indexed by name as well as by position:

        s.u[0]        s.u["density"]
        s.q[0]        s.q["density"]
        s.sigma[i]    the stored flux, sigma = -sigma_hat
        s.sigmaHat[i] the physical flux, what SigmaFn returned
        s.phi[i]      s.phi["potential"]
        s.geom[g]     a field model's geometry, derived rather than declared --
                      see the note on StateView::geom() below
        s.scalars[i]  s.scalars["current"]

    It is valid only for the duration of the call it was passed to; the
    underlying State belongs to the solver. Holding one past the hook's return
    and reading it is a use-after-free, which is why there is no way to
    construct one from Python.
*/

/// One field of a StateView: a window onto a Vector, with names attached.
class StateField
{
public:
    StateField() = default;

    StateField(double *data, Index size, std::vector<std::string> const *names, bool negated)
        : m_data(data), m_size(size), m_names(names), m_negated(negated)
    {
    }

    Index size() const { return m_size; }

    double get(py::object const &key) const
    {
        const Index i = resolve(key);
        return m_negated ? -m_data[i] : m_data[i];
    }

    void set(py::object const &key, double value)
    {
        if (m_negated)
            throw py::attribute_error(
                "sigmaHat is read-only: it is the negation of the stored flux, so there "
                "is nothing to assign to. Assign to sigma instead.");
        m_data[resolve(key)] = value;
    }

    /// numpy interop, so `np.asarray(s.u)` and `s.u @ w` work.
    ///
    /// `copy` must be honoured. numpy takes what __array__ returns at its word:
    /// asked for a copy and handed a view, it does not copy again, and the
    /// caller is left holding a window onto a State that the solver destroys
    /// when the hook returns. `np.array(s.u, copy=True)` inside a hook -- which
    /// is exactly how one records a state for later -- then reads freed memory.
    py::array asArray(bool copy)
    {
        if (copy && !m_negated)
        {
            py::array_t<double> out(static_cast<py::ssize_t>(m_size));
            if (m_size > 0)
                std::copy(m_data, m_data + m_size, out.mutable_data());
            return std::move(out);
        }

        // The negated field is always a copy: there is no buffer holding it.
        if (m_negated)
        {
            py::array_t<double> out(static_cast<py::ssize_t>(m_size));
            auto r = out.mutable_unchecked<1>();
            for (Index i = 0; i < m_size; ++i)
                r(i) = -m_data[i];
            return std::move(out);
        }
        // An empty field -- nAux = 0 is the common one -- has a null data
        // pointer, and py::capsule rejects null. Hand back an owned empty array
        // rather than a view of nothing.
        if (m_size == 0 || m_data == nullptr)
            return py::array_t<double>(0);

        // The capsule frees nothing: the buffer belongs to the State, which
        // outlives the call. Writable, so a hook can fill an out-parameter
        // through it.
        return py::array_t<double>({static_cast<py::ssize_t>(m_size)}, {sizeof(double)}, m_data,
                                   py::capsule(m_data, [](void *) {}));
    }

    std::string repr() const
    {
        std::string out = "<";
        for (Index i = 0; i < m_size; ++i)
        {
            if (i)
                out += ", ";
            if (m_names && static_cast<size_t>(i) < m_names->size())
                out += (*m_names)[i] + "=";
            out += std::to_string(m_negated ? -m_data[i] : m_data[i]);
        }
        return out + ">";
    }

private:
    /// An index, from either an integer or a declared name.
    ///
    /// Both spellings are checked. A name that is not declared reports the ones
    /// that are, because the usual cause is a typo or a variable that belongs
    /// to a different field.
    Index resolve(py::object const &key) const
    {
        if (py::isinstance<py::str>(key))
        {
            const std::string name = key.cast<std::string>();
            if (m_names)
                for (size_t i = 0; i < m_names->size(); ++i)
                    if ((*m_names)[i] == name)
                        return static_cast<Index>(i);

            std::string known;
            if (m_names)
                for (auto const &n : *m_names)
                    known += (known.empty() ? "" : ", ") + n;
            throw py::key_error("no field named '" + name + "'; this one has " +
                                (known.empty() ? "no named entries" : known));
        }

        Index i = key.cast<Index>();
        if (i < 0)
            i += m_size; // Python's negative indexing
        if (i < 0 || i >= m_size)
            throw py::index_error("index " + std::to_string(key.cast<Index>()) +
                                  " out of range for a field of length " +
                                  std::to_string(m_size));
        return i;
    }

    double *m_data = nullptr;
    Index m_size = 0;
    std::vector<std::string> const *m_names = nullptr;
    bool m_negated = false;
};

/// A non-owning view of a State, with the case's names attached.
class StateView
{
public:
    StateView(State &s, SystemSpec const &spec) : m_state(&s), m_spec(&spec) {}

    /// Without a spec: the fields index by position only, and a name raises.
    /// AdjointProblem does not hold the transport system, so its hooks get
    /// this one.
    explicit StateView(State &s) : m_state(&s), m_spec(nullptr) {}

    StateField u() const { return {m_state->u().data(), m_state->u().size(), &varNames(), false}; }
    StateField q() const { return {m_state->q().data(), m_state->q().size(), &varNames(), false}; }
    StateField sigma() const
    {
        return {m_state->sigma().data(), m_state->sigma().size(), &varNames(), false};
    }
    StateField sigmaHat() const
    {
        return {m_state->sigma().data(), m_state->sigma().size(), &varNames(), true};
    }
    StateField phi() const
    {
        return {m_state->phi().data(), m_state->phi().size(), &auxNames(), false};
    }
    /// The field model's geometry -- a derived metric field, not an unknown.
    /// Indexable by position only, and still is: the slot names live in the
    /// field model's own FieldModelSpec, while this view takes its names from
    /// the TransportSystem's SystemSpec, which knows nothing about whichever
    /// model happens to be attached. There is no route from one to the other,
    /// so there is nothing to look a name up in. See TODO for the entry.
    StateField geom() const
    {
        return {m_state->geom().data(), m_state->geom().size(), &noNames(), false};
    }
    StateField scalars() const
    {
        return {m_state->scalars().data(), m_state->scalars().size(), &scalarNames(), false};
    }

    std::string repr() const
    {
        return "State(u=" + u().repr() + ", q=" + q().repr() + ", sigma=" + sigma().repr() + ")";
    }

private:
    // Cached per view rather than per access: the spec stores whole specs, and
    // StateField wants just the names.
    std::vector<std::string> const &varNames() const
    {
        if (m_spec && m_varNames.empty() && !m_spec->variables.empty())
            for (auto const &v : m_spec->variables)
                m_varNames.push_back(v.name);
        return m_varNames;
    }
    std::vector<std::string> const &auxNames() const
    {
        if (m_spec && m_auxNames.empty() && !m_spec->aux.empty())
            for (auto const &a : m_spec->aux)
                m_auxNames.push_back(a.name);
        return m_auxNames;
    }
    std::vector<std::string> const &scalarNames() const
    {
        if (m_spec && m_scalarNames.empty() && !m_spec->scalars.empty())
            for (auto const &s : m_spec->scalars)
                m_scalarNames.push_back(s.name);
        return m_scalarNames;
    }
    /// geom() has no name source yet -- see its doc comment -- so it always
    /// gets this empty table, and a string key reports "no named entries"
    /// rather than resolving anything.
    static std::vector<std::string> const &noNames()
    {
        static const std::vector<std::string> empty;
        return empty;
    }

    State *m_state;
    SystemSpec const *m_spec;
    mutable std::vector<std::string> m_varNames, m_auxNames, m_scalarNames;
};

inline void bindState(py::module_ &m)
{
    py::class_<StateField>(m, "StateField",
                           "One field of a State: indexable by position or by declared name.")
        .def("__getitem__", &StateField::get)
        .def("__setitem__", &StateField::set)
        .def("__len__", &StateField::size)
        .def("__array__",
             [](StateField &f, py::object, py::object copy)
             { return f.asArray(!copy.is_none() && copy.cast<bool>()); },
             py::arg("dtype") = py::none(), py::arg("copy") = py::none())
        .def("__repr__", &StateField::repr);

    py::class_<StateView>(m, "State",
                          "A view of the solution at one point. Valid only inside the hook it "
                          "was passed to.")
        .def_property_readonly("u", &StateView::u, "the variables")
        .def_property_readonly("q", &StateView::q, "d(variable)/dx")
        .def_property_readonly("sigma", &StateView::sigma,
                               "the stored flux, sigma = -sigma_hat")
        .def_property_readonly("sigmaHat", &StateView::sigmaHat,
                               "the physical flux, the quantity SigmaFn returns (read-only)")
        .def_property_readonly("phi", &StateView::phi, "the auxiliary variables")
        .def_property_readonly("geom", &StateView::geom,
                               "the field model's geometry (derived, not an unknown)")
        .def_property_readonly("scalars", &StateView::scalars, "the global scalars")
        .def("__repr__", &StateView::repr);
}

#endif // PYSTATE_HPP
