#ifndef PYEQUILIBRIUMMODEL_HPP
#define PYEQUILIBRIUMMODEL_HPP

#include "EquilibriumModel.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

/*
    An EquilibriumModel written in Python: tiers 0 and 1 of
    docs/superpowers/specs/2026-10-23-field-coupling-tiers-design.md, which a
    black-box equilibrium code driven from Python can supply.

    A subclass defines

      Solve(states, points, weights, t, warm) -> any object
      Geometry(equilibrium, points)           -> (nGeometry, len(points)) array

    and optionally applyResponse / applyResponseTranspose (tier 1, declared
    with response = EquilibriumResponse.Model), responseStep, setTolerances and
    resetForRun. `states` crosses as the GlobalState dict, (nPoints, nVars)
    arrays like a vectorised case's; the object Solve returns is opaque to the
    solver, held in a PyEquilibrium and handed back as `warm` and to Geometry.

    The solver calls these with the GIL held -- PyRunner never releases it --
    and an exception from Solve is what it is from a physics hook: inside a
    residual, a recoverable failure the integrator answers by cutting its step.
 */
class PyEquilibrium : public Equilibrium
{
public:
    explicit PyEquilibrium(py::object v) : value(std::move(v)) {}
    ~PyEquilibrium() override
    {
        // The last reference may be dropped from a C++ frame that does not
        // hold the GIL; releasing a Python object there is undefined.
        py::gil_scoped_acquire gil;
        value = py::object();
    }
    py::object value;
};

class PyEquilibriumModel : public EquilibriumModel, public py::trampoline_self_life_support
{
public:
    explicit PyEquilibriumModel(EquilibriumSpec spec) : EquilibriumModel(std::move(spec)) {}

    std::unique_ptr<Equilibrium> Solve(GlobalState const &states, std::vector<Position> const &points,
                                       Vector const &weights, Time t, Equilibrium const *warm) override
    {
        py::gil_scoped_acquire gil;
        py::function f = required("Solve");
        py::object w = py::none();
        if (warm)
            w = held(*warm);
        return std::make_unique<PyEquilibrium>(f(states, asArray(points), weights, t, w));
    }

    void Geometry(MatrixRef g, Equilibrium const &e, std::vector<Position> const &points) override
    {
        py::gil_scoped_acquire gil;
        py::function f = required("Geometry");
        g = shaped(f(held(e), asArray(points)), g.rows(), g.cols(), "Geometry");
    }

    void applyResponse(MatrixRef dg, Equilibrium const &e, GlobalState const &states, GlobalState const &dp,
                       std::vector<Position> const &inputPoints, Vector const &weights, Time t,
                       std::vector<Position> const &geometryPoints) override
    {
        {
            py::gil_scoped_acquire gil;
            if (py::function f = py::get_override(this, "applyResponse"))
            {
                dg = shaped(f(held(e), states, dp, asArray(inputPoints), weights, t, asArray(geometryPoints)),
                            dg.rows(), dg.cols(), "applyResponse");
                return;
            }
        }
        EquilibriumModel::applyResponse(dg, e, states, dp, inputPoints, weights, t, geometryPoints);
    }

    void applyResponseTranspose(GlobalState &dp, Equilibrium const &e, GlobalState const &states,
                                Matrix const &dg, std::vector<Position> const &inputPoints,
                                Vector const &weights, Time t,
                                std::vector<Position> const &geometryPoints) override
    {
        {
            py::gil_scoped_acquire gil;
            if (py::function f = py::get_override(this, "applyResponseTranspose"))
            {
                // A dict, the GlobalState caster's layout; only the rows the
                // transpose fills are read back, each checked for shape, since
                // the caster builds a state that knows none of the solver's.
                py::dict out = f(held(e), states, dg, asArray(inputPoints), weights, t, asArray(geometryPoints));
                auto row = [&](char const *key, Matrix &into)
                {
                    if (!out.contains(key))
                        return;
                    into = shaped(out[key], into.cols(), into.rows(), "applyResponseTranspose")
                               .transpose();
                };
                row("Variable", dp.Variable());
                row("Derivative", dp.Derivative());
                row("Flux", dp.Flux());
                row("Aux", dp.Aux());
                return;
            }
        }
        EquilibriumModel::applyResponseTranspose(dp, e, states, dg, inputPoints, weights, t, geometryPoints);
    }

    double responseStep() const override
    {
        PYBIND11_OVERRIDE(double, EquilibriumModel, responseStep, );
    }

    bool setTolerances(double relative, double absolute) override
    {
        PYBIND11_OVERRIDE(bool, EquilibriumModel, setTolerances, relative, absolute);
    }

    void resetForRun() override { PYBIND11_OVERRIDE(void, EquilibriumModel, resetForRun, ); }

private:
    py::function required(char const *name)
    {
        py::function f = py::get_override(this, name);
        if (!f)
            throw std::runtime_error(std::string("manta.EquilibriumModel subclass does not define ") + name);
        return f;
    }

    static py::object held(Equilibrium const &e)
    {
        auto const *p = dynamic_cast<PyEquilibrium const *>(&e);
        if (!p)
            throw std::logic_error("a Python EquilibriumModel was handed an equilibrium it did not make");
        return p->value;
    }

    static py::array_t<double> asArray(std::vector<Position> const &points)
    {
        return py::array_t<double>(static_cast<py::ssize_t>(points.size()), points.data());
    }

    // An array of exactly (rows, cols), or a message naming the hook. Eigen's
    // own resize would accept the transpose of a square one silently.
    static Matrix shaped(py::handle h, Index rows, Index cols, char const *hook)
    {
        auto m = py::cast<Matrix>(h);
        if (m.rows() != rows || m.cols() != cols)
            throw std::invalid_argument(std::string(hook) + " returned an array of shape (" +
                                        std::to_string(m.rows()) + ", " + std::to_string(m.cols()) +
                                        "); expected (" + std::to_string(rows) + ", " +
                                        std::to_string(cols) + ")");
        return m;
    }
};

#endif // PYEQUILIBRIUMMODEL_HPP
