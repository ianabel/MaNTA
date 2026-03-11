#ifndef PYADJOINTRPOBLEM_HPP
#define PYADJOINTRPOBLEM_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <boost/math/quadrature/gauss.hpp>

#include "AdjointProblem.hpp"

namespace py = pybind11;

class PyAdjointProblem : public AdjointProblem, public py::trampoline_self_life_support
{
public:
    using AdjointProblem::AdjointProblem;

    void initializeOverrides()
    {
        auto make_override = [this](const char *method_name)
        {
            py::gil_scoped_acquire gil;
            py::function _override = py::get_override(this, method_name);

            if (_override)
            {
                return _override;
            }
            else
            {
                throw std::runtime_error(std::string("Pure virtual method ") + method_name + " not overridden in Python subclass");
            }
        };
        method_overrides.insert(std::make_pair("dgFn_dphi", make_override("dgFn_dphi")));
        method_overrides.insert(std::make_pair("dAux_dp", make_override("dAux_dp")));

        initialized = true;
    }
    // PyTransportSystem(TransportSystem &&base) : Tr

    using IntegratorType = boost::math::quadrature::gauss<double, 30>;
    static IntegratorType integrator;

    // We don't have the DGSoln object in Python, so we implement GFn and dGFndp here
    Value GFn(Index i, DGSoln &y) const override
    {
        auto g_wrapper = [&](const DGSoln &y, Position x)
        {
            State s = y.eval(x);
            return gFn(i, s, x);
        };
        return y.EvaluateIntegral(g_wrapper);
    };
    Value dGFndp(Index i, Index pIndex, DGSoln &y) const override
    {
        throw std::runtime_error("Non-vectorized version of function \"dGFndp\" depracated.");
    };
    Values dGFndp(Index gIndex, DGSoln &y) const override
    {
        auto const &grid = y.getGrid();
        auto const &x_vals = y.getBasis().abscissae();
        auto const &x_wgts = y.getBasis().weights();
        const size_t n_abscissa = x_vals.size();

        Values out(np);
        out.setZero();

        for (size_t i = 0; i < grid.getNCells(); i++)
        {
            const auto &I = grid[i];
            for (size_t j = 0; j < n_abscissa; ++j)
            {
                // Pull the loop over the gaussian integration points
                // outside so we can evaluate u, q, dX_dZ once and store the values

                // All for loops inside here can be parallelised as they all
                // write to separate entries in mat

                double wgt = x_wgts[i] * (I.h() / 2.0);

                double y_plus = I.x_l + (1.0 + x_vals[i]) * (I.h() / 2.0);
                double y_minus = I.x_l + (1.0 - x_vals[i]) * (I.h() / 2.0);
                State Y_plus = y.eval(y_plus), Y_minus = y.eval(y_minus);

                Values dgdp_plus = dgFndp(gIndex, Y_plus, y_plus);
                Values dgdp_minus = dgFndp(gIndex, Y_minus, y_minus);

                out += wgt * dgdp_plus + wgt * dgdp_minus;
            }
        }

        return out;
    }
    Value gFn(Index i, const State &s, Position x) const override
    {
        PYBIND11_OVERRIDE_PURE(Value, AdjointProblem, gFn, i, s, x);
    };

    Values dgFndp(Index i, const State &s, Position x) const override
    {
        PYBIND11_OVERRIDE_PURE(Values, AdjointProblem, dgFndp, i, s, x);
    };

    void dgFn_du(Index i, VectorRef out, const State &s, Position x) override
    {
        throw std::runtime_error("Individual derivative function \"dgFn_du\" depracated; use vectorized version dg instead.");
        // if (!initialized)
        //     initializeOverrides();
        // out = method_overrides["dgFn_du"](i, s, x).cast<Values>();
    };
    void dgFn_dq(Index i, VectorRef out, const State &s, Position x) override
    {
        throw std::runtime_error("Individual derivative function \"dgFn_dq\" depracated; use vectorized version dg instead.");
        // if (!initialized)
        //     initializeOverrides();
        // out = method_overrides["dgFn_dq"](i, s, x).cast<Values>();
    };
    void dgFn_dsigma(Index i, VectorRef out, const State &s, Position x) override
    {
        throw std::runtime_error("Individual derivative function \"dgFn_dsigma\" depracated; use vectorized version dg instead.");
        // if (!initialized)
        //     initializeOverrides();
        // out = method_overrides["dgFn_dsigma"](i, s, x).cast<Values>();
    };
    void dgFn_dphi(Index i, VectorRef out, const State &s, Position x) override
    {
        if (!initialized)
            initializeOverrides();
        out = method_overrides["dgFn_dphi"](i, s, x).cast<Values>();
    };

    void dg(Index gIndex, GlobalState &out, GlobalState const &states, std::vector<Position> const &abscissae) override
    {
        std::string method_name = "dg";
        py::gil_scoped_acquire gil;
        py::function _override = py::get_override(this, method_name.c_str());

        if (!_override)
        {
            throw std::runtime_error("Vectorized function \"dg\" not found in Python subclass");
            // std::cerr << "WARNING: Vectorized function \"dSigma\" not found in Python subclass" << std::endl;
            // TransportSystem::dSigma(i, out, states, abscissae, time);
            // return;
        }

        out = _override(gIndex, states, abscissae).cast<GlobalState>();
    };

    void dSigmaFn_dp(Index i, Index pIndex, Value &out, const State &s, Position x) override
    {
        throw std::runtime_error("Individual derivative functions depracated; use vectorized version dSigma instead.");
        // if (!initialized)
        //     initializeOverrides();
        // out = method_overrides["dSigmaFn_dp"](i, pIndex, s, x).cast<Value>();
    };

    void dSources_dp(Index i, Index pIndex, Value &out, const State &s, Position x) override
    {
        throw std::runtime_error("Individual derivative functions depracated; use vectorized version dSources instead.");
        // if (!initialized)
        //     initializeOverrides();
        // out = method_overrides["dSources_dp"](i, pIndex, s, x).cast<Value>();
    };

    void dSigma(Index i, GlobalState &out, GlobalState const &states, std::vector<Position> const &abscissae) override
    {
        std::string method_name = "dSigma";
        py::gil_scoped_acquire gil;
        py::function _override = py::get_override(this, method_name.c_str());

        if (!_override)
        {
            throw std::runtime_error("Vectorized function \"dSigma\" not found in Python subclass");
            // std::cerr << "WARNING: Vectorized function \"dSigma\" not found in Python subclass" << std::endl;
            // TransportSystem::dSigma(i, out, states, abscissae, time);
            // return;
        }

        out.Variable() = _override(i, states, abscissae).cast<Matrix>();
    };

    void dSources(Index i, GlobalState &out, GlobalState const &states, std::vector<Position> const &abscissae) override
    {
        std::string method_name = "dSources";
        py::gil_scoped_acquire gil;
        py::function _override = py::get_override(this, method_name.c_str());

        if (!_override)
        {
            throw std::runtime_error("Vectorized function \"dSources\" not found in Python subclass");
            // std::cerr << "WARNING: Vectorized function \"dSigma\" not found in Python subclass" << std::endl;
            // TransportSystem::dSigma(i, out, states, abscissae, time);
            // return;
        }

        out.Variable() = _override(i, states, abscissae).cast<Matrix>();
    };

    void dAux_dp(Index i, Index pIndex, Value &out, const State &s, Position x) override
    {

        if (!initialized)
            initializeOverrides();
        out = method_overrides["dAux_dp"](i, pIndex, s, x).cast<Value>();
    }

    std::string getName(Index pIndex) const override
    {
        PYBIND11_OVERRIDE(std::string, AdjointProblem, getName, pIndex);
    };

    bool computeUpperBoundarySensitivity(Index i, Index pIndex) override
    {
        PYBIND11_OVERRIDE(bool, AdjointProblem, computeUpperBoundarySensitivity, i, pIndex);
    };
    bool computeLowerBoundarySensitivity(Index i, Index pIndex) override
    {
        PYBIND11_OVERRIDE(bool, AdjointProblem, computeLowerBoundarySensitivity, i, pIndex);
    };

public:
    using AdjointProblem::np;
    using AdjointProblem::np_boundary;

private:
    bool initialized = false;
    std::map<std::string, py::function> method_overrides;
};

#endif // PYADJOINTPROBLEM_HPP