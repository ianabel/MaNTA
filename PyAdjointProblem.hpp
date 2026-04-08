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
    Matrix dGFndp(Index gIndex, DGSoln &y) const override
    {
        // auto const &grid = y.getGrid();
        // auto const &x_vals = y.getBasis().abscissae();
        // auto const &x_wgts = y.getBasis().weights();
        // const size_t n_abscissa = x_vals.size();
        const auto states = y.evalOnNodes();
        const auto points = y.getPoints();
        Matrix out;

        // If parameters are spatial, int dg/dp dx = int dg/dp_cell delta(x - x_cell) dx, so we just return dg/dp evaluated at the nodes. Otherwise, we need to integrate dg/dp over the domain to get the total sensitivity with respect to that parameter.
        if (areParametersSpatial())
        {
            return dgFndp(gIndex, states, points);
            // out.resize(1, np * y.getGrid().getNCells());
        }
        else
        {
            out.resize(1, np);
        }

        out.setZero();

        Matrix dgdp = dgFndp(gIndex, states, points);

        for (size_t i = 0; i < y.getGrid().getNCells(); i++)
        {
            const Interval &I = y.getGrid()[i];

            // interpolate dgFndp onto the quadrature points
            // integrate interpolation to get weights
            // compute integral as sum dgFndp * weights
            const auto k = y.getBasis().Order();

            const auto ind = Eigen::seq(i * (k + 1), (i + 1) * (k + 1) - 1);

            const auto weights = y.getBasis().getIntegrationWeights(I);
            for (Index p = 0; p < np; p++)
            {
                const Vector dgdp_cellwise = dgdp(p, ind);
                auto coeffs = y.getBasis().InterpolateOntoBasis(I, dgdp_cellwise);

                out(p) += coeffs.dot(weights);
            }
        }

        return out;
    }
    Value gFn(Index i, const State &s, Position x) const override
    {
        PYBIND11_OVERRIDE_PURE(Value, AdjointProblem, gFn, i, s, x);
    };

    Matrix dgFndp(Index i, const GlobalState &states, std::vector<Position> const &abscissae) const override
    {

        std::string method_name = "dgFndp";
        py::gil_scoped_acquire gil;
        py::function _override = py::get_override(this, method_name.c_str());

        if (!_override)
        {
            throw std::runtime_error("Vectorized function \"dSigma\" not found in Python subclass");
            // std::cerr << "WARNING: Vectorized function \"dSigma\" not found in Python subclass" << std::endl;
            // TransportSystem::dSigma(i, out, states, abscissae, time);
            // return;
        }
        auto out = _override(i, states, abscissae).cast<Matrix>();
        return out;

        // PYBIND11_OVERRIDE_PURE(GlobalState, AdjointProblem, dgFndp, i, s, x);
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
    using AdjointProblem::ng;
    using AdjointProblem::np;
    using AdjointProblem::np_boundary;

    using AdjointProblem::spatialParameters;

private:
    bool initialized = false;
    std::map<std::string, py::function> method_overrides;
};

#endif // PYADJOINTPROBLEM_HPP