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
        method_overrides.insert(std::make_pair("dgFn_du", make_override("dgFn_du")));
        method_overrides.insert(std::make_pair("dgFn_dq", make_override("dgFn_dq")));
        method_overrides.insert(std::make_pair("dgFn_dsigma", make_override("dgFn_dsigma")));
        method_overrides.insert(std::make_pair("dgFn_dphi", make_override("dgFn_dphi")));
        method_overrides.insert(std::make_pair("dSigmaFn_dp", make_override("dSigmaFn_dp")));
        method_overrides.insert(std::make_pair("dSources_dp", make_override("dSources_dp")));

        initialized = true;
    }
    // PyTransportSystem(TransportSystem &&base) : Tr

    using IntegratorType = boost::math::quadrature::gauss<double, 30>;
    static IntegratorType integrator;

    Value GFn(Index i, DGSoln &y) const override
    {
        auto g_wrapper = [&](const DGSoln &y, Position x)
        {
            State s = y.eval(x);
            return gFn(i, s, x);
        };
        return y.EvaluateIntegral(g_wrapper);
    };
    Value dGFndp(Index i, DGSoln &y) const override
    {
        auto g_wrapper = [&](const DGSoln &y, Position x)
        {
            State s = y.eval(x);
            return dgFndp(i, s, x);
        };
        return y.EvaluateIntegral(g_wrapper);
    };
    Value gFn(Index i, const State &s, Position x) const override
    {
        PYBIND11_OVERRIDE_PURE(Value, AdjointProblem, gFn, i, s, x);
    };

    virtual Value dgFndp(Index i, const State &s, Position x) const
    {
        PYBIND11_OVERRIDE_PURE(Value, PyAdjointProblem, dgFndp, i, s, x);
    };

    void dgFn_du(Index i, Values &out, const State &s, Position x) override
    {
        if (!initialized)
            initializeOverrides();
        out = method_overrides["dgFn_du"](i, s, x).cast<Values>();
    };
    void dgFn_dq(Index i, Values &out, const State &s, Position x) override
    {
        if (!initialized)
            initializeOverrides();
        out = method_overrides["dgFn_dq"](i, s, x).cast<Values>();
    };
    void dgFn_dsigma(Index i, Values &out, const State &s, Position x) override
    {
        if (!initialized)
            initializeOverrides();
        out = method_overrides["dgFn_dsigma"](i, s, x).cast<Values>();
    };
    void dgFn_dphi(Index i, Values &out, const State &s, Position x) override
    {
        if (!initialized)
            initializeOverrides();
        out = method_overrides["dgFn_dphi"](i, s, x).cast<Values>();
    };
    void dSigmaFn_dp(Index i, Index pIndex, Value &out, const State &s, Position x) override
    {
        if (!initialized)
            initializeOverrides();
        out = method_overrides["dSigmaFn_dp"](i, pIndex, s, x).cast<Value>();
    };
    void dSources_dp(Index i, Index pIndex, Value &out, const State &s, Position x) override
    {
        if (!initialized)
            initializeOverrides();
        out = method_overrides["dSources_dp"](i, pIndex, s, x).cast<Value>();
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