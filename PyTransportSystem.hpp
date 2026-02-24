#ifndef PYTRANSPORTSYSTEM_HPP
#define PYTRANSPORTSYSTEM_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "TransportSystem.hpp"

namespace py = pybind11;

class PyTransportSystem : public TransportSystem, public py::trampoline_self_life_support
{
public:
	using TransportSystem::TransportSystem;

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
		method_overrides.insert(std::make_pair("SigmaFn", make_override("SigmaFn")));
		method_overrides.insert(std::make_pair("Sources", make_override("Sources")));
		method_overrides.insert(std::make_pair("dSigmaFn_du", make_override("dSigmaFn_du")));
		method_overrides.insert(std::make_pair("dSigmaFn_dq", make_override("dSigmaFn_dq")));
		method_overrides.insert(std::make_pair("dSources_dq", make_override("dSources_dq")));
		method_overrides.insert(std::make_pair("dSources_du", make_override("dSources_du")));
		method_overrides.insert(std::make_pair("dSources_dsigma", make_override("dSources_dsigma")));
		if (nAux > 0)
		{
			method_overrides.insert(std::make_pair("AuxGPrime", make_override("AuxGPrime")));
			method_overrides.insert(std::make_pair("dSources_dPhi", make_override("dSources_dPhi")));
			method_overrides.insert(std::make_pair("dSigma_dPhi", make_override("dSigma_dPhi")));
		}

		initialized = true;
	}
	// PyTransportSystem(TransportSystem &&base) : TransportSystem(std::move(base)) {}
	Value LowerBoundary(Index i, Time t) const override { PYBIND11_OVERRIDE(Value, TransportSystem, LowerBoundary, i, t); };
	Value UpperBoundary(Index i, Time t) const override { PYBIND11_OVERRIDE(Value, TransportSystem, UpperBoundary, i, t); };
	bool isLowerBoundaryDirichlet(Index i) const override { PYBIND11_OVERRIDE(Value, TransportSystem, isLowerBoundaryDirichlet, i); };
	bool isUpperBoundaryDirichlet(Index i) const override { PYBIND11_OVERRIDE(Value, TransportSystem, isUpperBoundaryDirichlet, i); };

	// The guts of the physics problem (these are non-const as they
	// are allowed to alter internal state such as to store computations
	// for future calls)
	Value SigmaFn(Index i, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		return method_overrides["SigmaFn"](i, s, x, t).cast<Value>();
	};
	Values SigmaFn(Index i, std::vector<State> const &states, std::vector<Position> const &abscissae, Time time) override
	{
		py::gil_scoped_acquire gil;

		auto _override = py::get_override(this, "SigmaFn_v");
		if (_override)
		{
			return _override(i, states, abscissae, time).cast<Values>();
		}
		else
		{
			throw std::runtime_error("Attempted to call method SigmaFn_v not overridden in Python subclass");
		}
	};
	Value Sources(Index i, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		return method_overrides["Sources"](i, s, x, t).cast<Value>();
	};

	Values Sources(Index i, std::vector<State> const &states, std::vector<Position> const &abscissae, Time time) override
	{
		py::gil_scoped_acquire gil;

		auto _override = py::get_override(this, "Sources_v");
		if (_override)
		{
			return _override(i, states, abscissae, time).cast<Values>();
		}
		else
		{
			throw std::runtime_error("Attempted to call method Sources_v not overridden in Python subclass");
		}
	};
	void dSigmaFn_du(Index i, VectorRef out, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		out = method_overrides["dSigmaFn_du"](i, s, x, t).cast<Values>();
	};
	void dSigmaFn_dq(Index i, VectorRef out, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		out = method_overrides["dSigmaFn_dq"](i, s, x, t).cast<Values>();
	};

	void dSources_du(Index i, VectorRef v, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		v = method_overrides["dSources_du"](i, s, x, t).cast<Values>();
	};

	void dSources_dq(Index i, VectorRef v, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		v = method_overrides["dSources_dq"](i, s, x, t).cast<Values>();
	};

	void dSources_dsigma(Index i, VectorRef v, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		v = method_overrides["dSources_dsigma"](i, s, x, t).cast<Values>();
	};

	void dSigma(std::vector<std::vector<State>> &out, std::vector<State> const &states, std::vector<Position> const &abscissae, Time time) override
	{
		std::string method_name = "dSigma";
		py::gil_scoped_acquire gil;
		py::function _override = py::get_override(this, method_name.c_str());

		if (!_override)
		{
			throw std::runtime_error(std::string("Virtual method ") + method_name + " not overridden in Python subclass");
		}

		for (Index i = 0; i < nVars; ++i)
		{
			auto &out_var = out[i];
			auto retval = _override(i, states, abscissae, time).cast<std::vector<py::dict>>();
			for (size_t j = 0; j < states.size(); ++j)
			{
				out_var.emplace_back(nVars, nScalars, nAux);
				out_var[j].Variable = retval[j]["Variable"].cast<Vector>();
				out_var[j].Derivative = retval[j]["Derivative"].cast<Vector>();
			}
		}
	};

	void dSources(std::vector<std::vector<State>> &out, std::vector<State> const &states, std::vector<Position> const &abscissae, Time time) override
	{
		std::string method_name = "dSources";
		py::gil_scoped_acquire gil;
		py::function _override = py::get_override(this, method_name.c_str());

		if (!_override)
		{
			throw std::runtime_error(std::string("Virtual method ") + method_name + " not overridden in Python subclass");
		}

		for (Index i = 0; i < nVars; ++i)
		{
			auto &out_var = out[i];
			auto retval = _override(i, states, abscissae, time).cast<std::vector<py::dict>>();
			for (size_t j = 0; j < states.size(); ++j)
			{
				out_var.emplace_back(nVars, nScalars, nAux);
				out_var[j].Variable = retval[j]["Variable"].cast<Vector>();
				out_var[j].Derivative = retval[j]["Derivative"].cast<Vector>();
				out_var[j].Flux = retval[j]["Flux"].cast<Vector>();
				out_var[j].Aux = retval[j]["Aux"].cast<Vector>();
				out_var[j].Scalars = retval[j]["Scalars"].cast<Vector>();
			}
		}
	};

	// Finally one has to provide initial conditions for u & q
	Value InitialValue(Index i, Position x) const override
	{
		PYBIND11_OVERRIDE_PURE(Value, TransportSystem, InitialValue, i, x);
	};

	Value InitialDerivative(Index i, Position x) const override
	{
		PYBIND11_OVERRIDE_PURE(Value, TransportSystem, InitialDerivative, i, x);
	};

	Value InitialAuxValue(Index i, Position x) const override
	{
		PYBIND11_OVERRIDE(Value, TransportSystem, InitialAuxValue, i, x);
	}

	Value AuxG(Index i, const State &s, Position x, Time t) override
	{
		PYBIND11_OVERRIDE(Value, TransportSystem, AuxG, i, s, x, t);
	}

	void AuxGPrime(Index i, State &out, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();

		out = method_overrides["AuxGPrime"](i, s, x, t).cast<State>();
	}

	void dSources_dPhi(Index i, VectorRef v, const State &s, Position x, Time t) override
	{
		if (nAux == 0)
		{
			v.setZero();
			return;
		}
		if (!initialized)
			initializeOverrides();
		v = method_overrides["dSources_dPhi"](i, s, x, t).cast<Values>();
	}

	void dSigma_dPhi(Index i, VectorRef v, const State &s, Position x, Time t) override
	{
		if (nAux == 0)
		{
			v.setZero();
			return;
		}
		if (!initialized)
			initializeOverrides();
		v = method_overrides["dSigma_dPhi"](i, s, x, t).cast<Values>();
	}

	std::unique_ptr<AdjointProblem> createAdjointProblem() override
	{
		PYBIND11_OVERRIDE(std::unique_ptr<AdjointProblem>, TransportSystem, createAdjointProblem);
	}

public:
	using TransportSystem::isLowerDirichlet;
	using TransportSystem::isUpperDirichlet;
	using TransportSystem::nAux;
	using TransportSystem::nVars;

private:
	bool initialized = false;
	std::map<std::string, py::function> method_overrides;
};

#endif // PYTRANSPORTSYSTEM_HPP
