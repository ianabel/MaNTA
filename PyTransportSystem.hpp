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
	Value Sources(Index i, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		return method_overrides["Sources"](i, s, x, t).cast<Value>();
	};
	void dSigmaFn_du(Index i, Values &out, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		out = method_overrides["dSigmaFn_du"](i, s, x, t).cast<Values>();
	};
	void dSigmaFn_dq(Index i, Values &out, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		out = method_overrides["dSigmaFn_dq"](i, s, x, t).cast<Values>();
	};

	void dSources_du(Index i, Values &v, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		v = method_overrides["dSources_du"](i, s, x, t).cast<Values>();
	};

	void dSources_dq(Index i, Values &v, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		v = method_overrides["dSources_dq"](i, s, x, t).cast<Values>();
	};

	void dSources_dsigma(Index i, Values &v, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		v = method_overrides["dSources_dsigma"](i, s, x, t).cast<Values>();
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

	void dSources_dPhi(Index i, Values &v, const State &s, Position x, Time t) override
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

	void dSigma_dPhi(Index i, Values &v, const State &s, Position x, Time t) override
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
