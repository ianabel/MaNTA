#ifndef PYTRANSPORTSYSTEM_HPP
#define PYTRANSPORTSYSTEM_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <string_view>

#include "TransportSystem.hpp"

constexpr std::array<std::string_view, 7> required_method_names = {"SigmaFn", "Sources", "dSigmaFn_du", "dSigmaFn_dq", "dSources_du", "dSources_dq", "dSources_dsigma"};
constexpr std::array<std::string_view, 4> required_method_names_vectorized = {"SigmaFn_v", "Sources_v", "dSigma", "dSources"};

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
			return _override;
		};

		std::vector<std::string_view> missing_methods;
		std::vector<std::string_view> missing_vectorized_methods;

		for (const auto &method_name : required_method_names)
		{
			auto _override = make_override(method_name.data());
			if (!_override)
			{
				missing_methods.push_back(method_name);
			}
			else
			{
				method_overrides.insert(std::make_pair(method_name, std::move(_override)));
			}
		}

		for (const auto &method_name : required_method_names_vectorized)
		{
			auto _override = make_override(method_name.data());
			if (!_override)
			{
				missing_methods.push_back(method_name);
			}
			else
			{
				method_overrides.insert(std::make_pair(method_name, std::move(_override)));
			}
		}

		bool non_vectorized = missing_methods.empty();
		bool vectorized = missing_vectorized_methods.empty();

		if (!vectorized || !non_vectorized)
		{
			if (vectorized || non_vectorized)
			{
				// do nothing
			}
			else
			{
				std::string error_message = "The following required methods are missing in the Python subclass:\n";
				error_message += "Non-vectorized methods:\n";
				for (const auto &method_name : missing_methods)
				{
					error_message += std::string(method_name) + "\n";
				}
				error_message += "Vectorized methods:\n";
				for (const auto &method_name : missing_vectorized_methods)
				{
					error_message += std::string(method_name) + "\n";
				}
				error_message += "MaNTA requires either all vectorized or all non-vectorized methods to be implemented. Please implement the missing methods and try again.\n";
				throw std::runtime_error(error_message);
			}
		}

		if (nAux > 0)
		{
			method_overrides.insert(std::make_pair("AuxGPrime", make_override("AuxGPrime")));
			method_overrides.insert(std::make_pair("dSources_dPhi", make_override("dSources_dPhi")));
			method_overrides.insert(std::make_pair("dSigma_dPhi", make_override("dSigma_dPhi")));
		}

		initialized = true;
	}

	Value LowerBoundary(Index i, Time t) const override { PYBIND11_OVERRIDE(Value, TransportSystem, LowerBoundary, i, t); };
	Value UpperBoundary(Index i, Time t) const override { PYBIND11_OVERRIDE(Value, TransportSystem, UpperBoundary, i, t); };
	bool isLowerBoundaryDirichlet(Index i) const override { PYBIND11_OVERRIDE(Value, TransportSystem, isLowerBoundaryDirichlet, i); };
	bool isUpperBoundaryDirichlet(Index i) const override { PYBIND11_OVERRIDE(Value, TransportSystem, isUpperBoundaryDirichlet, i); };

	Value SigmaFn(Index i, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		try
		{
			return method_overrides["SigmaFn"](i, s, x, t).cast<Value>();
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error(std::string("Error occurred when trying to calculate SigmaFn: ") + e.what());
		}
	};
	Values SigmaFn(Index i, GlobalState const &states, std::vector<Position> const &abscissae, Time time) override
	{
		if (!initialized)
			initializeOverrides();
		try
		{
			return method_overrides["SigmaFn_v"](i, states, abscissae, time).cast<Values>();
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error(std::string("Error occurred when trying to calculate SigmaFn: ") + e.what());
		}
	};

	Value Sources(Index i, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();

		try
		{
			return method_overrides["Sources"](i, s, x, t).cast<Value>();
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error(std::string("Error occurred when trying to calculate Sources: ") + e.what());
		}
	};

	Values Sources(Index i, GlobalState const &states, std::vector<Position> const &abscissae, Time time) override
	{
		if (!initialized)
			initializeOverrides();

		try
		{
			return method_overrides["Sources_v"](i, states, abscissae, time).cast<Values>();
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error(std::string("Error occurred when trying to calculate Sources: ") + e.what());
		}
	};

	void dSigmaFn_du(Index i, VectorRef out, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();

		try
		{
			out = method_overrides["dSigmaFn_du"](i, s, x, t).cast<Values>();
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error(std::string("Error occurred when trying to calculate dSigmaFn_du: ") + e.what());
		}
	};
	void dSigmaFn_dq(Index i, VectorRef out, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		try
		{
			out = method_overrides["dSigmaFn_dq"](i, s, x, t).cast<Values>();
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error(std::string("Error occurred when trying to calculate dSources_dq: ") + e.what());
		}
	};

	void dSources_du(Index i, VectorRef v, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();

		try
		{
			v = method_overrides["dSources_du"](i, s, x, t).cast<Values>();
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error(std::string("Error occurred when trying to calculate dSources_du: ") + e.what());
		}
	};

	void dSources_dq(Index i, VectorRef v, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();

		try
		{
			v = method_overrides["dSources_dq"](i, s, x, t).cast<Values>();
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error(std::string("Error occurred when trying to calculate dSources_dq: ") + e.what());
		}
	};

	void dSources_dsigma(Index i, VectorRef v, const State &s, Position x, Time t) override
	{
		if (!initialized)
			initializeOverrides();
		try
		{
			v = method_overrides["dSources_dsigma"](i, s, x, t).cast<Values>();
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error(std::string("Error occurred when trying to calculate dSources_dsigma: ") + e.what());
		}
	};

	void dSigma(Index i, GlobalState &out, GlobalState const &states, std::vector<Position> const &abscissae, Time time) override
	{
		if (!initialized)
			initializeOverrides();
		try
		{
			out = method_overrides["dSigma"](i, states, abscissae, time).cast<GlobalState>();
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error(std::string("Error occurred when trying to calculate dSigma: ") + e.what());
		}
	};

	void dSources(Index i, GlobalState &out, GlobalState const &states, std::vector<Position> const &abscissae, Time time) override
	{
		if (!initialized)
			initializeOverrides();
		try
		{
			out = method_overrides["dSigma"](i, states, abscissae, time).cast<GlobalState>();
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error(std::string("Error occurred when trying to calculate dSources: ") + e.what());
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
		py::gil_scoped_acquire gil;
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
		py::gil_scoped_acquire gil;
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
		py::gil_scoped_acquire gil;
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
